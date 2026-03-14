"""
decay.py — Temporal scoring for the memory system.

Two mechanisms work together:
  1. LLM assigns an initial temporal_class (short / medium / long) at extraction.
  2. Recency decay continuously adjusts decay_score (0.0–1.0) based on how
     long ago and how often each item was seen. Items can be promoted to longer
     temporal classes but are never demoted (only forgotten when score → 0).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Union

from .config import DECAY_RATES, FORGET_THRESHOLD


def _to_aware(dt: Union[datetime, str, None]) -> datetime:
    """Normalise a datetime to UTC-aware."""
    if dt is None:
        return datetime.now(timezone.utc)
    if isinstance(dt, str):
        # DuckDB may return ISO strings
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_decay_score(
    last_seen_at: Union[datetime, str, None],
    session_count: int,
    temporal_class: str,
) -> float:
    """
    Score = exp(-effective_rate * days_since_last_seen)

    The effective rate is divided by a reinforcement factor derived from
    how many sessions this item has survived, making older survivors more
    resistant to decay.  A brand-new item (age=0) always scores 1.0.

    Returns a float in [0.0, 1.0].
    """
    now = datetime.now(timezone.utc)
    last = _to_aware(last_seen_at)
    days_old = max(0.0, (now - last).total_seconds() / 86_400)

    base_rate = DECAY_RATES.get(temporal_class, DECAY_RATES["medium"])

    # Each additional session halves the effective decay rate (capped at 10x)
    reinforcement = min(10.0, 1.0 + 0.5 * max(0, session_count - 1))
    effective_rate = base_rate / reinforcement

    return min(1.0, math.exp(-effective_rate * days_old))


def should_forget(
    decay_score: float,
    temporal_class: str,
) -> bool:
    """
    Only short-term items can be auto-forgotten when their score collapses.
    Medium and long items persist unless explicitly deleted.
    """
    return temporal_class == "short" and decay_score < FORGET_THRESHOLD


def temporal_weight(temporal_class: str, decay_score: float) -> float:
    """
    A single composite recall weight used for ranking.
    Long items get a class-level boost so they surface even when
    their embedding score is slightly lower than a fresh short-term item.
    """
    class_boost = {"long": 1.5, "medium": 1.1, "short": 0.8}
    return decay_score * class_boost.get(temporal_class, 1.0)
