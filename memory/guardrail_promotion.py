"""
guardrail_promotion.py — Detect facts that should be promoted to guardrails.

A fact is a guardrail candidate when it:
  1. Contains directive language (always, never, don't, must, do not)
  2. Has been reinforced across 3+ sessions (high conviction)
  3. Has importance >= 7
  4. Is NOT already covered by an existing guardrail (cosine > 0.85)
"""
from __future__ import annotations

import re
from typing import Optional

from . import db as memdb
from .config import DEDUP_THRESHOLD


# ── Directive language patterns ───────────────────────────────────────────

_DIRECTIVE_PATTERNS = [
    re.compile(r"\b(?:always|never|must|shall|do\s+not|don'?t|cannot|should\s+not|shouldn'?t)\b", re.I),
    re.compile(r"\b(?:required|forbidden|prohibited|mandatory|critical)\b", re.I),
    re.compile(r"\b(?:no\s+exceptions?|without\s+exception|every\s+time)\b", re.I),
]

_MIN_SESSION_COUNT = 3
_MIN_IMPORTANCE = 7


def _is_directive(text: str) -> bool:
    """Check if text contains directive/rule-like language."""
    return any(p.search(text) for p in _DIRECTIVE_PATTERNS)


def detect_guardrail_candidates(
    conn,
    scope: Optional[str] = None,
    min_session_count: int = _MIN_SESSION_COUNT,
    min_importance: int = _MIN_IMPORTANCE,
) -> list[dict]:
    """
    Find facts that should be promoted to guardrails.

    Returns list of candidate dicts with: id, text, session_count, importance, category, scope.
    """
    scope_sql, scope_params = "", []
    if scope:
        scope_sql = " AND (scope = ? OR scope = '__global__')"
        scope_params = [scope]

    # Get high-conviction facts
    try:
        rows = conn.execute(f"""
            SELECT id, text, session_count, importance, category, scope, embedding
            FROM facts
            WHERE is_active = TRUE
              AND session_count >= ?
              AND importance >= ?
              {scope_sql}
            ORDER BY importance DESC, session_count DESC
        """, [min_session_count, min_importance] + scope_params).fetchall()
    except Exception:
        return []

    candidates = []
    for fid, text, scount, imp, cat, fscope, emb in rows:
        # Must contain directive language
        if not _is_directive(text):
            continue

        # Check if already covered by an existing guardrail
        if emb and _has_matching_guardrail(conn, emb):
            continue

        candidates.append({
            "id": fid,
            "text": text,
            "session_count": scount,
            "importance": imp,
            "category": cat or "",
            "scope": fscope or "__global__",
        })

    return candidates


def _has_matching_guardrail(conn, embedding: list[float], threshold: float = 0.85) -> bool:
    """Check if there's already a guardrail with high cosine similarity to this embedding."""
    try:
        results = memdb.search_guardrails(conn, embedding, limit=1, threshold=threshold)
        return len(results) > 0
    except Exception:
        return False


def format_guardrail_proposals(candidates: list[dict]) -> str:
    """Format candidates as a markdown proposal for the user."""
    if not candidates:
        return "No guardrail candidates found. All high-conviction directive facts are already covered."

    lines = [
        f"## Proposed Guardrails ({len(candidates)} candidates)\n",
        "These facts appear repeatedly as rules/directives and may warrant promotion to guardrails.\n",
    ]

    for i, c in enumerate(candidates, 1):
        scope_label = c["scope"].split("/")[-1] if "/" in c["scope"] else c["scope"]
        lines.append(
            f"**{i}. {c['text']}**\n"
            f"   Seen in {c['session_count']} sessions | importance: {c['importance']} | "
            f"category: {c['category']} | scope: {scope_label}\n"
            f"   → To promote: `/remember guardrail: {c['text']}`\n"
        )

    lines.append("---")
    lines.append("To promote a fact to a guardrail, use `/remember guardrail: <text>`")
    return "\n".join(lines)
