"""
corrections.py — Detect and apply user corrections to previously recalled facts.

Layer 1: Regex/keyword heuristics for definite corrections.
Layer 2: (Future) LLM classification for ambiguous corrections.
"""
from __future__ import annotations

import re
from typing import Optional

from .config import GLOBAL_SCOPE
from .decay import compute_decay_score

# ── Detection patterns ────────────────────────────────────────────────────

_DEFINITE_PATTERNS = [
    re.compile(r"(?i)^correction:\s*(.+)"),
    re.compile(r"(?i)^update\s+memory:\s*(.+)"),
    re.compile(r"(?i)\bthat'?s\s+(?:wrong|incorrect|not\s+right)\b[,.]?\s*(.*)"),
    re.compile(r"(?i)\bno,?\s+it'?s\s+actually\b\s*(.*)"),
    re.compile(r"(?i)\bno,?\s+(?:the\s+)?(?:correct|right)\s+(?:answer|thing|fact|info)\s+is\b\s*(.*)"),
    re.compile(r"(?i)\bactually,?\s+(?:it'?s|that'?s|the)\s+(?:not|wrong)\b[,.]?\s*(.*)"),
    re.compile(r"(?i)\bthat\s+(?:fact|info|memory)\s+is\s+(?:wrong|incorrect|outdated)\b[,.]?\s*(.*)"),
]

_AMBIGUOUS_PATTERNS = [
    re.compile(r"(?i)^(?:no|actually|wait|not\s+quite)[,.\s]"),
    re.compile(r"(?i)\bI\s+(?:don'?t\s+think|disagree)\b"),
]


def detect_correction(prompt_text: str) -> Optional[dict]:
    """
    Detect if the user's prompt is correcting a previously recalled fact.

    Returns:
        {"type": "definite"|"ambiguous", "trigger": str, "corrected_text": str}
        or None if no correction detected.
    """
    text = prompt_text.strip()
    if not text or len(text) < 5:
        return None

    # Check definite patterns first
    for pattern in _DEFINITE_PATTERNS:
        m = pattern.search(text)
        if m:
            corrected = m.group(1).strip() if m.lastindex else ""
            if not corrected:
                # Try to extract everything after the trigger
                corrected = text[m.end():].strip()
            return {
                "type": "definite",
                "trigger": m.group(0).strip(),
                "corrected_text": corrected or text,
            }

    # Check ambiguous patterns
    for pattern in _AMBIGUOUS_PATTERNS:
        m = pattern.search(text)
        if m:
            return {
                "type": "ambiguous",
                "trigger": m.group(0).strip(),
                "corrected_text": text,
            }

    return None


def resolve_correction(
    prompt_text: str,
    recalled_items: list[dict],
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """
    Determine which recalled item is being corrected and extract the new fact.

    For definite corrections: match by keyword overlap with recalled items.
    For ambiguous corrections: requires LLM (not yet implemented, returns None).

    Returns:
        {"old_item_id": str, "old_table": str, "new_text": str, "confidence": str}
        or None if unable to resolve.
    """
    detection = detect_correction(prompt_text)
    if not detection:
        return None

    if detection["type"] == "ambiguous":
        # TODO: LLM classification for ambiguous corrections
        return None

    # For definite corrections, find the best matching recalled item
    corrected_text = detection["corrected_text"]
    prompt_lower = prompt_text.lower()

    best_match = None
    best_score = 0

    for item in recalled_items:
        item_text = item.get("text", "")
        item_words = set(item_text.lower().split())
        prompt_words = set(prompt_lower.split())

        # Score by word overlap
        overlap = len(item_words & prompt_words)
        # Boost if item text appears as a substring in the correction
        if any(w in prompt_lower for w in item_words if len(w) > 4):
            overlap += 3

        if overlap > best_score:
            best_score = overlap
            best_match = item

    if not best_match or best_score < 2:
        return None

    # Extract the new fact text from the correction
    new_text = _extract_new_fact(prompt_text, best_match.get("text", ""), corrected_text)

    return {
        "old_item_id": best_match["id"],
        "old_table": best_match.get("table", "facts"),
        "new_text": new_text,
        "confidence": "high" if detection["type"] == "definite" else "medium",
    }


def _extract_new_fact(prompt: str, old_text: str, corrected_text: str) -> str:
    """Extract the corrected fact from the user's prompt."""
    # Common patterns: "X is actually Y", "X not Y but Z"
    # If corrected_text is substantial, use it directly
    if corrected_text and len(corrected_text) > 15:
        return corrected_text

    # Try to extract after "actually", "should be", "is really", etc.
    for pattern in [
        re.compile(r"(?i)(?:actually|really|should\s+be|is\s+actually)\s+(.{10,})"),
        re.compile(r"(?i)(?:not\s+\w+\s+but)\s+(.{10,})"),
        re.compile(r"(?i)(?:it'?s|the\s+correct\s+\w+\s+is)\s+(.{10,})"),
    ]:
        m = pattern.search(prompt)
        if m:
            return m.group(1).strip().rstrip(".")

    # Fallback: use the full prompt as the corrected text
    return prompt


def apply_correction(
    conn,
    correction: dict,
    session_id: str,
    scope: str = GLOBAL_SCOPE,
) -> bool:
    """
    Supersede the old item and insert the corrected version.

    Returns True on success.
    """
    from . import db, embeddings

    old_id = correction["old_item_id"]
    old_table = correction["old_table"]
    new_text = correction["new_text"]

    # Supersede the old item
    try:
        db.supersede_item(conn, old_id, old_table, "", f"Corrected by user")
    except Exception:
        return False

    # Insert the corrected version
    try:
        emb = embeddings.embed(new_text)
        if old_table == "facts":
            new_id, _ = db.upsert_fact(
                conn, new_text, "operational", "long", correction.get("confidence", "high"),
                emb, session_id, compute_decay_score, scope=scope,
            )
        elif old_table == "decisions":
            new_id, _ = db.upsert_decision(
                conn, new_text, "long", emb, session_id, compute_decay_score, scope=scope,
            )
        else:
            # Generic: store as fact
            new_id, _ = db.upsert_fact(
                conn, new_text, "operational", "long", "high",
                emb, session_id, compute_decay_score, scope=scope,
            )
        return True
    except Exception:
        return False
