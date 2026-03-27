"""
validation.py — Validate extracted knowledge before storage.

Layer A: Inline rejection of obvious garbage.
Layer B: Flag borderline items for human review.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Rejection patterns ────────────────────────────────────────────────────

_REJECTION_PATTERNS = [
    re.compile(r"^https?://\S+$"),                           # bare URLs
    re.compile(r"^/[a-zA-Z][\w/.-]+$"),                      # bare file paths
    re.compile(r"(?i)^(?:I|we|the assistant)\s+(?:extracted|noted|stored|recorded|learned)"),
    re.compile(r"(?i)^(?:this (?:fact|item|entry) (?:was|is))"),  # meta-commentary
    re.compile(r"(?i)^(?:todo|fixme|hack|xxx)\b"),            # transient markers
    re.compile(r"(?i)^(?:N/A|none|null|undefined|n\.a\.)$"),  # empty values
]

_UNCERTAINTY_MARKERS = [
    "might", "maybe", "possibly", "unclear", "not sure",
    "I think", "probably", "seems like", "appears to",
    "could be", "uncertain", "unconfirmed",
]

MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 500


# ── Validation functions ──────────────────────────────────────────────────

def validate_facts(facts: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Validate a list of extracted facts.

    Returns (accepted, flagged_for_review, rejected).
    """
    accepted = []
    flagged = []
    rejected = []
    seen_texts: set[str] = set()

    for fact in facts:
        text = fact.get("text", "").strip()
        confidence = fact.get("confidence", "medium")
        importance = fact.get("importance", 5)

        # Dedup within batch
        text_lower = text.lower()
        if text_lower in seen_texts:
            rejected.append({**fact, "_rejection_reason": "duplicate_in_batch"})
            continue
        seen_texts.add(text_lower)

        # Length checks
        if len(text) < MIN_TEXT_LENGTH:
            rejected.append({**fact, "_rejection_reason": "too_short"})
            continue
        if len(text) > MAX_TEXT_LENGTH:
            rejected.append({**fact, "_rejection_reason": "too_long"})
            continue

        # Rejection patterns
        is_rejected = False
        for pattern in _REJECTION_PATTERNS:
            if pattern.search(text):
                rejected.append({**fact, "_rejection_reason": "pattern_match"})
                is_rejected = True
                break
        if is_rejected:
            continue

        # Low confidence + low importance → reject
        if confidence == "low" and (importance or 5) < 4:
            rejected.append({**fact, "_rejection_reason": "low_confidence_low_importance"})
            continue

        # Low confidence + high importance → flag for review
        if confidence == "low" and (importance or 5) >= 4:
            flagged.append({
                **fact,
                "reason": "low_confidence_high_importance",
                "item_table": "facts",
            })
            continue

        # Uncertainty markers with medium confidence → flag
        if confidence == "medium":
            text_lower_check = text.lower()
            has_uncertainty = any(marker in text_lower_check for marker in _UNCERTAINTY_MARKERS)
            if has_uncertainty:
                flagged.append({
                    **fact,
                    "reason": "uncertainty_markers",
                    "item_table": "facts",
                })
                continue

        accepted.append(fact)

    return accepted, flagged, rejected


def validate_knowledge(knowledge: dict) -> tuple[dict, list[dict]]:
    """
    Validate all extracted knowledge items.

    Returns (cleaned_knowledge, review_queue_items).
    cleaned_knowledge has the same structure as input but with bad items removed.
    review_queue_items is a list of dicts ready for the review_queue table.
    """
    cleaned = {}
    review_items: list[dict] = []

    # Validate facts
    raw_facts = knowledge.get("facts", [])
    if raw_facts:
        accepted, flagged, rejected = validate_facts(raw_facts)
        cleaned["facts"] = accepted
        review_items.extend(flagged)
    else:
        cleaned["facts"] = []

    # Pass through other types with minimal validation
    for key in ("key_decisions", "ideas", "relationships", "entities",
                "guardrails", "procedures", "error_solutions", "open_questions"):
        items = knowledge.get(key, [])
        if items:
            # Basic length check for text fields
            valid = []
            for item in items:
                text = item.get("text", "") or item.get("warning", "") or item.get("task_description", "") or item.get("error_pattern", "")
                if len(text.strip()) >= MIN_TEXT_LENGTH:
                    valid.append(item)
            cleaned[key] = valid
        else:
            cleaned[key] = []

    # Preserve non-list fields (session_summary, narrative_summary, supersedes, etc.)
    for key in ("session_summary", "narrative_summary", "supersedes"):
        if key in knowledge:
            cleaned[key] = knowledge[key]

    return cleaned, review_items
