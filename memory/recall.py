"""
recall.py — Retrieval functions used by SessionStart and UserPromptSubmit hooks.

Two recall modes:
  session_recall  — broad, temporally-weighted; injected once per session
  prompt_recall   — narrow, embedding-focused; injected per user prompt

Both formatters enforce a token budget. Project-local items get priority;
global items fill the remaining budget.
"""
from __future__ import annotations

import re
from typing import Optional

import duckdb

from . import db
from .config import (
    SESSION_LONG_FACTS_LIMIT, SESSION_MEDIUM_FACTS_LIMIT,
    SESSION_DECISIONS_LIMIT, SESSION_ENTITIES_LIMIT,
    PROMPT_FACTS_LIMIT, PROMPT_IDEAS_LIMIT, PROMPT_RELS_LIMIT,
    PROMPT_QUESTIONS_LIMIT,
    SESSION_TOKEN_BUDGET, PROMPT_TOKEN_BUDGET, CHARS_PER_TOKEN,
    GLOBAL_SCOPE,
    NARRATIVE_SEARCH_LIMIT,
)
from .decay import temporal_weight


# ── Token budget helper ──────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // CHARS_PER_TOKEN


def _budget_append(lines: list[str], section_lines: list[str], budget: int) -> int:
    """
    Append as many section_lines as fit within the remaining budget.
    Returns the remaining budget after appending.
    """
    for line in section_lines:
        cost = _estimate_tokens(line + "\n")
        if budget - cost < 0:
            lines.append("  ... (truncated to stay within token budget)")
            return 0
        lines.append(line)
        budget -= cost
    return budget


def _dedup_by_id(items: list[dict]) -> list[dict]:
    """Remove duplicate items by id, keeping the first occurrence."""
    seen = set()
    result = []
    for item in items:
        iid = item.get("id")
        if iid and iid in seen:
            continue
        if iid:
            seen.add(iid)
        result.append(item)
    return result


# ── Session-level recall (SessionStart) ───────────────────────────────────

def session_recall(
    conn: duckdb.DuckDBPyConnection,
    scope: Optional[str] = None,
) -> dict:
    """
    Retrieve broad context for the beginning of a new session.
    Project-local items first, then global items to fill limits.
    """
    if scope and scope != GLOBAL_SCOPE:
        # Project-local first, then global fills remaining slots
        local_long = db.get_facts_by_temporal(conn, "long", SESSION_LONG_FACTS_LIMIT, scope=scope)
        global_long = db.get_facts_by_temporal(conn, "long", SESSION_LONG_FACTS_LIMIT, scope=GLOBAL_SCOPE)
        long_facts = _dedup_by_id(local_long + global_long)[:SESSION_LONG_FACTS_LIMIT]

        local_med = db.get_facts_by_temporal(conn, "medium", SESSION_MEDIUM_FACTS_LIMIT, scope=scope)
        global_med = db.get_facts_by_temporal(conn, "medium", SESSION_MEDIUM_FACTS_LIMIT, scope=GLOBAL_SCOPE)
        medium_facts = _dedup_by_id(local_med + global_med)[:SESSION_MEDIUM_FACTS_LIMIT]

        local_dec = db.get_decisions(conn, SESSION_DECISIONS_LIMIT, scope=scope)
        global_dec = db.get_decisions(conn, SESSION_DECISIONS_LIMIT, scope=GLOBAL_SCOPE)
        decisions = _dedup_by_id(local_dec + global_dec)[:SESSION_DECISIONS_LIMIT]

        entities = db.get_top_entities(conn, SESSION_ENTITIES_LIMIT, scope=scope)
        all_rels = db.get_all_relationships(conn, scope=scope)
    else:
        long_facts = db.get_facts_by_temporal(conn, "long", SESSION_LONG_FACTS_LIMIT)
        medium_facts = db.get_facts_by_temporal(conn, "medium", SESSION_MEDIUM_FACTS_LIMIT)
        decisions = db.get_decisions(conn, SESSION_DECISIONS_LIMIT)
        entities = db.get_top_entities(conn, SESSION_ENTITIES_LIMIT)
        all_rels = db.get_all_relationships(conn)

    medium_facts.sort(
        key=lambda f: temporal_weight(f["temporal_class"], f["decay_score"]),
        reverse=True,
    )

    return {
        "long_facts":    long_facts,
        "medium_facts":  medium_facts,
        "decisions":     decisions,
        "entities":      entities,
        "relationships": all_rels,
    }


def format_session_context(recall: dict) -> str:
    """Render session recall as a markdown systemMessage string, respecting token budget."""
    header = "## Memory Context (from previous sessions)\n"
    lines: list[str] = [header]
    budget = SESSION_TOKEN_BUDGET - _estimate_tokens(header)

    # Priority 1: Long-term facts
    if recall["long_facts"] and budget > 0:
        section = ["### Established Knowledge (long-term)"]
        for f in recall["long_facts"]:
            section.append(f"- [{f['category']}] {f['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 2: Decisions
    if recall["decisions"] and budget > 0:
        section = ["### Key Decisions"]
        for d in recall["decisions"]:
            section.append(f"- {d['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 3: Medium-term facts
    if recall["medium_facts"] and budget > 0:
        section = ["### Working Context (medium-term)"]
        for f in recall["medium_facts"]:
            section.append(f"- {f['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 4: Entities (cheap — single line)
    if recall["entities"] and budget > 0:
        section = ["### Known Entities", ", ".join(recall["entities"]), ""]
        budget = _budget_append(lines, section, budget)

    # Priority 5: Relationships (lowest priority)
    if recall["relationships"] and budget > 0:
        section = ["### Relationship Graph"]
        for r in recall["relationships"][:20]:
            section.append(f"- {r['from']} —[{r['rel_type']}]-> {r['to']}: {r['description']}")
        if len(recall["relationships"]) > 20:
            section.append(f"  ... (+{len(recall['relationships'])-20} more in memory)")
        section.append("")
        budget = _budget_append(lines, section, budget)

    if len(lines) <= 2:
        return ""  # nothing to inject
    return "\n".join(lines)


# ── Prompt-level recall (UserPromptSubmit) ─────────────────────────────────

def prompt_recall(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    prompt_text: str,
    scope: Optional[str] = None,
) -> dict:
    """
    Retrieve narrow, prompt-specific context via vector similarity + entity graph.
    Project-local results first, global fills remaining.
    """
    if scope and scope != GLOBAL_SCOPE:
        # Project-local search first, then global
        local_facts = db.search_facts(conn, query_embedding, PROMPT_FACTS_LIMIT, scope=scope)
        global_facts = db.search_facts(conn, query_embedding, PROMPT_FACTS_LIMIT, scope=GLOBAL_SCOPE)
        facts = _dedup_by_id(local_facts + global_facts)[:PROMPT_FACTS_LIMIT]

        local_ideas = db.search_ideas(conn, query_embedding, PROMPT_IDEAS_LIMIT, scope=scope)
        global_ideas = db.search_ideas(conn, query_embedding, PROMPT_IDEAS_LIMIT, scope=GLOBAL_SCOPE)
        ideas = _dedup_by_id(local_ideas + global_ideas)[:PROMPT_IDEAS_LIMIT]

        questions = db.search_questions(conn, query_embedding, PROMPT_QUESTIONS_LIMIT, scope=scope)

        known_entities = db.get_top_entities(conn, 100, scope=scope)
        prompt_entities = _entities_in_text(prompt_text, known_entities)
        relationships = db.get_relationships_for_entities(conn, prompt_entities, PROMPT_RELS_LIMIT, scope=scope)
    else:
        facts = db.search_facts(conn, query_embedding, PROMPT_FACTS_LIMIT)
        ideas = db.search_ideas(conn, query_embedding, PROMPT_IDEAS_LIMIT)
        questions = db.search_questions(conn, query_embedding, PROMPT_QUESTIONS_LIMIT)

        known_entities = db.get_top_entities(conn, 100)
        prompt_entities = _entities_in_text(prompt_text, known_entities)
        relationships = db.get_relationships_for_entities(conn, prompt_entities, PROMPT_RELS_LIMIT)

    # Re-rank facts by composite weight (embedding score x temporal boost)
    for f in facts:
        f["_weight"] = temporal_weight(f.get("temporal_class", "short"), f.get("decay_score", 1.0)) * f.get("score", 0.5)
    facts.sort(key=lambda f: f["_weight"], reverse=True)

    # Search narratives for relevant session context
    try:
        narratives = db.search_narratives(
            conn, query_embedding, NARRATIVE_SEARCH_LIMIT, scope=scope,
        )
    except Exception:
        narratives = []  # table may not exist yet (pre-migration-4)

    return {
        "facts":         facts,
        "ideas":         ideas,
        "relationships": relationships,
        "questions":     questions,
        "entities_hit":  prompt_entities,
        "narratives":    narratives,
    }


def format_prompt_context(recall: dict) -> str:
    """Render prompt recall as an additionalContext markdown string, respecting token budget."""
    has_content = any([
        recall.get("facts"), recall.get("ideas"),
        recall.get("relationships"), recall.get("questions"),
        recall.get("narratives"),
    ])
    if not has_content:
        return ""

    header = "## Recalled Memory (relevant to this prompt)\n"
    lines: list[str] = [header]
    budget = PROMPT_TOKEN_BUDGET - _estimate_tokens(header)

    # Priority 1: Facts
    if recall.get("facts") and budget > 0:
        section = ["### Relevant Facts"]
        for f in recall["facts"]:
            tc = f.get("temporal_class", "")
            tc_label = f"[{tc}] " if tc else ""
            section.append(f"- {tc_label}{f['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 2: Related Session Context (narratives)
    if recall.get("narratives") and budget > 0:
        section = ["### Related Session Context"]
        for n in recall["narratives"]:
            section.append(f"- {n['narrative']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 3: Ideas
    if recall.get("ideas") and budget > 0:
        section = ["### Related Ideas"]
        for i in recall["ideas"]:
            section.append(f"- [{i.get('idea_type','')}] {i['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 4: Relationships
    if recall.get("relationships") and budget > 0:
        section = ["### Related Connections"]
        for r in recall["relationships"]:
            section.append(f"- {r['from_entity']} —[{r['rel_type']}]-> {r['to_entity']}: {r['description']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 5: Open questions
    if recall.get("questions") and budget > 0:
        section = ["### Open Questions to Keep in Mind"]
        for q in recall["questions"]:
            section.append(f"- ? {q['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    return "\n".join(lines)


# ── Entity extraction from free text ──────────────────────────────────────

def _entities_in_text(text: str, known_entities: list[str]) -> list[str]:
    """
    Return which known entities appear (case-insensitive) in the given text.
    """
    text_lower = text.lower()
    found = []
    for entity in known_entities:
        pattern = re.escape(entity.lower())
        if re.search(r'\b' + pattern + r'\b', text_lower):
            found.append(entity)
    return found
