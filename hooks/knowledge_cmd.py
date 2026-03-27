#!/usr/bin/env python3
"""
knowledge_cmd.py — Entry point for the /knowledge custom slash command.

Cross-type search: shows everything the memory system knows about a topic.
Searches facts, decisions, observations, guardrails, procedures,
error_solutions, entities, and relationships with semantic search
(embedding-based) when Ollama is available, falling back to text search.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))


# ── Type search configs ──────────────────────────────────────────────────
# Each entry: (label, search_func_name, text_key, extra_fields)
# extra_fields are tuples of (dict_key, display_format)

_SEARCH_TYPES = [
    {
        "label": "Facts",
        "search": "search_facts",
        "text_key": "text",
        "limit": 8,
        "extras": ["temporal_class", "decay_score", "scope", "category"],
    },
    {
        "label": "Decisions",
        "search": None,  # use _vector_search directly
        "table": "decisions",
        "cols": "id, text, temporal_class, decay_score, session_count, last_seen_at, scope",
        "text_key": "text",
        "limit": 5,
        "extras": ["temporal_class", "decay_score", "scope"],
    },
    {
        "label": "Observations",
        "search": "search_observations",
        "text_key": "text",
        "limit": 5,
        "extras": ["temporal_class", "decay_score", "scope", "proof_count"],
    },
    {
        "label": "Guardrails",
        "search": "search_guardrails",
        "text_key": "warning",
        "limit": 4,
        "extras": ["scope", "importance", "file_paths"],
    },
    {
        "label": "Procedures",
        "search": "search_procedures",
        "text_key": "task_description",
        "limit": 4,
        "extras": ["scope", "importance", "file_paths"],
    },
    {
        "label": "Error Solutions",
        "search": "search_error_solutions",
        "text_key": "error_pattern",
        "limit": 4,
        "extras": ["scope", "confidence", "solution", "file_paths"],
    },
]

# Total cap across all types
MAX_RESULTS = 30


def _format_scope(scope: str) -> str:
    """Shorten a scope path to just the project name."""
    if scope == "__global__":
        return "global"
    return Path(scope).name if "/" in scope else scope


def _format_item(item: dict, text_key: str, extras: list[str]) -> str:
    """Format a single result line."""
    score = item.get("score", 0)
    text = item.get(text_key, "").replace("\n", " ").strip()
    if len(text) > 120:
        text = text[:117] + "..."

    parts = [f"[{score:.2f}]"]

    # Temporal class + decay
    tc = item.get("temporal_class")
    ds = item.get("decay_score")
    if tc and ds is not None:
        parts.append(f"[{tc} {ds:.2f}]")
    elif tc:
        parts.append(f"[{tc}]")

    # Proof count for observations
    pc = item.get("proof_count")
    if pc and pc > 1:
        parts.append(f"(proof: {pc})")

    # Importance
    imp = item.get("importance")
    if imp is not None and "importance" in extras:
        parts.append(f"imp:{imp}")

    # Confidence
    conf = item.get("confidence")
    if conf and "confidence" in extras:
        parts.append(f"conf:{conf}")

    parts.append(text)

    # Scope
    scope = item.get("scope")
    if scope:
        parts.append(f"({_format_scope(scope)})")

    line = "  " + " ".join(parts)

    # File paths on a sub-line
    fps = item.get("file_paths")
    if fps and "file_paths" in extras:
        if isinstance(fps, list) and fps:
            line += "\n    files: " + ", ".join(str(f) for f in fps[:3])

    # Solution on a sub-line for error_solutions
    sol = item.get("solution")
    if sol and "solution" in extras:
        sol_text = sol.replace("\n", " ").strip()
        if len(sol_text) > 100:
            sol_text = sol_text[:97] + "..."
        line += f"\n    fix: {sol_text}"

    return line


def _search_semantic(conn, topic: str, scope: str) -> dict[str, list[dict]]:
    """Search all types using embedding-based semantic search."""
    from memory.embeddings import embed_query
    from memory import db

    embedding = embed_query(topic)
    if not embedding:
        return {}

    results: dict[str, list[dict]] = {}
    total = 0

    for stype in _SEARCH_TYPES:
        if total >= MAX_RESULTS:
            break

        remaining = MAX_RESULTS - total
        limit = min(stype["limit"], remaining)

        try:
            if stype["search"]:
                fn = getattr(db, stype["search"])
                hits = fn(conn, embedding, limit=limit, scope=scope)
            else:
                # Direct _vector_search for types without a dedicated search function
                hits = db._vector_search(
                    conn, stype["table"], embedding,
                    stype["cols"],
                    db._scope_filter(scope), limit, db.RECALL_THRESHOLD,
                )
        except Exception:
            hits = []

        if hits:
            results[stype["label"]] = hits
            total += len(hits)

    return results


def _search_text_fallback(conn, topic: str, scope: str) -> dict[str, list[dict]]:
    """Fall back to text-based search when embeddings are unavailable."""
    from memory import db

    raw = db.search_all_by_text(conn, topic, scope=scope)
    if not raw:
        return {}

    # Group by table name, map to our labels
    table_to_label = {
        "facts": "Facts",
        "decisions": "Decisions",
        "observations": "Observations",
        "guardrails": "Guardrails",
        "procedures": "Procedures",
        "error_solutions": "Error Solutions",
        "entities": "Entities",
        "ideas": "Ideas",
        "open_questions": "Open Questions",
    }

    grouped: dict[str, list[dict]] = {}
    total = 0
    for item in raw:
        if total >= MAX_RESULTS:
            break
        label = table_to_label.get(item["table"], item["table"])
        grouped.setdefault(label, [])
        # Text fallback items don't have scores; assign 0 for display
        item["score"] = 0.0
        grouped[label].append(item)
        total += 1

    return grouped


def _find_matching_entities(conn, topic: str, scope: str) -> tuple[list[str], list[dict]]:
    """Find entities whose names match the topic, plus their relationships."""
    from memory import db

    topic_lower = topic.lower()
    try:
        scope_sql, scope_params = db._scope_filter(scope)
        rows = conn.execute(f"""
            SELECT name, entity_type, session_count, scope
            FROM entities
            WHERE is_active = TRUE
              AND LOWER(name) LIKE ?
              {scope_sql}
            ORDER BY session_count DESC
            LIMIT 10
        """, [f"%{topic_lower}%"] + scope_params).fetchall()
        entities = [
            dict(zip(["name", "entity_type", "session_count", "scope"], r))
            for r in rows
        ]
        entity_names = [e["name"] for e in entities]
    except Exception:
        entities = []
        entity_names = []

    # Get relationships for found entities
    rels = []
    if entity_names:
        try:
            rels = db.get_relationships_for_entities(conn, entity_names, limit=15, scope=scope)
        except Exception:
            pass

    return entity_names, rels


def main() -> None:
    topic = os.environ.get("MEMORY_TEXT", "").strip()
    if not topic:
        print(
            "Usage: /knowledge <topic>\n\n"
            "Search everything the memory system knows about a topic.\n"
            "Shows facts, decisions, observations, guardrails, procedures,\n"
            "error solutions, entities, and relationships."
        )
        return

    from memory import db
    from memory.scope import resolve_scope

    scope = resolve_scope(os.getcwd())
    conn = db.get_connection(read_only=True)

    try:
        # Try semantic search first, fall back to text
        use_semantic = True
        results = _search_semantic(conn, topic, scope)
        if not results:
            use_semantic = False
            results = _search_text_fallback(conn, topic, scope)

        # Find entities and relationships
        entity_names, rels = _find_matching_entities(conn, topic, scope)

        # Count total results
        total_items = sum(len(v) for v in results.values())
        total_items += len(rels)
        type_count = len(results) + (1 if rels else 0)

        if total_items == 0 and not entity_names:
            print(f'No knowledge found for "{topic}".')
            return

        # Format output
        print(f'## Knowledge: "{topic}"\n')

        # Print each type section
        for stype in _SEARCH_TYPES:
            label = stype["label"]
            if label not in results:
                continue
            items = results[label]
            count_label = f"{len(items)} match" + ("es" if len(items) != 1 else "")
            print(f"### {label} ({count_label})")
            for item in items:
                text_key = stype["text_key"]
                extras = stype.get("extras", [])
                print(_format_item(item, text_key, extras))
            print()

        # Print any extra types from text fallback that aren't in _SEARCH_TYPES
        known_labels = {s["label"] for s in _SEARCH_TYPES}
        for label, items in results.items():
            if label in known_labels:
                continue
            count_label = f"{len(items)} match" + ("es" if len(items) != 1 else "")
            print(f"### {label} ({count_label})")
            for item in items:
                print(_format_item(item, "text", []))
            print()

        # Entities section
        if entity_names:
            print(f"### Related Entities")
            for name in entity_names:
                print(f"  {name}")
            print()

        # Relationships section
        if rels:
            print(f"### Relationships")
            for r in rels:
                fr = r.get("from_entity", "?")
                to = r.get("to_entity", "?")
                rt = r.get("rel_type", "?")
                desc = r.get("description", "")
                line = f"  {fr} --[{rt}]--> {to}"
                if desc:
                    desc_short = desc.replace("\n", " ").strip()
                    if len(desc_short) > 60:
                        desc_short = desc_short[:57] + "..."
                    line += f"  ({desc_short})"
                print(line)
            print()

        # Footer
        search_method = "semantic" if use_semantic else "text"
        scope_label = _format_scope(scope) + " + global"
        print("---")
        print(f"Searched {type_count} types | {total_items} results | "
              f"scope: {scope_label} | method: {search_method}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
