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
    SESSION_OBSERVATIONS_LIMIT,
    SESSION_GUARDRAILS_LIMIT, SESSION_PROCEDURES_LIMIT,
    PROMPT_FACTS_LIMIT, PROMPT_IDEAS_LIMIT, PROMPT_RELS_LIMIT,
    PROMPT_QUESTIONS_LIMIT,
    PROMPT_GUARDRAILS_LIMIT, PROMPT_PROCEDURES_LIMIT, PROMPT_ERROR_SOLUTIONS_LIMIT,
    SESSION_TOKEN_BUDGET, PROMPT_TOKEN_BUDGET, CHARS_PER_TOKEN,
    GLOBAL_SCOPE,
    NARRATIVE_SEARCH_LIMIT,
    PROMPT_RECALL_TIMEOUT_MS,
    CHUNKS_ENABLED, PROMPT_CHUNKS_LIMIT, PROMPT_SIBLINGS_LIMIT, CHUNK_MAX_DISPLAY_CHARS,
    SESSION_COMMUNITY_LIMIT, PROMPT_COMMUNITY_LIMIT,
    PROMPT_CODE_CONTEXT_LIMIT,
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

    # Fetch observations (consolidated knowledge)
    try:
        observations = db.get_observations_by_temporal(conn, "long", SESSION_OBSERVATIONS_LIMIT, scope=scope)
        observations += db.get_observations_by_temporal(conn, "medium", SESSION_OBSERVATIONS_LIMIT, scope=scope)
        observations = _dedup_by_id(observations)[:SESSION_OBSERVATIONS_LIMIT]
    except Exception:
        observations = []

    # Fetch guardrails
    try:
        guardrails = db.get_all_guardrails(conn, limit=SESSION_GUARDRAILS_LIMIT, scope=scope)
    except Exception:
        guardrails = []

    # Fetch procedures
    try:
        procedures = db.get_procedures(conn, limit=SESSION_PROCEDURES_LIMIT, scope=scope)
    except Exception:
        procedures = []

    # Fetch community summaries (hierarchical knowledge)
    try:
        community_summaries = db.get_community_summaries(conn, level=1, limit=SESSION_COMMUNITY_LIMIT, scope=scope)
    except Exception:
        community_summaries = []

    return {
        "long_facts":    long_facts,
        "medium_facts":  medium_facts,
        "decisions":     decisions,
        "entities":      entities,
        "relationships": all_rels,
        "observations":  observations,
        "guardrails":    guardrails,
        "procedures":    procedures,
        "community_summaries": community_summaries,
    }


def format_session_context(recall: dict) -> str:
    """Render session recall as a markdown systemMessage string, respecting token budget."""
    header = "## Memory Context (from previous sessions)\n"
    lines: list[str] = [header]
    budget = SESSION_TOKEN_BUDGET - _estimate_tokens(header)

    # Priority 0: Guardrails (always surface — highest priority)
    if recall.get("guardrails") and budget > 0:
        section = ["### ⚠ Guardrails (do not violate)"]
        for g in recall["guardrails"]:
            warning = g.get("warning", "")
            rationale = g.get("rationale", "")
            consequence = g.get("consequence", "")
            files = g.get("file_paths", [])
            line = f"- **{warning}**"
            if rationale:
                line += f" — {rationale}"
            if consequence:
                line += f" [consequence: {consequence}]"
            if files:
                line += f" (files: {', '.join(files)})"
            section.append(line)
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 0b: Procedures
    if recall.get("procedures") and budget > 0:
        section = ["### How-To Procedures"]
        for p in recall["procedures"]:
            section.append(f"- **{p['task_description']}**: {p['steps']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 0c: Community summaries (hierarchical knowledge)
    if recall.get("community_summaries") and budget > 0:
        section = ["### Architecture & Module Summaries"]
        for cs in recall["community_summaries"]:
            entities = cs.get("entity_ids", [])
            ent_label = f" ({', '.join(entities[:5])})" if entities else ""
            section.append(f"- {cs['summary']}{ent_label}")
        section.append("")
        budget = _budget_append(lines, section, budget)

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

    # Priority 3: Observations (synthesized knowledge)
    if recall.get("observations") and budget > 0:
        section = ["### Synthesized Knowledge (observations)"]
        for o in recall["observations"]:
            proof = o.get("proof_count", 1)
            section.append(f"- {o['text']} (evidence: {proof} facts)")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 4: Medium-term facts
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

def _extract_file_paths(prompt_text: str) -> list[str]:
    """
    Extract file paths mentioned in a prompt.
    Matches patterns like path/to/file.py, ./file.js, src/module/thing.rs, etc.
    """
    # Match file paths with extensions (at least one directory separator or dot-extension)
    pattern = r'(?:^|\s|[`"\'])([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})(?:\s|[`"\']|$|[,;:)])'
    matches = re.findall(pattern, prompt_text)
    # Filter out common non-file patterns (URLs, version numbers)
    return [m for m in matches if '/' in m or m.count('.') == 1]


def _recall_defensive_knowledge(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: Optional[list[float]],
    prompt_text: str,
    scope: Optional[str] = None,
) -> dict:
    """
    Multi-layered retrieval for guardrails, procedures, and error_solutions.

    Defense against false negatives:
      Layer 1 (deterministic): File-path JOIN — if the prompt mentions a file
              that has a guardrail linked, it ALWAYS surfaces. Zero semantic risk.
      Layer 2 (semantic):      Vector search on query embedding — catches
              conceptual matches like "I'll use exponential backoff" → "don't do that".
      Layer 3 (keyword):       BM25 text search — catches exact term matches that
              embeddings might miss (e.g., function names, error codes).

    Results are deduplicated by ID across layers, so no duplicate injection.
    """
    guardrails = []
    procedures = []
    error_solutions = []

    seen_guardrail_ids = set()
    seen_procedure_ids = set()
    seen_error_ids = set()

    # ── Layer 1: Deterministic file-path linking ─────────────────────────
    # This is the strongest defense — no semantic similarity needed.
    file_paths = _extract_file_paths(prompt_text)
    if file_paths:
        try:
            path_guardrails = db.get_guardrails_for_files(
                conn, file_paths, limit=PROMPT_GUARDRAILS_LIMIT, scope=scope,
            )
            for g in path_guardrails:
                if g["id"] not in seen_guardrail_ids:
                    seen_guardrail_ids.add(g["id"])
                    g["_source"] = "file_path"
                    guardrails.append(g)
        except Exception:
            pass

        # Also get path-linked procedures and error solutions
        try:
            for fp in file_paths:
                for item in db.get_items_by_file_paths(conn, [fp], "procedures", limit=3, scope=scope):
                    if item["id"] not in seen_procedure_ids:
                        seen_procedure_ids.add(item["id"])
                        # Fetch full procedure
                        full = conn.execute(
                            "SELECT id, task_description, steps, file_paths, importance, scope "
                            "FROM procedures WHERE id = ? AND is_active = TRUE",
                            [item["id"]],
                        ).fetchone()
                        if full:
                            procedures.append(dict(zip(
                                ["id", "task_description", "steps", "file_paths", "importance", "scope"], full
                            )))
        except Exception:
            pass

    # ── Layer 2: Semantic search (embedding similarity) ──────────────────
    if query_embedding:
        try:
            sem_guardrails = db.search_guardrails(
                conn, query_embedding, limit=PROMPT_GUARDRAILS_LIMIT, scope=scope,
            )
            for g in sem_guardrails:
                if g["id"] not in seen_guardrail_ids:
                    seen_guardrail_ids.add(g["id"])
                    g["_source"] = "semantic"
                    guardrails.append(g)
        except Exception:
            pass

        try:
            sem_procedures = db.search_procedures(
                conn, query_embedding, limit=PROMPT_PROCEDURES_LIMIT, scope=scope,
            )
            for p in sem_procedures:
                if p["id"] not in seen_procedure_ids:
                    seen_procedure_ids.add(p["id"])
                    procedures.append(p)
        except Exception:
            pass

        try:
            sem_errors = db.search_error_solutions(
                conn, query_embedding, limit=PROMPT_ERROR_SOLUTIONS_LIMIT, scope=scope,
            )
            for e in sem_errors:
                if e["id"] not in seen_error_ids:
                    seen_error_ids.add(e["id"])
                    error_solutions.append(e)
        except Exception:
            pass

    # ── Layer 3: BM25 keyword search ─────────────────────────────────────
    # Catches exact terms that embeddings might miss (error codes, function names)
    if prompt_text and len(prompt_text) >= 10:
        try:
            bm25_guardrails = db.search_bm25(
                conn, "guardrails", prompt_text, text_col="warning",
                select_cols="id, warning, rationale, consequence, file_paths, line_range, importance, scope",
                limit=5, scope=scope,
            )
            for g in bm25_guardrails:
                if g["id"] not in seen_guardrail_ids:
                    seen_guardrail_ids.add(g["id"])
                    g["_source"] = "bm25"
                    guardrails.append(g)
        except Exception:
            pass

        try:
            bm25_errors = db.search_bm25(
                conn, "error_solutions", prompt_text, text_col="error_pattern",
                select_cols="id, error_pattern, error_context, solution, file_paths, confidence, times_applied, scope",
                limit=5, scope=scope,
            )
            for e in bm25_errors:
                if e["id"] not in seen_error_ids:
                    seen_error_ids.add(e["id"])
                    error_solutions.append(e)
        except Exception:
            pass

    return {
        "guardrails": guardrails[:PROMPT_GUARDRAILS_LIMIT],
        "procedures": procedures[:PROMPT_PROCEDURES_LIMIT],
        "error_solutions": error_solutions[:PROMPT_ERROR_SOLUTIONS_LIMIT],
    }


def _recall_code_context(conn, prompt_text: str, scope: Optional[str]) -> list[dict]:
    """Extract file paths from prompt and fetch code structure."""
    try:
        from .code_graph import get_file_symbols, get_dependencies, get_dependents
        from .retrieval import _extract_file_paths
    except ImportError:
        return []

    paths = _extract_file_paths(prompt_text)
    if not paths:
        return []

    results = []
    for fp in paths[:PROMPT_CODE_CONTEXT_LIMIT]:
        syms = get_file_symbols(conn, fp)
        if not syms:
            # Try partial match
            rows = conn.execute(
                "SELECT DISTINCT file_path FROM code_symbols WHERE file_path LIKE ? LIMIT 3",
                [f"%{fp}%"]
            ).fetchall()
            for row in rows:
                syms = get_file_symbols(conn, row[0])
                if syms:
                    fp = row[0]
                    break

        if syms:
            deps = get_dependencies(conn, fp, scope=scope)
            dependents = get_dependents(conn, fp, scope=scope)
            sym_summary = ", ".join(f"{s['symbol_name']}" for s in syms[:15])
            dep_summary = ", ".join(d["to_file"] for d in deps[:5])
            results.append({
                "file": fp,
                "symbols": sym_summary,
                "dependencies": dep_summary,
                "dependent_count": len(dependents),
                "symbol_count": len(syms),
            })

    return results


def prompt_recall(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    prompt_text: str,
    scope: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Retrieve narrow, prompt-specific context via 4-way parallel retrieval + RRF.
    Falls back to legacy single-strategy search when retrieval module unavailable.

    Additionally performs multi-layered defensive retrieval for guardrails,
    procedures, and error_solutions (file-path JOIN + semantic + BM25).
    """
    # ── Defensive knowledge retrieval (guardrails, procedures, errors) ───
    # This runs on every prompt, independent of the main retrieval path.
    defensive = _recall_defensive_knowledge(conn, query_embedding, prompt_text, scope)

    code_context = _recall_code_context(conn, prompt_text, scope)

    # Try 4-way parallel retrieval
    try:
        from .retrieval import parallel_retrieve, ScoredItem
        import memory.config as _config

        path = db_path or str(_config.DB_PATH)
        result = parallel_retrieve(
            db_path=path,
            query_text=prompt_text,
            query_embedding=query_embedding,
            scope=scope,
            limit=PROMPT_FACTS_LIMIT + PROMPT_IDEAS_LIMIT + 5,
            timeout_ms=PROMPT_RECALL_TIMEOUT_MS,
        )

        # Partition results by table
        facts = []
        ideas = []
        observations = []
        other = []
        for item in result.items:
            d = {"id": item.id, "text": item.text, "score": item.score,
                 "table": item.table, **item.metadata}
            if item.table == "facts":
                facts.append(d)
            elif item.table == "ideas":
                ideas.append(d)
            elif item.table == "observations":
                observations.append(d)
            else:
                other.append(d)

        # Still get relationships and entities from the existing connection
        known_entities = db.get_top_entities(conn, 100, scope=scope)
        prompt_entities = _entities_in_text(prompt_text, known_entities)
        relationships = db.get_relationships_for_entities(
            conn, prompt_entities, PROMPT_RELS_LIMIT, scope=scope,
        )

        # Search narratives
        try:
            narratives = db.search_narratives(
                conn, query_embedding, NARRATIVE_SEARCH_LIMIT, scope=scope,
            )
        except Exception:
            narratives = []

        # Log timing to stderr
        if result.exceeded_budget:
            import sys
            timings = " ".join(f"{k}={v:.0f}ms" for k, v in result.strategy_timings.items())
            print(
                f"[memory] Recall took {result.elapsed_ms:.0f}ms "
                f"(>{PROMPT_RECALL_TIMEOUT_MS}ms budget) {timings} — retrieving for accuracy",
                file=sys.stderr,
            )

        # Load source chunks and sibling facts for retrieved facts
        top_facts = facts[:PROMPT_FACTS_LIMIT]
        chunks = {}
        sibling_facts = []
        if CHUNKS_ENABLED:
            chunks = _load_chunks_for_facts(conn, top_facts, query_embedding, prompt_text, scope)
            sibling_facts = _expand_sibling_facts(conn, top_facts)

        return {
            "facts": top_facts,
            "ideas": ideas[:PROMPT_IDEAS_LIMIT],
            "observations": observations[:5],
            "relationships": relationships,
            "questions": [],  # covered by semantic retrieval
            "entities_hit": prompt_entities,
            "narratives": narratives,
            "retrieval_stats": result.strategy_counts,
            "chunks": chunks,
            "sibling_facts": sibling_facts,
            "code_context": code_context,
            **defensive,
        }

    except Exception:
        # Fallback: legacy single-strategy search
        legacy = _legacy_prompt_recall(conn, query_embedding, prompt_text, scope)
        # Merge defensive knowledge into legacy results
        legacy.update(defensive)
        legacy["code_context"] = code_context
        return legacy


def _load_chunks_for_facts(
    conn: duckdb.DuckDBPyConnection,
    facts: list[dict],
    query_embedding: Optional[list[float]] = None,
    prompt_text: str = "",
    scope: Optional[str] = None,
) -> dict[str, dict]:
    """
    Load chunks via three paths:
    1. Post-hoc: chunks linked to retrieved facts via source_chunk_id
    2. Vector search: direct search on chunk embeddings
    3. BM25 keyword search: finds chunks by exact keyword match
    Returns {chunk_id: chunk_dict}.
    """
    chunks = {}

    # Path 1: Post-hoc from fact links
    chunk_ids = list({
        f["source_chunk_id"] for f in facts
        if f.get("source_chunk_id")
    })
    if chunk_ids:
        chunks.update(db.get_chunks_by_ids(conn, chunk_ids[:PROMPT_CHUNKS_LIMIT]))

    # Path 2: Direct vector search on chunks
    if query_embedding and CHUNKS_ENABLED and len(chunks) < PROMPT_CHUNKS_LIMIT:
        try:
            remaining = PROMPT_CHUNKS_LIMIT - len(chunks)
            direct_hits = db.search_chunks(conn, query_embedding, limit=remaining, scope=scope)
            for hit in direct_hits:
                if hit["id"] not in chunks:
                    chunks[hit["id"]] = hit
        except Exception:
            pass

    # Path 3: BM25 keyword search on chunks (catches exact name/number matches)
    if prompt_text and CHUNKS_ENABLED and len(chunks) < PROMPT_CHUNKS_LIMIT:
        try:
            remaining = PROMPT_CHUNKS_LIMIT - len(chunks)
            bm25_hits = db.search_chunks_bm25(conn, prompt_text, limit=remaining, scope=scope)
            for hit in bm25_hits:
                if hit["id"] not in chunks:
                    chunks[hit["id"]] = hit
        except Exception:
            pass

    return dict(list(chunks.items())[:PROMPT_CHUNKS_LIMIT])


def _expand_sibling_facts(
    conn: duckdb.DuckDBPyConnection,
    facts: list[dict],
    limit: int = PROMPT_SIBLINGS_LIMIT,
) -> list[dict]:
    """
    Chunk sibling expansion: find additional facts from the same source chunks
    as the retrieved facts. Returns sibling facts not already in the original set.
    """
    chunk_ids = list({
        f["source_chunk_id"] for f in facts
        if f.get("source_chunk_id")
    })
    if not chunk_ids:
        return []
    existing_ids = {f.get("id") for f in facts if f.get("id")}
    return db.get_sibling_facts(conn, chunk_ids, exclude_ids=existing_ids, limit=limit)


def _legacy_prompt_recall(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    prompt_text: str,
    scope: Optional[str] = None,
) -> dict:
    """Original single-strategy prompt recall (fallback)."""
    if scope and scope != GLOBAL_SCOPE:
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

    for f in facts:
        base = temporal_weight(f.get("temporal_class", "short"), f.get("decay_score", 1.0)) * f.get("score", 0.5)
        utility = f.get("recall_utility", 1.0) if f.get("recall_utility") else 1.0
        fail_boost = 1.0 + f.get("failure_probability", 0.0)
        f["_weight"] = base * utility * fail_boost
    facts.sort(key=lambda f: f["_weight"], reverse=True)

    try:
        narratives = db.search_narratives(
            conn, query_embedding, NARRATIVE_SEARCH_LIMIT, scope=scope,
        )
    except Exception:
        narratives = []

    # Load source chunks and sibling facts
    chunks = {}
    sibling_facts = []
    if CHUNKS_ENABLED:
        chunks = _load_chunks_for_facts(conn, facts, query_embedding, prompt_text, scope)
        sibling_facts = _expand_sibling_facts(conn, facts)

    return {
        "facts":         facts,
        "ideas":         ideas,
        "observations":  [],
        "relationships": relationships,
        "questions":     questions,
        "entities_hit":  prompt_entities,
        "narratives":    narratives,
        "chunks":        chunks,
        "sibling_facts": sibling_facts,
    }


def format_prompt_context(recall: dict) -> str:
    """Render prompt recall as an additionalContext markdown string, respecting token budget."""
    has_content = any([
        recall.get("facts"), recall.get("ideas"),
        recall.get("relationships"), recall.get("questions"),
        recall.get("narratives"), recall.get("observations"),
        recall.get("chunks"), recall.get("sibling_facts"),
        recall.get("guardrails"), recall.get("procedures"),
        recall.get("error_solutions"), recall.get("code_context"),
    ])
    if not has_content:
        return ""

    header = "## Recalled Memory (relevant to this prompt)\n"
    lines: list[str] = [header]
    budget = PROMPT_TOKEN_BUDGET - _estimate_tokens(header)

    # Priority 0: Guardrails (always surface — highest priority)
    if recall.get("guardrails") and budget > 0:
        section = ["### ⚠ Guardrails"]
        for g in recall["guardrails"]:
            warning = g.get("warning", "")
            rationale = g.get("rationale", "")
            consequence = g.get("consequence", "")
            line = f"- **{warning}**"
            if rationale:
                line += f" — {rationale}"
            if consequence:
                line += f" [breaks: {consequence}]"
            section.append(line)
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 0b: Error solutions
    if recall.get("error_solutions") and budget > 0:
        section = ["### Known Error Solutions"]
        for e in recall["error_solutions"]:
            section.append(f"- **{e.get('error_pattern', '')}** → {e.get('solution', '')}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 0c: Procedures
    if recall.get("procedures") and budget > 0:
        section = ["### Relevant Procedures"]
        for p in recall["procedures"]:
            section.append(f"- **{p.get('task_description', '')}**: {p.get('steps', '')}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Code structure section
    code_ctx = recall.get("code_context", [])
    if code_ctx and budget > 0:
        section = ["### Code Structure (active files)"]
        for c in code_ctx:
            line = f"- **{c['file']}** ({c['symbol_count']} symbols): {c['symbols']}"
            if c.get("dependencies"):
                line += f" | imports: {c['dependencies']}"
            if c.get("dependent_count", 0) > 0:
                line += f" | {c['dependent_count']} dependents"
            section.append(line)
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 1: Facts
    if recall.get("facts") and budget > 0:
        section = ["### Relevant Facts"]
        for f in recall["facts"]:
            tc = f.get("temporal_class", "")
            tc_label = f"[{tc}] " if tc else ""
            section.append(f"- {tc_label}{f['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 2: Observations (synthesized knowledge)
    if recall.get("observations") and budget > 0:
        section = ["### Synthesized Knowledge"]
        for o in recall["observations"]:
            proof = o.get("proof_count", "")
            proof_label = f" (evidence: {proof} facts)" if proof else ""
            section.append(f"- {o['text']}{proof_label}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 3: Related Session Context (narratives)
    if recall.get("narratives") and budget > 0:
        section = ["### Related Session Context"]
        for n in recall["narratives"]:
            section.append(f"- {n['narrative']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 3b: Sibling facts (from same conversation as retrieved facts)
    if recall.get("sibling_facts") and budget > 0:
        section = ["### Additional Context (from same conversations)"]
        for f in recall["sibling_facts"]:
            section.append(f"- {f['text']}")
        section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 3c: Source conversation chunks (verbatim detail backup)
    chunks = recall.get("chunks", {})
    if chunks and budget > 0:
        section = ["### Source Conversation Context"]
        for chunk_id, chunk in chunks.items():
            text = chunk.get("text", "")
            from . import config as _cfg
            max_chars = _cfg.CHUNK_MAX_DISPLAY_CHARS
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            section.append(f'> {text}')
            section.append("")
        budget = _budget_append(lines, section, budget)

    # Priority 4: Ideas
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
