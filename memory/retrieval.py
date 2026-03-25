"""
retrieval.py — 4-way parallel retrieval with reciprocal rank fusion.

Strategies: semantic (vector), BM25 (keyword), graph (entity traversal),
temporal (date-range filtering). Results are fused via RRF and optionally
reranked by a cross-encoder model.
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import db
from .config import (
    RETRIEVAL_STRATEGIES,
    RRF_K,
    RERANK_ENABLED,
    RECALL_THRESHOLD,
    PROMPT_RECALL_TIMEOUT_MS,
    GLOBAL_SCOPE,
)
from .decay import temporal_weight


# ── Data types ────────────────────────────────────────────────────────────

@dataclass
class ScoredItem:
    id: str
    table: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    items: list[ScoredItem]
    elapsed_ms: float
    exceeded_budget: bool
    strategy_counts: dict[str, int]
    strategy_timings: dict[str, float] = field(default_factory=dict)


# ── Individual retrieval strategies ───────────────────────────────────────

def retrieve_semantic(
    db_path: str,
    query_embedding: list[float],
    scope: Optional[str],
    limit: int,
) -> list[ScoredItem]:
    """Vector similarity search across facts, ideas, decisions, observations."""
    conn = db.get_connection(read_only=True, db_path=db_path)
    try:
        items = []
        for table, cols in _SEARCH_TABLES.items():
            results = db._vector_search(
                conn, table, query_embedding,
                cols, db._scope_filter(scope), limit, RECALL_THRESHOLD,
            )
            for r in results:
                items.append(ScoredItem(
                    id=r["id"], table=table, text=r["text"],
                    score=r.get("score", 0),
                    metadata={k: v for k, v in r.items() if k not in ("id", "text", "score", "embedding")},
                ))
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]
    finally:
        conn.close()


def retrieve_bm25(
    db_path: str,
    query_text: str,
    scope: Optional[str],
    limit: int,
) -> list[ScoredItem]:
    """BM25 full-text keyword search across text tables."""
    conn = db.get_connection(read_only=True, db_path=db_path)
    try:
        items = []
        for table, info in _BM25_TABLES.items():
            results = db.search_bm25(
                conn, table, query_text, info["text_col"],
                info["select_cols"], limit, scope=scope,
            )
            for r in results:
                items.append(ScoredItem(
                    id=r["id"], table=table, text=r.get("text", r.get("narrative", "")),
                    score=r.get("score", 0),
                    metadata={k: v for k, v in r.items() if k not in ("id", "text", "narrative", "score")},
                ))
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]
    finally:
        conn.close()


def retrieve_graph(
    db_path: str,
    query_text: str,
    scope: Optional[str],
    limit: int,
) -> list[ScoredItem]:
    """Graph traversal: find entities in query, follow relationships, pull linked facts."""
    conn = db.get_connection(read_only=True, db_path=db_path)
    try:
        known_entities = db.get_top_entities(conn, 200, scope=scope)
        prompt_entities = _entities_in_text(query_text, known_entities, db_path=db_path)
        if not prompt_entities:
            return []

        # BFS: 2 hops through relationships
        visited_entities = set(e.lower() for e in prompt_entities)
        hop_scores: dict[str, float] = {e.lower(): 1.0 for e in prompt_entities}

        for hop in range(2):
            decay = 0.5 ** hop
            current_entities = [e for e in visited_entities if hop_scores.get(e, 0) >= decay * 0.5]
            rels = db.get_relationships_for_entities(
                conn, list(current_entities), limit=50, scope=scope,
            )
            for r in rels:
                for end in (r["from_entity"], r["to_entity"]):
                    end_lower = end.lower()
                    if end_lower not in visited_entities:
                        visited_entities.add(end_lower)
                        parent_score = max(
                            hop_scores.get(r["from_entity"].lower(), 0),
                            hop_scores.get(r["to_entity"].lower(), 0),
                        )
                        hop_scores[end_lower] = parent_score * 0.5 * r.get("strength", 1.0)

        # Pull facts linked to discovered entities
        all_entity_names = list(visited_entities)
        items = []
        try:
            placeholders = ", ".join("?" * len(all_entity_names))
            rows = conn.execute(f"""
                SELECT f.id, f.text, f.temporal_class, f.decay_score,
                       fel.entity_name
                FROM fact_entity_links fel
                JOIN facts f ON f.id = fel.fact_id
                WHERE f.is_active = TRUE
                  AND LOWER(fel.entity_name) IN ({placeholders})
            """, all_entity_names).fetchall()

            seen_ids = set()
            for row in rows:
                fid, text, tc, ds, ent = row
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                entity_score = hop_scores.get(ent.lower(), 0.1)
                tw = temporal_weight(tc, ds)
                items.append(ScoredItem(
                    id=fid, table="facts", text=text,
                    score=entity_score * tw,
                    metadata={"temporal_class": tc, "decay_score": ds, "via_entity": ent},
                ))
        except Exception:
            pass

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]
    finally:
        conn.close()


def retrieve_temporal(
    db_path: str,
    query_text: str,
    scope: Optional[str],
    limit: int,
) -> list[ScoredItem]:
    """Date-range retrieval: extract temporal references from query, filter by recency."""
    date_range = _extract_date_range(query_text)
    if not date_range:
        return []

    start_date, end_date = date_range
    conn = db.get_connection(read_only=True, db_path=db_path)
    try:
        scope_sql, scope_params = db._scope_filter(scope)
        items = []
        for table in ("facts", "ideas", "decisions", "observations"):
            try:
                rows = conn.execute(f"""
                    SELECT id, text, temporal_class, decay_score, created_at
                    FROM {table}
                    WHERE is_active = TRUE
                      AND created_at >= ? AND created_at <= ?
                      {scope_sql}
                    ORDER BY created_at DESC
                    LIMIT ?
                """, [start_date, end_date] + scope_params + [limit]).fetchall()
                mid = start_date + (end_date - start_date) / 2
                for row in rows:
                    fid, text, tc, ds, created = row
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    days_from_mid = abs((created - mid).total_seconds()) / 86400
                    total_days = max((end_date - start_date).total_seconds() / 86400, 1)
                    proximity = max(0.1, 1.0 - (days_from_mid / total_days))
                    items.append(ScoredItem(
                        id=fid, table=table, text=text,
                        score=proximity * temporal_weight(tc, ds),
                        metadata={"temporal_class": tc, "decay_score": ds},
                    ))
            except Exception:
                continue

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]
    finally:
        conn.close()


def retrieve_code(
    db_path: str,
    query_text: str,
    scope: Optional[str],
    limit: int,
) -> list[ScoredItem]:
    """Code graph retrieval: find symbols and dependencies matching the query."""
    from . import code_graph

    conn = db.get_connection(read_only=True, db_path=db_path)
    try:
        items: list[ScoredItem] = []
        seen: set[str] = set()

        # Strategy 1: Symbol name search
        # Extract potential symbol references from query
        symbol_refs = _extract_symbol_refs(query_text)
        for ref in symbol_refs:
            results = code_graph.search_symbol(conn, ref, scope=scope)
            for sym in results:
                key = f"{sym['file_path']}:{sym['symbol_name']}"
                if key not in seen:
                    seen.add(key)
                    items.append(ScoredItem(
                        id=sym["id"], table="code_graph",
                        text=f"{sym['file_path']}:{sym['line_number']} {sym['symbol_type']} {sym['symbol_name']}{sym.get('signature', '')}",
                        score=1.0 if ref.lower() == sym["symbol_name"].lower() else 0.7,
                        metadata={"file_path": sym["file_path"], "symbol_type": sym["symbol_type"],
                                  "line_number": sym["line_number"], "signature": sym.get("signature", "")},
                    ))

        # Strategy 2: File path matching — pull symbols for matched files
        file_paths = _extract_file_paths(query_text)
        for fp in file_paths:
            matched_files: list[tuple[str, list[dict]]] = []
            syms = code_graph.get_file_symbols(conn, fp)
            if syms:
                matched_files.append((fp, syms))
            else:
                # Try partial match
                rows = conn.execute(
                    "SELECT DISTINCT file_path FROM code_symbols WHERE file_path LIKE ?",
                    [f"%{fp}%"]
                ).fetchall()
                for row in rows:
                    partial_syms = code_graph.get_file_symbols(conn, row[0])
                    if partial_syms:
                        matched_files.append((row[0], partial_syms))

            for matched_path, matched_syms in matched_files:
                key = f"file:{matched_path}"
                if key not in seen:
                    seen.add(key)
                    sym_names = ", ".join(s["symbol_name"] for s in matched_syms[:10])
                    items.append(ScoredItem(
                        id=matched_path, table="code_graph",
                        text=f"{matched_path} defines: {sym_names}",
                        score=0.9 if matched_path == fp else 0.8,
                        metadata={"file_path": matched_path, "symbol_count": len(matched_syms)},
                    ))

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]
    finally:
        conn.close()


_SYMBOL_RE = re.compile(r'\b([A-Z][a-zA-Z0-9]+|[a-z_][a-z0-9_]{2,})\b')

def _extract_symbol_refs(text: str) -> list[str]:
    """Extract potential symbol references (function/class names) from text."""
    # Filter out common English words
    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
                 "her", "was", "one", "our", "out", "how", "does", "what", "this",
                 "that", "with", "have", "from", "they", "been", "said", "each",
                 "which", "their", "will", "other", "about", "many", "then", "them",
                 "these", "some", "would", "make", "like", "into", "time", "very",
                 "when", "come", "could", "now", "than", "first", "been", "its",
                 "who", "way", "may", "down", "should", "called", "use", "show",
                 "work", "right", "memory", "file", "code", "function", "class",
                 "module", "import", "return", "value", "type", "name", "data",
                 "list", "dict", "string", "number", "true", "false", "none"}
    matches = _SYMBOL_RE.findall(text)
    return [m for m in matches if m.lower() not in stopwords and len(m) >= 3]


def retrieve_path(
    db_path: str,
    query_text: str,
    scope: Optional[str],
    limit: int,
) -> list[ScoredItem]:
    """File-path retrieval: find items whose file_paths overlap with paths in the query."""
    paths = _extract_file_paths(query_text)
    if not paths:
        return []

    conn = db.get_connection(read_only=True, db_path=db_path)
    try:
        scope_sql, scope_params = db._scope_filter(scope)
        items: list[ScoredItem] = []
        seen: set[str] = set()

        # Tables that carry file_paths: facts, guardrails, procedures, error_solutions
        tables = [
            ("facts", "id, text, temporal_class, decay_score, file_paths"),
            ("guardrails", "id, warning AS text, temporal_class, decay_score, file_paths"),
            ("procedures", "id, task_description AS text, temporal_class, decay_score, file_paths"),
            ("error_solutions", "id, error_pattern AS text, confidence, file_paths"),
        ]

        for table, cols in tables:
            try:
                rows = conn.execute(f"""
                    SELECT {cols}
                    FROM {table}
                    WHERE is_active = TRUE AND file_paths IS NOT NULL
                      AND len(file_paths) > 0
                      {scope_sql}
                    LIMIT 500
                """, scope_params).fetchall()
            except Exception:
                continue

            col_names = [c.strip().split(" AS ")[-1].split()[-1] for c in cols.split(",")]
            for row in rows:
                row_dict = dict(zip(col_names, row))
                rid = row_dict["id"]
                if rid in seen:
                    continue
                row_fps = row_dict.get("file_paths") or []
                # Score by how many query paths match
                matches = sum(1 for qp in paths if any(qp in fp or fp in qp for fp in row_fps))
                if matches > 0:
                    seen.add(rid)
                    ds = row_dict.get("decay_score", 1.0) or 1.0
                    items.append(ScoredItem(
                        id=rid, table=table, text=row_dict["text"],
                        score=matches * ds,
                        metadata={"file_paths": row_fps, "path_matches": matches},
                    ))

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]
    finally:
        conn.close()


_FILE_PATH_RE = re.compile(
    r'(?:^|[\s\'"])('
    r'(?:[a-zA-Z]:)?'              # optional drive letter (Windows)
    r'(?:\.{0,2}/)?'               # optional leading ./ ../ /
    r'(?:[\w.@~-]+/)*'             # directory components
    r'[\w.@~-]+\.[a-zA-Z0-9]{1,10}'  # filename with extension
    r')'
)


def _extract_file_paths(text: str) -> list[str]:
    """Extract plausible file paths from query text."""
    return _FILE_PATH_RE.findall(text)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[ScoredItem]],
    k: int = RRF_K,
) -> list[ScoredItem]:
    """Merge multiple ranked lists using RRF. Returns items sorted by fused score."""
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, ScoredItem] = {}

    for results in result_lists:
        for rank, item in enumerate(results, 1):
            scores[item.id] += 1.0 / (k + rank)
            if item.id not in items:
                items[item.id] = item

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for item_id, rrf_score in merged:
        item = items[item_id]
        result.append(ScoredItem(
            id=item.id, table=item.table, text=item.text,
            score=rrf_score, metadata=item.metadata,
        ))
    return result


# ── Main parallel retrieval entry point ───────────────────────────────────

def parallel_retrieve(
    db_path: str,
    query_text: str,
    query_embedding: Optional[list[float]],
    scope: Optional[str],
    limit: int = 10,
    timeout_ms: int = PROMPT_RECALL_TIMEOUT_MS,
    strategies: Optional[list[str]] = None,
) -> RetrievalResult:
    """
    Run enabled retrieval strategies in parallel, fuse with RRF.
    Returns results within timeout_ms budget; stragglers are skipped.
    """
    active_strategies = strategies or RETRIEVAL_STRATEGIES
    start = time.monotonic()

    strategy_fns = {
        "semantic": lambda: retrieve_semantic(db_path, query_embedding, scope, limit * 2) if query_embedding else [],
        "bm25": lambda: retrieve_bm25(db_path, query_text, scope, limit * 2),
        "graph": lambda: retrieve_graph(db_path, query_text, scope, limit * 2),
        "temporal": lambda: retrieve_temporal(db_path, query_text, scope, limit * 2),
        "path": lambda: retrieve_path(db_path, query_text, scope, limit * 2),
        "code": lambda: retrieve_code(db_path, query_text, scope, limit * 2),
    }

    results: dict[str, list[ScoredItem]] = {}
    timings: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=len(active_strategies)) as pool:
        futures = {}
        for name in active_strategies:
            if name in strategy_fns:
                futures[pool.submit(strategy_fns[name])] = name

        deadline = timeout_ms / 1000.0
        try:
            for future in as_completed(futures, timeout=deadline):
                name = futures[future]
                t0 = time.monotonic()
                try:
                    results[name] = future.result(timeout=0)
                except Exception as e:
                    print(f"[memory] {name} retrieval failed: {e}", file=sys.stderr)
                    results[name] = []
                timings[name] = (time.monotonic() - t0) * 1000
        except TimeoutError:
            # Some strategies didn't finish in time — use whatever we have
            for future, name in futures.items():
                if future.done() and name not in results:
                    try:
                        results[name] = future.result(timeout=0)
                    except Exception:
                        results[name] = []

    elapsed_ms = (time.monotonic() - start) * 1000
    exceeded = elapsed_ms > timeout_ms

    # Timing trace (always, to stderr)
    total_results = sum(len(results.get(s, [])) for s in active_strategies)
    if total_results > 0:
        strategy_names = ["semantic", "bm25", "graph", "temporal", "path", "code"]
        parts = []
        for s in strategy_names:
            if s in timings:
                parts.append(f"{s}={int(timings[s])}ms")
            else:
                parts.append(f"{s}=\u2014")
        print(
            f"[memory] Recall: {' '.join(parts)} total={int(elapsed_ms)}ms",
            file=sys.stderr,
        )

    # Fuse results
    all_lists = [results.get(s, []) for s in active_strategies if s in results]
    if all_lists:
        merged = reciprocal_rank_fusion(all_lists)
    else:
        merged = []

    return RetrievalResult(
        items=merged[:limit],
        elapsed_ms=elapsed_ms,
        exceeded_budget=exceeded,
        strategy_counts={s: len(results.get(s, [])) for s in active_strategies},
        strategy_timings=timings,
    )


# ── Helpers ───────────────────────────────────────────────────────────────

# Tables searched by semantic retrieval
_SEARCH_TABLES = {
    "facts": "id, text, temporal_class, decay_score, session_count, last_seen_at, category, confidence, scope",
    "ideas": "id, text, temporal_class, decay_score, session_count, last_seen_at, idea_type, scope",
    "decisions": "id, text, temporal_class, decay_score, session_count, last_seen_at, scope",
    "observations": "id, text, proof_count, source_fact_ids, temporal_class, decay_score, session_count, scope",
}

# Tables searched by BM25
_BM25_TABLES = {
    "facts": {"text_col": "text", "select_cols": "id, text, temporal_class, decay_score, scope"},
    "ideas": {"text_col": "text", "select_cols": "id, text, idea_type, scope"},
    "decisions": {"text_col": "text", "select_cols": "id, text, temporal_class, scope"},
    "observations": {"text_col": "text", "select_cols": "id, text, proof_count, scope"},
}


def _entities_in_text(
    text: str,
    known_entities: list[str],
    db_path: Optional[str] = None,
    top_k: int = 5,
) -> list[str]:
    """Return which known entities are relevant to text.

    First tries exact substring matching (fast). If no matches, falls back
    to embedding-based similarity search against entity embeddings.
    """
    text_lower = text.lower()
    found = []
    for entity in known_entities:
        pattern = re.escape(entity.lower())
        if re.search(r'\b' + pattern + r'\b', text_lower):
            found.append(entity)
    if found:
        return found

    # Fallback: embedding-based entity matching
    try:
        from .embeddings import embed_query
        query_vec = embed_query(text)
        if not query_vec:
            return []

        from .config import DB_PATH
        path = db_path or str(DB_PATH)
        conn = db.get_connection(read_only=True, db_path=path)
        try:
            rows = conn.execute("""
                SELECT name,
                       list_cosine_similarity(embedding, ?::FLOAT[]) AS score
                FROM entities
                WHERE is_active = TRUE AND embedding IS NOT NULL
                ORDER BY score DESC
                LIMIT ?
            """, [query_vec, top_k]).fetchall()
            for name, score in rows:
                if score and score >= 0.40:
                    found.append(name)
        finally:
            conn.close()
    except Exception:
        pass

    return found


def _extract_date_range(text: str) -> Optional[tuple[datetime, datetime]]:
    """
    Extract temporal references from query text.
    Returns (start_date, end_date) or None if no temporal reference found.
    """
    now = datetime.now(timezone.utc)
    text_lower = text.lower()

    # Relative patterns
    patterns = {
        r"\byesterday\b": (now - timedelta(days=1), now),
        r"\blast week\b": (now - timedelta(days=7), now),
        r"\bthis week\b": (now - timedelta(days=now.weekday()), now),
        r"\blast month\b": (now - timedelta(days=30), now),
        r"\btoday\b": (now.replace(hour=0, minute=0, second=0), now),
        r"\brecently\b": (now - timedelta(days=7), now),
        r"\b(\d+)\s+days?\s+ago\b": None,  # handled specially
        r"\blast\s+(\d+)\s+days?\b": None,  # handled specially
    }

    for pattern, date_range in patterns.items():
        if date_range is not None:
            if re.search(pattern, text_lower):
                return date_range

    # "N days ago"
    m = re.search(r"\b(\d+)\s+days?\s+ago\b", text_lower)
    if m:
        days = int(m.group(1))
        return (now - timedelta(days=days + 1), now - timedelta(days=max(0, days - 1)))

    # "last N days"
    m = re.search(r"\blast\s+(\d+)\s+days?\b", text_lower)
    if m:
        days = int(m.group(1))
        return (now - timedelta(days=days), now)

    # ISO date: 2024-01-15
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text_lower)
    if m:
        try:
            d = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
            return (d, d + timedelta(days=1))
        except ValueError:
            pass

    return None
