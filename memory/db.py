"""
db.py — DuckDB schema, schema migrations, and CRUD for the memory system.

Schema is append-friendly: new facts are deduplicated via cosine similarity.
Migrations are versioned; new columns/tables are added non-destructively.
"""
from __future__ import annotations

import uuid
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

from .config import (
    DB_PATH, EMBEDDING_DIM, DEDUP_THRESHOLD, RECALL_THRESHOLD,
    SESSION_LONG_FACTS_LIMIT, SESSION_MEDIUM_FACTS_LIMIT,
    SESSION_DECISIONS_LIMIT, SESSION_ENTITIES_LIMIT,
    PROMPT_FACTS_LIMIT, PROMPT_IDEAS_LIMIT, PROMPT_RELS_LIMIT,
    PROMPT_QUESTIONS_LIMIT,
    GLOBAL_SCOPE, AUTO_PROMOTE_PROJECT_COUNT,
)

# ── Schema migrations ──────────────────────────────────────────────────────
# Each entry: (version: int, description: str, sql: str)
# sql may contain multiple statements separated by semicolons.
# Each migration is applied once and recorded in schema_migrations.

MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "Initial schema", f"""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description VARCHAR
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id              VARCHAR PRIMARY KEY,
            trigger         VARCHAR DEFAULT 'unknown',
            cwd             VARCHAR,
            transcript_path VARCHAR,
            message_count   INTEGER DEFAULT 0,
            summary         TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS facts (
            id              VARCHAR PRIMARY KEY,
            text            TEXT NOT NULL,
            category        VARCHAR DEFAULT 'contextual',
            temporal_class  VARCHAR DEFAULT 'short',
            confidence      VARCHAR DEFAULT 'medium',
            decay_score     DOUBLE  DEFAULT 1.0,
            session_count   INTEGER DEFAULT 1,
            source_session  VARCHAR,
            embedding       DOUBLE[],
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE
        );

        CREATE TABLE IF NOT EXISTS ideas (
            id              VARCHAR PRIMARY KEY,
            text            TEXT NOT NULL,
            idea_type       VARCHAR DEFAULT 'insight',
            temporal_class  VARCHAR DEFAULT 'short',
            decay_score     DOUBLE  DEFAULT 1.0,
            session_count   INTEGER DEFAULT 1,
            source_session  VARCHAR,
            embedding       DOUBLE[],
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE
        );

        CREATE TABLE IF NOT EXISTS entities (
            id              VARCHAR PRIMARY KEY,
            name            VARCHAR NOT NULL,
            name_lower      VARCHAR NOT NULL,
            entity_type     VARCHAR DEFAULT 'general',
            first_seen_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_count   INTEGER DEFAULT 1,
            embedding       DOUBLE[]
        );

        CREATE UNIQUE INDEX IF NOT EXISTS entities_name_lower ON entities(name_lower);

        CREATE TABLE IF NOT EXISTS relationships (
            id              VARCHAR PRIMARY KEY,
            from_entity     VARCHAR NOT NULL,
            to_entity       VARCHAR NOT NULL,
            rel_type        VARCHAR NOT NULL,
            description     TEXT,
            strength        DOUBLE  DEFAULT 1.0,
            session_count   INTEGER DEFAULT 1,
            source_session  VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS rels_unique
            ON relationships(from_entity, to_entity, rel_type);

        CREATE TABLE IF NOT EXISTS decisions (
            id              VARCHAR PRIMARY KEY,
            text            TEXT NOT NULL,
            temporal_class  VARCHAR DEFAULT 'medium',
            decay_score     DOUBLE  DEFAULT 1.0,
            session_count   INTEGER DEFAULT 1,
            embedding       DOUBLE[],
            source_session  VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE
        );

        CREATE TABLE IF NOT EXISTS open_questions (
            id              VARCHAR PRIMARY KEY,
            text            TEXT NOT NULL,
            resolved        BOOLEAN DEFAULT FALSE,
            embedding       DOUBLE[],
            source_session  VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE
        );

        CREATE TABLE IF NOT EXISTS fact_entity_links (
            fact_id         VARCHAR NOT NULL,
            entity_name     VARCHAR NOT NULL,
            PRIMARY KEY (fact_id, entity_name)
        );
    """),
    (2, "Add project scope columns", """
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS scope VARCHAR DEFAULT '__global__';
        ALTER TABLE ideas ADD COLUMN IF NOT EXISTS scope VARCHAR DEFAULT '__global__';
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS scope VARCHAR DEFAULT '__global__';
        ALTER TABLE open_questions ADD COLUMN IF NOT EXISTS scope VARCHAR DEFAULT '__global__';
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS scope VARCHAR DEFAULT '__global__';
        ALTER TABLE relationships ADD COLUMN IF NOT EXISTS scope VARCHAR DEFAULT '__global__';
        ALTER TABLE sessions ADD COLUMN IF NOT EXISTS scope VARCHAR DEFAULT '__global__';

        CREATE TABLE IF NOT EXISTS item_scopes (
            item_id     VARCHAR NOT NULL,
            item_table  VARCHAR NOT NULL,
            scope       VARCHAR NOT NULL,
            PRIMARY KEY (item_id, item_table, scope)
        );
    """),
    (3, "Add deactivated_at columns and entities is_active for /forget support", """
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMP;
        ALTER TABLE ideas ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMP;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMP;
        ALTER TABLE relationships ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMP;
        ALTER TABLE open_questions ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMP;
        DROP INDEX IF EXISTS entities_name_lower;
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;
        ALTER TABLE entities ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMP;
        CREATE UNIQUE INDEX IF NOT EXISTS entities_name_lower ON entities(name_lower);
    """),
]


# ── Connection ─────────────────────────────────────────────────────────────

def get_connection(
    read_only: bool = False,
    db_path: Optional[str] = None,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, initialise schema on first run.

    db_path overrides the config path — useful for tests.
    """
    import memory.config as _config  # read at call time so test overrides work
    path = Path(db_path) if db_path else _config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        _run_migrations(conn)
    _try_load_vss(conn)
    return conn


def _try_load_vss(conn: duckdb.DuckDBPyConnection) -> None:
    """Optionally load the VSS extension for HNSW index acceleration."""
    try:
        conn.execute("LOAD vss")
    except Exception:
        pass  # VSS not installed; list_cosine_similarity still works via scan


def _run_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply any outstanding schema migrations in version order."""
    # Bootstrap: the migrations table may not exist yet
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description VARCHAR
        )
    """)
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}

    for version, description, sql in sorted(MIGRATIONS, key=lambda m: m[0]):
        if version in applied:
            continue
        # Execute each statement individually (DuckDB doesn't run multi-statement strings)
        for stmt in _split_sql(sql):
            if stmt:
                conn.execute(stmt)
        conn.execute(
            "INSERT INTO schema_migrations(version, description) VALUES (?, ?)",
            [version, description],
        )


def _split_sql(sql: str) -> list[str]:
    """Split a multi-statement SQL string on semicolons."""
    return [s.strip() for s in sql.split(";") if s.strip()]


# ── Helpers ────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return str(uuid.uuid4())


def _cosine_py(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity fallback."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _vector_search(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    query_embedding: list[float],
    select_cols: str,   # comma-separated logical columns (NOT embedding or score)
    where_extra: str,
    limit: int,
    threshold: float,
) -> list[dict]:
    """
    Find rows in `table` ordered by cosine similarity to query_embedding.

    select_cols controls which columns are returned in the result dicts.
    embedding is always fetched internally for the fallback path but is
    stripped from the returned dicts.

    Falls back to Python cosine similarity if list_cosine_similarity fails.
    """
    col_list = [c.strip() for c in select_cols.split(",") if c.strip()]
    all_col_names = col_list + ["embedding", "score"]

    try:
        rows = conn.execute(f"""
            SELECT {select_cols}, embedding,
                   list_cosine_similarity(embedding, ?) AS score
            FROM {table}
            WHERE is_active = TRUE AND embedding IS NOT NULL
              {where_extra}
            ORDER BY score DESC
            LIMIT ?
        """, [query_embedding, limit]).fetchall()
    except Exception:
        rows_all = conn.execute(f"""
            SELECT {select_cols}, embedding
            FROM {table}
            WHERE is_active = TRUE AND embedding IS NOT NULL
              {where_extra}
        """).fetchall()
        scored = [
            (*r, _cosine_py(r[-1], query_embedding))
            for r in rows_all
            if r[-1] is not None
        ]
        scored.sort(key=lambda x: x[-1], reverse=True)
        rows = scored[:limit]

    result = []
    for row in rows:
        d = dict(zip(all_col_names, row))
        if d.get("score", 0) >= threshold:
            d.pop("embedding", None)
            result.append(d)
    return result


# ── Scope helpers ─────────────────────────────────────────────────────────

def _scope_filter(scope: Optional[str]) -> str:
    """
    Return a SQL WHERE fragment to filter by scope.
    If scope is None, returns empty string (no filter = all scopes).
    If scope is a project path, matches that scope OR global.
    """
    if scope is None:
        return ""
    # Match project-local items AND global items
    return f"AND (scope = '{_sql_escape(scope)}' OR scope = '{GLOBAL_SCOPE}')"


def _scope_filter_exact(scope: str) -> str:
    """Strict filter — only items with exactly this scope."""
    return f"AND scope = '{_sql_escape(scope)}'"


def _sql_escape(s: str) -> str:
    """Escape single quotes for inline SQL."""
    return s.replace("'", "''")


def _track_item_scope(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    item_table: str,
    scope: str,
) -> None:
    """Record that an item was seen in a given project scope."""
    try:
        conn.execute(
            "INSERT INTO item_scopes(item_id, item_table, scope) VALUES (?, ?, ?)"
            " ON CONFLICT DO NOTHING",
            [item_id, item_table, scope],
        )
    except Exception:
        pass


def _maybe_auto_promote(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    item_table: str,
) -> None:
    """
    If an item has been seen in N+ distinct project scopes,
    promote it to global scope.
    """
    try:
        row = conn.execute("""
            SELECT COUNT(DISTINCT scope) FROM item_scopes
            WHERE item_id = ? AND item_table = ? AND scope != ?
        """, [item_id, item_table, GLOBAL_SCOPE]).fetchone()
        if row and row[0] >= AUTO_PROMOTE_PROJECT_COUNT:
            conn.execute(
                f"UPDATE {item_table} SET scope = ? WHERE id = ?",
                [GLOBAL_SCOPE, item_id],
            )
    except Exception:
        pass


def promote_to_global(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    item_table: str,
) -> bool:
    """
    Manually promote an item to global scope.
    Returns True if the item was found and promoted.
    """
    row = conn.execute(
        f"SELECT id, scope FROM {item_table} WHERE id = ?", [item_id]
    ).fetchone()
    if not row:
        return False
    if row[1] == GLOBAL_SCOPE:
        return True  # already global
    conn.execute(
        f"UPDATE {item_table} SET scope = ? WHERE id = ?",
        [GLOBAL_SCOPE, item_id],
    )
    return True


# ── Session ────────────────────────────────────────────────────────────────

def upsert_session(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    trigger: str,
    cwd: str,
    transcript_path: str,
    message_count: int,
    summary: str,
    scope: str = GLOBAL_SCOPE,
) -> None:
    conn.execute("""
        INSERT INTO sessions(id, trigger, cwd, transcript_path, message_count, summary, scope)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            message_count = excluded.message_count,
            summary       = excluded.summary
    """, [session_id, trigger, cwd, transcript_path, message_count, summary, scope])


# ── Facts ──────────────────────────────────────────────────────────────────

def upsert_fact(
    conn: duckdb.DuckDBPyConnection,
    text: str,
    category: str,
    temporal_class: str,
    confidence: str,
    embedding: Optional[list[float]],
    session_id: str,
    decay_fn,
    scope: str = GLOBAL_SCOPE,
) -> tuple[str, bool]:
    """
    Insert a new fact or reinforce an existing similar one.
    Returns (fact_id, is_new).
    """
    now = _now()

    # Deduplication: find a near-identical existing fact (across all scopes)
    existing = None
    if embedding:
        hits = _vector_search(
            conn, "facts", embedding,
            "id, text, temporal_class, decay_score, session_count, last_seen_at",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if existing:
        new_session_count = existing["session_count"] + 1
        new_decay = decay_fn(
            existing["last_seen_at"], new_session_count, existing["temporal_class"]
        )
        new_class = _promote_class(
            existing["temporal_class"], new_session_count, existing["last_seen_at"]
        )
        conn.execute("""
            UPDATE facts SET
                last_seen_at   = ?,
                session_count  = ?,
                decay_score    = ?,
                temporal_class = ?
            WHERE id = ?
        """, [now, new_session_count, new_decay, new_class, existing["id"]])
        _track_item_scope(conn, existing["id"], "facts", scope)
        _maybe_auto_promote(conn, existing["id"], "facts")
        return existing["id"], False

    fid = _uid()
    conn.execute("""
        INSERT INTO facts
            (id, text, category, temporal_class, confidence, decay_score,
             embedding, source_session, created_at, last_seen_at, scope)
        VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, ?, ?, ?)
    """, [fid, text, category, temporal_class, confidence, embedding,
          session_id, now, now, scope])
    _track_item_scope(conn, fid, "facts", scope)
    return fid, True


def get_facts_by_temporal(
    conn: duckdb.DuckDBPyConnection,
    temporal_class: str,
    limit: int,
    scope: Optional[str] = None,
) -> list[dict]:
    scope_filter = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, text, category, temporal_class, confidence, decay_score, scope
        FROM facts
        WHERE is_active = TRUE AND temporal_class = ?
          {scope_filter}
        ORDER BY decay_score DESC, last_seen_at DESC
        LIMIT ?
    """, [temporal_class, limit]).fetchall()
    return [
        dict(zip(["id","text","category","temporal_class","confidence","decay_score","scope"], r))
        for r in rows
    ]


def search_facts(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = PROMPT_FACTS_LIMIT,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    return _vector_search(
        conn, "facts", query_embedding,
        "id, text, temporal_class, decay_score, session_count, last_seen_at, category, confidence, scope",
        _scope_filter(scope), limit, threshold,
    )


# ── Ideas ──────────────────────────────────────────────────────────────────

def upsert_idea(
    conn: duckdb.DuckDBPyConnection,
    text: str,
    idea_type: str,
    temporal_class: str,
    embedding: Optional[list[float]],
    session_id: str,
    decay_fn,
    scope: str = GLOBAL_SCOPE,
) -> tuple[str, bool]:
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "ideas", embedding,
            "id, text, temporal_class, decay_score, session_count, last_seen_at",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if existing:
        new_session_count = existing["session_count"] + 1
        new_decay = decay_fn(
            existing["last_seen_at"], new_session_count, existing["temporal_class"]
        )
        new_class = _promote_class(
            existing["temporal_class"], new_session_count, existing["last_seen_at"]
        )
        conn.execute("""
            UPDATE ideas SET last_seen_at=?, session_count=?,
                             decay_score=?, temporal_class=?
            WHERE id=?
        """, [now, new_session_count, new_decay, new_class, existing["id"]])
        _track_item_scope(conn, existing["id"], "ideas", scope)
        _maybe_auto_promote(conn, existing["id"], "ideas")
        return existing["id"], False

    iid = _uid()
    conn.execute("""
        INSERT INTO ideas
            (id, text, idea_type, temporal_class, decay_score,
             embedding, source_session, created_at, last_seen_at, scope)
        VALUES (?, ?, ?, ?, 1.0, ?, ?, ?, ?, ?)
    """, [iid, text, idea_type, temporal_class, embedding, session_id, now, now, scope])
    _track_item_scope(conn, iid, "ideas", scope)
    return iid, True


def search_ideas(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = PROMPT_IDEAS_LIMIT,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    return _vector_search(
        conn, "ideas", query_embedding,
        "id, text, temporal_class, decay_score, session_count, last_seen_at, idea_type, scope",
        _scope_filter(scope), limit, threshold,
    )


# ── Entities ───────────────────────────────────────────────────────────────

def upsert_entity(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    entity_type: str = "general",
    embedding: Optional[list[float]] = None,
    scope: str = GLOBAL_SCOPE,
) -> str:
    now = _now()
    name_lower = name.strip().lower()

    existing = conn.execute(
        "SELECT id FROM entities WHERE name_lower = ?", [name_lower]
    ).fetchone()

    if existing:
        conn.execute("""
            UPDATE entities SET
                last_seen_at  = ?,
                session_count = session_count + 1
            WHERE name_lower = ?
        """, [now, name_lower])
        _track_item_scope(conn, existing[0], "entities", scope)
        _maybe_auto_promote(conn, existing[0], "entities")
        return existing[0]

    eid = _uid()
    conn.execute("""
        INSERT INTO entities
            (id, name, name_lower, entity_type, embedding, first_seen_at, last_seen_at, scope)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [eid, name, name_lower, entity_type, embedding, now, now, scope])
    _track_item_scope(conn, eid, "entities", scope)
    return eid


def get_top_entities(
    conn: duckdb.DuckDBPyConnection,
    limit: int = SESSION_ENTITIES_LIMIT,
    scope: Optional[str] = None,
) -> list[str]:
    scope_filter = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT name FROM entities
        WHERE 1=1 {scope_filter}
        ORDER BY session_count DESC, last_seen_at DESC
        LIMIT ?
    """, [limit]).fetchall()
    return [r[0] for r in rows]


# ── Relationships ──────────────────────────────────────────────────────────

def upsert_relationship(
    conn: duckdb.DuckDBPyConnection,
    from_entity: str,
    to_entity: str,
    rel_type: str,
    description: str,
    session_id: str,
    scope: str = GLOBAL_SCOPE,
) -> str:
    now = _now()

    existing = conn.execute("""
        SELECT id FROM relationships
        WHERE from_entity = ? AND to_entity = ? AND rel_type = ?
    """, [from_entity, to_entity, rel_type]).fetchone()

    if existing:
        conn.execute("""
            UPDATE relationships SET
                last_seen_at  = ?,
                session_count = session_count + 1,
                strength      = LEAST(strength + 0.1, 2.0)
            WHERE id = ?
        """, [now, existing[0]])
        _track_item_scope(conn, existing[0], "relationships", scope)
        _maybe_auto_promote(conn, existing[0], "relationships")
        return existing[0]

    rid = _uid()
    conn.execute("""
        INSERT INTO relationships
            (id, from_entity, to_entity, rel_type, description,
             source_session, created_at, last_seen_at, scope)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [rid, from_entity, to_entity, rel_type, description, session_id, now, now, scope])
    _track_item_scope(conn, rid, "relationships", scope)
    return rid


def get_relationships_for_entities(
    conn: duckdb.DuckDBPyConnection,
    entity_names: list[str],
    limit: int = PROMPT_RELS_LIMIT,
    scope: Optional[str] = None,
) -> list[dict]:
    if not entity_names:
        return []
    placeholders = ", ".join("?" * len(entity_names))
    names_lower = [n.lower() for n in entity_names]
    scope_filter = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT from_entity, to_entity, rel_type, description, strength, session_count
        FROM relationships
        WHERE is_active = TRUE
          AND (LOWER(from_entity) IN ({placeholders})
               OR LOWER(to_entity) IN ({placeholders}))
          {scope_filter}
        ORDER BY strength DESC, session_count DESC
        LIMIT ?
    """, names_lower + names_lower + [limit]).fetchall()
    return [
        dict(zip(["from_entity","to_entity","rel_type","description","strength","session_count"], r))
        for r in rows
    ]


def get_all_relationships(
    conn: duckdb.DuckDBPyConnection,
    scope: Optional[str] = None,
) -> list[dict]:
    """All active relationships — used for graph visualisation."""
    scope_filter = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT from_entity, to_entity, rel_type, description, strength, session_count
        FROM relationships
        WHERE is_active = TRUE {scope_filter}
        ORDER BY strength DESC
    """).fetchall()
    return [
        dict(zip(["from","to","rel_type","description","strength","session_count"], r))
        for r in rows
    ]


# ── Decisions ──────────────────────────────────────────────────────────────

def upsert_decision(
    conn: duckdb.DuckDBPyConnection,
    text: str,
    temporal_class: str,
    embedding: Optional[list[float]],
    session_id: str,
    decay_fn,
    scope: str = GLOBAL_SCOPE,
) -> tuple[str, bool]:
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "decisions", embedding,
            "id, text, temporal_class, decay_score, session_count, last_seen_at",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if existing:
        conn.execute("""
            UPDATE decisions SET last_seen_at=?, session_count=session_count+1 WHERE id=?
        """, [now, existing["id"]])
        _track_item_scope(conn, existing["id"], "decisions", scope)
        _maybe_auto_promote(conn, existing["id"], "decisions")
        return existing["id"], False

    did = _uid()
    conn.execute("""
        INSERT INTO decisions
            (id, text, temporal_class, decay_score, embedding,
             source_session, created_at, last_seen_at, scope)
        VALUES (?, ?, ?, 1.0, ?, ?, ?, ?, ?)
    """, [did, text, temporal_class, embedding, session_id, now, now, scope])
    _track_item_scope(conn, did, "decisions", scope)
    return did, True


def get_decisions(
    conn: duckdb.DuckDBPyConnection,
    limit: int = SESSION_DECISIONS_LIMIT,
    scope: Optional[str] = None,
) -> list[dict]:
    scope_filter = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, text, temporal_class, decay_score
        FROM decisions
        WHERE is_active = TRUE {scope_filter}
        ORDER BY decay_score DESC, last_seen_at DESC
        LIMIT ?
    """, [limit]).fetchall()
    return [dict(zip(["id","text","temporal_class","decay_score"], r)) for r in rows]


# ── Open questions ─────────────────────────────────────────────────────────

def upsert_question(
    conn: duckdb.DuckDBPyConnection,
    text: str,
    embedding: Optional[list[float]],
    session_id: str,
    scope: str = GLOBAL_SCOPE,
) -> tuple[str, bool]:
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "open_questions", embedding,
            "id, text, resolved, last_seen_at",
            "AND resolved = FALSE", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if existing:
        conn.execute("UPDATE open_questions SET last_seen_at=? WHERE id=?", [now, existing["id"]])
        return existing["id"], False

    qid = _uid()
    conn.execute("""
        INSERT INTO open_questions
            (id, text, embedding, source_session, created_at, last_seen_at, scope)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [qid, text, embedding, session_id, now, now, scope])
    return qid, True


def search_questions(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = PROMPT_QUESTIONS_LIMIT,
    scope: Optional[str] = None,
) -> list[dict]:
    scope_filter = _scope_filter(scope)
    return _vector_search(
        conn, "open_questions", query_embedding,
        "id, text, resolved, last_seen_at",
        f"AND resolved = FALSE {scope_filter}", limit, RECALL_THRESHOLD,
    )


# ── Fact-entity links ──────────────────────────────────────────────────────

def link_fact_entities(
    conn: duckdb.DuckDBPyConnection,
    fact_id: str,
    entity_names: list[str],
) -> None:
    for name in entity_names:
        try:
            conn.execute(
                "INSERT INTO fact_entity_links(fact_id, entity_name) VALUES (?, ?)"
                " ON CONFLICT DO NOTHING",
                [fact_id, name],
            )
        except Exception:
            pass


# ── Decay & reclassification ───────────────────────────────────────────────

def _promote_class(current: str, session_count: int, created_at) -> str:
    """
    Rules: session_count and age can only promote, never demote a class.
    (LLM demotions happen at extraction time.)
    """
    from .config import (
        SESSION_COUNT_MEDIUM, SESSION_COUNT_LONG,
        AGE_MEDIUM_DAYS, AGE_LONG_DAYS,
    )
    if current == "long":
        return "long"

    now = _now()
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (now - created_at).days

    if session_count >= SESSION_COUNT_LONG or age_days >= AGE_LONG_DAYS:
        return "long"
    if current == "medium":
        return "medium"
    if session_count >= SESSION_COUNT_MEDIUM or age_days >= AGE_MEDIUM_DAYS:
        return "medium"
    return current


def apply_decay_pass(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Update decay_scores for all active items; mark forgotten ones inactive;
    reclassify temporal classes upward.
    Returns a summary dict.
    """
    from .decay import compute_decay_score

    now = _now()
    stats = {"updated": 0, "forgotten": 0, "promoted": 0}

    for table in ("facts", "ideas", "decisions"):
        rows = conn.execute(f"""
            SELECT id, temporal_class, session_count, last_seen_at, created_at
            FROM {table}
            WHERE is_active = TRUE
        """).fetchall()

        for row_id, tc, sc, last_seen, created in rows:
            new_score = compute_decay_score(last_seen, sc, tc)
            new_class = _promote_class(tc, sc, created)
            if new_class != tc:
                stats["promoted"] += 1
            if new_score < 0.05 and tc == "short":
                conn.execute(f"UPDATE {table} SET is_active=FALSE WHERE id=?", [row_id])
                stats["forgotten"] += 1
            else:
                conn.execute(f"""
                    UPDATE {table} SET decay_score=?, temporal_class=? WHERE id=?
                """, [new_score, new_class, row_id])
                stats["updated"] += 1

    return stats


# ── Database statistics ────────────────────────────────────────────────────

# ── Forget (search, soft-delete, purge) ───────────────────────────────────

# Tables that have a 'text' column and support soft-delete
_TEXT_TABLES = {
    "facts":          "text",
    "ideas":          "text",
    "decisions":      "text",
    "open_questions": "text",
}

# Tables where the display text comes from a different column
_OTHER_TABLES = {
    "entities":      "name",
    "relationships": "description",
}

_ALL_FORGET_TABLES = {**_TEXT_TABLES, **_OTHER_TABLES}


def search_all_by_text(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    scope: Optional[str] = None,
) -> list[dict]:
    """Search all tables for items whose text/name/description contains query (case-insensitive)."""
    results: list[dict] = []
    query_lower = query.lower()

    for table, col in _ALL_FORGET_TABLES.items():
        active_filter = "is_active = TRUE AND" if table != "entities" else (
            "is_active = TRUE AND" if "is_active" in _get_columns(conn, table) else ""
        )
        # After migration 3, entities always has is_active
        try:
            active_clause = "is_active = TRUE AND " if table in ("facts", "ideas", "decisions", "relationships", "open_questions", "entities") else ""
            scope_filter = _scope_filter(scope) if scope else ""
            rows = conn.execute(f"""
                SELECT id, {col} FROM {table}
                WHERE {active_clause}LOWER({col}) LIKE ?
                {scope_filter}
            """, [f"%{query_lower}%"]).fetchall()
            for row in rows:
                results.append({"id": row[0], "text": row[1], "table": table})
        except Exception:
            pass

    return results


def search_all_by_id(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
) -> list[dict]:
    """Search all tables for an item with the exact ID."""
    results: list[dict] = []

    for table, col in _ALL_FORGET_TABLES.items():
        try:
            row = conn.execute(
                f"SELECT id, {col} FROM {table} WHERE id = ?", [item_id]
            ).fetchone()
            if row:
                results.append({"id": row[0], "text": row[1], "table": table})
        except Exception:
            pass

    return results


def soft_delete(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    table: str,
) -> bool:
    """Mark an item as inactive and set deactivated_at. Returns True if found."""
    if table not in _ALL_FORGET_TABLES:
        return False

    row = conn.execute(f"SELECT id FROM {table} WHERE id = ?", [item_id]).fetchone()
    if not row:
        return False

    now = _now()
    conn.execute(
        f"UPDATE {table} SET is_active = FALSE, deactivated_at = ? WHERE id = ?",
        [now, item_id],
    )
    return True


def purge_deleted(
    conn: duckdb.DuckDBPyConnection,
    max_age_days: int = 30,
) -> dict:
    """Hard-delete items that were soft-deleted more than max_age_days ago."""
    cutoff = _now() - __import__("datetime").timedelta(days=max_age_days)
    stats = {"purged": 0}

    for table in _ALL_FORGET_TABLES:
        try:
            result = conn.execute(
                f"DELETE FROM {table} WHERE is_active = FALSE AND deactivated_at IS NOT NULL AND deactivated_at < ?",
                [cutoff],
            )
            stats["purged"] += result.fetchone()[0] if result.description else 0
        except Exception:
            # Count via separate query
            try:
                before = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE is_active = FALSE AND deactivated_at IS NOT NULL AND deactivated_at < ?", [cutoff]).fetchone()[0]
                conn.execute(f"DELETE FROM {table} WHERE is_active = FALSE AND deactivated_at IS NOT NULL AND deactivated_at < ?", [cutoff])
                stats["purged"] += before
            except Exception:
                pass

    return stats


def _get_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Get column names for a table."""
    return {r[0] for r in conn.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}'"
    ).fetchall()}


def get_stats(conn: duckdb.DuckDBPyConnection) -> dict:
    def count(table, where="is_active = TRUE"):
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]
        except Exception:
            return 0

    return {
        "facts":      {"total": count("facts"), "long": count("facts", "is_active=TRUE AND temporal_class='long'"),
                       "medium": count("facts", "is_active=TRUE AND temporal_class='medium'"),
                       "short": count("facts", "is_active=TRUE AND temporal_class='short'"),
                       "inactive": count("facts", "is_active=FALSE")},
        "ideas":      {"total": count("ideas"), "inactive": count("ideas","is_active=FALSE")},
        "entities":   {"total": conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]},
        "relationships": {"total": count("relationships")},
        "decisions":  {"total": count("decisions")},
        "questions":  {"total": count("open_questions","is_active=TRUE AND resolved=FALSE"),
                       "resolved": count("open_questions","resolved=TRUE")},
        "sessions":   {"total": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]},
    }
