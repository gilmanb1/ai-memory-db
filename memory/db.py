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
    (4, "Add session_narratives table and superseded_by columns", """
        CREATE TABLE IF NOT EXISTS session_narratives (
            id              VARCHAR PRIMARY KEY,
            session_id      VARCHAR NOT NULL,
            pass_number     INTEGER NOT NULL,
            narrative       TEXT NOT NULL,
            embedding       DOUBLE[],
            is_final        BOOLEAN DEFAULT FALSE,
            is_active       BOOLEAN DEFAULT TRUE,
            scope           VARCHAR DEFAULT '__global__',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS narratives_session ON session_narratives(session_id);
        CREATE INDEX IF NOT EXISTS narratives_final ON session_narratives(is_final);

        ALTER TABLE facts ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
        ALTER TABLE ideas ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
        ALTER TABLE open_questions ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
    """),
    (5, "Add observations table, consolidation log, and consolidated_at on facts", """
        CREATE TABLE IF NOT EXISTS observations (
            id              VARCHAR PRIMARY KEY,
            text            TEXT NOT NULL,
            proof_count     INTEGER DEFAULT 1,
            source_fact_ids VARCHAR[],
            history         TEXT,
            embedding       DOUBLE[],
            temporal_class  VARCHAR DEFAULT 'medium',
            decay_score     DOUBLE  DEFAULT 1.0,
            session_count   INTEGER DEFAULT 1,
            scope           VARCHAR DEFAULT '__global__',
            is_active       BOOLEAN DEFAULT TRUE,
            superseded_by   VARCHAR,
            deactivated_at  TIMESTAMP,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS consolidation_log (
            id              VARCHAR PRIMARY KEY,
            session_id      VARCHAR,
            action          VARCHAR NOT NULL,
            observation_id  VARCHAR,
            source_ids      VARCHAR[],
            reason          TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        ALTER TABLE facts ADD COLUMN IF NOT EXISTS consolidated_at TIMESTAMP;
    """),

    (6, "Conversation chunks table and fact-chunk linking", """
        CREATE TABLE IF NOT EXISTS conversation_chunks (
            id              VARCHAR PRIMARY KEY,
            session_id      VARCHAR NOT NULL,
            text            TEXT NOT NULL,
            embedding       FLOAT[],
            scope           VARCHAR DEFAULT '__global__',
            is_active       BOOLEAN DEFAULT TRUE,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deactivated_at  TIMESTAMP
        );

        ALTER TABLE facts ADD COLUMN IF NOT EXISTS source_chunk_id VARCHAR;
    """),

    (7, "Cast DOUBLE[] embeddings to FLOAT[] for HNSW index support", "-- handled by _cast_embeddings_to_float"),

    (8, "Add coding-oriented tables: guardrails, procedures, error_solutions; add importance and file_paths", """
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS importance INTEGER DEFAULT 5;
        ALTER TABLE ideas ADD COLUMN IF NOT EXISTS importance INTEGER DEFAULT 5;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS importance INTEGER DEFAULT 5;
        ALTER TABLE observations ADD COLUMN IF NOT EXISTS importance INTEGER DEFAULT 5;

        ALTER TABLE facts ADD COLUMN IF NOT EXISTS valid_from TIMESTAMP;
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS valid_until TIMESTAMP;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS valid_from TIMESTAMP;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS valid_until TIMESTAMP;

        CREATE TABLE IF NOT EXISTS fact_file_links (
            fact_id         VARCHAR NOT NULL,
            file_path       VARCHAR NOT NULL,
            item_table      VARCHAR DEFAULT 'facts',
            PRIMARY KEY (fact_id, file_path, item_table)
        );

        CREATE TABLE IF NOT EXISTS guardrails (
            id              VARCHAR PRIMARY KEY,
            warning         TEXT NOT NULL,
            rationale       TEXT,
            consequence     TEXT,
            file_paths      VARCHAR[],
            line_range      VARCHAR,
            scope           VARCHAR DEFAULT '__global__',
            importance      INTEGER DEFAULT 9,
            embedding       FLOAT[],
            session_count   INTEGER DEFAULT 1,
            source_session  VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE,
            deactivated_at  TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS procedures (
            id              VARCHAR PRIMARY KEY,
            task_description TEXT NOT NULL,
            steps           TEXT NOT NULL,
            file_paths      VARCHAR[],
            scope           VARCHAR DEFAULT '__global__',
            temporal_class  VARCHAR DEFAULT 'long',
            importance      INTEGER DEFAULT 7,
            decay_score     DOUBLE DEFAULT 1.0,
            embedding       FLOAT[],
            session_count   INTEGER DEFAULT 1,
            source_session  VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE,
            deactivated_at  TIMESTAMP,
            superseded_by   VARCHAR
        );

        CREATE TABLE IF NOT EXISTS error_solutions (
            id              VARCHAR PRIMARY KEY,
            error_pattern   TEXT NOT NULL,
            error_context   TEXT,
            solution        TEXT NOT NULL,
            file_paths      VARCHAR[],
            scope           VARCHAR DEFAULT '__global__',
            confidence      VARCHAR DEFAULT 'medium',
            times_applied   INTEGER DEFAULT 1,
            embedding       FLOAT[],
            source_session  VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active       BOOLEAN DEFAULT TRUE,
            deactivated_at  TIMESTAMP
        );
    """),

    (9, "Outcome scoring, failure probability, community summaries", """
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS times_recalled INTEGER DEFAULT 0;
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS times_applied INTEGER DEFAULT 0;
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS recall_utility DOUBLE DEFAULT 1.0;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS times_recalled INTEGER DEFAULT 0;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS times_applied INTEGER DEFAULT 0;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS recall_utility DOUBLE DEFAULT 1.0;
        ALTER TABLE guardrails ADD COLUMN IF NOT EXISTS times_recalled INTEGER DEFAULT 0;
        ALTER TABLE guardrails ADD COLUMN IF NOT EXISTS times_applied INTEGER DEFAULT 0;
        ALTER TABLE guardrails ADD COLUMN IF NOT EXISTS recall_utility DOUBLE DEFAULT 1.0;
        ALTER TABLE procedures ADD COLUMN IF NOT EXISTS times_recalled INTEGER DEFAULT 0;
        ALTER TABLE procedures ADD COLUMN IF NOT EXISTS times_applied INTEGER DEFAULT 0;
        ALTER TABLE procedures ADD COLUMN IF NOT EXISTS recall_utility DOUBLE DEFAULT 1.0;
        ALTER TABLE error_solutions ADD COLUMN IF NOT EXISTS times_recalled INTEGER DEFAULT 0;
        ALTER TABLE error_solutions ADD COLUMN IF NOT EXISTS times_applied INTEGER DEFAULT 0;
        ALTER TABLE error_solutions ADD COLUMN IF NOT EXISTS recall_utility DOUBLE DEFAULT 1.0;

        ALTER TABLE facts ADD COLUMN IF NOT EXISTS failure_probability DOUBLE DEFAULT 0.0;
        ALTER TABLE decisions ADD COLUMN IF NOT EXISTS failure_probability DOUBLE DEFAULT 0.0;
        ALTER TABLE guardrails ADD COLUMN IF NOT EXISTS failure_probability DOUBLE DEFAULT 0.5;

        CREATE TABLE IF NOT EXISTS community_summaries (
            id              VARCHAR PRIMARY KEY,
            level           INTEGER NOT NULL DEFAULT 1,
            summary         TEXT NOT NULL,
            entity_ids      VARCHAR[],
            source_item_ids VARCHAR[],
            scope           VARCHAR DEFAULT '__global__',
            embedding       FLOAT[],
            is_active       BOOLEAN DEFAULT TRUE,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deactivated_at  TIMESTAMP
        );
    """),
]


def _create_hnsw_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Create HNSW indexes on embedding columns for fast vector search.
    Safe to call multiple times — skips if index already exists or VSS not loaded."""
    tables = ["facts", "ideas", "decisions", "observations", "open_questions", "session_narratives",
              "guardrails", "procedures", "error_solutions", "community_summaries"]
    for table in tables:
        try:
            idx_name = f"hnsw_{table}_embedding"
            # Check if index exists
            existing = conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE index_name = ?", [idx_name]
            ).fetchone()
            if existing:
                continue
            # Check if table has any rows with embeddings (HNSW needs at least 1)
            has_data = conn.execute(
                f"SELECT 1 FROM {table} WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
            if not has_data:
                continue
            conn.execute(
                f"CREATE INDEX {idx_name} ON {table} USING HNSW (embedding) WITH (metric = 'cosine')"
            )
        except Exception:
            pass  # VSS not loaded or table doesn't exist yet


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
        conn.execute("SET hnsw_enable_experimental_persistence = true")
    except Exception:
        pass  # VSS not installed; list_cosine_similarity still works via scan


# ── BM25 Full-Text Search ─────────────────────────────────────────────────

_fts_available: bool = False


def ensure_fts(conn: duckdb.DuckDBPyConnection) -> None:
    """Try to INSTALL and LOAD the DuckDB fts extension. Sets _fts_available on success."""
    global _fts_available
    try:
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        _fts_available = True
    except Exception:
        pass  # FTS not available; leave flag False


def rebuild_fts_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate FTS indexes on all text-searchable tables."""
    global _fts_available
    if not _fts_available:
        ensure_fts(conn)
    if not _fts_available:
        return

    tables = [
        ("facts", "text"),
        ("ideas", "text"),
        ("decisions", "text"),
        ("observations", "text"),
        ("session_narratives", "narrative"),
        ("conversation_chunks", "text"),
        ("guardrails", "warning"),
        ("procedures", "task_description"),
        ("error_solutions", "error_pattern"),
        ("community_summaries", "summary"),
    ]
    for table, text_col in tables:
        try:
            conn.execute(f"PRAGMA drop_fts_index('{table}')")
        except Exception:
            pass
        try:
            conn.execute(f"PRAGMA create_fts_index('{table}', 'id', '{text_col}')")
        except Exception:
            pass  # Table may not exist yet

    # Also rebuild HNSW indexes
    _create_hnsw_indexes(conn)


def search_bm25(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    query: str,
    text_col: str = "text",
    select_cols: str = "id, text",
    limit: int = 10,
    scope: Optional[str] = None,
) -> list[dict]:
    """Search a table using BM25 full-text scoring, with LIKE fallback."""
    scope_sql = ""
    params: list = []

    if scope:
        scope_sql = " AND (scope = ? OR scope = ?)"

    if _fts_available:
        sql = (
            f"SELECT {select_cols}, fts_main_{table}.match_bm25(id, ?, fields := '{text_col}') AS score "
            f"FROM {table} "
            f"WHERE score IS NOT NULL AND is_active = TRUE{scope_sql} "
            f"ORDER BY score DESC LIMIT ?"
        )
        params = [query]
        if scope:
            params += [scope, GLOBAL_SCOPE]
        params.append(limit)
    else:
        sql = (
            f"SELECT {select_cols}, 0.5 AS score "
            f"FROM {table} "
            f"WHERE is_active = TRUE AND LOWER({text_col}) LIKE ?{scope_sql} "
            f"LIMIT ?"
        )
        params = [f"%{query.lower()}%"]
        if scope:
            params += [scope, GLOBAL_SCOPE]
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    col_names = [c.strip() for c in select_cols.split(",")] + ["score"]
    return [dict(zip(col_names, row)) for row in rows]


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
        if version == 7:
            _cast_embeddings_to_float(conn)
        else:
            # Execute each statement individually (DuckDB doesn't run multi-statement strings)
            for stmt in _split_sql(sql):
                if stmt:
                    conn.execute(stmt)
        conn.execute(
            "INSERT INTO schema_migrations(version, description) VALUES (?, ?)",
            [version, description],
        )


def _cast_embeddings_to_float(conn: duckdb.DuckDBPyConnection) -> None:
    """Cast DOUBLE[] embedding columns to FLOAT[768] for HNSW index compatibility."""
    from .config import EMBEDDING_DIM
    tables = ["facts", "ideas", "entities", "open_questions", "observations", "session_narratives"]
    for table in tables:
        try:
            conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE FLOAT[{EMBEDDING_DIM}]")
        except Exception:
            pass  # Column may not exist or already correct type


def _split_sql(sql: str) -> list[str]:
    """Split a multi-statement SQL string on semicolons."""
    return [s.strip() for s in sql.split(";") if s.strip()]


# ── Helpers ────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return str(uuid.uuid4())


def _text_dedup(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    text: str,
    select_cols: str,
    scope: Optional[str] = None,
) -> Optional[dict]:
    """
    Fallback deduplication by exact text match.
    Used when embedding is None (Ollama down) or existing items lack embeddings.
    Respects scope filtering: only matches within the same scope or global.
    Returns the first matching active row as a dict, or None.
    """
    text_col = {
        "entities": "name",
        "guardrails": "warning",
        "procedures": "task_description",
        "error_solutions": "error_pattern",
    }.get(table, "text")
    active_clause = "AND is_active = TRUE"
    scope_sql, scope_params = _scope_filter(scope)
    try:
        row = conn.execute(f"""
            SELECT {select_cols}
            FROM {table}
            WHERE {text_col} = ?
              {active_clause}
              {scope_sql}
            LIMIT 1
        """, [text] + scope_params).fetchone()
        if row:
            col_names = [c.strip() for c in select_cols.split(",")]
            return dict(zip(col_names, row))
    except Exception:
        pass
    return None


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
    where_extra: str | tuple[str, list],
    limit: int,
    threshold: float,
) -> list[dict]:
    """
    Find rows in `table` ordered by cosine similarity to query_embedding.

    select_cols controls which columns are returned in the result dicts.
    where_extra can be a (sql_fragment, params) tuple or a plain string for
    static fragments like "AND resolved = FALSE".
    embedding is always fetched internally for the fallback path but is
    stripped from the returned dicts.

    Falls back to Python cosine similarity if list_cosine_similarity fails.
    """
    # Normalize where_extra to (sql, params)
    if isinstance(where_extra, tuple):
        extra_sql, extra_params = where_extra
    else:
        extra_sql, extra_params = where_extra, []

    col_list = [c.strip() for c in select_cols.split(",") if c.strip()]
    all_col_names = col_list + ["embedding", "score"]

    try:
        rows = conn.execute(f"""
            SELECT {select_cols}, embedding,
                   list_cosine_similarity(embedding, ?) AS score
            FROM {table}
            WHERE is_active = TRUE AND embedding IS NOT NULL
              {extra_sql}
            ORDER BY score DESC
            LIMIT ?
        """, [query_embedding] + extra_params + [limit]).fetchall()
    except Exception:
        rows_all = conn.execute(f"""
            SELECT {select_cols}, embedding
            FROM {table}
            WHERE is_active = TRUE AND embedding IS NOT NULL
              {extra_sql}
        """, extra_params).fetchall()
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

def _scope_filter(scope: Optional[str]) -> tuple[str, list]:
    """
    Return a (sql_fragment, params) tuple to filter by scope.
    If scope is None, returns ("", []) (no filter = all scopes).
    If scope is a project path, matches that scope OR global.
    """
    if scope is None:
        return ("", [])
    return ("AND (scope = ? OR scope = ?)", [scope, GLOBAL_SCOPE])


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


# ── Conversation Chunks ───────────────────────────────────────────────────

def insert_chunk(
    conn: duckdb.DuckDBPyConnection,
    text: str,
    session_id: str,
    scope: str = GLOBAL_SCOPE,
    embedding: Optional[list[float]] = None,
) -> str:
    """Insert a raw conversation chunk with optional embedding. Returns chunk_id."""
    chunk_id = _uid()
    conn.execute("""
        INSERT INTO conversation_chunks (id, session_id, text, embedding, scope, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [chunk_id, session_id, text, embedding, scope, _now()])
    return chunk_id


def get_chunks_by_ids(
    conn: duckdb.DuckDBPyConnection,
    chunk_ids: list[str],
) -> dict[str, dict]:
    """Batch-fetch chunks by ID. Returns {chunk_id: {id, text, session_id}}."""
    if not chunk_ids:
        return {}
    placeholders = ", ".join(["?"] * len(chunk_ids))
    rows = conn.execute(f"""
        SELECT id, text, session_id
        FROM conversation_chunks
        WHERE id IN ({placeholders}) AND is_active = TRUE
    """, chunk_ids).fetchall()
    return {
        row[0]: {"id": row[0], "text": row[1], "session_id": row[2]}
        for row in rows
    }


def search_chunks(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = 5,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    """Vector search on conversation chunks. Returns chunks with text and score."""
    return _vector_search(
        conn, "conversation_chunks", query_embedding,
        "id, text, session_id, scope",
        _scope_filter(scope), limit, threshold,
    )


def search_chunks_bm25(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    limit: int = 5,
    scope: Optional[str] = None,
) -> list[dict]:
    """BM25 keyword search on conversation chunks."""
    return search_bm25(
        conn, "conversation_chunks", query,
        text_col="text",
        select_cols="id, text, session_id, scope",
        limit=limit,
        scope=scope,
    )


def get_sibling_facts(
    conn: duckdb.DuckDBPyConnection,
    chunk_ids: list[str],
    exclude_ids: Optional[set[str]] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Get all facts that share the same source_chunk_id as the given chunks.
    Excludes facts already in exclude_ids (the originally retrieved ones).
    Returns sibling facts sorted by decay_score DESC.
    """
    if not chunk_ids:
        return []
    exclude_ids = exclude_ids or set()
    placeholders = ", ".join(["?"] * len(chunk_ids))
    rows = conn.execute(f"""
        SELECT id, text, category, temporal_class, confidence, decay_score, scope, source_chunk_id
        FROM facts
        WHERE source_chunk_id IN ({placeholders})
          AND is_active = TRUE
        ORDER BY decay_score DESC
        LIMIT ?
    """, chunk_ids + [limit + len(exclude_ids)]).fetchall()
    cols = ["id", "text", "category", "temporal_class", "confidence", "decay_score", "scope", "source_chunk_id"]
    results = []
    for row in rows:
        d = dict(zip(cols, row))
        if d["id"] not in exclude_ids:
            results.append(d)
    return results[:limit]


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
    source_chunk_id: Optional[str] = None,
    importance: int = 5,
    file_paths: Optional[list[str]] = None,
    failure_probability: float = 0.0,
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

    # Fallback: text-based dedup when embedding search didn't find a match
    if not existing:
        existing = _text_dedup(
            conn, "facts", text,
            "id, text, temporal_class, decay_score, session_count, last_seen_at",
            scope=scope,
        )

    if existing:
        new_session_count = existing["session_count"] + 1
        new_decay = decay_fn(
            existing["last_seen_at"], new_session_count, existing["temporal_class"]
        )
        new_class = _promote_class(
            existing["temporal_class"], new_session_count, existing["last_seen_at"]
        )
        # Backfill embedding if the existing row lacks one
        emb_update = ", embedding = ?" if embedding else ""
        params = [now, new_session_count, new_decay, new_class]
        if embedding:
            params.append(embedding)
        params.append(existing["id"])
        conn.execute(f"""
            UPDATE facts SET
                last_seen_at   = ?,
                session_count  = ?,
                decay_score    = ?,
                temporal_class = ?
                {emb_update}
            WHERE id = ?
        """, params)
        _track_item_scope(conn, existing["id"], "facts", scope)
        _maybe_auto_promote(conn, existing["id"], "facts")
        return existing["id"], False

    fid = _uid()
    conn.execute("""
        INSERT INTO facts
            (id, text, category, temporal_class, confidence, decay_score,
             embedding, source_session, created_at, last_seen_at, scope,
             source_chunk_id, importance, valid_from, failure_probability)
        VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [fid, text, category, temporal_class, confidence, embedding,
          session_id, now, now, scope, source_chunk_id, importance, now, failure_probability])
    _track_item_scope(conn, fid, "facts", scope)
    # Link file paths if provided
    if file_paths:
        for fp in file_paths:
            _link_file_path(conn, fid, fp, "facts")
    return fid, True


def get_facts_by_temporal(
    conn: duckdb.DuckDBPyConnection,
    temporal_class: str,
    limit: int,
    scope: Optional[str] = None,
) -> list[dict]:
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, text, category, temporal_class, confidence, decay_score, scope,
               source_chunk_id, importance, failure_probability
        FROM facts
        WHERE is_active = TRUE AND temporal_class = ?
          {scope_sql}
        ORDER BY importance DESC, failure_probability DESC, decay_score DESC, last_seen_at DESC
        LIMIT ?
    """, [temporal_class] + scope_params + [limit]).fetchall()
    return [
        dict(zip(["id","text","category","temporal_class","confidence","decay_score","scope",
                  "source_chunk_id","importance","failure_probability"], r))
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
        "id, text, temporal_class, decay_score, session_count, last_seen_at, category, confidence, scope, source_chunk_id, importance, recall_utility, failure_probability",
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

    # Fallback: text-based dedup
    if not existing:
        existing = _text_dedup(
            conn, "ideas", text,
            "id, text, temporal_class, decay_score, session_count, last_seen_at",
            scope=scope,
        )

    if existing:
        new_session_count = existing["session_count"] + 1
        new_decay = decay_fn(
            existing["last_seen_at"], new_session_count, existing["temporal_class"]
        )
        new_class = _promote_class(
            existing["temporal_class"], new_session_count, existing["last_seen_at"]
        )
        emb_update = ", embedding = ?" if embedding else ""
        params = [now, new_session_count, new_decay, new_class]
        if embedding:
            params.append(embedding)
        params.append(existing["id"])
        conn.execute(f"""
            UPDATE ideas SET last_seen_at=?, session_count=?,
                             decay_score=?, temporal_class=?
                             {emb_update}
            WHERE id=?
        """, params)
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
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT name FROM entities
        WHERE 1=1 {scope_sql}
        ORDER BY session_count DESC, last_seen_at DESC
        LIMIT ?
    """, scope_params + [limit]).fetchall()
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
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT from_entity, to_entity, rel_type, description, strength, session_count
        FROM relationships
        WHERE is_active = TRUE
          AND (LOWER(from_entity) IN ({placeholders})
               OR LOWER(to_entity) IN ({placeholders}))
          {scope_sql}
        ORDER BY strength DESC, session_count DESC
        LIMIT ?
    """, names_lower + names_lower + scope_params + [limit]).fetchall()
    return [
        dict(zip(["from_entity","to_entity","rel_type","description","strength","session_count"], r))
        for r in rows
    ]


def get_all_relationships(
    conn: duckdb.DuckDBPyConnection,
    scope: Optional[str] = None,
) -> list[dict]:
    """All active relationships — used for graph visualisation."""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT from_entity, to_entity, rel_type, description, strength, session_count
        FROM relationships
        WHERE is_active = TRUE {scope_sql}
        ORDER BY strength DESC
    """, scope_params).fetchall()
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

    # Fallback: text-based dedup
    if not existing:
        existing = _text_dedup(
            conn, "decisions", text,
            "id, text, temporal_class, decay_score, session_count, last_seen_at",
            scope=scope,
        )

    if existing:
        emb_update = ", embedding = ?" if embedding else ""
        params = [now]
        if embedding:
            params.append(embedding)
        params.append(existing["id"])
        conn.execute(f"""
            UPDATE decisions SET last_seen_at=?, session_count=session_count+1
            {emb_update} WHERE id=?
        """, params)
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
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, text, temporal_class, decay_score
        FROM decisions
        WHERE is_active = TRUE {scope_sql}
        ORDER BY decay_score DESC, last_seen_at DESC
        LIMIT ?
    """, scope_params + [limit]).fetchall()
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
    scope_sql, scope_params = _scope_filter(scope)
    return _vector_search(
        conn, "open_questions", query_embedding,
        "id, text, resolved, last_seen_at",
        (f"AND resolved = FALSE {scope_sql}", scope_params), limit, RECALL_THRESHOLD,
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

    for table in ("facts", "ideas", "decisions", "observations", "procedures"):
        try:
            rows = conn.execute(f"""
                SELECT id, temporal_class, session_count, last_seen_at, created_at
                FROM {table}
                WHERE is_active = TRUE
            """).fetchall()
        except Exception:
            continue  # table may not exist yet (migration 5 not applied)

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
    "observations":   "text",
}

# Tables where the display text comes from a different column
_OTHER_TABLES = {
    "entities":        "name",
    "relationships":   "description",
    "guardrails":      "warning",
    "procedures":      "task_description",
    "error_solutions": "error_pattern",
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
        try:
            active_clause = "is_active = TRUE AND "
            scope_sql, scope_params = _scope_filter(scope) if scope else ("", [])
            rows = conn.execute(f"""
                SELECT id, {col} FROM {table}
                WHERE {active_clause}LOWER({col}) LIKE ?
                {scope_sql}
            """, [f"%{query_lower}%"] + scope_params).fetchall()
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

    stats = {
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
    try:
        stats["observations"] = {
            "total": count("observations"),
            "long": count("observations", "is_active=TRUE AND temporal_class='long'"),
            "medium": count("observations", "is_active=TRUE AND temporal_class='medium'"),
            "inactive": count("observations", "is_active=FALSE"),
        }
    except Exception:
        stats["observations"] = {"total": 0}
    try:
        stats["guardrails"] = {"total": count("guardrails")}
    except Exception:
        stats["guardrails"] = {"total": 0}
    try:
        stats["procedures"] = {"total": count("procedures")}
    except Exception:
        stats["procedures"] = {"total": 0}
    try:
        stats["error_solutions"] = {"total": count("error_solutions")}
    except Exception:
        stats["error_solutions"] = {"total": 0}
    return stats


# ══════════════════════════════════════════════════════════════════════════
# Narrative storage (incremental extraction)
# ══════════════════════════════════════════════════════════════════════════

def upsert_narrative(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    pass_number: int,
    narrative: str,
    embedding: Optional[list[float]] = None,
    is_final: bool = False,
    scope: str = "__global__",
) -> str:
    """Insert or replace a narrative for a session+pass. Returns the narrative ID."""
    # Check if one already exists for this session + pass
    existing = conn.execute(
        "SELECT id FROM session_narratives WHERE session_id = ? AND pass_number = ?",
        [session_id, pass_number],
    ).fetchone()

    if existing:
        nid = existing[0]
        conn.execute(
            "UPDATE session_narratives SET narrative = ?, embedding = ?, is_final = ?, scope = ? WHERE id = ?",
            [narrative, embedding, is_final, scope, nid],
        )
        return nid

    nid = _uid()
    conn.execute(
        """INSERT INTO session_narratives(id, session_id, pass_number, narrative, embedding, is_final, scope)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [nid, session_id, pass_number, narrative, embedding, is_final, scope],
    )
    return nid


def finalize_narratives(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
) -> None:
    """
    Mark the highest-pass narrative as final and delete intermediate ones.
    Only the final narrative persists for long-term recall.
    """
    rows = conn.execute(
        "SELECT id, pass_number FROM session_narratives WHERE session_id = ? ORDER BY pass_number DESC",
        [session_id],
    ).fetchall()

    if not rows:
        return

    # Mark highest pass as final
    final_id = rows[0][0]
    conn.execute(
        "UPDATE session_narratives SET is_final = TRUE WHERE id = ?",
        [final_id],
    )

    # Delete all non-final narratives for this session
    if len(rows) > 1:
        non_final_ids = [r[0] for r in rows[1:]]
        for nfid in non_final_ids:
            conn.execute("DELETE FROM session_narratives WHERE id = ?", [nfid])


def search_narratives(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = 3,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    """Vector search on final narratives only."""
    scope_sql, scope_params = _scope_filter(scope)
    extra_sql = f"AND is_final = TRUE {scope_sql}"
    return _vector_search(
        conn,
        table="session_narratives",
        query_embedding=query_embedding,
        select_cols="id, session_id, pass_number, narrative, scope, created_at",
        where_extra=(extra_sql, scope_params),
        limit=limit,
        threshold=threshold,
    )


# ══════════════════════════════════════════════════════════════════════════
# Observations (consolidation engine)
# ══════════════════════════════════════════════════════════════════════════

def upsert_observation(
    conn: duckdb.DuckDBPyConnection,
    text: str,
    source_fact_ids: list[str],
    embedding: Optional[list[float]],
    scope: str = GLOBAL_SCOPE,
) -> tuple[str, bool]:
    """Insert a new observation or reinforce an existing one. Returns (obs_id, is_new)."""
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "observations", embedding,
            "id, text, proof_count, source_fact_ids, temporal_class, decay_score, session_count, last_seen_at",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if not existing:
        existing = _text_dedup(
            conn, "observations", text,
            "id, text, proof_count, source_fact_ids, temporal_class, decay_score, session_count, last_seen_at",
            scope=scope,
        )

    if existing:
        old_sources = existing.get("source_fact_ids") or []
        merged_sources = list(set(old_sources) | set(source_fact_ids))
        new_proof = len(merged_sources)
        new_session_count = existing["session_count"] + 1
        emb_update = ", embedding = ?" if embedding else ""
        params = [now, now, new_proof, merged_sources, new_session_count]
        if embedding:
            params.append(embedding)
        params.append(existing["id"])
        conn.execute(f"""
            UPDATE observations SET
                last_seen_at   = ?,
                updated_at     = ?,
                proof_count    = ?,
                source_fact_ids = ?,
                session_count  = ?
                {emb_update}
            WHERE id = ?
        """, params)
        _track_item_scope(conn, existing["id"], "observations", scope)
        _maybe_auto_promote(conn, existing["id"], "observations")
        return existing["id"], False

    oid = _uid()
    conn.execute("""
        INSERT INTO observations
            (id, text, proof_count, source_fact_ids, embedding,
             scope, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [oid, text, len(source_fact_ids), source_fact_ids, embedding,
          scope, now, now])
    _track_item_scope(conn, oid, "observations", scope)
    return oid, True


def update_observation(
    conn: duckdb.DuckDBPyConnection,
    obs_id: str,
    text: str,
    embedding: Optional[list[float]],
    new_source_ids: list[str],
) -> bool:
    """Update an observation's text, embedding, and merge new source IDs."""
    now = _now()
    row = conn.execute(
        "SELECT source_fact_ids, history, proof_count FROM observations WHERE id = ? AND is_active = TRUE",
        [obs_id],
    ).fetchone()
    if not row:
        return False

    old_sources = row[0] or []
    old_history = row[1] or "[]"
    merged_sources = list(set(old_sources) | set(new_source_ids))

    # Append previous state to history
    try:
        history = json.loads(old_history)
    except (json.JSONDecodeError, TypeError):
        history = []
    history.append({"updated_at": now.isoformat(), "previous_text": text, "proof_count": row[2]})

    emb_update = ", embedding = ?" if embedding else ""
    params = [text, merged_sources, len(merged_sources), json.dumps(history), now]
    if embedding:
        params.append(embedding)
    params.append(obs_id)
    conn.execute(f"""
        UPDATE observations SET
            text            = ?,
            source_fact_ids = ?,
            proof_count     = ?,
            history         = ?,
            updated_at      = ?
            {emb_update}
        WHERE id = ?
    """, params)
    return True


def search_observations(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = 10,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    scope_sql, scope_params = _scope_filter(scope)
    return _vector_search(
        conn, "observations", query_embedding,
        "id, text, proof_count, source_fact_ids, temporal_class, decay_score, session_count, scope",
        (scope_sql, scope_params), limit, threshold,
    )


def get_observations_by_temporal(
    conn: duckdb.DuckDBPyConnection,
    temporal_class: str,
    limit: int = 10,
    scope: Optional[str] = None,
) -> list[dict]:
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, text, proof_count, temporal_class, decay_score, scope
        FROM observations
        WHERE is_active = TRUE AND temporal_class = ?
          {scope_sql}
        ORDER BY proof_count DESC, decay_score DESC
        LIMIT ?
    """, [temporal_class] + scope_params + [limit]).fetchall()
    return [
        dict(zip(["id", "text", "proof_count", "temporal_class", "decay_score", "scope"], r))
        for r in rows
    ]


def get_unconsolidated_facts(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 10,
    scope: Optional[str] = None,
) -> list[dict]:
    """Fetch facts that haven't been processed by the consolidation engine."""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, text, category, temporal_class, confidence, embedding, scope
        FROM facts
        WHERE is_active = TRUE AND consolidated_at IS NULL
          {scope_sql}
        ORDER BY created_at ASC
        LIMIT ?
    """, scope_params + [limit]).fetchall()
    return [
        dict(zip(["id", "text", "category", "temporal_class", "confidence", "embedding", "scope"], r))
        for r in rows
    ]


def mark_facts_consolidated(
    conn: duckdb.DuckDBPyConnection,
    fact_ids: list[str],
) -> None:
    """Mark facts as processed by the consolidation engine."""
    now = _now()
    for fid in fact_ids:
        conn.execute("UPDATE facts SET consolidated_at = ? WHERE id = ?", [now, fid])


def log_consolidation_action(
    conn: duckdb.DuckDBPyConnection,
    action: str,
    observation_id: str,
    source_ids: list[str],
    reason: str,
    session_id: Optional[str] = None,
) -> None:
    conn.execute("""
        INSERT INTO consolidation_log(id, session_id, action, observation_id, source_ids, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [_uid(), session_id, action, observation_id, source_ids, reason])


def get_all_observation_embeddings(
    conn: duckdb.DuckDBPyConnection,
    scope: Optional[str] = None,
) -> list[dict]:
    """Get all active observations with their embeddings for similarity sweep."""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, text, proof_count, embedding, created_at
        FROM observations
        WHERE is_active = TRUE AND embedding IS NOT NULL
          {scope_sql}
        ORDER BY proof_count DESC, created_at DESC
    """, scope_params).fetchall()
    return [
        dict(zip(["id", "text", "proof_count", "embedding", "created_at"], r))
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════
# Superseding (incremental extraction)
# ══════════════════════════════════════════════════════════════════════════

_SUPERSEDE_TABLES = {"facts", "ideas", "decisions", "open_questions", "observations"}


def supersede_item(
    conn: duckdb.DuckDBPyConnection,
    old_id: str,
    old_table: str,
    new_id: str,
    reason: str,
) -> bool:
    """
    Mark an item as superseded by a newer item.
    Sets is_active=FALSE, deactivated_at=now, superseded_by=new_id.
    Returns True if the old item was found and updated.
    """
    if old_table not in _SUPERSEDE_TABLES:
        return False

    # Check the column exists (migration 4 may not have run yet)
    cols = _get_columns(conn, old_table)
    if "superseded_by" not in cols:
        # Fall back to simple soft-delete
        return soft_delete(conn, old_id, old_table)

    row = conn.execute(f"SELECT id FROM {old_table} WHERE id = ?", [old_id]).fetchone()
    if not row:
        return False

    now = _now()
    conn.execute(
        f"UPDATE {old_table} SET is_active = FALSE, deactivated_at = ?, superseded_by = ? WHERE id = ?",
        [now, new_id, old_id],
    )
    return True


# ══════════════════════════════════════════════════════════════════════════
# Guardrails (defensive knowledge — "don't touch this" warnings)
# ══════════════════════════════════════════════════════════════════════════

def upsert_guardrail(
    conn: duckdb.DuckDBPyConnection,
    warning: str,
    rationale: str = "",
    consequence: str = "",
    file_paths: Optional[list[str]] = None,
    line_range: str = "",
    embedding: Optional[list[float]] = None,
    session_id: str = "",
    scope: str = GLOBAL_SCOPE,
    importance: int = 9,
    failure_probability: float = 0.5,
) -> tuple[str, bool]:
    """Insert or reinforce a guardrail. Returns (guardrail_id, is_new)."""
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "guardrails", embedding,
            "id, warning, session_count, last_seen_at",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if not existing:
        existing = _text_dedup(
            conn, "guardrails", warning,
            "id, warning, session_count, last_seen_at",
            scope=scope,
        )

    if existing:
        conn.execute("""
            UPDATE guardrails SET
                last_seen_at = ?,
                session_count = session_count + 1
            WHERE id = ?
        """, [now, existing["id"]])
        _track_item_scope(conn, existing["id"], "guardrails", scope)
        return existing["id"], False

    gid = _uid()
    conn.execute("""
        INSERT INTO guardrails
            (id, warning, rationale, consequence, file_paths, line_range,
             scope, importance, embedding, source_session, created_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [gid, warning, rationale, consequence, file_paths or [],
          line_range, scope, importance, embedding, session_id, now, now])
    _track_item_scope(conn, gid, "guardrails", scope)
    # Link file paths
    for fp in (file_paths or []):
        _link_file_path(conn, gid, fp, "guardrails")
    return gid, True


def get_guardrails_for_files(
    conn: duckdb.DuckDBPyConnection,
    file_paths: list[str],
    limit: int = 20,
    scope: Optional[str] = None,
) -> list[dict]:
    """Get guardrails associated with specific files. Always surface these."""
    if not file_paths:
        return []
    scope_sql, scope_params = _scope_filter(scope)
    placeholders = ", ".join(["?"] * len(file_paths))
    rows = conn.execute(f"""
        SELECT DISTINCT g.id, g.warning, g.rationale, g.consequence,
               g.file_paths, g.line_range, g.importance, g.scope
        FROM guardrails g
        JOIN fact_file_links ffl ON g.id = ffl.fact_id AND ffl.item_table = 'guardrails'
        WHERE ffl.file_path IN ({placeholders})
          AND g.is_active = TRUE
          {scope_sql}
        ORDER BY g.importance DESC
        LIMIT ?
    """, file_paths + scope_params + [limit]).fetchall()
    return [
        dict(zip(["id", "warning", "rationale", "consequence",
                  "file_paths", "line_range", "importance", "scope"], r))
        for r in rows
    ]


def get_all_guardrails(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 50,
    scope: Optional[str] = None,
) -> list[dict]:
    """Get all active guardrails."""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, warning, rationale, consequence, file_paths,
               line_range, importance, scope
        FROM guardrails
        WHERE is_active = TRUE {scope_sql}
        ORDER BY importance DESC, last_seen_at DESC
        LIMIT ?
    """, scope_params + [limit]).fetchall()
    return [
        dict(zip(["id", "warning", "rationale", "consequence",
                  "file_paths", "line_range", "importance", "scope"], r))
        for r in rows
    ]


def search_guardrails(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = 10,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    """Vector search on guardrails."""
    return _vector_search(
        conn, "guardrails", query_embedding,
        "id, warning, rationale, consequence, file_paths, line_range, importance, scope",
        _scope_filter(scope), limit, threshold,
    )


# ══════════════════════════════════════════════════════════════════════════
# Procedures ("how to do X" knowledge)
# ══════════════════════════════════════════════════════════════════════════

def upsert_procedure(
    conn: duckdb.DuckDBPyConnection,
    task_description: str,
    steps: str,
    file_paths: Optional[list[str]] = None,
    embedding: Optional[list[float]] = None,
    session_id: str = "",
    decay_fn=None,
    scope: str = GLOBAL_SCOPE,
    importance: int = 7,
) -> tuple[str, bool]:
    """Insert or reinforce a procedure. Returns (procedure_id, is_new)."""
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "procedures", embedding,
            "id, task_description, session_count, last_seen_at, temporal_class, decay_score",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if not existing:
        existing = _text_dedup(
            conn, "procedures", task_description,
            "id, task_description, session_count, last_seen_at, temporal_class, decay_score",
            scope=scope,
        )

    if existing:
        new_session_count = existing["session_count"] + 1
        conn.execute("""
            UPDATE procedures SET
                last_seen_at = ?,
                session_count = ?,
                steps = ?
            WHERE id = ?
        """, [now, new_session_count, steps, existing["id"]])
        _track_item_scope(conn, existing["id"], "procedures", scope)
        return existing["id"], False

    pid = _uid()
    conn.execute("""
        INSERT INTO procedures
            (id, task_description, steps, file_paths, scope, importance,
             embedding, source_session, created_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [pid, task_description, steps, file_paths or [], scope, importance,
          embedding, session_id, now, now])
    _track_item_scope(conn, pid, "procedures", scope)
    for fp in (file_paths or []):
        _link_file_path(conn, pid, fp, "procedures")
    return pid, True


def search_procedures(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = 5,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    """Vector search on procedures."""
    return _vector_search(
        conn, "procedures", query_embedding,
        "id, task_description, steps, file_paths, importance, scope",
        _scope_filter(scope), limit, threshold,
    )


def get_procedures(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 10,
    scope: Optional[str] = None,
) -> list[dict]:
    """Get all active procedures."""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, task_description, steps, file_paths, importance, scope
        FROM procedures
        WHERE is_active = TRUE {scope_sql}
        ORDER BY importance DESC, last_seen_at DESC
        LIMIT ?
    """, scope_params + [limit]).fetchall()
    return [
        dict(zip(["id", "task_description", "steps", "file_paths", "importance", "scope"], r))
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════
# Error→Solution catalog
# ══════════════════════════════════════════════════════════════════════════

def upsert_error_solution(
    conn: duckdb.DuckDBPyConnection,
    error_pattern: str,
    solution: str,
    error_context: str = "",
    file_paths: Optional[list[str]] = None,
    embedding: Optional[list[float]] = None,
    session_id: str = "",
    scope: str = GLOBAL_SCOPE,
    confidence: str = "medium",
) -> tuple[str, bool]:
    """Insert or reinforce an error→solution pair. Returns (id, is_new)."""
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "error_solutions", embedding,
            "id, error_pattern, times_applied, last_applied_at",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if not existing:
        existing = _text_dedup(
            conn, "error_solutions", error_pattern,
            "id, error_pattern, times_applied, last_applied_at",
            scope=scope,
        )

    if existing:
        conn.execute("""
            UPDATE error_solutions SET
                last_applied_at = ?,
                times_applied = times_applied + 1,
                solution = ?
            WHERE id = ?
        """, [now, solution, existing["id"]])
        _track_item_scope(conn, existing["id"], "error_solutions", scope)
        return existing["id"], False

    eid = _uid()
    conn.execute("""
        INSERT INTO error_solutions
            (id, error_pattern, error_context, solution, file_paths, scope,
             confidence, embedding, source_session, created_at, last_applied_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [eid, error_pattern, error_context, solution, file_paths or [],
          scope, confidence, embedding, session_id, now, now])
    _track_item_scope(conn, eid, "error_solutions", scope)
    for fp in (file_paths or []):
        _link_file_path(conn, eid, fp, "error_solutions")
    return eid, True


def search_error_solutions(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = 5,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    """Vector search on error→solution pairs."""
    return _vector_search(
        conn, "error_solutions", query_embedding,
        "id, error_pattern, error_context, solution, file_paths, confidence, times_applied, scope",
        _scope_filter(scope), limit, threshold,
    )


# ══════════════════════════════════════════════════════════════════════════
# File path linking (cross-table)
# ══════════════════════════════════════════════════════════════════════════

def _link_file_path(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    file_path: str,
    item_table: str = "facts",
) -> None:
    """Link an item to a file path for path-scoped recall."""
    try:
        conn.execute(
            "INSERT INTO fact_file_links(fact_id, file_path, item_table) VALUES (?, ?, ?)"
            " ON CONFLICT DO NOTHING",
            [item_id, file_path, item_table],
        )
    except Exception:
        pass


def link_item_file_paths(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    file_paths: list[str],
    item_table: str = "facts",
) -> None:
    """Link an item to multiple file paths."""
    for fp in file_paths:
        _link_file_path(conn, item_id, fp, item_table)


def get_items_by_file_paths(
    conn: duckdb.DuckDBPyConnection,
    file_paths: list[str],
    item_table: str = "facts",
    limit: int = 20,
    scope: Optional[str] = None,
) -> list[dict]:
    """Get items linked to specific file paths via fact_file_links."""
    if not file_paths:
        return []
    scope_sql, scope_params = _scope_filter(scope)
    placeholders = ", ".join(["?"] * len(file_paths))

    text_col = "task_description" if item_table == "procedures" else (
        "error_pattern" if item_table == "error_solutions" else (
        "warning" if item_table == "guardrails" else "text"
    ))

    rows = conn.execute(f"""
        SELECT DISTINCT t.id, t.{text_col}, t.scope
        FROM {item_table} t
        JOIN fact_file_links ffl ON t.id = ffl.fact_id AND ffl.item_table = ?
        WHERE ffl.file_path IN ({placeholders})
          AND t.is_active = TRUE
          {scope_sql}
        LIMIT ?
    """, [item_table] + file_paths + scope_params + [limit]).fetchall()
    return [dict(zip(["id", "text", "scope"], r)) for r in rows]


# ══════════════════════════════════════════════════════════════════════════
# Bi-temporal helpers
# ══════════════════════════════════════════════════════════════════════════

def invalidate_fact(
    conn: duckdb.DuckDBPyConnection,
    fact_id: str,
    valid_until: Optional[datetime] = None,
) -> bool:
    """Set valid_until on a fact (bi-temporal invalidation without deleting)."""
    now = valid_until or _now()
    row = conn.execute("SELECT id FROM facts WHERE id = ?", [fact_id]).fetchone()
    if not row:
        return False
    conn.execute(
        "UPDATE facts SET valid_until = ? WHERE id = ?",
        [now, fact_id],
    )
    return True


def get_current_facts(
    conn: duckdb.DuckDBPyConnection,
    scope: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Get facts that are currently valid (valid_until IS NULL or in the future)."""
    scope_sql, scope_params = _scope_filter(scope)
    now = _now()
    rows = conn.execute(f"""
        SELECT id, text, category, temporal_class, importance, decay_score, scope
        FROM facts
        WHERE is_active = TRUE
          AND (valid_until IS NULL OR valid_until > ?)
          {scope_sql}
        ORDER BY importance DESC, decay_score DESC
        LIMIT ?
    """, [now] + scope_params + [limit]).fetchall()
    return [
        dict(zip(["id", "text", "category", "temporal_class", "importance", "decay_score", "scope"], r))
        for r in rows
    ]


def get_items_by_ids(
    conn: duckdb.DuckDBPyConnection,
    item_ids: dict[str, list[str]],
) -> list[dict]:
    """
    Fetch items by their IDs for inclusion in extraction prompts.

    Args:
        item_ids: dict mapping table name to list of IDs, e.g.
                  {"facts": ["id1", "id2"], "ideas": ["id3"]}

    Returns list of {"id", "text", "table"} dicts.
    """
    results = []
    table_text_col = {
        "facts": "text",
        "ideas": "text",
        "decisions": "text",
        "open_questions": "text",
    }

    for table, ids in item_ids.items():
        if not ids or table not in table_text_col:
            continue
        col = table_text_col[table]
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id, {col} FROM {table} WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        for row in rows:
            results.append({"id": row[0], "text": row[1], "table": table})

    return results


# ══════════════════════════════════════════════════════════════════════════
# Feature 3: Outcome-based memory scoring
# ══════════════════════════════════════════════════════════════════════════

_OUTCOME_TABLES = {"facts", "decisions", "guardrails", "procedures", "error_solutions", "ideas", "observations"}


def increment_recalled(
    conn: duckdb.DuckDBPyConnection,
    item_ids_by_table: dict[str, list[str]],
) -> int:
    """Increment times_recalled for a batch of items. Returns total updated count."""
    total = 0
    for table, ids in item_ids_by_table.items():
        if not ids or table not in _OUTCOME_TABLES:
            continue
        try:
            placeholders = ", ".join(["?"] * len(ids))
            conn.execute(
                f"UPDATE {table} SET times_recalled = times_recalled + 1 WHERE id IN ({placeholders})",
                ids,
            )
            total += len(ids)
        except Exception:
            pass  # Column may not exist yet
    return total


def mark_applied(
    conn: duckdb.DuckDBPyConnection,
    item_id: str,
    item_table: str,
) -> bool:
    """Mark an item as applied (used by the agent). Recomputes recall_utility."""
    if item_table not in _OUTCOME_TABLES:
        return False
    try:
        row = conn.execute(f"SELECT id FROM {item_table} WHERE id = ?", [item_id]).fetchone()
        if not row:
            return False
        conn.execute(f"""
            UPDATE {item_table} SET
                times_applied = times_applied + 1,
                recall_utility = 1.0 + LN(1 + times_applied + 1) / (1 + LN(1 + times_recalled))
            WHERE id = ?
        """, [item_id])
        return True
    except Exception:
        return False


def recompute_recall_utility(
    conn: duckdb.DuckDBPyConnection,
    item_table: str,
) -> int:
    """Batch recompute recall_utility for all items in a table."""
    if item_table not in _OUTCOME_TABLES:
        return 0
    try:
        result = conn.execute(f"""
            UPDATE {item_table} SET
                recall_utility = CASE
                    WHEN times_recalled = 0 THEN 1.0
                    ELSE 1.0 + LN(1.0 + times_applied) / (1.0 + LN(1.0 + times_recalled))
                END
            WHERE times_recalled > 0
        """)
        return result.fetchone()[0] if result.description else 0
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════
# Feature 1: Community summaries (hierarchical memory)
# ══════════════════════════════════════════════════════════════════════════

def upsert_community_summary(
    conn: duckdb.DuckDBPyConnection,
    level: int,
    summary: str,
    entity_ids: list[str],
    source_item_ids: list[str],
    embedding: Optional[list[float]] = None,
    scope: str = GLOBAL_SCOPE,
) -> tuple[str, bool]:
    """Insert or update a community summary. Returns (id, is_new)."""
    now = _now()

    existing = None
    if embedding:
        hits = _vector_search(
            conn, "community_summaries", embedding,
            "id, summary, entity_ids, source_item_ids, level",
            "", 1, DEDUP_THRESHOLD,
        )
        if hits:
            existing = hits[0]

    if existing:
        old_sources = existing.get("source_item_ids") or []
        merged = list(set(old_sources) | set(source_item_ids))
        emb_update = ", embedding = ?" if embedding else ""
        params = [summary, merged, now]
        if embedding:
            params.append(embedding)
        params.append(existing["id"])
        conn.execute(f"""
            UPDATE community_summaries SET
                summary = ?, source_item_ids = ?, updated_at = ?
                {emb_update}
            WHERE id = ?
        """, params)
        return existing["id"], False

    cid = _uid()
    conn.execute("""
        INSERT INTO community_summaries
            (id, level, summary, entity_ids, source_item_ids, embedding, scope, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [cid, level, summary, entity_ids, source_item_ids, embedding, scope, now, now])
    return cid, True


def get_community_summaries(
    conn: duckdb.DuckDBPyConnection,
    level: int = 1,
    limit: int = 10,
    scope: Optional[str] = None,
) -> list[dict]:
    """Get active community summaries at a given level."""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(f"""
        SELECT id, level, summary, entity_ids, source_item_ids, scope
        FROM community_summaries
        WHERE is_active = TRUE AND level = ?
          {scope_sql}
        ORDER BY updated_at DESC
        LIMIT ?
    """, [level] + scope_params + [limit]).fetchall()
    return [
        dict(zip(["id", "level", "summary", "entity_ids", "source_item_ids", "scope"], r))
        for r in rows
    ]


def search_community_summaries(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: list[float],
    limit: int = 5,
    threshold: float = RECALL_THRESHOLD,
    scope: Optional[str] = None,
) -> list[dict]:
    """Vector search on community summaries."""
    return _vector_search(
        conn, "community_summaries", query_embedding,
        "id, level, summary, entity_ids, source_item_ids, scope",
        _scope_filter(scope), limit, threshold,
    )


# ══════════════════════════════════════════════════════════════════════════
# Feature 7: Coherence validation
# ══════════════════════════════════════════════════════════════════════════

def find_potential_contradictions(
    conn: duckdb.DuckDBPyConnection,
    scope: Optional[str] = None,
    similarity_low: float = 0.88,
    similarity_high: float = 0.92,
    limit: int = 50,
) -> list[tuple[dict, dict, float]]:
    """
    Find active fact pairs with high but sub-dedup similarity (potential contradictions).
    Returns list of (fact_a, fact_b, similarity) tuples.
    """
    scope_sql, scope_params = _scope_filter(scope)
    try:
        rows = conn.execute(f"""
            SELECT id, text, category, temporal_class, importance, decay_score,
                   embedding, created_at
            FROM facts
            WHERE is_active = TRUE AND embedding IS NOT NULL
              {scope_sql}
            ORDER BY created_at DESC
            LIMIT ?
        """, scope_params + [limit * 2]).fetchall()
    except Exception:
        return []

    cols = ["id", "text", "category", "temporal_class", "importance", "decay_score", "embedding", "created_at"]
    items = [dict(zip(cols, r)) for r in rows]

    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["embedding"] is None or b["embedding"] is None:
                continue
            sim = _cosine_py(a["embedding"], b["embedding"])
            if similarity_low <= sim < similarity_high:
                a_clean = {k: v for k, v in a.items() if k != "embedding"}
                b_clean = {k: v for k, v in b.items() if k != "embedding"}
                pairs.append((a_clean, b_clean, sim))
    return pairs[:limit]


def resolve_contradiction(
    conn: duckdb.DuckDBPyConnection,
    keep_id: str,
    invalidate_id: str,
) -> bool:
    """Resolve a contradiction by invalidating the older fact using bi-temporal."""
    return invalidate_fact(conn, invalidate_id)
