"""Chat streaming endpoint — NDJSON-streamed reasoning over the memory database."""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    scope: Optional[str] = None
    conversation_history: list[dict] = []  # [{role, content}]


# ── Direct query patterns ─────────────────────────────────────────────────
# These bypass semantic search and run real SQL for questions about
# recency, ordering, counts, and listings.

_TABLE_MAP = {
    "fact":            ("facts",            "text",             "created_at"),
    "facts":           ("facts",            "text",             "created_at"),
    "decision":        ("decisions",        "text",             "created_at"),
    "decisions":       ("decisions",        "text",             "created_at"),
    "observation":     ("observations",     "text",             "created_at"),
    "observations":    ("observations",     "text",             "created_at"),
    "entity":          ("entities",         "name",             "last_seen_at"),
    "entities":        ("entities",         "name",             "last_seen_at"),
    "guardrail":       ("guardrails",       "warning",          "created_at"),
    "guardrails":      ("guardrails",       "warning",          "created_at"),
    "procedure":       ("procedures",       "task_description", "created_at"),
    "procedures":      ("procedures",       "task_description", "created_at"),
    "error_solution":  ("error_solutions",  "error_pattern",    "created_at"),
    "error_solutions": ("error_solutions",  "error_pattern",    "created_at"),
    "error solution":  ("error_solutions",  "error_pattern",    "created_at"),
    "error solutions": ("error_solutions",  "error_pattern",    "created_at"),
    "idea":            ("ideas",            "text",             "created_at"),
    "ideas":           ("ideas",            "text",             "created_at"),
    "relationship":    ("relationships",    "description",      "created_at"),
    "relationships":   ("relationships",    "description",      "created_at"),
    "session":         ("sessions",         "summary",          "created_at"),
    "sessions":        ("sessions",         "summary",          "created_at"),
}

# Patterns: "N most recent/latest/newest <type>", "last N <type>", "oldest N <type>"
_RECENCY_RE = re.compile(
    r"(?:(?:the\s+)?(\d+)\s+(?:most\s+)?(recent|latest|newest|oldest|earliest)\s+([\w\s]+?))"
    r"|(?:(?:the\s+)?(?:last|latest|newest)\s+(\d+)\s+([\w\s]+?))"
    r"|(?:(?:the\s+)?(?:oldest|earliest|first)\s+(\d+)\s+([\w\s]+?))",
    re.IGNORECASE,
)

# Patterns: "list all <type>", "show all <type>", "all <type>"
_LIST_ALL_RE = re.compile(
    r"(?:list|show|display|get)\s+(?:all\s+)?([\w\s]+?)(?:\s+stored|\s+in\s+the|\s*$)",
    re.IGNORECASE,
)


_AGGREGATION_PATTERNS = [
    # (regex, handler_name)
    (re.compile(r"(?:most|highest|top)\s+(?:recalled|retrieved|used)\s+(\w+)", re.I), "most_recalled"),
    (re.compile(r"(?:most|highest)\s+(?:important|valued)\s+(\w+)", re.I), "most_important"),
    (re.compile(r"(?:which|what)\s+(\w+)\s+(?:have|has)\s+(?:been\s+)?(?:recalled|retrieved)\s+(?:the\s+)?most", re.I), "most_recalled"),
    (re.compile(r"(?:which|what)\s+(?:are\s+)?(?:the\s+)?most\s+important\s+(\w+)", re.I), "most_important"),
    (re.compile(r"(?:which|what)\s+(?:entit|node)\w*\s+(?:have|has)\s+(?:the\s+)?most\s+(?:relationship|connection|edge)", re.I), "most_connected"),
    (re.compile(r"most\s+(?:connected|linked|related)\s+entit", re.I), "most_connected"),
]

_SESSION_PATTERNS = [
    (re.compile(r"(?:what\s+was\s+)?(?:learned|extracted|captured)\s+in\s+(?:the\s+)?(?:last|latest|most\s+recent)\s+session", re.I), "last_session"),
    (re.compile(r"(?:summarize|summary\s+of|show|describe|what\s+(?:was|happened)\s+in)\s+session\s+([\w-]+)", re.I), "specific_session"),
    (re.compile(r"(?:last|latest|most\s+recent)\s+session", re.I), "last_session"),
]

_FILE_PATTERNS = [
    (re.compile(r"(?:what|which)\s+(?:guardrails?)\s+(?:protect|apply\s+to|cover|are\s+on|for)\s+(.+?)(?:\s*\??\s*$)", re.I), "file_guardrails"),
    (re.compile(r"(?:what\s+do\s+we\s+know\s+about|knowledge\s+about|facts?\s+about|info\s+about)\s+(?:file\s+)?(.+?)(?:\s*\??\s*$)", re.I), "file_knowledge"),
    (re.compile(r"(?:guardrails?|procedures?|facts?|error.?solutions?)\s+(?:for|about|on|linked\s+to)\s+(.+?)(?:\s*\??\s*$)", re.I), "file_knowledge"),
]

_SCOPE_PATTERNS = [
    (re.compile(r"(?:what(?:'s|\s+is)\s+in|show|list)\s+(?:scope|project)\s+(.+?)(?:\s*\??\s*$)", re.I), "scope_items"),
    (re.compile(r"(?:compare|diff|difference)\s+(?:between\s+)?(?:scope|project)\s+(.+?)\s+(?:and|vs|with)\s+(.+?)(?:\s*\??\s*$)", re.I), "scope_compare"),
]

_CONTRADICTION_PATTERNS = [
    re.compile(r"(?:contradiction|conflict|inconsistenc|disagree)", re.I),
]


def _try_direct_query(conn, query: str, scope: Optional[str]) -> Optional[str]:
    """Attempt to answer the query with a direct SQL query. Returns context string or None."""
    query_lower = query.lower().strip().rstrip("?.")

    scope_sql, scope_params = "", []
    if scope:
        scope_sql = " AND (scope = ? OR scope = '__global__')"
        scope_params = [scope]

    # Try recency patterns
    m = _RECENCY_RE.search(query_lower)
    if m:
        if m.group(1):  # "N most recent <type>"
            n = int(m.group(1))
            direction = m.group(2).lower()
            type_name = m.group(3).strip()
        elif m.group(4):  # "last N <type>"
            n = int(m.group(4))
            direction = "recent"
            type_name = m.group(5).strip()
        elif m.group(6):  # "oldest N <type>"
            n = int(m.group(6))
            direction = "oldest"
            type_name = m.group(7).strip()
        else:
            return None

        n = min(n, 50)  # safety cap
        ascending = direction in ("oldest", "earliest", "first")
        return _query_by_recency(conn, type_name, n, ascending, scope_sql, scope_params)

    # Try "list all" patterns
    m = _LIST_ALL_RE.search(query_lower)
    if m:
        type_name = m.group(1).strip()
        return _query_by_recency(conn, type_name, 30, False, scope_sql, scope_params)

    # Try aggregation patterns
    for pattern, handler in _AGGREGATION_PATTERNS:
        m = pattern.search(query_lower)
        if m:
            if handler == "most_recalled":
                type_name = m.group(1).strip() if m.lastindex else "facts"
                return _query_aggregation(conn, type_name, "times_recalled", scope_sql, scope_params)
            elif handler == "most_important":
                type_name = m.group(1).strip() if m.lastindex else "facts"
                return _query_aggregation(conn, type_name, "importance", scope_sql, scope_params)
            elif handler == "most_connected":
                return _query_most_connected_entities(conn, scope_sql, scope_params)

    # Try session patterns
    for pattern, handler in _SESSION_PATTERNS:
        m = pattern.search(query_lower)
        if m:
            if handler == "last_session":
                return _query_session(conn, None, scope_sql, scope_params)
            elif handler == "specific_session":
                sid = m.group(1).strip()
                return _query_session(conn, sid, scope_sql, scope_params)

    # Try file patterns
    for pattern, handler in _FILE_PATTERNS:
        m = pattern.search(query_lower)
        if m:
            file_ref = m.group(1).strip().strip("'\"")
            if handler == "file_guardrails":
                return _query_file_items(conn, file_ref, "guardrails")
            elif handler == "file_knowledge":
                return _query_file_items(conn, file_ref, None)

    # Try scope patterns
    for pattern, handler in _SCOPE_PATTERNS:
        m = pattern.search(query_lower)
        if m:
            if handler == "scope_items":
                scope_name = m.group(1).strip()
                return _query_scope(conn, scope_name)
            elif handler == "scope_compare":
                return _query_scope_compare(conn, m.group(1).strip(), m.group(2).strip())

    # Try contradiction patterns
    for pattern in _CONTRADICTION_PATTERNS:
        if pattern.search(query_lower):
            return _query_contradictions(conn, scope_sql, scope_params)

    return None


def _query_by_recency(
    conn, type_name: str, n: int, ascending: bool,
    scope_sql: str, scope_params: list,
) -> Optional[str]:
    """Query a table ordered by time. Returns formatted context string or None."""
    # Resolve type name to table
    info = _TABLE_MAP.get(type_name.rstrip("s"))
    if not info:
        info = _TABLE_MAP.get(type_name)
    if not info:
        # Try partial match
        for key, val in _TABLE_MAP.items():
            if key in type_name or type_name in key:
                info = val
                break
    if not info:
        return None

    table, text_col, time_col = info
    order = "ASC" if ascending else "DESC"
    label = "oldest" if ascending else "most recent"

    # entities table doesn't have is_active in the base schema (added by migration)
    active_filter = "is_active = TRUE" if table not in ("sessions",) else "1=1"

    try:
        # Build select with available columns
        table_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        select_parts = ["id", text_col, time_col]
        if "temporal_class" in table_cols:
            select_parts.append("temporal_class")
        if "importance" in table_cols:
            select_parts.append("importance")
        if "scope" in table_cols:
            select_parts.append("scope")
        if "category" in table_cols:
            select_parts.append("category")
        if "session_count" in table_cols:
            select_parts.append("session_count")

        select = ", ".join(select_parts)
        rows = conn.execute(f"""
            SELECT {select} FROM {table}
            WHERE {active_filter} {scope_sql}
            ORDER BY {time_col} {order}
            LIMIT ?
        """, scope_params + [n]).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    col_names = select_parts
    lines = [f"## Direct Query: {label} {n} {table}\n"]
    for row in rows:
        d = dict(zip(col_names, row))
        rid = d["id"]
        text = d.get(text_col, "") or ""
        if len(text) > 200:
            text = text[:197] + "..."
        ts = str(d.get(time_col, ""))[:19]
        node_id = f"{table}:{rid}" if table != "entities" else d.get("name", rid)

        parts = [f"[{node_id}]"]
        if ts:
            parts.append(f"({ts})")
        tc = d.get("temporal_class")
        if tc:
            parts.append(f"[{tc}]")
        imp = d.get("importance")
        if imp is not None:
            parts.append(f"imp:{imp}")
        scope_val = d.get("scope", "")
        if scope_val and scope_val != "__global__":
            parts.append(f"scope:{scope_val.split('/')[-1]}")
        parts.append(text)
        lines.append(" ".join(parts))

    return "\n".join(lines)


def _query_aggregation(
    conn, type_name: str, order_col: str,
    scope_sql: str, scope_params: list, limit: int = 15,
) -> Optional[str]:
    """Query items ordered by an aggregation column (importance, times_recalled, etc.)."""
    info = _resolve_table(type_name)
    if not info:
        return None

    table, text_col, time_col = info
    try:
        table_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return None

    if order_col not in table_cols:
        return None

    active_filter = "is_active = TRUE" if "is_active" in table_cols else "1=1"
    select_parts = ["id", text_col, order_col]
    if "scope" in table_cols:
        select_parts.append("scope")
    if time_col in table_cols and time_col not in select_parts:
        select_parts.append(time_col)

    try:
        select = ", ".join(select_parts)
        rows = conn.execute(f"""
            SELECT {select} FROM {table}
            WHERE {active_filter} {scope_sql}
            ORDER BY {order_col} DESC
            LIMIT ?
        """, scope_params + [limit]).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    lines = [f"## Direct Query: {table} by {order_col} (top {limit})\n"]
    for row in rows:
        d = dict(zip(select_parts, row))
        rid = d["id"]
        text = str(d.get(text_col, "") or "")[:200]
        node_id = f"{table}:{rid}"
        val = d.get(order_col, 0)
        lines.append(f"[{node_id}] {order_col}={val} {text}")

    return "\n".join(lines)


def _query_most_connected_entities(
    conn, scope_sql: str, scope_params: list, limit: int = 15,
) -> Optional[str]:
    """Find entities with the most relationships."""
    try:
        rows = conn.execute(f"""
            SELECT entity, COUNT(*) as rel_count FROM (
                SELECT from_entity AS entity FROM relationships WHERE is_active = TRUE {scope_sql}
                UNION ALL
                SELECT to_entity AS entity FROM relationships WHERE is_active = TRUE {scope_sql}
            ) GROUP BY entity ORDER BY rel_count DESC LIMIT ?
        """, scope_params + scope_params + [limit]).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    lines = ["## Direct Query: Most connected entities\n"]
    for name, count in rows:
        # Get relationship details
        try:
            rels = conn.execute("""
                SELECT rel_type, CASE WHEN from_entity = ? THEN to_entity ELSE from_entity END AS other
                FROM relationships WHERE is_active = TRUE AND (from_entity = ? OR to_entity = ?)
                LIMIT 5
            """, [name, name, name]).fetchall()
            rel_desc = ", ".join(f"--[{rt}]-->{other}" for rt, other in rels)
        except Exception:
            rel_desc = ""
        lines.append(f"[{name}] relationships={count} {rel_desc}")

    return "\n".join(lines)


def _query_session(
    conn, session_id: Optional[str],
    scope_sql: str, scope_params: list,
) -> Optional[str]:
    """Get session details + items extracted from it."""
    try:
        if session_id:
            row = conn.execute(
                "SELECT id, trigger, cwd, message_count, summary, created_at, scope "
                "FROM sessions WHERE id LIKE ? ORDER BY created_at DESC LIMIT 1",
                [session_id + "%"],
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, trigger, cwd, message_count, summary, created_at, scope "
                "FROM sessions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
    except Exception:
        return None

    if not row:
        return None

    sid, trigger, cwd, msg_count, summary, created, sess_scope = row
    lines = [
        f"## Direct Query: Session {sid}\n",
        f"Timestamp: {str(created)[:19]}",
        f"Trigger: {trigger}",
        f"Messages: {msg_count}",
        f"Scope: {sess_scope or 'unknown'}",
    ]
    if summary:
        lines.append(f"Summary: {summary}")

    # Get narratives
    try:
        narratives = conn.execute(
            "SELECT pass_number, narrative, is_final FROM session_narratives "
            "WHERE session_id = ? ORDER BY pass_number", [sid],
        ).fetchall()
        if narratives:
            lines.append("\n### Narratives")
            for pn, narr, is_final in narratives:
                tag = " (final)" if is_final else ""
                lines.append(f"Pass {pn}{tag}: {narr}")
    except Exception:
        pass

    # Get items from this session
    item_tables = [
        ("Facts", "facts", "text"),
        ("Ideas", "ideas", "text"),
        ("Decisions", "decisions", "text"),
        ("Guardrails", "guardrails", "warning"),
        ("Procedures", "procedures", "task_description"),
        ("Error Solutions", "error_solutions", "error_pattern"),
    ]
    for label, table, text_col in item_tables:
        try:
            table_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if "source_session" not in table_cols:
                continue
            items = conn.execute(f"""
                SELECT id, {text_col} FROM {table}
                WHERE source_session = ? AND is_active = TRUE
                ORDER BY created_at LIMIT 20
            """, [sid]).fetchall()
            if items:
                lines.append(f"\n### {label} ({len(items)})")
                for iid, txt in items:
                    txt_short = (txt or "")[:150]
                    lines.append(f"[{table}:{iid}] {txt_short}")
        except Exception:
            continue

    # Get relationships from this session
    try:
        rels = conn.execute(
            "SELECT from_entity, to_entity, rel_type FROM relationships "
            "WHERE source_session = ? LIMIT 20", [sid],
        ).fetchall()
        if rels:
            lines.append(f"\n### Relationships ({len(rels)})")
            for fr, to, rt in rels:
                lines.append(f"{fr} --[{rt}]--> {to}")
    except Exception:
        pass

    return "\n".join(lines)


def _query_file_items(
    conn, file_ref: str, restrict_table: Optional[str] = None,
) -> Optional[str]:
    """Find knowledge items linked to a file path via fact_file_links."""
    # Normalize the file reference — user might say "db.py" meaning "memory/db.py"
    file_ref_clean = file_ref.strip().rstrip("?.")

    tables_to_query = [
        ("facts", "fact", "text"),
        ("guardrails", "guardrail", "warning"),
        ("procedures", "procedure", "task_description"),
        ("error_solutions", "error_solution", "error_pattern"),
    ]
    if restrict_table:
        tables_to_query = [(t, p, c) for t, p, c in tables_to_query if t.startswith(restrict_table)]

    lines = [f"## Direct Query: Knowledge linked to '{file_ref_clean}'\n"]
    total = 0

    for item_table, _, text_col in tables_to_query:
        try:
            rows = conn.execute(f"""
                SELECT DISTINCT t.id, t.{text_col}, ffl.file_path
                FROM {item_table} t
                JOIN fact_file_links ffl ON t.id = ffl.fact_id AND ffl.item_table = ?
                WHERE t.is_active = TRUE AND ffl.file_path LIKE ?
                LIMIT 20
            """, [item_table, f"%{file_ref_clean}%"]).fetchall()

            if rows:
                lines.append(f"### {item_table} ({len(rows)})")
                for rid, txt, fp in rows:
                    txt_short = (txt or "")[:150]
                    lines.append(f"[{item_table}:{rid}] (file: {fp}) {txt_short}")
                total += len(rows)
        except Exception:
            continue

    # Also check guardrails with inline file_paths array
    if not restrict_table or restrict_table == "guardrails":
        try:
            rows = conn.execute("""
                SELECT id, warning, file_paths FROM guardrails
                WHERE is_active = TRUE AND file_paths IS NOT NULL AND len(file_paths) > 0
            """).fetchall()
            for gid, warning, fps in rows:
                if fps and any(file_ref_clean in fp for fp in fps):
                    lines.append(f"[guardrails:{gid}] (files: {', '.join(fps)}) {warning}")
                    total += 1
        except Exception:
            pass

    if total == 0:
        return None

    return "\n".join(lines)


def _query_scope(conn, scope_name: str) -> Optional[str]:
    """List items in a specific project scope."""
    # Try to match scope_name to actual scope values
    try:
        scopes = conn.execute("""
            SELECT DISTINCT scope FROM (
                SELECT scope FROM facts WHERE is_active = TRUE
                UNION ALL SELECT scope FROM decisions WHERE is_active = TRUE
                UNION ALL SELECT scope FROM ideas WHERE is_active = TRUE
            )
        """).fetchall()
        scope_values = [r[0] for r in scopes]
    except Exception:
        return None

    matched = None
    for sv in scope_values:
        if scope_name in sv or sv.endswith(scope_name) or scope_name.lower() in sv.lower():
            matched = sv
            break

    if not matched:
        return f"## Direct Query: Scope '{scope_name}'\n\nNo matching scope found. Known scopes: {', '.join(scope_values)}"

    lines = [f"## Direct Query: Scope '{matched}'\n"]
    tables = [
        ("Facts", "facts", "text"),
        ("Decisions", "decisions", "text"),
        ("Ideas", "ideas", "text"),
        ("Observations", "observations", "text"),
        ("Guardrails", "guardrails", "warning"),
    ]
    total = 0
    for label, table, text_col in tables:
        try:
            rows = conn.execute(f"""
                SELECT id, {text_col} FROM {table}
                WHERE is_active = TRUE AND scope = ?
                ORDER BY created_at DESC LIMIT 10
            """, [matched]).fetchall()
            if rows:
                lines.append(f"### {label} ({len(rows)})")
                for rid, txt in rows:
                    lines.append(f"[{table}:{rid}] {(txt or '')[:150]}")
                total += len(rows)
        except Exception:
            continue

    lines.insert(1, f"Total items shown: {total}")
    return "\n".join(lines)


def _query_scope_compare(conn, scope_a: str, scope_b: str) -> Optional[str]:
    """Compare item counts between two scopes."""
    lines = [f"## Direct Query: Scope comparison '{scope_a}' vs '{scope_b}'\n"]

    tables = ["facts", "decisions", "ideas", "observations", "guardrails", "procedures", "error_solutions"]
    for table in tables:
        try:
            count_a = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE is_active = TRUE AND scope LIKE ?",
                [f"%{scope_a}%"],
            ).fetchone()[0]
            count_b = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE is_active = TRUE AND scope LIKE ?",
                [f"%{scope_b}%"],
            ).fetchone()[0]
            lines.append(f"{table:20s}  {scope_a}: {count_a:>4}  |  {scope_b}: {count_b:>4}")
        except Exception:
            continue

    return "\n".join(lines)


def _query_contradictions(
    conn, scope_sql: str, scope_params: list, limit: int = 10,
) -> Optional[str]:
    """Find potentially contradictory facts by looking for pairs with high embedding
    similarity but different content (the coherence detection approach)."""
    lines = ["## Direct Query: Potential contradictions in facts\n"]

    try:
        # Get facts with embeddings
        rows = conn.execute(f"""
            SELECT id, text, embedding, temporal_class
            FROM facts
            WHERE is_active = TRUE AND embedding IS NOT NULL {scope_sql}
            ORDER BY importance DESC, decay_score DESC
            LIMIT 100
        """, scope_params).fetchall()
    except Exception:
        return "## Direct Query: Contradiction check\n\nCould not query facts for contradiction analysis."

    if len(rows) < 2:
        return "## Direct Query: Contradiction check\n\nNot enough facts to check for contradictions."

    # Compare pairs — look for high similarity (>0.85) but not dedup-level (>0.92)
    pairs_found = []
    try:
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                id_a, text_a, emb_a, tc_a = rows[i]
                id_b, text_b, emb_b, tc_b = rows[j]
                if emb_a is None or emb_b is None:
                    continue
                # Cosine similarity
                dot = sum(a * b for a, b in zip(emb_a, emb_b))
                norm_a = sum(a * a for a in emb_a) ** 0.5
                norm_b = sum(b * b for b in emb_b) ** 0.5
                if norm_a == 0 or norm_b == 0:
                    continue
                sim = dot / (norm_a * norm_b)
                if 0.82 <= sim <= 0.93:
                    pairs_found.append((sim, id_a, text_a, id_b, text_b))
                if len(pairs_found) >= limit:
                    break
            if len(pairs_found) >= limit:
                break
    except Exception:
        pass

    if not pairs_found:
        lines.append("No potential contradictions detected. All high-similarity fact pairs appear to be consistent or duplicates.")
        return "\n".join(lines)

    pairs_found.sort(key=lambda x: x[0], reverse=True)
    lines.append(f"Found {len(pairs_found)} potentially contradictory pairs (similarity 0.82-0.93):\n")
    for sim, id_a, text_a, id_b, text_b in pairs_found:
        lines.append(f"Pair (similarity={sim:.3f}):")
        lines.append(f"  [facts:{id_a}] {text_a[:120]}")
        lines.append(f"  [facts:{id_b}] {text_b[:120]}")
        lines.append("")

    return "\n".join(lines)


def _resolve_table(type_name: str):
    """Resolve a type name string to (table, text_col, time_col) tuple."""
    info = _TABLE_MAP.get(type_name.rstrip("s"))
    if not info:
        info = _TABLE_MAP.get(type_name)
    if not info:
        for key, val in _TABLE_MAP.items():
            if key in type_name or type_name in key:
                info = val
                break
    return info


# ── Main endpoint ─────────────────────────────────────────────────────────

@router.post("/chat/stream")
def chat_stream(body: ChatRequest):
    """Stream a chat response with memory retrieval context."""

    def generate():
        import anthropic
        from memory.config import REFLECT_MODEL, DB_PATH
        from memory.retrieval import parallel_retrieve
        from memory.embeddings import embed_query

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            yield json.dumps({"type": "error", "data": "ANTHROPIC_API_KEY not set"}) + "\n"
            return

        # 1. Get database stats (always — cheap query, gives Claude accurate numbers)
        from memory import db as memdb
        stats_context = ""
        direct_context = ""
        try:
            stats_conn = memdb.get_connection(read_only=True)
            stats = memdb.get_stats(stats_conn)

            # 1b. Try direct query for recency/listing questions
            direct_context = _try_direct_query(stats_conn, body.query, body.scope) or ""

            stats_conn.close()
            stats_context = (
                "\n## Database Statistics\n"
                f"Facts: {stats['facts']['total']} active ({stats['facts']['long']} long, {stats['facts']['medium']} medium, {stats['facts']['short']} short), {stats['facts']['inactive']} inactive\n"
                f"Ideas: {stats['ideas']['total']} active\n"
                f"Entities: {stats['entities']['total']}\n"
                f"Relationships: {stats['relationships']['total']}\n"
                f"Decisions: {stats['decisions']['total']}\n"
                f"Observations: {stats.get('observations', {}).get('total', 0)} active\n"
                f"Open Questions: {stats['questions']['total']} ({stats['questions']['resolved']} resolved)\n"
                f"Sessions: {stats['sessions']['total']}\n"
                f"Guardrails: {stats.get('guardrails', {}).get('total', 0)}\n"
                f"Procedures: {stats.get('procedures', {}).get('total', 0)}\n"
                f"Error Solutions: {stats.get('error_solutions', {}).get('total', 0)}\n"
            )
        except Exception:
            pass

        # 2. Retrieve relevant context (semantic + BM25 + graph + temporal)
        retrieval_context = ""
        sources = []
        try:
            query_emb = None
            try:
                query_emb = embed_query(body.query)
            except Exception:
                pass

            retrieval = parallel_retrieve(
                db_path=str(DB_PATH),
                query_text=body.query,
                query_embedding=query_emb,
                scope=body.scope,
                limit=25,
                timeout_ms=5000,
            )

            for item in retrieval.items:
                node_id = f"{item.table}:{item.id}" if item.table != "entities" else item.id
                sources.append({
                    "id": node_id,
                    "node_type": item.table,
                    "text": item.text[:150],
                    "score": round(item.score, 3),
                })

            context_lines = []
            for item in retrieval.items:
                node_id = f"{item.table}:{item.id}" if item.table != "entities" else item.id
                context_lines.append(f"[{node_id}] ({item.table}, score={item.score:.2f}) {item.text}")
            retrieval_context = "\n".join(context_lines)
        except Exception:
            pass

        # 3. Emit sources event
        yield json.dumps({"type": "sources", "data": sources}) + "\n"

        if not retrieval_context and not direct_context and not stats_context:
            yield json.dumps({"type": "text", "data": "I don't have any relevant knowledge about that topic."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        # 4. Build system prompt with all context
        system_parts = [
            "You are a knowledge assistant with access to a memory database. "
            "Answer the user's question based on the context below. "
            "When asked about counts or 'how many', use the Database Statistics section for accurate numbers. "
            "When a Direct Query section is present, it contains exact results from the database — use those as your primary source. "
            "Cite your sources by including the source ID in brackets like [fact:abc123] when referencing specific items. "
            "If the context doesn't contain relevant information, say so honestly.",
        ]
        if stats_context:
            system_parts.append(stats_context)
        if direct_context:
            system_parts.append(f"\n{direct_context}")
        if retrieval_context:
            system_parts.append(f"\n## Semantic Search Results ({len(sources)} most relevant items)\n{retrieval_context}")

        system_prompt = "\n".join(system_parts)

        # 5. Build messages with conversation history
        messages = []
        for msg in (body.conversation_history or [])[-10:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": body.query})

        # 6. Stream Claude response
        try:
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model=REFLECT_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield json.dumps({"type": "text", "data": text}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": f"LLM error: {e}"}) + "\n"
            return

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
