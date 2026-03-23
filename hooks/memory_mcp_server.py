#!/usr/bin/env python3
"""
memory_mcp_server.py — Minimal MCP (Model Context Protocol) server for the memory system.

Communicates via stdio using JSON-RPC 2.0. Exposes memory tools to Claude Code:
  - memory_search: Search all memory types by query
  - memory_store: Store a fact, decision, guardrail, procedure, or error_solution
  - memory_guardrail: Create a guardrail (convenience wrapper)
  - memory_check_file: Get all memory associated with a file path
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure memory package is importable
sys.path.insert(0, str(Path.home() / ".claude"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Logging ────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[memory-mcp] {msg}", file=sys.stderr, flush=True)


# ── Lazy imports ───────────────────────────────────────────────────────────

def _get_scope() -> str:
    """Resolve project scope from CWD env var."""
    try:
        from memory.config import GLOBAL_SCOPE
        from memory.scope import resolve_scope
        cwd = os.environ.get("CWD", os.getcwd())
        return resolve_scope(cwd)
    except Exception:
        return "__global__"


def _get_connection():
    """Get a DB connection, or None if unavailable."""
    try:
        from memory import db
        return db.get_connection()
    except Exception as e:
        _log(f"DB connection failed: {e}")
        return None


def _embed(text: str):
    """Embed document text, returns list[float] or None."""
    try:
        from memory.embeddings import embed
        return embed(text)
    except Exception:
        return None


def _embed_query(text: str):
    """Embed query text, returns list[float] or None."""
    try:
        from memory.embeddings import embed_query
        return embed_query(text)
    except Exception:
        return None


# ── Tool definitions ───────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "memory_search",
        "description": "Search all memory types (facts, guardrails, procedures, error_solutions, decisions) by semantic query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query text",
                },
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory types to search. Default: all. Options: facts, guardrails, procedures, error_solutions, decisions",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results per type (default 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_store",
        "description": "Store a fact, decision, guardrail, procedure, or error_solution in memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to store",
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "decision", "guardrail", "procedure", "error_solution"],
                    "description": "The type of memory to store",
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance score 1-10 (default 7)",
                },
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to associate with this memory",
                },
            },
            "required": ["text", "type"],
        },
    },
    {
        "name": "memory_guardrail",
        "description": "Create a guardrail — a warning to surface when working with specific files or patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "warning": {
                    "type": "string",
                    "description": "The guardrail warning text",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this guardrail exists",
                },
                "consequence": {
                    "type": "string",
                    "description": "What happens if the guardrail is violated",
                },
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths this guardrail applies to",
                },
            },
            "required": ["warning"],
        },
    },
    {
        "name": "memory_check_file",
        "description": "Get all memory (guardrails, procedures, facts, error_solutions) associated with a file path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The file path to check",
                },
            },
            "required": ["file_path"],
        },
    },
]


# ── Tool handlers ──────────────────────────────────────────────────────────

def handle_memory_search(arguments: dict) -> str:
    """Search memory by semantic query."""
    query = arguments.get("query", "").strip()
    if not query:
        return "Error: query is required"

    types = arguments.get("types", ["facts", "guardrails", "procedures", "error_solutions", "decisions"])
    limit = arguments.get("limit", 10)

    query_vec = _embed_query(query)
    if not query_vec:
        return "Error: embedding unavailable — cannot perform semantic search"

    conn = _get_connection()
    if not conn:
        return "Error: database unavailable"

    scope = _get_scope()
    results = []

    try:
        from memory import db

        if "facts" in types or "decisions" in types:
            facts = db.search_facts(conn, query_vec, limit=limit, scope=scope)
            for f in facts:
                category = f.get("category", "")
                is_decision = category in ("decision", "decision_rationale")
                type_label = "decision" if is_decision else "fact"
                if type_label == "decision" and "decisions" not in types:
                    continue
                if type_label == "fact" and "facts" not in types:
                    continue
                results.append(f"[{type_label}] {f.get('text', '')} (importance={f.get('importance', '?')}, scope={f.get('scope', '?')})")

        if "guardrails" in types:
            guardrails = db.search_guardrails(conn, query_vec, limit=limit, scope=scope)
            for g in guardrails:
                line = f"[guardrail] {g.get('warning', '')}"
                if g.get("rationale"):
                    line += f" — {g['rationale']}"
                results.append(line)

        if "procedures" in types:
            procedures = db.search_procedures(conn, query_vec, limit=limit, scope=scope)
            for p in procedures:
                results.append(f"[procedure] {p.get('task_description', '')}: {p.get('steps', '')}")

        if "error_solutions" in types:
            errors = db.search_error_solutions(conn, query_vec, limit=limit, scope=scope)
            for e in errors:
                results.append(f"[error_solution] {e.get('error_pattern', '')} -> {e.get('solution', '')}")

    except Exception as e:
        _log(f"Search error: {e}")
        return f"Error during search: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not results:
        return f"No results found for: {query}"

    return f"Found {len(results)} result(s) for '{query}':\n\n" + "\n\n".join(results)


def handle_memory_store(arguments: dict) -> str:
    """Store a memory item."""
    text = arguments.get("text", "").strip()
    mem_type = arguments.get("type", "").strip()
    importance = arguments.get("importance", 7)
    file_paths = arguments.get("file_paths", [])

    if not text:
        return "Error: text is required"
    if mem_type not in ("fact", "decision", "guardrail", "procedure", "error_solution"):
        return f"Error: invalid type '{mem_type}'. Must be one of: fact, decision, guardrail, procedure, error_solution"

    embedding = _embed(text)
    conn = _get_connection()
    if not conn:
        return "Error: database unavailable"

    scope = _get_scope()
    session_id = os.environ.get("SESSION_ID", "mcp-session")

    try:
        from memory import db
        from memory.decay import compute_decay_score

        if mem_type in ("fact", "decision"):
            category = "decision_rationale" if mem_type == "decision" else "contextual"
            item_id, is_new = db.upsert_fact(
                conn, text, category=category, temporal_class="long",
                confidence="high", embedding=embedding, session_id=session_id,
                decay_fn=compute_decay_score, scope=scope, importance=importance,
                file_paths=file_paths or None,
            )
            action = "Stored" if is_new else "Reinforced"
            return f"{action} {mem_type} (id={item_id})"

        elif mem_type == "guardrail":
            warning = text
            rationale = ""
            if " — " in text:
                parts = text.split(" — ", 1)
                warning = parts[0]
                rationale = parts[1]
            item_id, is_new = db.upsert_guardrail(
                conn, warning=warning, rationale=rationale,
                file_paths=file_paths or None, embedding=embedding,
                session_id=session_id, scope=scope, importance=importance,
            )
            action = "Stored" if is_new else "Reinforced"
            return f"{action} guardrail (id={item_id})"

        elif mem_type == "procedure":
            task_desc = text
            steps = ""
            if ": " in text:
                parts = text.split(": ", 1)
                task_desc = parts[0]
                steps = parts[1]
            item_id, is_new = db.upsert_procedure(
                conn, task_description=task_desc, steps=steps,
                file_paths=file_paths or None, embedding=embedding,
                session_id=session_id, decay_fn=compute_decay_score,
                scope=scope, importance=importance,
            )
            action = "Stored" if is_new else "Reinforced"
            return f"{action} procedure (id={item_id})"

        elif mem_type == "error_solution":
            error_pattern = text
            solution = ""
            if " -> " in text:
                parts = text.split(" -> ", 1)
                error_pattern = parts[0]
                solution = parts[1]
            item_id, is_new = db.upsert_error_solution(
                conn, error_pattern=error_pattern, solution=solution,
                file_paths=file_paths or None, embedding=embedding,
                session_id=session_id, scope=scope,
            )
            action = "Stored" if is_new else "Reinforced"
            return f"{action} error_solution (id={item_id})"

    except Exception as e:
        _log(f"Store error: {e}")
        return f"Error storing {mem_type}: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return "Error: unexpected state"


def handle_memory_guardrail(arguments: dict) -> str:
    """Create a guardrail (convenience wrapper)."""
    warning = arguments.get("warning", "").strip()
    if not warning:
        return "Error: warning is required"

    rationale = arguments.get("rationale", "")
    consequence = arguments.get("consequence", "")
    file_paths = arguments.get("file_paths", [])

    embedding = _embed(warning)
    conn = _get_connection()
    if not conn:
        return "Error: database unavailable"

    scope = _get_scope()
    session_id = os.environ.get("SESSION_ID", "mcp-session")

    try:
        from memory import db

        item_id, is_new = db.upsert_guardrail(
            conn, warning=warning, rationale=rationale, consequence=consequence,
            file_paths=file_paths or None, embedding=embedding,
            session_id=session_id, scope=scope,
        )
        action = "Created" if is_new else "Reinforced"
        return f"{action} guardrail (id={item_id})"

    except Exception as e:
        _log(f"Guardrail error: {e}")
        return f"Error creating guardrail: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def handle_memory_check_file(arguments: dict) -> str:
    """Get all memory associated with a file path."""
    file_path = arguments.get("file_path", "").strip()
    if not file_path:
        return "Error: file_path is required"

    conn = _get_connection()
    if not conn:
        return "Error: database unavailable"

    scope = _get_scope()
    results = []

    try:
        from memory import db

        for table in ("guardrails", "procedures", "facts", "error_solutions"):
            items = db.get_items_by_file_paths(conn, [file_path], item_table=table, scope=scope)
            for item in items:
                results.append(f"[{table}] {item.get('text', '')} (scope={item.get('scope', '?')})")

    except Exception as e:
        _log(f"Check file error: {e}")
        return f"Error checking file: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not results:
        return f"No memory items linked to: {file_path}"

    return f"Found {len(results)} item(s) for '{file_path}':\n\n" + "\n\n".join(results)


# ── Tool dispatch ──────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "memory_search": handle_memory_search,
    "memory_store": handle_memory_store,
    "memory_guardrail": handle_memory_guardrail,
    "memory_check_file": handle_memory_check_file,
}


# ── JSON-RPC handling ──────────────────────────────────────────────────────

def _make_response(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _make_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _handle_initialize(req_id, _params: dict) -> dict:
    return _make_response(req_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "memory", "version": "1.0.0"},
    })


def _handle_tools_list(req_id, _params: dict) -> dict:
    return _make_response(req_id, {"tools": TOOLS})


def _handle_tools_call(req_id, params: dict) -> dict:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return _make_error(req_id, -32602, f"Unknown tool: {tool_name}")

    try:
        result_text = handler(arguments)
        return _make_response(req_id, {
            "content": [{"type": "text", "text": result_text}],
        })
    except Exception as e:
        _log(f"Tool call error ({tool_name}): {e}")
        return _make_response(req_id, {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        })


METHOD_HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
}

# Notifications that require no response
NOTIFICATIONS = {
    "notifications/initialized",
    "notifications/cancelled",
}


# ── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    _log("Starting memory MCP server")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _log(f"Malformed JSON: {e}")
            # Only send error if there might be an id
            error = _make_error(None, -32700, f"Parse error: {e}")
            sys.stdout.write(json.dumps(error) + "\n")
            sys.stdout.flush()
            continue

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        # Notifications: no response
        if method in NOTIFICATIONS:
            _log(f"Notification: {method}")
            continue

        # No id means it's a notification we don't recognize — skip
        if req_id is None:
            _log(f"Unknown notification: {method}")
            continue

        handler = METHOD_HANDLERS.get(method)
        if handler:
            response = handler(req_id, params)
        else:
            response = _make_error(req_id, -32601, f"Method not found: {method}")

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
