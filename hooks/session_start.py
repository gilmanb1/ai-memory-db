#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
#   "numpy>=1.24.0",
#   "tree-sitter>=0.24.0",
#   "tree-sitter-python>=0.25.0",
#   "tree-sitter-javascript>=0.25.0",
#   "tree-sitter-typescript>=0.23.0",
#   "tree-sitter-go>=0.25.0",
#   "tree-sitter-rust>=0.24.0",
# ]
# ///
"""
session_start.py — Claude Code SessionStart hook.

Fires at the beginning of every new Claude Code session.
Queries the memory database for long-term and medium-term knowledge,
and injects it as a systemMessage so Claude has persistent context
from day one of the session.

Output (stdout) — JSON with a `systemMessage` field.
Stderr is reserved for errors/warnings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

from memory import db, recall
from memory.scope import resolve_scope


def main(payload: dict) -> None:
    # If the DB doesn't exist yet there's nothing to recall
    from memory.config import DB_PATH, OLLAMA_MODEL
    if not DB_PATH.exists():
        sys.exit(0)

    # Warn if Ollama is down — per-prompt recall will be disabled
    from memory.embeddings import is_ollama_available
    if not is_ollama_available():
        print(
            f"[memory] Ollama not available or '{OLLAMA_MODEL}' not pulled. "
            f"Per-prompt recall disabled. Run: ollama pull {OLLAMA_MODEL}",
            file=sys.stderr,
        )

    # Resolve project scope from cwd
    cwd = payload.get("cwd", "")
    scope = resolve_scope(cwd) if cwd else None

    conn = None
    try:
        conn = db.get_connection(read_only=True)

        stats = db.get_stats(conn)

        # Nothing stored yet → skip injection
        total_facts = stats["facts"]["total"]
        if total_facts == 0:
            sys.exit(0)

        context = recall.session_recall(conn, scope=scope)
    except Exception as exc:
        print(f"[session_start] Memory recall failed: {exc}", file=sys.stderr)
        sys.exit(0)
    finally:
        if conn is not None:
            conn.close()

    system_message, truncation_stats = recall.format_session_context(context)
    if not system_message:
        sys.exit(0)

    # Output JSON that Claude Code reads and prepends as a system message
    print(json.dumps({"systemMessage": system_message}))

    # Cache recall item IDs for the first extraction pass
    session_id = payload.get("session_id", "")
    if session_id:
        try:
            from memory.extraction_state import save_recall_cache
            recall_items = []
            for f in context.get("long_facts", []):
                recall_items.append({"id": f["id"], "text": f["text"], "table": "facts"})
            for f in context.get("medium_facts", []):
                recall_items.append({"id": f["id"], "text": f["text"], "table": "facts"})
            for d in context.get("decisions", []):
                recall_items.append({"id": d["id"], "text": d["text"], "table": "decisions"})
            if recall_items:
                save_recall_cache(session_id, recall_items)
        except Exception:
            pass  # Non-critical — extraction will fall back to DB query

    # Terminal hint
    fact_counts = stats["facts"]
    parts = [
        f"[memory] Injected {fact_counts['long']} long + {fact_counts['medium']} medium facts",
        f"{stats['decisions']['total']} decisions",
        f"{stats['entities']['total']} entities",
    ]
    guardrail_count = stats.get("guardrails", {}).get("total", 0)
    procedure_count = stats.get("procedures", {}).get("total", 0)
    if guardrail_count:
        parts.append(f"{guardrail_count} guardrails")
    if procedure_count:
        parts.append(f"{procedure_count} procedures")
    if truncation_stats.get("truncated", 0) > 0:
        parts.append(f"truncated {truncation_stats['truncated']} items")
    print(" | ".join(parts), file=sys.stderr)

    # ── Incremental code graph parse ──────────────────────────────────
    try:
        from memory.config import CODE_GRAPH_ENABLED
        if CODE_GRAPH_ENABLED and cwd:
            from memory.code_graph import parse_repo
            cg_conn = db.get_connection(read_only=False)
            try:
                cg_stats = parse_repo(cwd, cg_conn, scope)
                if cg_stats.get("files_parsed", 0) > 0:
                    print(
                        f"[memory] Code graph: {cg_stats['files_parsed']} files, "
                        f"{cg_stats['symbols_found']} symbols",
                        file=sys.stderr,
                    )
            finally:
                cg_conn.close()
    except Exception as exc:
        print(f"[memory] Code graph skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        payload = {}
    main(payload)
