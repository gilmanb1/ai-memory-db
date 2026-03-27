#!/usr/bin/env python3
"""
post_tool_use.py — Re-parse edited files to keep code graph current.

Fires after Write or Edit tool calls. Parses the modified file
and updates code_symbols and code_dependencies in DuckDB.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))


def main(payload: dict) -> None:
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Write", "Edit"):
        return

    # Extract file path from tool input
    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return

    # Only parse if we have a parser for this extension
    ext = Path(file_path).suffix.lower()
    parseable = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}
    if ext not in parseable:
        return

    from memory.config import CODE_GRAPH_ENABLED
    if not CODE_GRAPH_ENABLED:
        return

    cwd = payload.get("cwd", os.getcwd())

    try:
        from memory import db
        from memory.code_graph import parse_single_file
        from memory.scope import resolve_scope

        scope = resolve_scope(cwd)
        conn = db.get_connection(read_only=False)
        try:
            stats = parse_single_file(file_path, cwd, conn, scope)
            if stats.get("parsed"):
                print(
                    f"[memory] Code graph updated: {Path(file_path).name} "
                    f"({stats.get('symbols', 0)} symbols)",
                    file=sys.stderr,
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] Code graph update failed: {exc}", file=sys.stderr)

    # ── Guardrail enforcement ─────────────────────────────────────────────
    try:
        from memory.config import GUARDRAIL_ENFORCEMENT_ENABLED, GUARDRAIL_AUTO_STASH
        if GUARDRAIL_ENFORCEMENT_ENABLED:
            from memory import db as _db
            from memory.guardrail_check import enforce_guardrail
            from memory.scope import resolve_scope

            scope = resolve_scope(cwd)
            gconn = _db.get_connection(read_only=True)
            try:
                violation = enforce_guardrail(gconn, file_path, cwd, auto_stash=GUARDRAIL_AUTO_STASH, scope=scope)
            finally:
                gconn.close()

            if violation:
                stash_info = ""
                if violation.get("stashed"):
                    stash_info = (
                        f" Changes stashed as '{violation['stash_msg']}'. "
                        f"Use 'git stash pop' to restore or 'git stash drop' to discard."
                    )
                print(
                    f"[memory] ⚠ GUARDRAIL VIOLATION: Edit to {file_path} may violate:\n"
                    f"{violation['warning_text']}{stash_info}",
                    file=sys.stderr,
                )
    except ImportError:
        pass  # Config keys not yet defined
    except Exception as exc:
        print(f"[memory] Guardrail check failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        payload = {}
    main(payload)
