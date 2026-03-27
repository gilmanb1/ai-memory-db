#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
#   "anthropic>=0.50.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
status_line.py — Claude Code status line script.

Called repeatedly with JSON on stdin containing context_window stats.
Triggers background knowledge extraction at multiple thresholds (40%, 70%, 90%)
for incremental multi-pass extraction.

Outputs a single line of text for the status bar display.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

from memory.config import EXTRACTION_THRESHOLDS

_WORKER = Path.home() / ".claude" / "hooks" / "_extract_worker.py"


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    # ── Extract context window info ───────────────────────────────────────
    ctx = data.get("context_window", {})
    used_pct = ctx.get("used_percentage", 0)
    total = ctx.get("context_window_size", 0)

    session_id = data.get("session_id", "")
    transcript_path = data.get("transcript_path", "")
    cwd = data.get("cwd", "")

    # ── Display status ────────────────────────────────────────────────────
    if total:
        status = f"mem: {used_pct}% ctx"
    else:
        status = "mem: --"

    # ── Trigger extraction at thresholds ──────────────────────────────────
    if session_id and transcript_path:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            from memory.extraction_state import (
                is_extraction_complete, load_state,
                acquire_running_lock, release_running_lock,
            )

            if not is_extraction_complete(session_id):
                state = load_state(session_id)
                pass_count = state["pass_count"] if state else 0

                # Determine which threshold to check
                threshold_idx = min(pass_count, len(EXTRACTION_THRESHOLDS) - 1)
                threshold = EXTRACTION_THRESHOLDS[threshold_idx]

                if used_pct >= threshold:
                    # Try to spawn a background extraction pass
                    try:
                        subprocess.Popen(
                            [sys.executable, str(_WORKER), session_id, transcript_path, cwd],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=open(Path.home() / ".claude" / "memory" / "extract.log", "a"),
                            start_new_session=True,
                            env={**os.environ, "ANTHROPIC_API_KEY": api_key},
                        )
                        status += f" [pass {pass_count + 1}]"
                    except Exception:
                        pass

    # ── Speculative pre-fetch recall ───────────────────────────────────────
    # Pre-compute recall context so user_prompt_submit can use it instantly
    if session_id and cwd:
        try:
            from memory.extraction_state import load_prefetch, save_prefetch
            from memory.config import DB_PATH

            # Only pre-fetch if DB exists and we haven't recently
            if DB_PATH.exists():
                cached = load_prefetch(session_id, max_age_s=30.0)
                if not cached:
                    from memory import db, recall, embeddings
                    from memory.scope import resolve_scope

                    # Use recent conversation context as the prefetch query
                    recent_summary = data.get("conversation_summary", "")
                    if recent_summary and len(recent_summary) > 20:
                        query_emb = embeddings.embed(recent_summary)
                        if query_emb:
                            conn = db.get_connection(read_only=True)
                            try:
                                scope = resolve_scope(cwd)
                                ctx = recall.prompt_recall(conn, query_emb, recent_summary, scope=scope)
                                formatted, _stats = recall.format_prompt_context(ctx)
                                if formatted:
                                    save_prefetch(session_id, formatted, query_emb)
                            finally:
                                conn.close()
        except Exception:
            pass  # Pre-fetch is best-effort, never crash

    print(status)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Status line must never crash — log the error for debugging
        import traceback
        try:
            with open(str(Path.home() / ".claude" / "memory" / "status_error.log"), "a") as f:
                traceback.print_exc(file=f)
        except Exception:
            pass
        print("mem: err")
