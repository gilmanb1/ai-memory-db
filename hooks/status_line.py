#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
#   "anthropic>=0.50.0",
# ]
# ///
"""
status_line.py — Claude Code status line script.

Called repeatedly with JSON on stdin containing context_window stats.
When used_percentage >= 90%, triggers background knowledge extraction
before compaction kicks in.

Outputs a single line of text for the status bar display.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

EXTRACTION_THRESHOLD = 90  # percent context window usage

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
        remaining_pct = 100 - used_pct
        status = f"mem: {used_pct}% ctx"
    else:
        status = "mem: --"
        remaining_pct = 100

    # ── Trigger extraction at threshold ───────────────────────────────────
    if used_pct >= EXTRACTION_THRESHOLD and session_id and transcript_path:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            from memory.ingest import _lock_path
            if not _lock_path(session_id).exists():
                # Spawn background extraction
                try:
                    subprocess.Popen(
                        [sys.executable, str(_WORKER), session_id, transcript_path, cwd],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=open(Path.home() / ".claude" / "memory" / "extract.log", "a"),
                        start_new_session=True,
                        env={**os.environ, "ANTHROPIC_API_KEY": api_key},
                    )
                    status += " [extracting]"
                except Exception:
                    pass

    print(status)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Status line must never crash
        print("mem: err")
