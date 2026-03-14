#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
#   "anthropic>=0.50.0",
# ]
# ///
"""
pre_compact.py — Claude Code PreCompact hook.

Fires before a conversation is compacted. Extracts knowledge from the
transcript and stores it to DuckDB. Skips if extraction already ran
for this session (e.g., triggered earlier by the status line at 90%).

Outputs:
  stdout — formatted summary (appears in Claude Code transcript log)
  stderr — diagnostics (shown directly in terminal)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

from memory.ingest import acquire_lock, run_extraction


def main(payload: dict) -> None:
    session_id = payload.get("session_id", "unknown")
    transcript_path = payload.get("transcript_path", "")
    trigger = payload.get("trigger", "auto")
    cwd = payload.get("cwd", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not acquire_lock(session_id):
        print(f"[pre_compact] Extraction already ran for session {session_id[:12]}..., skipping.", file=sys.stderr)
        sys.exit(0)

    run_extraction(
        session_id=session_id,
        transcript_path=transcript_path,
        trigger=f"pre_compact/{trigger}",
        cwd=cwd,
        api_key=api_key,
    )


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        print(f"[pre_compact] Failed to parse payload: {exc}", file=sys.stderr)
        sys.exit(1)
    main(payload)
