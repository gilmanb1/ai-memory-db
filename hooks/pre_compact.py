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

Fires before a conversation is compacted. Runs incremental extraction
with is_final=True to finalize narratives and mark extraction complete.

Skips if extraction already completed for this session.

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

from memory.extraction_state import (
    is_extraction_complete, acquire_running_lock, release_running_lock,
    load_recall_cache,
)
from memory.ingest import run_incremental_extraction


def main(payload: dict) -> None:
    session_id = payload.get("session_id", "unknown")
    transcript_path = payload.get("transcript_path", "")
    trigger = payload.get("trigger", "auto")
    cwd = payload.get("cwd", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if is_extraction_complete(session_id):
        print(f"[pre_compact] Extraction already complete for {session_id[:12]}...", file=sys.stderr)
        sys.exit(0)

    if not acquire_running_lock(session_id):
        print(f"[pre_compact] Another pass is running for {session_id[:12]}...", file=sys.stderr)
        sys.exit(0)

    try:
        recall_items = load_recall_cache(session_id)
        run_incremental_extraction(
            session_id=session_id,
            transcript_path=transcript_path,
            trigger=f"pre_compact/{trigger}",
            cwd=cwd,
            api_key=api_key,
            is_final=True,
            session_recall_items=recall_items,
        )
    finally:
        release_running_lock(session_id)


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:
        print(f"[pre_compact] Failed to parse payload: {exc}", file=sys.stderr)
        sys.exit(1)
    main(payload)
