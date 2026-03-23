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
_extract_worker.py — Background extraction worker.

Spawned by session_end.py and status_line.py to run incremental extraction
in a detached process. Uses running lock to prevent concurrent passes.

Usage: python _extract_worker.py <session_id> <transcript_path> <cwd> [--final]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

from memory.extraction_state import (
    acquire_running_lock, release_running_lock,
    is_extraction_complete, load_recall_cache,
)
from memory.ingest import run_incremental_extraction


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: _extract_worker.py <session_id> <transcript_path> <cwd> [--final]", file=sys.stderr)
        sys.exit(1)

    session_id = sys.argv[1]
    transcript_path = sys.argv[2]
    cwd = sys.argv[3]
    is_final = "--final" in sys.argv
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Skip if extraction already completed for this session
    if is_extraction_complete(session_id):
        print(f"[worker] Extraction already complete for {session_id[:12]}...", file=sys.stderr)
        return

    # Acquire running lock to prevent concurrent passes
    if not acquire_running_lock(session_id):
        print(f"[worker] Another pass is running for {session_id[:12]}...", file=sys.stderr)
        return

    try:
        # Load cached recall items for pass 1
        recall_items = load_recall_cache(session_id)

        result = run_incremental_extraction(
            session_id=session_id,
            transcript_path=transcript_path,
            trigger="background",
            cwd=cwd,
            api_key=api_key,
            quiet=True,
            is_final=is_final,
            session_recall_items=recall_items,
        )

        if result is None and not is_final:
            # Non-final pass failed — release lock for retry
            pass  # lock released in finally
    finally:
        release_running_lock(session_id)


if __name__ == "__main__":
    main()
