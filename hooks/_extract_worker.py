#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
#   "anthropic>=0.50.0",
# ]
# ///
"""
_extract_worker.py — Background extraction worker.

Spawned by session_end.py and status_line.py to run extraction
in a detached process. Acquires a per-session lock to prevent
duplicate work.

Usage: python _extract_worker.py <session_id> <transcript_path> <cwd>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

from memory.ingest import acquire_lock, release_lock, run_extraction


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: _extract_worker.py <session_id> <transcript_path> <cwd>", file=sys.stderr)
        sys.exit(1)

    session_id = sys.argv[1]
    transcript_path = sys.argv[2]
    cwd = sys.argv[3]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not acquire_lock(session_id):
        print(f"[worker] Extraction already ran for {session_id[:12]}...", file=sys.stderr)
        return

    result = run_extraction(
        session_id=session_id,
        transcript_path=transcript_path,
        trigger="background",
        cwd=cwd,
        api_key=api_key,
        quiet=True,  # no stdout — we're detached
    )

    if result is None:
        release_lock(session_id)  # allow retry on failure


if __name__ == "__main__":
    main()
