#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
#   "anthropic>=0.50.0",
# ]
# ///
"""
session_end.py — Claude Code SessionEnd hook.

Fires when a session terminates. Has a 1.5s default timeout, so we
fork the extraction into a background process and exit immediately.

Skips if extraction already ran for this session (lock file exists).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

from memory.ingest import _lock_path

# The background worker script that actually runs extraction
_WORKER = Path.home() / ".claude" / "hooks" / "_extract_worker.py"


def main(payload: dict) -> None:
    session_id = payload.get("session_id", "unknown")
    transcript_path = payload.get("transcript_path", "")
    cwd = payload.get("cwd", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key or not transcript_path:
        sys.exit(0)

    # Check if extraction already happened
    if _lock_path(session_id).exists():
        print(f"[session_end] Extraction already ran for {session_id[:12]}...", file=sys.stderr)
        sys.exit(0)

    # Fire-and-forget: spawn detached background worker
    subprocess.Popen(
        [sys.executable, str(_WORKER), session_id, transcript_path, cwd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(Path.home() / ".claude" / "memory" / "extract.log", "a"),
        start_new_session=True,  # detach from parent process group
        env={**os.environ, "ANTHROPIC_API_KEY": api_key},
    )

    print(f"[session_end] Background extraction started for {session_id[:12]}...", file=sys.stderr)


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        payload = {}
    try:
        main(payload)
    except Exception:
        # Never block session exit
        pass
