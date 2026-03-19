"""
extraction_state.py — Per-session extraction state management.

Replaces the one-shot lock system for incremental multi-pass extraction.
Each session gets a JSON state file that tracks:
  - How many extraction passes have run
  - Where in the transcript the last pass stopped (byte offset)
  - The cumulative narrative from the last pass
  - IDs of items extracted so far (for cross-pass dedup)
  - IDs of recalled items (for cross-session superseding)

Lock semantics:
  - .running lock: prevents concurrent passes within a session
  - .lock file: signals all extraction is complete (backward compat)
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


LOCK_DIR = Path.home() / ".claude" / "memory" / "locks"
STATE_DIR = Path.home() / ".claude" / "memory" / "extraction_state"


def _sanitize_id(session_id: str) -> str:
    """Sanitize session ID for use as filename."""
    return session_id.replace("/", "_").replace("..", "_").replace("\x00", "_")


def _state_path(session_id: str) -> Path:
    return STATE_DIR / f"{_sanitize_id(session_id)}.json"


def _running_lock_path(session_id: str) -> Path:
    return LOCK_DIR / f"{_sanitize_id(session_id)}.running"


def _complete_lock_path(session_id: str) -> Path:
    return LOCK_DIR / f"{_sanitize_id(session_id)}.lock"


def _recall_cache_path(session_id: str) -> Path:
    return STATE_DIR / f"{_sanitize_id(session_id)}_recall.json"


# ── State file operations ────────────────────────────────────────────────

def load_state(session_id: str) -> Optional[dict]:
    """Load extraction state for a session. Returns None if missing or corrupt."""
    path = _state_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def save_state(session_id: str, state: dict) -> None:
    """Atomically save extraction state (write to .tmp, rename)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path(session_id)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp.rename(path)


def delete_state(session_id: str) -> None:
    """Remove state file for a session."""
    try:
        _state_path(session_id).unlink()
    except FileNotFoundError:
        pass


# ── Recall cache (session-start → first extraction pass) ────────────────

def save_recall_cache(session_id: str, items: list[dict]) -> None:
    """Cache session-start recall items for the first extraction pass."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _recall_cache_path(session_id)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"items": items}, fh)
    tmp.rename(path)


def load_recall_cache(session_id: str) -> Optional[list[dict]]:
    """Load cached recall items. Returns None if missing."""
    try:
        with open(_recall_cache_path(session_id), "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data.get("items")
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def delete_recall_cache(session_id: str) -> None:
    """Remove recall cache file."""
    try:
        _recall_cache_path(session_id).unlink()
    except FileNotFoundError:
        pass


# ── Running lock (prevents concurrent passes) ───────────────────────────

def acquire_running_lock(session_id: str) -> bool:
    """
    Atomically acquire the running lock for this session.
    Returns True if acquired, False if another pass is running.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = _running_lock_path(session_id)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, datetime.now(timezone.utc).isoformat().encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_running_lock(session_id: str) -> None:
    """Release the running lock."""
    try:
        _running_lock_path(session_id).unlink()
    except FileNotFoundError:
        pass


# ── Completion lock (backward compat with existing hooks) ───────────────

def mark_extraction_complete(session_id: str) -> None:
    """
    Create the .lock file that signals all extraction is done.
    This is the same file the existing hooks check.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = _complete_lock_path(session_id)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, datetime.now(timezone.utc).isoformat().encode())
        os.close(fd)
    except FileExistsError:
        pass  # already marked complete


def is_extraction_complete(session_id: str) -> bool:
    """Check if extraction has already completed for this session."""
    return _complete_lock_path(session_id).exists()


# ── Cleanup ─────────────────────────────────────────────────────────────

def cleanup_old_state(max_age_hours: int = 48) -> None:
    """Remove stale state files and running locks older than max_age_hours."""
    now = time.time()
    for directory in (STATE_DIR, LOCK_DIR):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.suffix in (".json", ".running", ".tmp"):
                try:
                    age_hours = (now - path.stat().st_mtime) / 3600
                    if age_hours > max_age_hours:
                        path.unlink()
                except OSError:
                    pass
