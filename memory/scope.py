"""
scope.py — Project scope resolution.

Determines the project scope from cwd by finding the git repo root.
Falls back to the cwd itself if not inside a git repo.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import GLOBAL_SCOPE


def resolve_scope(cwd: str) -> str:
    """
    Return a canonical project scope string for the given working directory.
    Uses `git rev-parse --show-toplevel` to find the repo root.
    Falls back to the cwd path if not in a git repo.
    """
    if not cwd:
        return GLOBAL_SCOPE

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Not a git repo — use the cwd as scope
    return str(Path(cwd).resolve())


def scope_display_name(scope: str) -> str:
    """Short display name for a scope (last path component or 'global')."""
    if scope == GLOBAL_SCOPE:
        return "global"
    return Path(scope).name
