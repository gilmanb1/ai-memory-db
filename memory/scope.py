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


def resolve_scopes(primary_cwd: str) -> list[str]:
    """
    Resolve all active scopes for a session.

    Includes the primary scope (from cwd) plus any additional scopes
    specified via MEMORY_ADDITIONAL_SCOPES environment variable.

    The env var is a comma or colon-separated list of paths.
    Each path is resolved to its git root (same as resolve_scope).
    """
    import os

    scopes = []
    primary = resolve_scope(primary_cwd)
    if primary and primary != GLOBAL_SCOPE:
        scopes.append(primary)

    additional = os.environ.get("MEMORY_ADDITIONAL_SCOPES", "").strip()
    if additional:
        for path in additional.replace(":", ",").split(","):
            path = path.strip()
            if path:
                s = resolve_scope(path)
                if s and s != GLOBAL_SCOPE and s not in scopes:
                    scopes.append(s)

    return scopes if scopes else [primary]


def scope_display_name(scope: str) -> str:
    """Short display name for a scope (last path component or 'global')."""
    if scope == GLOBAL_SCOPE:
        return "global"
    return Path(scope).name
