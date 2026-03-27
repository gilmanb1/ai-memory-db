"""
guardrail_check.py — Detect and enforce guardrail violations on file edits.

Used by the PostToolUse hook to warn when Claude edits a guardrailed file.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def check_guardrail_violation(
    conn,
    file_path: str,
    scope: Optional[str] = None,
) -> Optional[dict]:
    """
    Check if editing this file violates any guardrails.

    Returns {"file": str, "guardrails": list, "warning_text": str} or None.
    """
    from . import db

    # Normalize: try both the exact path and just the filename/relative parts
    file_name = Path(file_path).name
    file_parts = file_path.replace("\\", "/")

    # Query guardrails that reference this file
    guardrails = []
    try:
        # Method 1: via fact_file_links join
        rows = db.get_guardrails_for_files(conn, [file_path], limit=10, scope=scope)
        guardrails.extend(rows)
    except Exception:
        pass

    # Method 2: check inline file_paths arrays for partial matches
    try:
        all_guardrails = conn.execute("""
            SELECT id, warning, rationale, consequence, file_paths
            FROM guardrails
            WHERE is_active = TRUE AND file_paths IS NOT NULL AND len(file_paths) > 0
        """).fetchall()
        seen_ids = {g["id"] for g in guardrails}
        for gid, warning, rationale, consequence, fps in all_guardrails:
            if gid in seen_ids:
                continue
            if fps and any(
                fp in file_parts or file_parts.endswith(fp) or file_name == Path(fp).name
                for fp in fps
            ):
                guardrails.append({
                    "id": gid,
                    "warning": warning,
                    "rationale": rationale or "",
                    "consequence": consequence or "",
                    "file_paths": fps,
                })
    except Exception:
        pass

    if not guardrails:
        return None

    # Build warning text
    warnings = []
    for g in guardrails:
        w = f"- {g['warning']}"
        if g.get("rationale"):
            w += f" (reason: {g['rationale']})"
        if g.get("consequence"):
            w += f" [breaks: {g['consequence']}]"
        warnings.append(w)

    return {
        "file": file_path,
        "guardrails": guardrails,
        "warning_text": "\n".join(warnings),
    }


def enforce_guardrail(
    conn,
    file_path: str,
    cwd: str,
    auto_stash: bool = True,
    scope: Optional[str] = None,
) -> Optional[dict]:
    """
    Check guardrails and optionally git-stash the change.

    Returns the violation dict (with stash info if applicable) or None.
    """
    violation = check_guardrail_violation(conn, file_path, scope)
    if not violation:
        return None

    violation["stashed"] = False

    if auto_stash:
        try:
            # Check if we're in a git repo
            check = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            )
            if check.returncode == 0:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                stash_msg = f"guardrail-check-{ts}-{Path(file_path).name}"
                result = subprocess.run(
                    ["git", "stash", "push", "-m", stash_msg, "--", file_path],
                    cwd=cwd, capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    violation["stashed"] = True
                    violation["stash_msg"] = stash_msg
        except Exception:
            pass  # Can't stash — warn only

    return violation
