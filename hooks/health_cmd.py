#!/usr/bin/env python3
"""
health_cmd.py — System health check for the memory system.

Checks: Ollama status, DB lock state, snapshot age, extraction backlog,
review queue, embedding coverage, and disk usage.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Prefer local project copy during development, fall back to installed
_project_root = Path(__file__).resolve().parent.parent
if (_project_root / "memory" / "__init__.py").exists():
    sys.path.insert(0, str(_project_root))
else:
    sys.path.insert(0, str(Path.home() / ".claude"))


def _check(name: str, ok: bool, detail: str = "") -> str:
    status = "OK" if ok else "FAIL"
    line = f"  {'[' + status + ']':>6}  {name}"
    if detail:
        line += f"  —  {detail}"
    return line


def main() -> None:
    from memory.config import (
        DB_PATH, MEMORY_DIR, OLLAMA_URL, OLLAMA_MODEL,
        SNAPSHOT_DIR, MAX_SNAPSHOTS,
    )

    print("## Memory System Health\n")

    checks = []

    # ── 1. Database file ──────────────────────────────────────────────
    db_exists = DB_PATH.exists()
    db_size = DB_PATH.stat().st_size / 1024 if db_exists else 0
    checks.append(_check("Database", db_exists, f"{db_size:.0f} KB at {DB_PATH}"))

    # ── 2. Database lock ──────────────────────────────────────────────
    if db_exists:
        try:
            import duckdb
            conn = duckdb.connect(str(DB_PATH), read_only=True)
            conn.close()
            checks.append(_check("DB Lock", True, "No lock contention"))
        except Exception as e:
            err = str(e).lower()
            if "lock" in err or "busy" in err:
                checks.append(_check("DB Lock", False, f"Database locked: {e}"))
            else:
                checks.append(_check("DB Lock", False, str(e)))
    else:
        checks.append(_check("DB Lock", True, "No database yet"))

    # ── 3. Ollama ─────────────────────────────────────────────────────
    try:
        from memory.embeddings import is_ollama_available
        ollama_ok = is_ollama_available()
        checks.append(_check("Ollama", ollama_ok,
                              f"{OLLAMA_MODEL} at {OLLAMA_URL}" if ollama_ok else f"Not running at {OLLAMA_URL}"))
    except Exception:
        checks.append(_check("Ollama", False, "Import error"))

    # ── 4. ONNX embeddings ───────────────────────────────────────────
    try:
        from memory.embeddings import _init_onnx
        onnx_ok = _init_onnx()
        checks.append(_check("ONNX Embeddings", onnx_ok,
                              "nomic-embed-text-v1.5 loaded" if onnx_ok else "Not available (Ollama fallback)"))
    except Exception:
        checks.append(_check("ONNX Embeddings", False, "Import error"))

    # ── 5. Anthropic API key ─────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    has_key = bool(api_key and len(api_key) > 10)
    checks.append(_check("API Key", has_key,
                          f"Set ({api_key[:8]}...)" if has_key else "ANTHROPIC_API_KEY not set — extraction disabled"))

    # ── 6. Snapshots ─────────────────────────────────────────────────
    snap_dir = Path(SNAPSHOT_DIR) if SNAPSHOT_DIR else MEMORY_DIR / "snapshots"
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob("knowledge_*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
        if snaps:
            latest_snap = snaps[0]
            snap_age = datetime.now() - datetime.fromtimestamp(latest_snap.stat().st_mtime)
            age_str = f"{snap_age.days}d" if snap_age.days > 0 else f"{snap_age.seconds // 3600}h"
            checks.append(_check("Snapshots", True, f"{len(snaps)} snapshots, latest: {age_str} ago"))
        else:
            checks.append(_check("Snapshots", False, "No snapshots — run a session to create one"))
    else:
        checks.append(_check("Snapshots", False, f"Snapshot directory not found: {snap_dir}"))

    # ── 7. Review queue ──────────────────────────────────────────────
    if db_exists:
        try:
            from memory import db
            conn = db.get_connection(read_only=True)
            try:
                pending = conn.execute(
                    "SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"
                ).fetchone()[0]
                checks.append(_check("Review Queue", pending == 0,
                                      f"{pending} pending items" if pending > 0 else "Empty"))
            finally:
                conn.close()
        except Exception:
            checks.append(_check("Review Queue", True, "Table not created yet"))

    # ── 8. Extraction state ──────────────────────────────────────────
    state_dir = MEMORY_DIR / "extraction_state"
    if state_dir.exists():
        active_states = list(state_dir.glob("*.json"))
        locks = list((MEMORY_DIR / "locks").glob("*.lock")) if (MEMORY_DIR / "locks").exists() else []
        if locks:
            checks.append(_check("Extraction", False, f"{len(locks)} active lock(s) — extraction may be running or stale"))
        else:
            checks.append(_check("Extraction", True, f"{len(active_states)} state file(s)"))
    else:
        checks.append(_check("Extraction", True, "No extraction state"))

    # ── 9. Embedding coverage ────────────────────────────────────────
    if db_exists:
        try:
            from memory import db
            conn = db.get_connection(read_only=True)
            try:
                total = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active = TRUE").fetchone()[0]
                with_emb = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE is_active = TRUE AND embedding IS NOT NULL"
                ).fetchone()[0]
                pct = (with_emb * 100 // total) if total > 0 else 100
                checks.append(_check("Embeddings", pct >= 90,
                                      f"{with_emb}/{total} facts have embeddings ({pct}%)"))
            finally:
                conn.close()
        except Exception:
            pass

    # ── 10. Disk usage ───────────────────────────────────────────────
    try:
        total_size = sum(f.stat().st_size for f in MEMORY_DIR.rglob("*") if f.is_file())
        checks.append(_check("Disk Usage", total_size < 500 * 1024 * 1024,
                              f"{total_size / 1024 / 1024:.1f} MB in {MEMORY_DIR}"))
    except Exception:
        pass

    # ── Output ───────────────────────────────────────────────────────
    for line in checks:
        print(line)

    ok_count = sum(1 for c in checks if "[  OK]" in c)
    fail_count = sum(1 for c in checks if "[FAIL]" in c)
    print(f"\n---\n{ok_count} passed, {fail_count} failed out of {len(checks)} checks")

    if fail_count > 0:
        print(f"[memory] Health: {fail_count} issue(s) detected", file=sys.stderr)
    else:
        print("[memory] Health: all systems operational", file=sys.stderr)


if __name__ == "__main__":
    main()
