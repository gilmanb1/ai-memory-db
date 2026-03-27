"""
backup.py — Snapshot, export/import, and restore for the memory database.

Snapshots: full DuckDB file copies with rotation.
Export: JSON dump of all active items.
Import: JSON load with dedup against existing items.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def create_snapshot(
    db_path: str,
    snapshot_dir: str,
    reason: str = "session_end",
    max_snapshots: int = 5,
) -> Path:
    """Copy the DB file to snapshot_dir with a timestamped name. Rotates old snapshots."""
    snap_dir = Path(snapshot_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    snap_name = f"knowledge_{ts}_{reason}.duckdb"
    snap_path = snap_dir / snap_name

    shutil.copy2(db_path, str(snap_path))

    # Rotate: keep only the newest max_snapshots
    existing = sorted(snap_dir.glob("knowledge_*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in existing[max_snapshots:]:
        try:
            old.unlink()
        except Exception:
            pass

    return snap_path


def list_snapshots(snapshot_dir: str) -> list[dict]:
    """List available snapshots sorted newest first."""
    snap_dir = Path(snapshot_dir)
    if not snap_dir.exists():
        return []

    snaps = []
    for p in sorted(snap_dir.glob("knowledge_*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True):
        snaps.append({
            "path": str(p),
            "name": p.name,
            "size_kb": round(p.stat().st_size / 1024, 1),
            "created": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return snaps


def restore_snapshot(
    snapshot_path: str,
    db_path: str,
    snapshot_dir: str,
) -> bool:
    """Replace the DB with a snapshot. Creates a pre-restore snapshot first."""
    snap = Path(snapshot_path)
    target = Path(db_path)

    if not snap.exists():
        return False

    # Create pre-restore safety snapshot
    if target.exists():
        create_snapshot(db_path, snapshot_dir, "pre_restore", max_snapshots=10)

    # Replace DB
    shutil.copy2(str(snap), str(target))

    # Also remove WAL file if present (stale WAL from old DB state)
    wal = Path(str(target) + ".wal")
    if wal.exists():
        wal.unlink()

    return True


def export_memory(
    db_path: str,
    output_path: str,
    scope: Optional[str] = None,
) -> dict:
    """Export all active items to a JSON file. Returns item counts."""
    from . import db as memdb

    conn = memdb.get_connection(read_only=True, db_path=db_path)
    try:
        data: dict = {}
        counts: dict = {}

        scope_sql, scope_params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            scope_params = [scope]

        tables = {
            "facts": "id, text, category, temporal_class, confidence, importance, scope, created_at",
            "decisions": "id, text, temporal_class, importance, scope, created_at",
            "ideas": "id, text, idea_type, temporal_class, scope, created_at",
            "entities": "id, name, entity_type, scope",
            "relationships": "id, from_entity, to_entity, rel_type, description, strength, scope",
            "guardrails": "id, warning, rationale, consequence, file_paths, importance, scope",
            "procedures": "id, task_description, steps, file_paths, importance, scope",
            "error_solutions": "id, error_pattern, solution, error_context, file_paths, confidence, scope",
            "observations": "id, text, proof_count, source_fact_ids, temporal_class, scope",
        }

        for table, cols in tables.items():
            try:
                active_filter = "is_active = TRUE" if table != "entities" else "1=1"
                # Check if is_active exists
                table_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                if "is_active" not in table_cols:
                    active_filter = "1=1"

                rows = conn.execute(f"""
                    SELECT {cols} FROM {table}
                    WHERE {active_filter} {scope_sql}
                """, scope_params).fetchall()

                col_names = [c.strip().split(" AS ")[-1].split(".")[-1] for c in cols.split(",")]
                items = []
                for row in rows:
                    d = dict(zip(col_names, row))
                    # Convert non-serializable types
                    for k, v in d.items():
                        if isinstance(v, datetime):
                            d[k] = v.isoformat()
                    items.append(d)

                data[table] = items
                counts[table] = len(items)
            except Exception:
                data[table] = []
                counts[table] = 0

        export = {
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "scope": scope or "__all__",
            "data": data,
        }

        Path(output_path).write_text(json.dumps(export, indent=2, default=str))
        return counts
    finally:
        conn.close()


def import_memory(
    db_path: str,
    input_path: str,
    merge: bool = True,
) -> dict:
    """Import items from a JSON export file. Returns item counts."""
    from . import db as memdb
    from .decay import compute_decay_score

    export = json.loads(Path(input_path).read_text())
    if export.get("version") != 1:
        raise ValueError(f"Unsupported export version: {export.get('version')}")

    data = export.get("data", {})
    counts: dict = {}

    conn = memdb.get_connection(db_path=db_path)
    try:
        # Import facts
        for fact in data.get("facts", []):
            try:
                memdb.upsert_fact(
                    conn, fact["text"],
                    fact.get("category", "operational"),
                    fact.get("temporal_class", "long"),
                    fact.get("confidence", "high"),
                    None,  # no embedding during import
                    "import", compute_decay_score,
                    scope=fact.get("scope", "__global__"),
                    importance=fact.get("importance", 5),
                )
                counts["facts"] = counts.get("facts", 0) + 1
            except Exception:
                pass

        # Import decisions
        for dec in data.get("decisions", []):
            try:
                memdb.upsert_decision(
                    conn, dec["text"],
                    dec.get("temporal_class", "long"),
                    None, "import", compute_decay_score,
                    scope=dec.get("scope", "__global__"),
                )
                counts["decisions"] = counts.get("decisions", 0) + 1
            except Exception:
                pass

        # Import entities
        for ent in data.get("entities", []):
            try:
                memdb.upsert_entity(
                    conn, ent["name"],
                    entity_type=ent.get("entity_type", "general"),
                    scope=ent.get("scope", "__global__"),
                )
                counts["entities"] = counts.get("entities", 0) + 1
            except Exception:
                pass

        # Import guardrails
        for g in data.get("guardrails", []):
            try:
                memdb.upsert_guardrail(
                    conn, warning=g["warning"],
                    rationale=g.get("rationale", ""),
                    consequence=g.get("consequence", ""),
                    file_paths=g.get("file_paths"),
                    session_id="import",
                    scope=g.get("scope", "__global__"),
                )
                counts["guardrails"] = counts.get("guardrails", 0) + 1
            except Exception:
                pass

        return counts
    finally:
        conn.close()
