from typing import Optional
from fastapi import APIRouter, HTTPException

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["observations"])


@router.get("/observations")
def list_observations(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, text, proof_count, source_fact_ids, temporal_class,
                   decay_score, session_count, scope, importance,
                   superseded_by, created_at, last_seen_at, updated_at
            FROM observations WHERE is_active = TRUE{scope_sql}
            ORDER BY proof_count DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "text", "proof_count", "source_fact_ids", "temporal_class",
                "decay_score", "session_count", "scope", "importance",
                "superseded_by", "created_at", "last_seen_at", "updated_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM observations WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.delete("/observations/{obs_id}")
def delete_observation(obs_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, obs_id, "observations")
            if not success:
                raise HTTPException(404, "Observation not found")
            return {"id": obs_id, "deleted": True}
        finally:
            conn.close()
