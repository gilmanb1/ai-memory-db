from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["procedures"])


class ProcedureCreate(BaseModel):
    task_description: str
    steps: str
    file_paths: Optional[list[str]] = None
    importance: int = 7
    scope: str = "__global__"


class ProcedureUpdate(BaseModel):
    task_description: Optional[str] = None
    steps: Optional[str] = None
    file_paths: Optional[list[str]] = None
    importance: Optional[int] = None


@router.get("/procedures")
def list_procedures(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, task_description, steps, file_paths, importance,
                   scope, temporal_class, decay_score, session_count,
                   times_recalled, times_applied,
                   created_at, last_seen_at
            FROM procedures WHERE is_active = TRUE{scope_sql}
            ORDER BY importance DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "task_description", "steps", "file_paths", "importance",
                "scope", "temporal_class", "decay_score", "session_count",
                "times_recalled", "times_applied",
                "created_at", "last_seen_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM procedures WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.post("/procedures")
def create_procedure(body: ProcedureCreate):
    from memory import db
    from memory.embeddings import embed
    from memory.decay import compute_decay_score

    embedding = embed(body.task_description)
    with write_lock:
        conn = get_write_conn()
        try:
            pid, is_new = db.upsert_procedure(
                conn,
                task_description=body.task_description,
                steps=body.steps,
                file_paths=body.file_paths,
                embedding=embedding,
                session_id="dashboard",
                decay_fn=compute_decay_score,
                scope=body.scope,
                importance=body.importance,
            )
            return {"id": pid, "is_new": is_new}
        finally:
            conn.close()


@router.put("/procedures/{procedure_id}")
def update_procedure(procedure_id: str, body: ProcedureUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM procedures WHERE id = ?", [procedure_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Procedure not found")

            updates, params = [], []
            if body.task_description is not None:
                updates.append("task_description = ?")
                params.append(body.task_description)
                new_emb = embed(body.task_description)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.steps is not None:
                updates.append("steps = ?")
                params.append(body.steps)
            if body.file_paths is not None:
                updates.append("file_paths = ?")
                params.append(body.file_paths)
            if body.importance is not None:
                updates.append("importance = ?")
                params.append(body.importance)

            if not updates:
                return {"id": procedure_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            conn.execute(
                f"UPDATE procedures SET {', '.join(updates)} WHERE id = ?",
                params + [procedure_id],
            )
            return {"id": procedure_id, "updated": True}
        finally:
            conn.close()


@router.delete("/procedures/{procedure_id}")
def delete_procedure(procedure_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, procedure_id, "procedures")
            if not success:
                raise HTTPException(404, "Procedure not found")
            return {"id": procedure_id, "deleted": True}
        finally:
            conn.close()
