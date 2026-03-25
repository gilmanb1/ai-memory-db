from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["error_solutions"])


class ErrorSolutionCreate(BaseModel):
    error_pattern: str
    solution: str
    error_context: str = ""
    file_paths: Optional[list[str]] = None
    scope: str = "__global__"


class ErrorSolutionUpdate(BaseModel):
    error_pattern: Optional[str] = None
    solution: Optional[str] = None
    error_context: Optional[str] = None
    file_paths: Optional[list[str]] = None


@router.get("/error_solutions")
def list_error_solutions(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, error_pattern, error_context, solution, file_paths,
                   scope, confidence, times_applied,
                   times_recalled, created_at, last_applied_at
            FROM error_solutions WHERE is_active = TRUE{scope_sql}
            ORDER BY times_applied DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "error_pattern", "error_context", "solution", "file_paths",
                "scope", "confidence", "times_applied",
                "times_recalled", "created_at", "last_applied_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM error_solutions WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.post("/error_solutions")
def create_error_solution(body: ErrorSolutionCreate):
    from memory import db
    from memory.embeddings import embed

    embedding = embed(body.error_pattern)
    with write_lock:
        conn = get_write_conn()
        try:
            eid, is_new = db.upsert_error_solution(
                conn,
                error_pattern=body.error_pattern,
                solution=body.solution,
                error_context=body.error_context,
                file_paths=body.file_paths,
                embedding=embedding,
                session_id="dashboard",
                scope=body.scope,
            )
            return {"id": eid, "is_new": is_new}
        finally:
            conn.close()


@router.put("/error_solutions/{es_id}")
def update_error_solution(es_id: str, body: ErrorSolutionUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM error_solutions WHERE id = ?", [es_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Error solution not found")

            updates, params = [], []
            if body.error_pattern is not None:
                updates.append("error_pattern = ?")
                params.append(body.error_pattern)
                new_emb = embed(body.error_pattern)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.solution is not None:
                updates.append("solution = ?")
                params.append(body.solution)
            if body.error_context is not None:
                updates.append("error_context = ?")
                params.append(body.error_context)
            if body.file_paths is not None:
                updates.append("file_paths = ?")
                params.append(body.file_paths)

            if not updates:
                return {"id": es_id, "updated": False}

            conn.execute(
                f"UPDATE error_solutions SET {', '.join(updates)} WHERE id = ?",
                params + [es_id],
            )
            return {"id": es_id, "updated": True}
        finally:
            conn.close()


@router.delete("/error_solutions/{es_id}")
def delete_error_solution(es_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, es_id, "error_solutions")
            if not success:
                raise HTTPException(404, "Error solution not found")
            return {"id": es_id, "deleted": True}
        finally:
            conn.close()
