from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["guardrails"])


class GuardrailCreate(BaseModel):
    warning: str
    rationale: str = ""
    consequence: str = ""
    file_paths: Optional[list[str]] = None
    importance: int = 9
    scope: str = "__global__"


class GuardrailUpdate(BaseModel):
    warning: Optional[str] = None
    rationale: Optional[str] = None
    consequence: Optional[str] = None
    file_paths: Optional[list[str]] = None
    importance: Optional[int] = None


@router.get("/guardrails")
def list_guardrails(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, warning, rationale, consequence, file_paths,
                   importance, scope, session_count,
                   times_recalled, times_applied,
                   created_at, last_seen_at
            FROM guardrails WHERE is_active = TRUE{scope_sql}
            ORDER BY importance DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "warning", "rationale", "consequence", "file_paths",
                "importance", "scope", "session_count",
                "times_recalled", "times_applied",
                "created_at", "last_seen_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM guardrails WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.post("/guardrails")
def create_guardrail(body: GuardrailCreate):
    from memory import db
    from memory.embeddings import embed

    embedding = embed(body.warning)
    with write_lock:
        conn = get_write_conn()
        try:
            gid, is_new = db.upsert_guardrail(
                conn,
                warning=body.warning,
                rationale=body.rationale,
                consequence=body.consequence,
                file_paths=body.file_paths,
                embedding=embedding,
                session_id="dashboard",
                scope=body.scope,
                importance=body.importance,
            )
            return {"id": gid, "is_new": is_new}
        finally:
            conn.close()


@router.put("/guardrails/{guardrail_id}")
def update_guardrail(guardrail_id: str, body: GuardrailUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM guardrails WHERE id = ?", [guardrail_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Guardrail not found")

            updates, params = [], []
            if body.warning is not None:
                updates.append("warning = ?")
                params.append(body.warning)
                new_emb = embed(body.warning)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.rationale is not None:
                updates.append("rationale = ?")
                params.append(body.rationale)
            if body.consequence is not None:
                updates.append("consequence = ?")
                params.append(body.consequence)
            if body.file_paths is not None:
                updates.append("file_paths = ?")
                params.append(body.file_paths)
            if body.importance is not None:
                updates.append("importance = ?")
                params.append(body.importance)

            if not updates:
                return {"id": guardrail_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            conn.execute(
                f"UPDATE guardrails SET {', '.join(updates)} WHERE id = ?",
                params + [guardrail_id],
            )
            return {"id": guardrail_id, "updated": True}
        finally:
            conn.close()


@router.delete("/guardrails/{guardrail_id}")
def delete_guardrail(guardrail_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, guardrail_id, "guardrails")
            if not success:
                raise HTTPException(404, "Guardrail not found")
            return {"id": guardrail_id, "deleted": True}
        finally:
            conn.close()
