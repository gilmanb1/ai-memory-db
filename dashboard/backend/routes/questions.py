from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["questions"])


class QuestionCreate(BaseModel):
    text: str
    scope: str = "__global__"


class QuestionUpdate(BaseModel):
    text: Optional[str] = None
    resolved: Optional[bool] = None


@router.get("/questions")
def list_questions(scope: Optional[str] = None, resolved: Optional[bool] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        filters = ["is_active = TRUE"]
        params: list = []
        if scope:
            filters.append("(scope = ? OR scope = '__global__')")
            params.append(scope)
        if resolved is not None:
            filters.append("resolved = ?")
            params.append(resolved)

        where = " AND ".join(filters)
        sql = f"""
            SELECT id, text, resolved, scope, created_at, last_seen_at
            FROM open_questions WHERE {where}
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "text", "resolved", "scope", "created_at", "last_seen_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM open_questions WHERE {where}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.post("/questions")
def create_question(body: QuestionCreate):
    from memory import db
    from memory.embeddings import embed

    embedding = embed(body.text)
    with write_lock:
        conn = get_write_conn()
        try:
            qid, is_new = db.upsert_question(
                conn, body.text,
                embedding=embedding,
                session_id="dashboard",
                scope=body.scope,
            )
            return {"id": qid, "is_new": is_new}
        finally:
            conn.close()


@router.put("/questions/{question_id}")
def update_question(question_id: str, body: QuestionUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM open_questions WHERE id = ?", [question_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Question not found")

            updates, params = [], []
            if body.text is not None:
                updates.append("text = ?")
                params.append(body.text)
                new_emb = embed(body.text)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.resolved is not None:
                updates.append("resolved = ?")
                params.append(body.resolved)

            if not updates:
                return {"id": question_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            conn.execute(
                f"UPDATE open_questions SET {', '.join(updates)} WHERE id = ?",
                params + [question_id],
            )
            return {"id": question_id, "updated": True}
        finally:
            conn.close()


@router.delete("/questions/{question_id}")
def delete_question(question_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, question_id, "open_questions")
            if not success:
                raise HTTPException(404, "Question not found")
            return {"id": question_id, "deleted": True}
        finally:
            conn.close()
