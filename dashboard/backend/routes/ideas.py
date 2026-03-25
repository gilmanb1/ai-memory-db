from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["ideas"])


class IdeaCreate(BaseModel):
    text: str
    idea_type: str = "insight"
    temporal_class: str = "medium"
    scope: str = "__global__"


class IdeaUpdate(BaseModel):
    text: Optional[str] = None
    idea_type: Optional[str] = None
    temporal_class: Optional[str] = None


@router.get("/ideas")
def list_ideas(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, text, idea_type, temporal_class, decay_score,
                   session_count, scope, created_at, last_seen_at
            FROM ideas WHERE is_active = TRUE{scope_sql}
            ORDER BY decay_score DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "text", "idea_type", "temporal_class", "decay_score",
                "session_count", "scope", "created_at", "last_seen_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM ideas WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.post("/ideas")
def create_idea(body: IdeaCreate):
    from memory import db
    from memory.embeddings import embed
    from memory.decay import compute_decay_score

    embedding = embed(body.text)
    with write_lock:
        conn = get_write_conn()
        try:
            item_id, is_new = db.upsert_idea(
                conn, body.text,
                idea_type=body.idea_type,
                temporal_class=body.temporal_class,
                embedding=embedding,
                session_id="dashboard",
                decay_fn=compute_decay_score,
                scope=body.scope,
            )
            return {"id": item_id, "is_new": is_new}
        finally:
            conn.close()


@router.put("/ideas/{idea_id}")
def update_idea(idea_id: str, body: IdeaUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM ideas WHERE id = ?", [idea_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Idea not found")

            updates, params = [], []
            if body.text is not None:
                updates.append("text = ?")
                params.append(body.text)
                new_emb = embed(body.text)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.idea_type is not None:
                updates.append("idea_type = ?")
                params.append(body.idea_type)
            if body.temporal_class is not None:
                updates.append("temporal_class = ?")
                params.append(body.temporal_class)

            if not updates:
                return {"id": idea_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            conn.execute(
                f"UPDATE ideas SET {', '.join(updates)} WHERE id = ?",
                params + [idea_id],
            )
            return {"id": idea_id, "updated": True}
        finally:
            conn.close()


@router.delete("/ideas/{idea_id}")
def delete_idea(idea_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, idea_id, "ideas")
            if not success:
                raise HTTPException(404, "Idea not found")
            return {"id": idea_id, "deleted": True}
        finally:
            conn.close()
