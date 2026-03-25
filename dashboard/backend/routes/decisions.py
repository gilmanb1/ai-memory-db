from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["decisions"])


class DecisionCreate(BaseModel):
    text: str
    temporal_class: str = "long"
    importance: int = 7
    scope: str = "__global__"


class DecisionUpdate(BaseModel):
    text: Optional[str] = None
    temporal_class: Optional[str] = None
    importance: Optional[int] = None


@router.get("/decisions")
def list_decisions(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, text, temporal_class, decay_score, session_count,
                   importance, scope, times_recalled, times_applied,
                   created_at, last_seen_at
            FROM decisions WHERE is_active = TRUE{scope_sql}
            ORDER BY decay_score DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "text", "temporal_class", "decay_score", "session_count",
                "importance", "scope", "times_recalled", "times_applied",
                "created_at", "last_seen_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM decisions WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.post("/decisions")
def create_decision(body: DecisionCreate):
    from memory import db
    from memory.embeddings import embed
    from memory.decay import compute_decay_score

    embedding = embed(body.text)
    with write_lock:
        conn = get_write_conn()
        try:
            item_id, is_new = db.upsert_decision(
                conn, body.text,
                temporal_class=body.temporal_class,
                embedding=embedding,
                session_id="dashboard",
                decay_fn=compute_decay_score,
                scope=body.scope,
                importance=body.importance,
            )
            return {"id": item_id, "is_new": is_new}
        finally:
            conn.close()


@router.put("/decisions/{decision_id}")
def update_decision(decision_id: str, body: DecisionUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM decisions WHERE id = ?", [decision_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Decision not found")

            updates, params = [], []
            if body.text is not None:
                updates.append("text = ?")
                params.append(body.text)
                new_emb = embed(body.text)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.temporal_class is not None:
                updates.append("temporal_class = ?")
                params.append(body.temporal_class)
            if body.importance is not None:
                updates.append("importance = ?")
                params.append(body.importance)

            if not updates:
                return {"id": decision_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            conn.execute(
                f"UPDATE decisions SET {', '.join(updates)} WHERE id = ?",
                params + [decision_id],
            )
            return {"id": decision_id, "updated": True}
        finally:
            conn.close()


@router.delete("/decisions/{decision_id}")
def delete_decision(decision_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, decision_id, "decisions")
            if not success:
                raise HTTPException(404, "Decision not found")
            return {"id": decision_id, "deleted": True}
        finally:
            conn.close()
