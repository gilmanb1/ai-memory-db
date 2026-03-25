from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["facts"])


class FactCreate(BaseModel):
    text: str
    category: str = "contextual"
    temporal_class: str = "long"
    confidence: str = "high"
    importance: int = 5
    scope: str = "__global__"
    file_paths: Optional[list[str]] = None


class FactUpdate(BaseModel):
    text: Optional[str] = None
    category: Optional[str] = None
    temporal_class: Optional[str] = None
    importance: Optional[int] = None


@router.get("/facts")
def list_facts(
    temporal_class: Optional[str] = None,
    scope: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
):
    conn = get_read_conn()
    try:
        scope_sql, scope_params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            scope_params = [scope]

        tc_sql = ""
        tc_params = []
        if temporal_class:
            tc_sql = " AND temporal_class = ?"
            tc_params = [temporal_class]

        sql = f"""
            SELECT id, text, category, temporal_class, confidence,
                   decay_score, session_count, importance, scope,
                   times_recalled, times_applied, recall_utility,
                   failure_probability, created_at, last_seen_at
            FROM facts
            WHERE is_active = TRUE{scope_sql}{tc_sql}
            ORDER BY decay_score DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, scope_params + tc_params + [limit, offset]).fetchall()
        cols = ["id", "text", "category", "temporal_class", "confidence",
                "decay_score", "session_count", "importance", "scope",
                "times_recalled", "times_applied", "recall_utility",
                "failure_probability", "created_at", "last_seen_at"]

        count_sql = f"SELECT COUNT(*) FROM facts WHERE is_active = TRUE{scope_sql}{tc_sql}"
        total = conn.execute(count_sql, scope_params + tc_params).fetchone()[0]

        return {
            "items": [dict(zip(cols, row)) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        conn.close()


@router.get("/facts/{fact_id}")
def get_fact(fact_id: str):
    conn = get_read_conn()
    try:
        row = conn.execute("""
            SELECT id, text, category, temporal_class, confidence,
                   decay_score, session_count, importance, scope,
                   times_recalled, times_applied, recall_utility,
                   failure_probability, superseded_by,
                   created_at, last_seen_at, is_active
            FROM facts WHERE id = ?
        """, [fact_id]).fetchone()
        if not row:
            raise HTTPException(404, "Fact not found")
        cols = ["id", "text", "category", "temporal_class", "confidence",
                "decay_score", "session_count", "importance", "scope",
                "times_recalled", "times_applied", "recall_utility",
                "failure_probability", "superseded_by",
                "created_at", "last_seen_at", "is_active"]
        return dict(zip(cols, row))
    finally:
        conn.close()


@router.post("/facts")
def create_fact(body: FactCreate):
    from memory import db
    from memory.embeddings import embed
    from memory.decay import compute_decay_score

    embedding = embed(body.text)
    with write_lock:
        conn = get_write_conn()
        try:
            item_id, is_new = db.upsert_fact(
                conn, body.text,
                category=body.category,
                temporal_class=body.temporal_class,
                confidence=body.confidence,
                embedding=embedding,
                session_id="dashboard",
                decay_fn=compute_decay_score,
                scope=body.scope,
                importance=body.importance,
                file_paths=body.file_paths,
            )
            return {"id": item_id, "is_new": is_new}
        finally:
            conn.close()


@router.put("/facts/{fact_id}")
def update_fact(fact_id: str, body: FactUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute("SELECT id, text FROM facts WHERE id = ?", [fact_id]).fetchone()
            if not existing:
                raise HTTPException(404, "Fact not found")

            updates, params = [], []
            if body.text is not None:
                updates.append("text = ?")
                params.append(body.text)
                # Recompute embedding if text changed
                new_emb = embed(body.text)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.category is not None:
                updates.append("category = ?")
                params.append(body.category)
            if body.temporal_class is not None:
                updates.append("temporal_class = ?")
                params.append(body.temporal_class)
            if body.importance is not None:
                updates.append("importance = ?")
                params.append(body.importance)

            if not updates:
                return {"id": fact_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            sql = f"UPDATE facts SET {', '.join(updates)} WHERE id = ?"
            params.append(fact_id)
            conn.execute(sql, params)
            return {"id": fact_id, "updated": True}
        finally:
            conn.close()


@router.delete("/facts/{fact_id}")
def delete_fact(fact_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, fact_id, "facts")
            if not success:
                raise HTTPException(404, "Fact not found")
            return {"id": fact_id, "deleted": True}
        finally:
            conn.close()
