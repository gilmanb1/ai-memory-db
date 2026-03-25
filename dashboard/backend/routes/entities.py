from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["entities"])


class EntityCreate(BaseModel):
    name: str
    entity_type: str = "general"
    scope: str = "__global__"


class EntityUpdate(BaseModel):
    name: Optional[str] = None
    entity_type: Optional[str] = None


@router.get("/entities")
def list_entities(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, name, entity_type, session_count, scope,
                   first_seen_at, last_seen_at
            FROM entities WHERE is_active = TRUE{scope_sql}
            ORDER BY session_count DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "name", "entity_type", "session_count", "scope",
                "first_seen_at", "last_seen_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM entities WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.post("/entities")
def create_entity(body: EntityCreate):
    from memory import db
    from memory.embeddings import embed

    embedding = embed(body.name)
    with write_lock:
        conn = get_write_conn()
        try:
            eid, is_new = db.upsert_entity(
                conn, body.name, body.entity_type,
                embedding=embedding, scope=body.scope,
            )
            return {"id": eid, "is_new": is_new}
        finally:
            conn.close()


@router.put("/entities/{entity_id}")
def update_entity(entity_id: str, body: EntityUpdate):
    from memory.embeddings import embed

    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM entities WHERE id = ?", [entity_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Entity not found")

            updates, params = [], []
            if body.name is not None:
                updates.append("name = ?")
                params.append(body.name)
                updates.append("name_lower = ?")
                params.append(body.name.lower())
                new_emb = embed(body.name)
                if new_emb:
                    updates.append("embedding = ?")
                    params.append(new_emb)
            if body.entity_type is not None:
                updates.append("entity_type = ?")
                params.append(body.entity_type)

            if not updates:
                return {"id": entity_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            conn.execute(
                f"UPDATE entities SET {', '.join(updates)} WHERE id = ?",
                params + [entity_id],
            )
            return {"id": entity_id, "updated": True}
        finally:
            conn.close()


@router.delete("/entities/{entity_id}")
def delete_entity(entity_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, entity_id, "entities")
            if not success:
                raise HTTPException(404, "Entity not found")
            return {"id": entity_id, "deleted": True}
        finally:
            conn.close()
