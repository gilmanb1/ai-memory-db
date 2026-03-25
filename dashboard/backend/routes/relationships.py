from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["relationships"])


class RelationshipCreate(BaseModel):
    from_entity: str
    to_entity: str
    rel_type: str
    description: str = ""
    strength: float = 1.0
    scope: str = "__global__"


class RelationshipUpdate(BaseModel):
    description: Optional[str] = None
    rel_type: Optional[str] = None
    strength: Optional[float] = None


@router.get("/relationships")
def list_relationships(scope: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (scope = ? OR scope = '__global__')"
            params = [scope]

        sql = f"""
            SELECT id, from_entity, to_entity, rel_type, description,
                   strength, session_count, scope, created_at, last_seen_at
            FROM relationships WHERE is_active = TRUE{scope_sql}
            ORDER BY strength DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["id", "from_entity", "to_entity", "rel_type", "description",
                "strength", "session_count", "scope", "created_at", "last_seen_at"]

        total = conn.execute(
            f"SELECT COUNT(*) FROM relationships WHERE is_active = TRUE{scope_sql}", params
        ).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.get("/relationships/graph")
def get_graph(scope: Optional[str] = None, limit: int = 100):
    """Return nodes (entities) and edges (relationships) for Cytoscape visualization."""
    conn = get_read_conn()
    try:
        scope_sql, params = "", []
        if scope:
            scope_sql = " AND (r.scope = ? OR r.scope = '__global__')"
            params = [scope]

        # Get relationships
        rel_sql = f"""
            SELECT r.id, r.from_entity, r.to_entity, r.rel_type,
                   r.description, r.strength
            FROM relationships r
            WHERE r.is_active = TRUE{scope_sql}
            ORDER BY r.strength DESC LIMIT ?
        """
        rels = conn.execute(rel_sql, params + [limit * 3]).fetchall()

        # Collect entity names from relationships
        entity_names = set()
        edges = []
        for rid, from_e, to_e, rtype, desc, strength in rels:
            entity_names.add(from_e)
            entity_names.add(to_e)
            edges.append({
                "id": rid,
                "source": from_e,
                "target": to_e,
                "rel_type": rtype,
                "description": desc or "",
                "strength": strength or 1.0,
            })

        # Get entity details
        nodes = []
        if entity_names:
            placeholders = ",".join(["?"] * len(entity_names))
            ent_rows = conn.execute(f"""
                SELECT name, entity_type, session_count
                FROM entities
                WHERE name IN ({placeholders}) AND is_active = TRUE
            """, list(entity_names)).fetchall()

            seen = set()
            for name, etype, scount in ent_rows:
                nodes.append({
                    "id": name,
                    "label": name,
                    "entity_type": etype or "general",
                    "session_count": scount or 1,
                })
                seen.add(name)

            # Add any entities referenced in rels but not in entities table
            for name in entity_names - seen:
                nodes.append({
                    "id": name,
                    "label": name,
                    "entity_type": "unknown",
                    "session_count": 1,
                })

        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()


@router.post("/relationships")
def create_relationship(body: RelationshipCreate):
    from memory import db

    with write_lock:
        conn = get_write_conn()
        try:
            rid, is_new = db.upsert_relationship(
                conn,
                from_entity=body.from_entity,
                to_entity=body.to_entity,
                rel_type=body.rel_type,
                description=body.description,
                strength=body.strength,
                session_id="dashboard",
                scope=body.scope,
            )
            return {"id": rid, "is_new": is_new}
        finally:
            conn.close()


@router.put("/relationships/{rel_id}")
def update_relationship(rel_id: str, body: RelationshipUpdate):
    with write_lock:
        conn = get_write_conn()
        try:
            existing = conn.execute(
                "SELECT id FROM relationships WHERE id = ?", [rel_id]
            ).fetchone()
            if not existing:
                raise HTTPException(404, "Relationship not found")

            updates, params = [], []
            if body.description is not None:
                updates.append("description = ?")
                params.append(body.description)
            if body.rel_type is not None:
                updates.append("rel_type = ?")
                params.append(body.rel_type)
            if body.strength is not None:
                updates.append("strength = ?")
                params.append(body.strength)

            if not updates:
                return {"id": rel_id, "updated": False}

            updates.append("last_seen_at = CURRENT_TIMESTAMP")
            conn.execute(
                f"UPDATE relationships SET {', '.join(updates)} WHERE id = ?",
                params + [rel_id],
            )
            return {"id": rel_id, "updated": True}
        finally:
            conn.close()


@router.delete("/relationships/{rel_id}")
def delete_relationship(rel_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.soft_delete(conn, rel_id, "relationships")
            if not success:
                raise HTTPException(404, "Relationship not found")
            return {"id": rel_id, "deleted": True}
        finally:
            conn.close()
