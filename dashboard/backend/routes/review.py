from typing import Optional
from fastapi import APIRouter, HTTPException

from ..server import get_read_conn, get_write_conn, write_lock

router = APIRouter(tags=["review"])


@router.get("/review")
def list_reviews(status: str = "pending", limit: int = 50):
    conn = get_read_conn()
    try:
        rows = conn.execute("""
            SELECT id, item_text, item_table, item_data, reason, status,
                   source_session, scope, created_at, reviewed_at
            FROM review_queue
            WHERE status = ?
            ORDER BY created_at DESC LIMIT ?
        """, [status, limit]).fetchall()
        cols = ["id", "item_text", "item_table", "item_data", "reason", "status",
                "source_session", "scope", "created_at", "reviewed_at"]
        total = conn.execute(
            "SELECT COUNT(*) FROM review_queue WHERE status = ?", [status]
        ).fetchone()[0]
        return {"items": [dict(zip(cols, r)) for r in rows], "total": total}
    except Exception:
        return {"items": [], "total": 0}
    finally:
        conn.close()


@router.post("/review/{review_id}/approve")
def approve_review(review_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.approve_review(conn, review_id)
            if not success:
                raise HTTPException(404, "Review item not found or already processed")
            # Store the approved item
            import json
            row = conn.execute(
                "SELECT item_text, item_table, item_data, scope FROM review_queue WHERE id = ?",
                [review_id],
            ).fetchone()
            stored_id = None
            if row:
                text, table, data_json, scope = row
                data = json.loads(data_json) if data_json else {}
                from memory.decay import compute_decay_score
                if table == "facts":
                    stored_id, _ = db.upsert_fact(
                        conn, text,
                        data.get("category", "operational"),
                        data.get("temporal_class", "long"),
                        "high", None, "review-approved", compute_decay_score,
                        scope=scope, importance=data.get("importance", 5),
                    )
            return {"id": review_id, "approved": True, "stored_id": stored_id}
        finally:
            conn.close()


@router.post("/review/{review_id}/reject")
def reject_review(review_id: str):
    from memory import db
    with write_lock:
        conn = get_write_conn()
        try:
            success = db.reject_review(conn, review_id)
            if not success:
                raise HTTPException(404, "Review item not found or already processed")
            return {"id": review_id, "rejected": True}
        finally:
            conn.close()
