from typing import Optional
from fastapi import APIRouter

from ..server import get_read_conn

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
def list_sessions(limit: int = 100, offset: int = 0):
    conn = get_read_conn()
    try:
        rows = conn.execute("""
            SELECT id, trigger, cwd, message_count, summary, scope, created_at
            FROM sessions
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, [limit, offset]).fetchall()
        cols = ["id", "trigger", "cwd", "message_count", "summary", "scope", "created_at"]

        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        return {
            "items": [dict(zip(cols, row)) for row in rows],
            "total": total,
        }
    finally:
        conn.close()
