from fastapi import APIRouter
from ..server import get_read_conn

router = APIRouter(tags=["stats"])


@router.get("/stats")
def get_stats():
    conn = get_read_conn()
    try:
        from memory import db
        stats = db.get_stats(conn)

        # Add guardrails/procedures/error_solutions counts
        for table in ("guardrails", "procedures", "error_solutions"):
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE is_active = TRUE"
                ).fetchone()
                stats[table] = {"total": row[0] if row else 0}
            except Exception:
                stats[table] = {"total": 0}

        return stats
    finally:
        conn.close()
