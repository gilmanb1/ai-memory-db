from fastapi import APIRouter

from ..server import get_read_conn

router = APIRouter(tags=["scopes"])


@router.get("/scopes")
def list_scopes():
    conn = get_read_conn()
    try:
        rows = conn.execute("""
            SELECT scope, COUNT(*) as cnt FROM (
                SELECT scope FROM facts WHERE is_active = TRUE
                UNION ALL
                SELECT scope FROM ideas WHERE is_active = TRUE
                UNION ALL
                SELECT scope FROM decisions WHERE is_active = TRUE
                UNION ALL
                SELECT scope FROM guardrails WHERE is_active = TRUE
                UNION ALL
                SELECT scope FROM procedures WHERE is_active = TRUE
                UNION ALL
                SELECT scope FROM error_solutions WHERE is_active = TRUE
            ) GROUP BY scope ORDER BY cnt DESC
        """).fetchall()

        items = []
        for scope, count in rows:
            display = scope
            if scope == "__global__":
                display = "Global"
            elif "/" in scope:
                display = scope.rstrip("/").split("/")[-1]
            items.append({"scope": scope, "display_name": display, "count": count})

        return {"items": items}
    finally:
        conn.close()
