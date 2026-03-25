from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from ..server import get_read_conn

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    types: Optional[list[str]] = None  # facts, decisions, guardrails, procedures, error_solutions
    scope: Optional[str] = None
    limit: int = 20


@router.post("/search")
def search_memory(body: SearchRequest):
    from memory.embeddings import embed_query
    from memory import db

    query_vec = embed_query(body.query)
    if not query_vec:
        return {"results": [], "error": "Embedding unavailable"}

    conn = get_read_conn()
    try:
        types = body.types or ["facts", "decisions", "guardrails", "procedures", "error_solutions"]
        results = []

        if "facts" in types:
            facts = db.search_facts(conn, query_vec, limit=body.limit, scope=body.scope)
            for f in facts:
                f["type"] = "fact"
                results.append(f)

        if "decisions" in types:
            # Search decisions via vector similarity
            try:
                rows = conn.execute("""
                    SELECT id, text, temporal_class, decay_score, scope,
                           list_cosine_similarity(embedding, ?::FLOAT[]) AS score
                    FROM decisions
                    WHERE is_active = TRUE AND embedding IS NOT NULL
                    ORDER BY score DESC LIMIT ?
                """, [query_vec, body.limit]).fetchall()
                for row in rows:
                    if row[5] and row[5] >= 0.3:
                        results.append({
                            "type": "decision",
                            "id": row[0], "text": row[1],
                            "temporal_class": row[2], "decay_score": row[3],
                            "scope": row[4], "score": row[5],
                        })
            except Exception:
                pass

        if "guardrails" in types:
            grs = db.search_guardrails(conn, query_vec, limit=body.limit, scope=body.scope)
            for g in grs:
                g["type"] = "guardrail"
                g["text"] = g.get("warning", "")
                results.append(g)

        if "procedures" in types:
            procs = db.search_procedures(conn, query_vec, limit=body.limit, scope=body.scope)
            for p in procs:
                p["type"] = "procedure"
                p["text"] = p.get("task_description", "")
                results.append(p)

        if "error_solutions" in types:
            errs = db.search_error_solutions(conn, query_vec, limit=body.limit, scope=body.scope)
            for e in errs:
                e["type"] = "error_solution"
                e["text"] = f"{e.get('error_pattern', '')} -> {e.get('solution', '')}"
                results.append(e)

        # Sort by score descending
        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return {"results": results[:body.limit]}
    finally:
        conn.close()
