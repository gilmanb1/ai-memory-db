from typing import Optional
from fastapi import APIRouter
from ..server import get_read_conn

router = APIRouter(tags=["code_graph"])


@router.get("/code-graph/stats")
def code_graph_stats():
    conn = get_read_conn()
    try:
        stats = {}
        # Total files and symbols
        try:
            row = conn.execute("SELECT COUNT(DISTINCT file_path), COUNT(*) FROM code_symbols").fetchone()
            stats["files"] = row[0] if row else 0
            stats["symbols"] = row[1] if row else 0
        except Exception:
            stats["files"] = 0
            stats["symbols"] = 0

        # By language
        try:
            rows = conn.execute("""
                SELECT language, COUNT(DISTINCT file_path) as files, COUNT(*) as symbols
                FROM code_symbols GROUP BY language ORDER BY files DESC
            """).fetchall()
            stats["by_language"] = [{"language": r[0], "files": r[1], "symbols": r[2]} for r in rows]
        except Exception:
            stats["by_language"] = []

        # Dependencies
        try:
            row = conn.execute("SELECT COUNT(*) FROM code_dependencies").fetchone()
            stats["dependencies"] = row[0] if row else 0
        except Exception:
            stats["dependencies"] = 0

        return stats
    finally:
        conn.close()


@router.get("/code-graph/files")
def list_code_files(language: Optional[str] = None, limit: int = 500, offset: int = 0):
    conn = get_read_conn()
    try:
        lang_sql, params = "", []
        if language:
            lang_sql = " WHERE language = ?"
            params = [language]

        sql = f"""
            SELECT file_path, language, symbol_count, parsed_at
            FROM code_file_index{lang_sql}
            ORDER BY symbol_count DESC LIMIT ? OFFSET ?
        """
        rows = conn.execute(sql, params + [limit, offset]).fetchall()
        cols = ["file_path", "language", "symbol_count", "parsed_at"]

        count_sql = f"SELECT COUNT(*) FROM code_file_index{lang_sql}"
        total = conn.execute(count_sql, params).fetchone()[0]

        return {"items": [dict(zip(cols, row)) for row in rows], "total": total}
    finally:
        conn.close()


@router.get("/code-graph/files/{file_path:path}/symbols")
def get_file_symbols(file_path: str):
    conn = get_read_conn()
    try:
        rows = conn.execute("""
            SELECT id, symbol_name, symbol_type, language, line_number, signature, docstring
            FROM code_symbols WHERE file_path = ?
            ORDER BY line_number
        """, [file_path]).fetchall()
        cols = ["id", "symbol_name", "symbol_type", "language", "line_number", "signature", "docstring"]
        return {"items": [dict(zip(cols, row)) for row in rows]}
    finally:
        conn.close()


@router.get("/code-graph/files/{file_path:path}/impact")
def get_file_impact(file_path: str):
    conn = get_read_conn()
    try:
        from memory.code_graph import get_impact_analysis
        result = get_impact_analysis(conn, file_path)
        return result
    finally:
        conn.close()


@router.get("/code-graph/graph")
def get_code_graph(scope: Optional[str] = None, limit: int = 200):
    """Return Cytoscape-format nodes and edges for the file dependency graph."""
    conn = get_read_conn()
    try:
        # Get files as nodes
        scope_sql, scope_params = "", []
        if scope:
            scope_sql = " WHERE scope = ? OR scope = '__global__'"
            scope_params = [scope]

        file_rows = conn.execute(f"""
            SELECT file_path, language, symbol_count
            FROM code_file_index{scope_sql}
            ORDER BY symbol_count DESC LIMIT ?
        """, scope_params + [limit]).fetchall()

        file_set = {r[0] for r in file_rows}
        nodes = []
        for fp, lang, sym_count in file_rows:
            label = fp.split("/")[-1] if "/" in fp else fp
            nodes.append({
                "id": fp,
                "label": label,
                "language": lang or "unknown",
                "symbol_count": sym_count or 0,
            })

        # Get dependencies as edges (only between files in our node set)
        edges = []
        if file_set:
            placeholders = ",".join(["?"] * len(file_set))
            dep_rows = conn.execute(f"""
                SELECT id, from_file, to_file, import_name
                FROM code_dependencies
                WHERE from_file IN ({placeholders}) AND to_file IN ({placeholders})
            """, list(file_set) + list(file_set)).fetchall()

            seen_edges = set()
            for did, from_f, to_f, imp_name in dep_rows:
                edge_key = f"{from_f}->{to_f}"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "id": did,
                        "source": from_f,
                        "target": to_f,
                        "import_name": imp_name or "",
                    })

        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()


@router.get("/code-graph/symbols/search")
def search_symbols(q: str, limit: int = 50):
    conn = get_read_conn()
    try:
        from memory.code_graph import search_symbol
        results = search_symbol(conn, q)
        return {"items": results[:limit]}
    finally:
        conn.close()
