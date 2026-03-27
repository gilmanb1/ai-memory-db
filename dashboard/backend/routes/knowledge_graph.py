"""Unified knowledge graph endpoint — returns all knowledge types and their real connections."""
from __future__ import annotations

from collections import Counter
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..server import get_read_conn

router = APIRouter(tags=["knowledge_graph"])


# ── Public API ────────────────────────────────────────────────────────────

VALID_TYPES = {"entity", "fact", "decision", "observation", "guardrail", "procedure", "error_solution", "file"}


@router.get("/knowledge-graph")
def get_knowledge_graph(
    types: str = "entity,fact,decision",
    scope: Optional[str] = None,
    limit: int = 150,
    cluster_by: str = "type",
):
    """Build a unified graph across all knowledge types with real join-table edges."""
    conn = get_read_conn()
    try:
        type_list = [t.strip() for t in types.split(",") if t.strip() in VALID_TYPES]
        if not type_list:
            type_list = ["entity", "fact", "decision"]
        return build_knowledge_graph(conn, type_list, limit, cluster_by, scope)
    finally:
        conn.close()


# ── Core graph builder (testable without FastAPI) ─────────────────────────

def build_knowledge_graph(
    conn,
    types: list[str],
    limit: int = 150,
    cluster_by: str = "type",
    scope: Optional[str] = None,
) -> dict:
    scope_sql, scope_params = _scope_filter(scope)
    per_type_limit = max(10, limit // max(len(types), 1))

    nodes: list[dict] = []
    node_id_set: set[str] = set()
    edges: list[dict] = []
    type_counts: Counter = Counter()

    # ── Collect nodes per type ────────────────────────────────────────
    if "entity" in types:
        _add_entity_nodes(conn, nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit)

    if "fact" in types:
        _add_typed_nodes(conn, "fact", "facts", "text", "category",
                         nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit)

    if "decision" in types:
        _add_typed_nodes(conn, "decision", "decisions", "text", None,
                         nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit)

    if "observation" in types:
        _add_typed_nodes(conn, "observation", "observations", "text", None,
                         nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit,
                         extra_cols="proof_count",
                         size_col="proof_count")

    if "guardrail" in types:
        _add_typed_nodes(conn, "guardrail", "guardrails", "warning", None,
                         nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit,
                         size_col="importance")

    if "procedure" in types:
        _add_typed_nodes(conn, "procedure", "procedures", "task_description", None,
                         nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit,
                         size_col="importance")

    if "error_solution" in types:
        _add_typed_nodes(conn, "error_solution", "error_solutions", "error_pattern", None,
                         nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit,
                         size_col="times_applied")

    if "file" in types:
        _add_file_nodes(conn, nodes, node_id_set, type_counts, scope_sql, scope_params, per_type_limit)

    # ── Build edges from real join tables ──────────────────────────────

    # Entity ↔ entity relationships
    if "entity" in types:
        _add_relationship_edges(conn, edges, node_id_set, scope_sql, scope_params, limit * 3)

    # Fact ↔ entity via fact_entity_links
    if "entity" in types and "fact" in types:
        _add_fact_entity_edges(conn, edges, node_id_set)

    # Observation → fact via source_fact_ids
    if "observation" in types and "fact" in types:
        _add_observation_fact_edges(conn, edges, node_id_set)

    # Item → file via fact_file_links
    if "file" in types:
        _add_file_link_edges(conn, edges, node_id_set, types)

    # File → file via code_dependencies
    if "file" in types:
        _add_code_dependency_edges(conn, edges, node_id_set)

    # ── Compute degree ────────────────────────────────────────────────
    degree: Counter = Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for n in nodes:
        n["degree"] = degree.get(n["id"], 0)

    # ── Assign clusters ──────────────────────────────────────────────
    for n in nodes:
        if cluster_by == "type":
            n["cluster"] = n["node_type"]
        elif cluster_by == "scope":
            n["cluster"] = n.get("scope", "__global__")
        elif cluster_by == "session":
            n["cluster"] = n.get("metadata", {}).get("source_session", "unknown")
        else:
            n["cluster"] = n["node_type"]  # default

    # ── Apply limit (sort by size_metric desc, keep top N) ────────────
    nodes.sort(key=lambda n: n.get("size_metric", 0), reverse=True)
    if len(nodes) > limit:
        nodes = nodes[:limit]
        node_id_set = {n["id"] for n in nodes}
        edges = [e for e in edges if e["source"] in node_id_set and e["target"] in node_id_set]

    # ── Build cluster list ────────────────────────────────────────────
    cluster_counter: Counter = Counter()
    for n in nodes:
        cluster_counter[n["cluster"]] += 1
    clusters = [
        {"id": cid, "label": cid, "node_count": cnt}
        for cid, cnt in cluster_counter.most_common()
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
        "type_counts": dict(type_counts),
    }


# ── Node builders ─────────────────────────────────────────────────────────

def _add_entity_nodes(conn, nodes, node_id_set, type_counts, scope_sql, scope_params, limit):
    try:
        rows = conn.execute(f"""
            SELECT name, entity_type, session_count, scope
            FROM entities WHERE is_active = TRUE {scope_sql}
            ORDER BY session_count DESC LIMIT ?
        """, scope_params + [limit]).fetchall()
    except Exception:
        return

    for name, etype, scount, esc in rows:
        nid = name
        if nid in node_id_set:
            continue
        node_id_set.add(nid)
        nodes.append({
            "id": nid,
            "label": name,
            "node_type": "entity",
            "size_metric": scount or 1,
            "metadata": {"entity_type": etype or "general", "session_count": scount or 1},
            "cluster": None,
            "scope": esc or "__global__",
        })
        type_counts["entity"] += 1


def _add_typed_nodes(conn, node_type, table, text_col, cat_col,
                     nodes, node_id_set, type_counts, scope_sql, scope_params, limit,
                     extra_cols="", size_col="importance"):
    # Check which optional columns exist on this table
    try:
        table_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return

    select = f"id, {text_col}, scope"
    if "source_session" in table_cols:
        select += ", source_session"
    if size_col and size_col != text_col and size_col in table_cols:
        select += f", {size_col}"
    elif size_col and size_col not in table_cols:
        size_col = "decay_score" if "decay_score" in table_cols else None
        if size_col:
            select += f", {size_col}"
    if cat_col and cat_col in table_cols:
        select += f", {cat_col}"
    if extra_cols:
        for ec in extra_cols.split(","):
            ec = ec.strip()
            if ec and ec not in select and ec in table_cols:
                select += f", {ec}"

    order_col = size_col if size_col and size_col in table_cols else "created_at"
    try:
        rows = conn.execute(f"""
            SELECT {select} FROM {table}
            WHERE is_active = TRUE {scope_sql}
            ORDER BY {order_col} DESC, created_at DESC LIMIT ?
        """, scope_params + [limit]).fetchall()
    except Exception:
        return

    col_names = [c.strip().split(" AS ")[-1].split(".")[-1] for c in select.split(",")]
    for row in rows:
        d = dict(zip(col_names, row))
        nid = f"{node_type}:{d['id']}"
        if nid in node_id_set:
            continue
        node_id_set.add(nid)
        raw_text = d.get(text_col, "")
        label = (raw_text[:77] + "...") if len(raw_text) > 80 else raw_text
        nodes.append({
            "id": nid,
            "label": label,
            "node_type": node_type,
            "size_metric": d.get(size_col, 1) or 1,
            "metadata": {k: v for k, v in d.items() if k not in ("id", text_col)},
            "cluster": None,
            "scope": d.get("scope", "__global__"),
        })
        type_counts[node_type] += 1


def _add_file_nodes(conn, nodes, node_id_set, type_counts, scope_sql, scope_params, limit):
    try:
        rows = conn.execute(f"""
            SELECT file_path, language, symbol_count
            FROM code_file_index
            ORDER BY symbol_count DESC LIMIT ?
        """, [limit]).fetchall()
    except Exception:
        return

    for fp, lang, sym_count in rows:
        nid = f"file:{fp}"
        if nid in node_id_set:
            continue
        node_id_set.add(nid)
        label = fp.split("/")[-1] if "/" in fp else fp
        nodes.append({
            "id": nid,
            "label": label,
            "node_type": "file",
            "size_metric": sym_count or 1,
            "metadata": {"language": lang or "unknown", "file_path": fp},
            "cluster": None,
            "scope": "__global__",
        })
        type_counts["file"] += 1


# ── Edge builders ─────────────────────────────────────────────────────────

def _add_relationship_edges(conn, edges, node_id_set, scope_sql, scope_params, limit):
    try:
        rows = conn.execute(f"""
            SELECT id, from_entity, to_entity, rel_type, description, strength
            FROM relationships WHERE is_active = TRUE {scope_sql}
            ORDER BY strength DESC LIMIT ?
        """, scope_params + [limit]).fetchall()
    except Exception:
        return

    for rid, from_e, to_e, rtype, desc, strength in rows:
        if from_e in node_id_set and to_e in node_id_set:
            edges.append({
                "id": f"rel:{rid}",
                "source": from_e,
                "target": to_e,
                "edge_type": rtype or "related",
                "strength": strength or 1.0,
                "metadata": {"description": desc or ""},
            })


def _add_fact_entity_edges(conn, edges, node_id_set):
    try:
        rows = conn.execute("""
            SELECT fact_id, entity_name FROM fact_entity_links
        """).fetchall()
    except Exception:
        return

    for fact_id, entity_name in rows:
        src = f"fact:{fact_id}"
        if src in node_id_set and entity_name in node_id_set:
            edges.append({
                "id": f"fel:{fact_id}:{entity_name}",
                "source": src,
                "target": entity_name,
                "edge_type": "mentions",
                "strength": 1.0,
                "metadata": {},
            })


def _add_observation_fact_edges(conn, edges, node_id_set):
    try:
        rows = conn.execute("""
            SELECT id, source_fact_ids FROM observations
            WHERE is_active = TRUE AND source_fact_ids IS NOT NULL
        """).fetchall()
    except Exception:
        return

    for obs_id, source_ids in rows:
        obs_nid = f"observation:{obs_id}"
        if obs_nid not in node_id_set or not source_ids:
            continue
        for fid in source_ids:
            fact_nid = f"fact:{fid}"
            if fact_nid in node_id_set:
                edges.append({
                    "id": f"proof:{obs_id}:{fid}",
                    "source": obs_nid,
                    "target": fact_nid,
                    "edge_type": "proven_by",
                    "strength": 1.0,
                    "metadata": {},
                })


def _add_file_link_edges(conn, edges, node_id_set, types):
    tables_to_check = []
    if "fact" in types:
        tables_to_check.append(("facts", "fact"))
    if "guardrail" in types:
        tables_to_check.append(("guardrails", "guardrail"))
    if "procedure" in types:
        tables_to_check.append(("procedures", "procedure"))
    if "error_solution" in types:
        tables_to_check.append(("error_solutions", "error_solution"))

    for item_table, node_prefix in tables_to_check:
        try:
            rows = conn.execute("""
                SELECT fact_id, file_path FROM fact_file_links WHERE item_table = ?
            """, [item_table]).fetchall()
        except Exception:
            continue

        for item_id, file_path in rows:
            src = f"{node_prefix}:{item_id}"
            tgt = f"file:{file_path}"
            if src in node_id_set and tgt in node_id_set:
                edges.append({
                    "id": f"ffl:{item_id}:{file_path}",
                    "source": src,
                    "target": tgt,
                    "edge_type": "references_file",
                    "strength": 1.0,
                    "metadata": {},
                })


def _add_code_dependency_edges(conn, edges, node_id_set):
    try:
        rows = conn.execute("""
            SELECT id, from_file, to_file, import_name FROM code_dependencies
        """).fetchall()
    except Exception:
        return

    seen = set()
    for did, from_f, to_f, imp_name in rows:
        src = f"file:{from_f}"
        tgt = f"file:{to_f}"
        key = f"{src}->{tgt}"
        if key in seen:
            continue
        seen.add(key)
        if src in node_id_set and tgt in node_id_set:
            edges.append({
                "id": f"dep:{did}",
                "source": src,
                "target": tgt,
                "edge_type": "imports",
                "strength": 1.0,
                "metadata": {"import_name": imp_name or ""},
            })


# ── Helpers ───────────────────────────────────────────────────────────────

def _scope_filter(scope: Optional[str]):
    if scope:
        return "AND (scope = ? OR scope = '__global__')", [scope]
    return "", []
