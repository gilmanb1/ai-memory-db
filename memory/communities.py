"""
communities.py — Entity-based clustering and hierarchical community summaries.

Clusters related facts by shared entity references, then uses Claude to
summarize each cluster into a community summary. Summaries are hierarchical:
  Level 1: summaries of raw fact clusters
  Level 2: summaries of Level 1 summaries that share entities
"""
from __future__ import annotations

import sys
import textwrap
from collections import defaultdict
from typing import Optional

import anthropic

from . import db, embeddings
from .config import (
    CLAUDE_MODEL,
    EXTRACT_MAX_TOKENS,
    COMMUNITY_MIN_CLUSTER_SIZE,
    COMMUNITY_MIN_ENTITY_OVERLAP,
    GLOBAL_SCOPE,
)


# ── Community summary tool schema ────────────────────────────────────────

COMMUNITY_SUMMARY_TOOL = {
    "name": "create_community_summary",
    "description": "Create a concise summary of a cluster of related facts about a software project.",
    "input_schema": {
        "type": "object",
        "required": ["summary", "key_entities"],
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "2-4 sentence synthesis of the cluster's knowledge. "
                    "Focus on architectural patterns, key decisions, and relationships."
                ),
            },
            "key_entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The most important entity names in this cluster.",
            },
        },
    },
}

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a knowledge synthesis engine for a software project memory system.
    Given a cluster of related facts about a codebase, produce a concise 2-4
    sentence summary that captures the key architectural patterns, decisions,
    and relationships. The summary should help a coding agent quickly understand
    this area of the codebase.
""")


# ── Union-Find for clustering ────────────────────────────────────────────

class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ── Clustering ───────────────────────────────────────────────────────────

def find_entity_clusters(
    conn,
    scope: Optional[str] = None,
    min_overlap: int = COMMUNITY_MIN_ENTITY_OVERLAP,
) -> list[list[dict]]:
    """
    Build clusters of facts that share entities via fact_entity_links.

    Two facts are connected if they share min_overlap or more entities.
    Returns list of clusters, each cluster is a list of fact dicts.
    Only returns clusters with >= COMMUNITY_MIN_CLUSTER_SIZE items.
    """
    # Build entity → fact_ids mapping
    scope_sql, scope_params = db._scope_filter(scope)
    try:
        rows = conn.execute(f"""
            SELECT fel.fact_id, fel.entity_name, f.text, f.category,
                   f.temporal_class, f.importance
            FROM fact_entity_links fel
            JOIN facts f ON fel.fact_id = f.id
            WHERE f.is_active = TRUE {scope_sql}
        """, scope_params).fetchall()
    except Exception:
        return []

    entity_to_facts: dict[str, set[str]] = defaultdict(set)
    fact_data: dict[str, dict] = {}
    fact_entities: dict[str, set[str]] = defaultdict(set)

    for fact_id, entity_name, text, category, tc, importance in rows:
        entity_to_facts[entity_name].add(fact_id)
        fact_entities[fact_id].add(entity_name)
        if fact_id not in fact_data:
            fact_data[fact_id] = {
                "id": fact_id,
                "text": text,
                "category": category,
                "temporal_class": tc,
                "importance": importance,
            }

    # Union-Find: connect facts that share enough entities
    uf = _UnionFind()
    fact_ids = list(fact_data.keys())
    for i in range(len(fact_ids)):
        for j in range(i + 1, len(fact_ids)):
            shared = fact_entities[fact_ids[i]] & fact_entities[fact_ids[j]]
            if len(shared) >= min_overlap:
                uf.union(fact_ids[i], fact_ids[j])

    # Group by connected component
    groups: dict[str, list[str]] = defaultdict(list)
    for fid in fact_ids:
        root = uf.find(fid)
        groups[root].append(fid)

    # Filter by minimum cluster size
    clusters = []
    for group_ids in groups.values():
        if len(group_ids) >= COMMUNITY_MIN_CLUSTER_SIZE:
            cluster = [fact_data[fid] for fid in group_ids]
            clusters.append(cluster)

    return clusters


# ── Summary generation ───────────────────────────────────────────────────

def build_community_summaries(
    conn,
    api_key: str,
    scope: str = GLOBAL_SCOPE,
    quiet: bool = False,
) -> dict:
    """
    Find entity clusters, summarize each via Claude, store as community_summaries.

    Returns stats dict: {"clusters_found", "summaries_created", "summaries_updated"}
    """
    stats = {"clusters_found": 0, "summaries_created": 0, "summaries_updated": 0}

    if not api_key:
        return stats

    clusters = find_entity_clusters(conn, scope=scope)
    stats["clusters_found"] = len(clusters)

    if not clusters:
        return stats

    client = anthropic.Anthropic(api_key=api_key)

    for cluster in clusters:
        # Build the facts text for the LLM
        fact_lines = []
        entity_set: set[str] = set()
        source_ids = []
        for fact in cluster:
            fact_lines.append(f"- {fact['text']}")
            source_ids.append(fact["id"])

        # Get entities for this cluster
        for fact in cluster:
            try:
                ent_rows = conn.execute(
                    "SELECT entity_name FROM fact_entity_links WHERE fact_id = ?",
                    [fact["id"]],
                ).fetchall()
                for r in ent_rows:
                    entity_set.add(r[0])
            except Exception:
                pass

        user_message = (
            f"Summarize this cluster of {len(cluster)} related facts:\n\n"
            + "\n".join(fact_lines)
        )

        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                tools=[COMMUNITY_SUMMARY_TOOL],
                tool_choice={"type": "tool", "name": "create_community_summary"},
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "create_community_summary":
                    summary_text = block.input.get("summary", "")
                    key_entities = block.input.get("key_entities", list(entity_set)[:10])
                    break
            else:
                continue

            emb = embeddings.embed(summary_text)
            _, is_new = db.upsert_community_summary(
                conn,
                level=1,
                summary=summary_text,
                entity_ids=key_entities,
                source_item_ids=source_ids,
                embedding=emb,
                scope=scope,
            )
            if is_new:
                stats["summaries_created"] += 1
            else:
                stats["summaries_updated"] += 1

        except Exception as exc:
            if not quiet:
                print(f"[memory] Community summary failed for cluster: {exc}", file=sys.stderr)

    return stats
