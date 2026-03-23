"""
consolidation.py — Observation synthesis and semantic forgetting.

Processes unconsolidated facts in batches, calls Claude to synthesize
observations that combine insights across multiple facts, and runs a
post-consolidation similarity sweep to deduplicate near-identical observations.
"""
from __future__ import annotations

import sys
import textwrap
from typing import Optional

import anthropic

from . import db
from .config import (
    CLAUDE_MODEL,
    CONSOLIDATION_BATCH_SIZE,
    CONSOLIDATION_MAX_OBS_CONTEXT,
    OBSERVATION_SIMILARITY_THRESHOLD,
)
from .embeddings import embed


# ── Consolidation tool schema ────────────────────────────────────────────

CONSOLIDATION_TOOL = {
    "name": "consolidate_observations",
    "description": "Create, update, or delete synthesized observations based on new facts.",
    "input_schema": {
        "type": "object",
        "required": ["creates", "updates", "deletes"],
        "properties": {
            "creates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "source_fact_ids"],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Synthesized observation that combines multiple facts",
                        },
                        "source_fact_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of facts this observation is derived from",
                        },
                    },
                },
            },
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["observation_id", "text", "additional_source_ids"],
                    "properties": {
                        "observation_id": {"type": "string"},
                        "text": {
                            "type": "string",
                            "description": "Updated observation text",
                        },
                        "additional_source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "deletes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["observation_id", "reason"],
                    "properties": {
                        "observation_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ── Consolidation system prompt ──────────────────────────────────────────

CONSOLIDATION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a knowledge consolidation engine analyzing newly extracted facts
    against existing observations in a persistent memory system.

    Your job is to synthesize higher-level observations from raw facts:

    CREATE observations that SYNTHESIZE insights across multiple facts.
    Never just restate a single fact — an observation must combine, generalize,
    or draw a conclusion from 2+ facts. Include all source_fact_ids for provenance.

    UPDATE observations when new facts extend, refine, or add supporting evidence
    to an existing observation. Provide the new text (full replacement, not a diff)
    and the additional source IDs.

    DELETE observations that are contradicted by newer facts or fully superseded
    by a more complete, more accurate observation. Provide a clear reason.

    RULES:
    - Be conservative: only create an observation when you see a clear pattern
      across 2+ facts.
    - Prefer updating existing observations over creating new near-duplicates.
    - Always include source_fact_ids so every observation has provenance.
    - If the new facts don't warrant any changes, return empty arrays for all
      three actions.
""")


# ── Main consolidation function ──────────────────────────────────────────

def run_consolidation(
    conn,
    api_key: str,
    scope: str,
    quiet: bool = False,
) -> dict:
    """
    Process unconsolidated facts through Claude to synthesize observations.

    Returns a stats dict with keys: batches, created, updated, deleted.
    """
    stats = {"batches": 0, "created": 0, "updated": 0, "deleted": 0}

    batch = db.get_unconsolidated_facts(conn, limit=CONSOLIDATION_BATCH_SIZE, scope=scope)
    if not batch:
        return stats

    # ── Gather related observations for context ──────────────────────
    related_obs: dict[str, dict] = {}
    for fact in batch:
        fact_embedding = fact.get("embedding")
        if not fact_embedding:
            continue
        hits = db.search_observations(conn, fact_embedding, limit=5, scope=scope)
        for obs in hits:
            related_obs[obs["id"]] = obs

    # Cap the number of observations shown to the LLM
    obs_list = list(related_obs.values())[:CONSOLIDATION_MAX_OBS_CONTEXT]

    # ── Build user message ───────────────────────────────────────────
    parts: list[str] = []

    parts.append("## New Facts")
    for i, fact in enumerate(batch, 1):
        parts.append(f"{i}. [{fact['id']}] {fact['text']}")

    parts.append("")
    parts.append("## Existing Observations")
    if obs_list:
        for i, obs in enumerate(obs_list, 1):
            source_ids = obs.get("source_fact_ids") or []
            sources_str = ", ".join(source_ids) if source_ids else "none"
            proof = obs.get("proof_count", 0)
            parts.append(f"{i}. [{obs['id']}] {obs['text']} (proof_count: {proof}, sources: {sources_str})")
    else:
        parts.append("(none yet)")

    user_message = "\n".join(parts)

    # ── Call Claude ──────────────────────────────────────────────────
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=CONSOLIDATION_SYSTEM_PROMPT,
            tools=[CONSOLIDATION_TOOL],
            tool_choice={"type": "tool", "name": "consolidate_observations"},
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as exc:
        print(f"[memory] Consolidation API call failed: {exc}", file=sys.stderr)
        return stats

    # ── Parse tool_use response ──────────────────────────────────────
    tool_input = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "consolidate_observations":
            tool_input = block.input
            break

    if not tool_input:
        print("[memory] Consolidation: no tool_use block in response", file=sys.stderr)
        return stats

    stats["batches"] = 1

    # ── Execute creates ──────────────────────────────────────────────
    for create in tool_input.get("creates", []):
        text = create.get("text", "")
        source_ids = create.get("source_fact_ids", [])
        if not text:
            continue
        try:
            emb = embed(text)
            obs_id, is_new = db.upsert_observation(conn, text, source_ids, emb, scope)
            db.log_consolidation_action(conn, "create", obs_id, source_ids, "synthesized from facts")
            stats["created"] += 1
            if not quiet:
                print(f"[memory] Consolidated: created observation {obs_id[:8]}…", file=sys.stderr)
        except Exception as exc:
            print(f"[memory] Consolidation create failed: {exc}", file=sys.stderr)

    # ── Execute updates ──────────────────────────────────────────────
    for update in tool_input.get("updates", []):
        obs_id = update.get("observation_id", "")
        text = update.get("text", "")
        additional_ids = update.get("additional_source_ids", [])
        if not obs_id or not text:
            continue
        try:
            emb = embed(text)
            ok = db.update_observation(conn, obs_id, text, emb, additional_ids)
            if ok:
                db.log_consolidation_action(conn, "update", obs_id, additional_ids, "refined with new evidence")
                stats["updated"] += 1
                if not quiet:
                    print(f"[memory] Consolidated: updated observation {obs_id[:8]}…", file=sys.stderr)
            else:
                print(f"[memory] Consolidation update: observation {obs_id[:8]}… not found", file=sys.stderr)
        except Exception as exc:
            print(f"[memory] Consolidation update failed: {exc}", file=sys.stderr)

    # ── Execute deletes ──────────────────────────────────────────────
    for delete in tool_input.get("deletes", []):
        obs_id = delete.get("observation_id", "")
        reason = delete.get("reason", "")
        if not obs_id:
            continue
        try:
            ok = db.supersede_item(conn, obs_id, "observations", "", reason)
            if ok:
                db.log_consolidation_action(conn, "delete", obs_id, [], reason)
                stats["deleted"] += 1
                if not quiet:
                    print(f"[memory] Consolidated: deleted observation {obs_id[:8]}…", file=sys.stderr)
            else:
                print(f"[memory] Consolidation delete: observation {obs_id[:8]}… not found", file=sys.stderr)
        except Exception as exc:
            print(f"[memory] Consolidation delete failed: {exc}", file=sys.stderr)

    # ── Mark facts as consolidated ───────────────────────────────────
    try:
        db.mark_facts_consolidated(conn, [f["id"] for f in batch])
    except Exception as exc:
        print(f"[memory] Failed to mark facts consolidated: {exc}", file=sys.stderr)

    return stats


# ── Semantic forgetting (post-consolidation similarity sweep) ────────────

def run_semantic_forgetting(conn, scope: str) -> dict:
    """
    Find near-duplicate observations and soft-delete the weaker one.

    Compares all active observation pairs; if cosine similarity >=
    OBSERVATION_SIMILARITY_THRESHOLD, keeps the one with higher proof_count
    (ties broken by newer created_at) and supersedes the other.

    Returns stats: {"pairs_checked": N, "superseded": M}
    """
    observations = db.get_all_observation_embeddings(conn, scope=scope)
    stats = {"pairs_checked": 0, "superseded": 0}

    # Track which IDs have already been superseded in this sweep
    superseded_ids: set[str] = set()

    n = len(observations)
    for i in range(n):
        if observations[i]["id"] in superseded_ids:
            continue
        for j in range(i + 1, n):
            if observations[j]["id"] in superseded_ids:
                continue

            stats["pairs_checked"] += 1
            sim = db._cosine_py(observations[i]["embedding"], observations[j]["embedding"])

            if sim >= OBSERVATION_SIMILARITY_THRESHOLD:
                a, b = observations[i], observations[j]

                # Decide winner: higher proof_count wins; ties: newer created_at wins
                if a["proof_count"] > b["proof_count"]:
                    winner, loser = a, b
                elif b["proof_count"] > a["proof_count"]:
                    winner, loser = b, a
                else:
                    # Tie: keep newer
                    if a["created_at"] >= b["created_at"]:
                        winner, loser = a, b
                    else:
                        winner, loser = b, a

                try:
                    ok = db.supersede_item(
                        conn, loser["id"], "observations", winner["id"], "semantic_dedup"
                    )
                    if ok:
                        superseded_ids.add(loser["id"])
                        stats["superseded"] += 1
                except Exception as exc:
                    print(
                        f"[memory] Semantic forgetting failed for {loser['id'][:8]}…: {exc}",
                        file=sys.stderr,
                    )

    # ── Orphaned short-term fact cleanup ────────────────────────────────
    stats["orphaned_facts_cleaned"] = 0
    try:
        for obs_id in superseded_ids:
            # Get source_fact_ids for this superseded observation
            row = conn.execute(
                "SELECT source_fact_ids FROM observations WHERE id = ?",
                [obs_id],
            ).fetchone()
            if not row or not row[0]:
                continue
            source_fact_ids = row[0]
            for fact_id in source_fact_ids:
                # Check if the fact is short-term
                fact_row = conn.execute(
                    "SELECT temporal_class FROM facts WHERE id = ? AND is_active = TRUE",
                    [fact_id],
                ).fetchone()
                if not fact_row or fact_row[0] != "short":
                    continue
                # Check if any other active observation references this fact
                count_row = conn.execute(
                    """SELECT COUNT(*) FROM observations
                       WHERE is_active = TRUE
                         AND list_contains(source_fact_ids, ?)""",
                    [fact_id],
                ).fetchone()
                if count_row and count_row[0] == 0:
                    ok = db.soft_delete(conn, fact_id, "facts")
                    if ok:
                        stats["orphaned_facts_cleaned"] += 1
    except Exception as exc:
        print(
            f"[memory] Orphaned fact cleanup failed: {exc}",
            file=sys.stderr,
        )

    return stats


# ══════════════════════════════════════════════════════════════════════════
# Feature 7: Memory coherence validation
# ══════════════════════════════════════════════════════════════════════════

def run_coherence_check(
    conn,
    scope: str,
    quiet: bool = False,
) -> dict:
    """
    Find and resolve contradictory facts (high similarity, different content).

    Heuristic approach (no LLM call):
    - Find fact pairs with cosine similarity between 0.88 and 0.92
    - These are "same topic, different content" — potential contradictions
    - Auto-resolve: keep the newer fact, invalidate the older via bi-temporal
    - Log to consolidation_log
    """
    from .config import COHERENCE_ENABLED, COHERENCE_SIMILARITY_LOW, COHERENCE_SIMILARITY_HIGH

    stats = {"pairs_checked": 0, "contradictions_found": 0, "resolved": 0}

    if not COHERENCE_ENABLED:
        return stats

    try:
        pairs = db.find_potential_contradictions(
            conn, scope=scope,
            similarity_low=COHERENCE_SIMILARITY_LOW,
            similarity_high=COHERENCE_SIMILARITY_HIGH,
            limit=50,
        )
    except Exception:
        return stats

    stats["pairs_checked"] = len(pairs)

    for fact_a, fact_b, similarity in pairs:
        stats["contradictions_found"] += 1

        # Keep newer, invalidate older
        a_time = fact_a.get("created_at", "")
        b_time = fact_b.get("created_at", "")
        if str(a_time) >= str(b_time):
            keep, invalidate = fact_a, fact_b
        else:
            keep, invalidate = fact_b, fact_a

        try:
            ok = db.resolve_contradiction(conn, keep["id"], invalidate["id"])
            if ok:
                stats["resolved"] += 1
                db.log_consolidation_action(
                    conn,
                    action="coherence_resolve",
                    observation_id=keep["id"],
                    source_ids=[invalidate["id"]],
                    reason=f"Contradiction resolved (sim={similarity:.3f}): kept newer fact, invalidated older",
                )
        except Exception as exc:
            if not quiet:
                print(f"[memory] Coherence resolution failed: {exc}", file=sys.stderr)

    return stats
