"""
ingest.py — Shared extraction + storage logic.

Used by three triggers:
  1. Status line (context window at 90%)
  2. PreCompact hook (safety net before compaction)
  3. SessionEnd hook (background process at session close)

All callers use a lock file per session to prevent duplicate extraction.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import DB_PATH, GLOBAL_SCOPE, CROSS_PASS_DEDUP_THRESHOLD, RECALL_THRESHOLD
from . import db, embeddings, extract, extraction_state
from .decay import compute_decay_score
from .scope import resolve_scope
from typing import Optional


LOCK_DIR = Path.home() / ".claude" / "memory" / "locks"


def _lock_path(session_id: str) -> Path:
    """Return the lock file path for a session."""
    safe_id = session_id.replace("/", "_").replace("..", "_")
    return LOCK_DIR / f"{safe_id}.lock"


def acquire_lock(session_id: str) -> bool:
    """
    Try to claim the extraction lock for this session.
    Returns True if we got it (and should proceed), False if already claimed.
    """
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(session_id)
    try:
        # O_CREAT | O_EXCL — atomic create-or-fail
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, datetime.now(timezone.utc).isoformat().encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(session_id: str) -> None:
    """Remove the lock file (e.g., on failure so it can be retried)."""
    try:
        _lock_path(session_id).unlink()
    except FileNotFoundError:
        pass


def cleanup_old_locks(max_age_hours: int = 24) -> None:
    """Remove lock files older than max_age_hours."""
    if not LOCK_DIR.exists():
        return
    now = time.time()
    for lock in LOCK_DIR.iterdir():
        if lock.suffix == ".lock":
            age_hours = (now - lock.stat().st_mtime) / 3600
            if age_hours > max_age_hours:
                try:
                    lock.unlink()
                except OSError:
                    pass


def run_extraction(
    session_id: str,
    transcript_path: str,
    trigger: str,
    cwd: str,
    api_key: str,
    quiet: bool = False,
) -> dict | None:
    """
    Full extraction pipeline: parse transcript, call Claude, embed, store to DB.

    Returns a summary dict on success, None on failure.
    Set quiet=True to suppress stdout (for background/SessionEnd use).
    """
    if not api_key:
        _err("[memory/ingest] ANTHROPIC_API_KEY not set — skipping.")
        return None

    if not transcript_path or not Path(transcript_path).exists():
        _err(f"[memory/ingest] Transcript not found: {transcript_path!r}")
        return None

    # ── Parse transcript ──────────────────────────────────────────────────
    messages = extract.parse_transcript(transcript_path)
    if not messages:
        _err("[memory/ingest] No user/assistant messages in transcript.")
        return None

    conversation_text = extract.build_conversation_text(messages)

    # ── Extract knowledge (with one retry) ────────────────────────────────
    knowledge = None
    for attempt in range(2):
        try:
            knowledge = extract.extract_knowledge(conversation_text, api_key)
            break
        except Exception as exc:
            if attempt == 0:
                _err(f"[memory/ingest] Extraction failed (attempt 1), retrying: {exc}")
                time.sleep(2)
            else:
                _err(f"[memory/ingest] Extraction failed after retry: {exc}")
                return None

    if knowledge is None:
        return None

    # ── Resolve project scope ────────────────────────────────────────────
    scope = resolve_scope(cwd)

    # ── Store to DB ───────────────────────────────────────────────────────
    conn = db.get_connection()

    db.upsert_session(
        conn,
        session_id=session_id,
        trigger=trigger,
        cwd=cwd,
        transcript_path=transcript_path,
        message_count=len(messages),
        summary=knowledge.get("session_summary", ""),
        scope=scope,
    )

    counters = {"facts": 0, "ideas": 0, "entities": 0, "rels": 0, "decisions": 0, "questions": 0, "chunks": 0, "guardrails": 0, "procedures": 0, "error_solutions": 0}

    # Store conversation as small overlapping chunks for precise retrieval
    from .config import CHUNKS_ENABLED
    chunk_ids_list = []
    if CHUNKS_ENABLED:
        from .chunking import split_into_chunks
        windows = split_into_chunks(conversation_text)
        for window_text in windows:
            chunk_emb = embeddings.embed(window_text)
            cid = db.insert_chunk(conn, window_text, session_id, scope, embedding=chunk_emb)
            chunk_ids_list.append(cid)
        counters["chunks"] += len(chunk_ids_list)
    # Use first chunk_id for fact linking (facts link to their session's first chunk)
    chunk_id = chunk_ids_list[0] if chunk_ids_list else None

    # Entities first
    for entity_name in knowledge.get("entities", []):
        emb = embeddings.embed(entity_name)
        db.upsert_entity(conn, entity_name, embedding=emb, scope=scope)
        counters["entities"] += 1

    # Facts
    for fact in knowledge.get("facts", []):
        text = fact.get("text", "").strip()
        if not text:
            continue
        emb = embeddings.embed(text)
        fact_file_paths = fact.get("file_paths", []) if isinstance(fact, dict) else []
        fid, is_new = db.upsert_fact(
            conn,
            text=text,
            category=fact.get("category", "contextual"),
            temporal_class=fact.get("temporal_class", "short"),
            confidence=fact.get("confidence", "medium"),
            embedding=emb,
            session_id=session_id,
            decay_fn=compute_decay_score,
            scope=scope,
            source_chunk_id=chunk_id,
            importance=fact.get("importance", 5) if isinstance(fact, dict) else 5,
            file_paths=fact_file_paths,
            failure_probability=fact.get("failure_probability", 0.0) if isinstance(fact, dict) else 0.0,
        )
        if is_new:
            counters["facts"] += 1
            entity_names = knowledge.get("entities", [])
            db.link_fact_entities(conn, fid, [
                e for e in entity_names if e.lower() in text.lower()
            ])

    # Ideas
    for idea in knowledge.get("ideas", []):
        text = idea.get("text", "").strip()
        if not text:
            continue
        emb = embeddings.embed(text)
        _, is_new = db.upsert_idea(
            conn,
            text=text,
            idea_type=idea.get("type", "insight"),
            temporal_class=idea.get("temporal_class", "short"),
            embedding=emb,
            session_id=session_id,
            decay_fn=compute_decay_score,
            scope=scope,
        )
        if is_new:
            counters["ideas"] += 1

    # Relationships
    for rel in knowledge.get("relationships", []):
        f = rel.get("from", "").strip()
        t = rel.get("to", "").strip()
        if f and t:
            db.upsert_relationship(
                conn,
                from_entity=f,
                to_entity=t,
                rel_type=rel.get("type", "relates_to"),
                description=rel.get("description", ""),
                session_id=session_id,
                scope=scope,
            )
            counters["rels"] += 1

    # Decisions
    for dec in knowledge.get("key_decisions", []):
        text = dec.get("text", dec) if isinstance(dec, dict) else dec
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            continue
        tc = dec.get("temporal_class", "medium") if isinstance(dec, dict) else "medium"
        emb = embeddings.embed(text)
        _, is_new = db.upsert_decision(
            conn,
            text=text,
            temporal_class=tc,
            embedding=emb,
            session_id=session_id,
            decay_fn=compute_decay_score,
            scope=scope,
        )
        if is_new:
            counters["decisions"] += 1

    # Open questions
    for q_text in knowledge.get("open_questions", []):
        q_text = q_text.strip() if isinstance(q_text, str) else ""
        if not q_text:
            continue
        emb = embeddings.embed(q_text)
        _, is_new = db.upsert_question(conn, q_text, emb, session_id, scope=scope)
        if is_new:
            counters["questions"] += 1

    # Guardrails
    for guard in knowledge.get("guardrails", []):
        warning = guard.get("warning", "").strip()
        if not warning:
            continue
        emb = embeddings.embed(warning)
        _, is_new = db.upsert_guardrail(
            conn, warning=warning,
            rationale=guard.get("rationale", ""),
            consequence=guard.get("consequence", ""),
            file_paths=guard.get("file_paths", []),
            line_range=guard.get("line_range", ""),
            embedding=emb, session_id=session_id, scope=scope,
        )
        if is_new:
            counters["guardrails"] += 1

    # Procedures
    for proc in knowledge.get("procedures", []):
        task_desc = proc.get("task_description", "").strip()
        steps = proc.get("steps", "").strip()
        if not task_desc or not steps:
            continue
        emb = embeddings.embed(task_desc)
        _, is_new = db.upsert_procedure(
            conn, task_description=task_desc, steps=steps,
            file_paths=proc.get("file_paths", []),
            embedding=emb, session_id=session_id, scope=scope,
        )
        if is_new:
            counters["procedures"] += 1

    # Error→Solution pairs
    for err in knowledge.get("error_solutions", []):
        error_pattern = err.get("error_pattern", "").strip()
        solution = err.get("solution", "").strip()
        if not error_pattern or not solution:
            continue
        emb = embeddings.embed(error_pattern)
        _, is_new = db.upsert_error_solution(
            conn, error_pattern=error_pattern, solution=solution,
            error_context=err.get("error_context", ""),
            file_paths=err.get("file_paths", []),
            embedding=emb, session_id=session_id, scope=scope,
        )
        if is_new:
            counters["error_solutions"] += 1

    # ── Consolidation (synthesize observations from facts) ────────────────
    consolidation_stats = {"batches": 0, "created": 0, "updated": 0, "deleted": 0}
    from .config import CONSOLIDATION_ENABLED
    if CONSOLIDATION_ENABLED and api_key:
        try:
            from .consolidation import run_consolidation, run_semantic_forgetting, run_coherence_check
            consolidation_stats = run_consolidation(conn, api_key, scope, quiet=quiet)
            if consolidation_stats.get("created", 0) > 0 or consolidation_stats.get("deleted", 0) > 0:
                run_semantic_forgetting(conn, scope)
            # Coherence validation
            coherence_stats = run_coherence_check(conn, scope, quiet=quiet)
            if coherence_stats.get("resolved", 0) > 0 and not quiet:
                _err(f"[memory] Coherence: resolved {coherence_stats['resolved']} contradictions")
        except Exception as exc:
            print(f"[memory] Consolidation failed: {exc}", file=sys.stderr)

    # ── Community summaries ────────────────────────────────────────────────
    if CONSOLIDATION_ENABLED and api_key:
        try:
            from .communities import build_community_summaries
            community_stats = build_community_summaries(conn, api_key, scope, quiet=quiet)
            if community_stats.get("summaries_created", 0) > 0 and not quiet:
                _err(f"[memory] Communities: {community_stats['summaries_created']} summaries created")
        except Exception as exc:
            print(f"[memory] Community summaries failed: {exc}", file=sys.stderr)

    # ── Decay pass ────────────────────────────────────────────────────────
    decay_stats = db.apply_decay_pass(conn)

    # ── Purge old soft-deleted items (>30 days) ──────────────────────────
    purge_stats = db.purge_deleted(conn)

    # ── Rebuild FTS indexes ───────────────────────────────────────────────
    db.rebuild_fts_indexes(conn)

    stats = db.get_stats(conn)
    conn.close()

    # ── Cleanup old locks ─────────────────────────────────────────────────
    cleanup_old_locks()

    # ── Output ────────────────────────────────────────────────────────────
    summary = {
        "session_id": session_id,
        "trigger": trigger,
        "counters": counters,
        "decay_stats": decay_stats,
        "db_stats": stats,
        "session_summary": knowledge.get("session_summary", ""),
    }

    if not quiet:
        _print_summary(summary)

    _err(f"[memory] Extracted via {trigger}: +{counters['facts']} facts, "
         f"+{counters['ideas']} ideas, +{counters['entities']} entities")

    return summary


# ══════════════════════════════════════════════════════════════════════════
# Incremental extraction pipeline
# ══════════════════════════════════════════════════════════════════════════

def _store_structured_items(
    conn,
    knowledge: dict,
    session_id: str,
    scope: str,
    skip_embeddings: Optional[list[list[float]]] = None,
    dedup_threshold: float = 0.92,
) -> tuple[dict, dict[str, list[str]]]:
    """
    Embed and upsert structured items (entities, facts, ideas, decisions, questions).

    If skip_embeddings is provided, any new item with cosine >= dedup_threshold
    against those embeddings is treated as a near-duplicate and not inserted
    (the existing upsert handles reinforcement).

    Returns (counters_dict, new_item_ids_dict).
    """
    counters = {"facts": 0, "ideas": 0, "entities": 0, "rels": 0, "decisions": 0, "questions": 0, "guardrails": 0, "procedures": 0, "error_solutions": 0}
    new_ids: dict[str, list[str]] = {"facts": [], "ideas": [], "decisions": []}

    def _should_skip(emb):
        """Check if this embedding is too similar to a known prior/recalled item."""
        if not skip_embeddings or emb is None:
            return False
        for skip_emb in skip_embeddings:
            if skip_emb is None:
                continue
            sim = _cosine(emb, skip_emb)
            if sim >= dedup_threshold:
                return True
        return False

    # Entities
    for entity_name in knowledge.get("entities", []):
        emb = embeddings.embed(entity_name)
        db.upsert_entity(conn, entity_name, embedding=emb, scope=scope)
        counters["entities"] += 1

    # Facts
    for fact in knowledge.get("facts", []):
        text = fact.get("text", "").strip()
        if not text:
            continue
        emb = embeddings.embed(text)
        if _should_skip(emb):
            continue
        fact_file_paths = fact.get("file_paths", []) if isinstance(fact, dict) else []
        fid, is_new = db.upsert_fact(
            conn, text=text,
            category=fact.get("category", "contextual"),
            temporal_class=fact.get("temporal_class", "short"),
            confidence=fact.get("confidence", "medium"),
            embedding=emb, session_id=session_id,
            decay_fn=compute_decay_score, scope=scope,
            importance=fact.get("importance", 5) if isinstance(fact, dict) else 5,
            file_paths=fact_file_paths,
            failure_probability=fact.get("failure_probability", 0.0) if isinstance(fact, dict) else 0.0,
        )
        if is_new:
            counters["facts"] += 1
            new_ids["facts"].append(fid)
            entity_names = knowledge.get("entities", [])
            db.link_fact_entities(conn, fid, [
                e for e in entity_names if e.lower() in text.lower()
            ])

    # Ideas
    for idea in knowledge.get("ideas", []):
        text = idea.get("text", "").strip()
        if not text:
            continue
        emb = embeddings.embed(text)
        if _should_skip(emb):
            continue
        iid, is_new = db.upsert_idea(
            conn, text=text,
            idea_type=idea.get("type", "insight"),
            temporal_class=idea.get("temporal_class", "short"),
            embedding=emb, session_id=session_id,
            decay_fn=compute_decay_score, scope=scope,
        )
        if is_new:
            counters["ideas"] += 1
            new_ids["ideas"].append(iid)

    # Relationships
    for rel in knowledge.get("relationships", []):
        f = rel.get("from", "").strip()
        t = rel.get("to", "").strip()
        if f and t:
            db.upsert_relationship(
                conn, from_entity=f, to_entity=t,
                rel_type=rel.get("type", "relates_to"),
                description=rel.get("description", ""),
                session_id=session_id, scope=scope,
            )
            counters["rels"] += 1

    # Decisions
    for dec in knowledge.get("key_decisions", []):
        text = dec.get("text", dec) if isinstance(dec, dict) else dec
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            continue
        tc = dec.get("temporal_class", "medium") if isinstance(dec, dict) else "medium"
        emb = embeddings.embed(text)
        if _should_skip(emb):
            continue
        did, is_new = db.upsert_decision(
            conn, text=text, temporal_class=tc,
            embedding=emb, session_id=session_id,
            decay_fn=compute_decay_score, scope=scope,
        )
        if is_new:
            counters["decisions"] += 1
            new_ids["decisions"].append(did)

    # Open questions
    for q_text in knowledge.get("open_questions", []):
        q_text = q_text.strip() if isinstance(q_text, str) else ""
        if not q_text:
            continue
        emb = embeddings.embed(q_text)
        _, is_new = db.upsert_question(conn, q_text, emb, session_id, scope=scope)
        if is_new:
            counters["questions"] += 1

    # Guardrails
    for guard in knowledge.get("guardrails", []):
        warning = guard.get("warning", "").strip()
        if not warning:
            continue
        emb = embeddings.embed(warning)
        if _should_skip(emb):
            continue
        _, is_new = db.upsert_guardrail(
            conn, warning=warning,
            rationale=guard.get("rationale", ""),
            consequence=guard.get("consequence", ""),
            file_paths=guard.get("file_paths", []),
            line_range=guard.get("line_range", ""),
            embedding=emb, session_id=session_id, scope=scope,
        )
        if is_new:
            counters["guardrails"] += 1

    # Procedures
    for proc in knowledge.get("procedures", []):
        task_desc = proc.get("task_description", "").strip()
        steps = proc.get("steps", "").strip()
        if not task_desc or not steps:
            continue
        emb = embeddings.embed(task_desc)
        if _should_skip(emb):
            continue
        _, is_new = db.upsert_procedure(
            conn, task_description=task_desc, steps=steps,
            file_paths=proc.get("file_paths", []),
            embedding=emb, session_id=session_id, scope=scope,
        )
        if is_new:
            counters["procedures"] += 1

    # Error→Solution pairs
    for err in knowledge.get("error_solutions", []):
        error_pattern = err.get("error_pattern", "").strip()
        solution = err.get("solution", "").strip()
        if not error_pattern or not solution:
            continue
        emb = embeddings.embed(error_pattern)
        if _should_skip(emb):
            continue
        _, is_new = db.upsert_error_solution(
            conn, error_pattern=error_pattern, solution=solution,
            error_context=err.get("error_context", ""),
            file_paths=err.get("file_paths", []),
            embedding=emb, session_id=session_id, scope=scope,
        )
        if is_new:
            counters["error_solutions"] += 1

    return counters, new_ids


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def run_incremental_extraction(
    session_id: str,
    transcript_path: str,
    trigger: str,
    cwd: str,
    api_key: str,
    quiet: bool = False,
    is_final: bool = False,
    session_recall_items: Optional[list[dict]] = None,
) -> dict | None:
    """
    Incremental extraction pipeline: parse delta, call Claude, embed, store.

    Processes only the conversation segment since the last extraction pass.
    Each pass receives:
      - The prior pass's narrative summary
      - Relevant existing DB items (for cross-session superseding)
      - IDs of items already extracted this session (to avoid re-extraction)

    When is_final=True, finalizes narratives and marks extraction complete.

    Returns a summary dict on success, None on failure.
    """
    if not api_key:
        _err("[memory/ingest] ANTHROPIC_API_KEY not set — skipping.")
        return None

    if not transcript_path or not Path(transcript_path).exists():
        _err(f"[memory/ingest] Transcript not found: {transcript_path!r}")
        return None

    # ── Load state ──────────────────────────────────────────────────────
    state = extraction_state.load_state(session_id)
    pass_count = state["pass_count"] if state else 0
    byte_offset = state["last_byte_offset"] if state else 0
    prior_narrative = state["last_narrative"] if state else None
    prior_item_ids = state["prior_item_ids"] if state else {"facts": [], "ideas": [], "decisions": []}
    recalled_item_ids = state.get("recalled_item_ids", []) if state else []

    # ── Parse delta ──────────────────────────────────────────────────────
    messages, new_offset = extract.parse_transcript_delta(transcript_path, byte_offset)
    if not messages:
        _err("[memory/ingest] No new messages in delta.")
        if is_final and pass_count > 0:
            # Still finalize even if no new content
            conn = db.get_connection()
            scope = resolve_scope(cwd)
            db.finalize_narratives(conn, session_id)
            conn.close()
            extraction_state.mark_extraction_complete(session_id)
            extraction_state.delete_state(session_id)
            extraction_state.delete_recall_cache(session_id)
        return None

    # ── Quality gate ─────────────────────────────────────────────────────
    if not is_final and not extract.is_delta_substantial(messages):
        _err(f"[memory/ingest] Delta not substantial ({len(messages)} msgs), deferring.")
        # Save updated offset so next pass includes this content
        new_state = {
            "session_id": session_id,
            "pass_count": pass_count,
            "last_byte_offset": new_offset,
            "last_narrative": prior_narrative or "",
            "prior_item_ids": prior_item_ids,
            "recalled_item_ids": recalled_item_ids,
            "last_pass_at": datetime.now(timezone.utc).isoformat(),
        }
        extraction_state.save_state(session_id, new_state)
        return None

    # ── Build delta text ─────────────────────────────────────────────────
    delta_text = extract.build_conversation_text(messages)

    # ── Resolve scope ────────────────────────────────────────────────────
    scope = resolve_scope(cwd)

    # ── Gather existing items for superseding (read-only connection) ─────
    existing_items = []
    ro_conn = db.get_connection(read_only=True)
    try:
        if pass_count == 0 and session_recall_items:
            # Pass 1: use cached session-start recall items
            existing_items = session_recall_items
        elif pass_count == 0:
            # Pass 1 without cache: load from recall cache file
            cached = extraction_state.load_recall_cache(session_id)
            if cached:
                existing_items = cached
            else:
                # Fallback: embed first 1000 chars and search
                snippet = delta_text[:4000]  # ~1000 tokens
                snippet_emb = embeddings.embed(snippet)
                if snippet_emb:
                    facts = db.search_facts(ro_conn, snippet_emb, limit=5, threshold=RECALL_THRESHOLD, scope=scope)
                    decisions = db.search_facts(ro_conn, snippet_emb, limit=3, threshold=RECALL_THRESHOLD, scope=scope)
                    existing_items = [
                        {"id": r["id"], "text": r["text"], "table": "facts"} for r in facts
                    ] + [
                        {"id": r["id"], "text": r["text"], "table": "decisions"} for r in decisions
                    ]
        else:
            # Pass 2+: use prior narrative to find relevant existing items
            if prior_narrative:
                narrative_emb = embeddings.embed(prior_narrative)
                if narrative_emb:
                    facts = db.search_facts(ro_conn, narrative_emb, limit=5, threshold=RECALL_THRESHOLD, scope=scope)
                    decs = db.search_facts(ro_conn, narrative_emb, limit=3, threshold=RECALL_THRESHOLD, scope=scope)
                    existing_items = [
                        {"id": r["id"], "text": r["text"], "table": "facts"} for r in facts
                    ] + [
                        {"id": r["id"], "text": r["text"], "table": "decisions"} for r in decs
                    ]

        # ── Gather prior pass items ──────────────────────────────────────
        prior_items = []
        if any(prior_item_ids.get(t) for t in ("facts", "ideas", "decisions")):
            prior_items = db.get_items_by_ids(ro_conn, prior_item_ids)
    finally:
        ro_conn.close()

    # ── Build skip embeddings for cross-pass dedup ───────────────────────
    skip_embeddings = []
    for item in prior_items + existing_items:
        emb = embeddings.embed(item["text"])
        if emb:
            skip_embeddings.append(emb)

    # ── Extract knowledge (no DB lock held during API call) ──────────────
    knowledge = None
    for attempt in range(2):
        try:
            knowledge = extract.extract_knowledge_incremental(
                delta_text=delta_text,
                api_key=api_key,
                prior_narrative=prior_narrative,
                existing_items=existing_items if existing_items else None,
                prior_items=prior_items if prior_items else None,
            )
            break
        except Exception as exc:
            if attempt == 0:
                _err(f"[memory/ingest] Incremental extraction failed (attempt 1), retrying: {exc}")
                time.sleep(2)
            else:
                _err(f"[memory/ingest] Incremental extraction failed after retry: {exc}")
                return None

    if knowledge is None:
        return None

    # ── Open write connection only for the storage phase ──────────────────
    conn = db.get_connection()

    # ── Upsert session ───────────────────────────────────────────────────
    db.upsert_session(
        conn, session_id=session_id, trigger=f"{trigger}/pass{pass_count+1}",
        cwd=cwd, transcript_path=transcript_path,
        message_count=len(messages), summary=knowledge.get("narrative_summary", ""),
        scope=scope,
    )

    # ── Process supersedes ───────────────────────────────────────────────
    supersede_count = 0
    for sup in knowledge.get("supersedes", []):
        old_id = sup.get("old_id", "")
        old_table = sup.get("old_table", "")
        reason = sup.get("reason", "")
        if old_id and old_table:
            # Find the best matching new item to link as the replacement
            # For now, use the first new item as a placeholder
            new_id = ""  # Will be set after storing items
            if db.supersede_item(conn, old_id, old_table, new_id, reason):
                supersede_count += 1

    # ── Store structured items ───────────────────────────────────────────
    counters, new_ids = _store_structured_items(
        conn, knowledge, session_id, scope,
        skip_embeddings=skip_embeddings,
        dedup_threshold=CROSS_PASS_DEDUP_THRESHOLD,
    )

    # ── Store narrative ──────────────────────────────────────────────────
    narrative = knowledge.get("narrative_summary", "")
    narrative_emb = embeddings.embed(narrative) if narrative else None
    db.upsert_narrative(
        conn, session_id, pass_count + 1, narrative,
        embedding=narrative_emb, is_final=is_final, scope=scope,
    )

    # ── Update state ─────────────────────────────────────────────────────
    # Merge new item IDs with prior
    merged_ids = {
        "facts": prior_item_ids.get("facts", []) + new_ids.get("facts", []),
        "ideas": prior_item_ids.get("ideas", []) + new_ids.get("ideas", []),
        "decisions": prior_item_ids.get("decisions", []) + new_ids.get("decisions", []),
    }
    new_state = {
        "session_id": session_id,
        "pass_count": pass_count + 1,
        "last_byte_offset": new_offset,
        "last_narrative": narrative,
        "prior_item_ids": merged_ids,
        "recalled_item_ids": [item["id"] for item in existing_items],
        "last_pass_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Finalize if this is the last pass ────────────────────────────────
    if is_final:
        db.finalize_narratives(conn, session_id)
        # Consolidation
        from .config import CONSOLIDATION_ENABLED
        if CONSOLIDATION_ENABLED and api_key:
            try:
                from .consolidation import run_consolidation, run_semantic_forgetting, run_coherence_check
                c_stats = run_consolidation(conn, api_key, scope, quiet=quiet)
                if c_stats.get("created", 0) > 0 or c_stats.get("deleted", 0) > 0:
                    run_semantic_forgetting(conn, scope)
                run_coherence_check(conn, scope, quiet=quiet)
            except Exception as exc:
                print(f"[memory] Consolidation failed: {exc}", file=sys.stderr)
            try:
                from .communities import build_community_summaries
                build_community_summaries(conn, api_key, scope, quiet=quiet)
            except Exception as exc:
                print(f"[memory] Community summaries failed: {exc}", file=sys.stderr)
        decay_stats = db.apply_decay_pass(conn)
        db.purge_deleted(conn)
        db.rebuild_fts_indexes(conn)
        conn.close()
        extraction_state.mark_extraction_complete(session_id)
        extraction_state.delete_state(session_id)
        extraction_state.delete_recall_cache(session_id)
        cleanup_old_locks()
    else:
        extraction_state.save_state(session_id, new_state)
        decay_stats = {"updated": 0, "forgotten": 0, "promoted": 0}
        conn.close()

    # ── Output ───────────────────────────────────────────────────────────
    summary = {
        "session_id": session_id,
        "trigger": trigger,
        "pass_number": pass_count + 1,
        "is_final": is_final,
        "counters": counters,
        "superseded": supersede_count,
        "narrative": narrative,
        "decay_stats": decay_stats,
    }

    if not quiet:
        _err(f"[memory] Pass {pass_count+1} via {trigger}: "
             f"+{counters['facts']} facts, +{counters['ideas']} ideas, "
             f"{supersede_count} superseded, final={is_final}")

    return summary


def _print_summary(summary: dict) -> None:
    """Print formatted summary to stdout."""
    counters = summary["counters"]
    stats = summary["db_stats"]
    decay_stats = summary["decay_stats"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 62)
    print(f"  MEMORY SNAPSHOT  |  {ts}")
    print(f"  Session: {summary['session_id'][:16]}...   Trigger: {summary['trigger']}")
    print("=" * 62)
    print(f"\n  {summary['session_summary']}\n")
    print(f"  Stored this session:")
    print(f"    Facts       +{counters['facts']:>3}   (total active: {stats['facts']['total']})")
    print(f"    Ideas       +{counters['ideas']:>3}   (total active: {stats['ideas']['total']})")
    print(f"    Entities    +{counters['entities']:>3}   (total: {stats['entities']['total']})")
    print(f"    Relations   +{counters['rels']:>3}   (total active: {stats['relationships']['total']})")
    print(f"    Decisions   +{counters['decisions']:>3}   (total active: {stats['decisions']['total']})")
    print(f"    Questions   +{counters['questions']:>3}")
    print(f"\n  Knowledge base by temporal class:")
    print(f"    Long  : {stats['facts']['long']:>4} facts  (permanent memory)")
    print(f"    Medium: {stats['facts']['medium']:>4} facts  (working memory)")
    print(f"    Short : {stats['facts']['short']:>4} facts  (session cache)")
    print(f"\n  Decay pass: {decay_stats['updated']} updated, "
          f"{decay_stats['forgotten']} forgotten, "
          f"{decay_stats['promoted']} promoted")
    print(f"\n  DB: {DB_PATH}")
    print("=" * 62)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)
