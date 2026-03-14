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

from .config import DB_PATH, GLOBAL_SCOPE
from . import db, embeddings, extract
from .decay import compute_decay_score
from .scope import resolve_scope


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

    counters = {"facts": 0, "ideas": 0, "entities": 0, "rels": 0, "decisions": 0, "questions": 0}

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

    # ── Decay pass ────────────────────────────────────────────────────────
    decay_stats = db.apply_decay_pass(conn)
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
