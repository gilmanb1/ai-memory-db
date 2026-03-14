#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
# ]
# ///
"""
user_prompt_submit.py — Claude Code UserPromptSubmit hook.

Two modes:
  1. /remember <text>  — Store a fact/decision to long-term memory immediately
  2. Normal prompt     — Recall relevant context via semantic search

/remember prefixes:
  /remember <text>               → store as long-term fact in current project scope
  /remember global: <text>       → store as long-term fact in global scope
  /remember decision: <text>     → store as long-term decision in current project scope
  /remember global decision: <text> → store as long-term decision in global scope

Output (stdout) — JSON with an `additionalContext` field (or nothing).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))


def _handle_remember(prompt_text: str, payload: dict) -> None:
    """Parse and store a /remember command."""
    from memory import db, embeddings
    from memory.scope import resolve_scope
    from memory.config import GLOBAL_SCOPE
    from memory.decay import compute_decay_score

    # Strip the /remember prefix
    text = prompt_text[len("/remember"):].strip()
    if not text:
        print(json.dumps({"additionalContext":
            "## /remember\nUsage: `/remember <text>` to store a fact. "
            "Prefixes: `global:` (shared across projects), `decision:` (store as decision). "
            "Example: `/remember global: My name is Ben`"
        }))
        return

    # Parse prefixes
    is_global = False
    is_decision = False

    # Check for "global:" or "global decision:" prefixes (order-independent)
    prefix_pattern = re.compile(r'^((?:global|decision)[:\s]+)+', re.IGNORECASE)
    match = prefix_pattern.match(text)
    if match:
        prefix = match.group(0).lower()
        is_global = "global" in prefix
        is_decision = "decision" in prefix
        text = text[match.end():].strip()

    if not text:
        print(json.dumps({"additionalContext": "## /remember\nNo content provided after prefix."}))
        return

    # Resolve scope
    cwd = payload.get("cwd", "")
    scope = GLOBAL_SCOPE if is_global else (resolve_scope(cwd) if cwd else GLOBAL_SCOPE)
    session_id = payload.get("session_id", "manual")

    # Embed and store
    emb = embeddings.embed(text)
    conn = db.get_connection()

    if is_decision:
        item_id, is_new = db.upsert_decision(
            conn, text=text, temporal_class="long", embedding=emb,
            session_id=session_id, decay_fn=compute_decay_score, scope=scope,
        )
        item_type = "decision"
    else:
        item_id, is_new = db.upsert_fact(
            conn, text=text, category="personal", temporal_class="long",
            confidence="high", embedding=emb, session_id=session_id,
            decay_fn=compute_decay_score, scope=scope,
        )
        item_type = "fact"

    conn.close()

    # Build confirmation
    action = "Stored" if is_new else "Reinforced"
    scope_label = "global" if scope == GLOBAL_SCOPE else Path(scope).name
    confirmation = (
        f"## Memory Stored\n"
        f"- **{action}** {item_type}: {text}\n"
        f"- **Scope:** {scope_label}\n"
        f"- **Temporal class:** long\n"
        f"- **ID:** `{item_id[:12]}...`"
    )

    print(json.dumps({"additionalContext": confirmation}))
    print(f"[memory] {action} {item_type} ({scope_label}): {text[:80]}", file=sys.stderr)


def main(payload: dict) -> None:
    prompt_text = payload.get("prompt", "").strip()

    # ── /remember command ─────────────────────────────────────────────────
    if prompt_text.lower().startswith("/remember"):
        try:
            _handle_remember(prompt_text, payload)
        except Exception as exc:
            print(f"[memory] /remember failed: {exc}", file=sys.stderr)
            print(json.dumps({"additionalContext": f"## /remember\nFailed to store: {exc}"}))
        return

    # ── Normal recall flow ────────────────────────────────────────────────
    from memory.config import DB_PATH
    if not DB_PATH.exists():
        sys.exit(0)

    if not prompt_text or len(prompt_text) < 10:
        sys.exit(0)

    from memory import embeddings
    query_embedding = embeddings.embed(prompt_text)
    if query_embedding is None:
        sys.exit(0)

    try:
        from memory import db, recall
        from memory.scope import resolve_scope
        cwd = payload.get("cwd", "")
        scope = resolve_scope(cwd) if cwd else None
        conn = db.get_connection(read_only=True)
        context = recall.prompt_recall(conn, query_embedding, prompt_text, scope=scope)
        conn.close()
    except Exception as exc:
        print(f"[user_prompt] Recall failed: {exc}", file=sys.stderr)
        sys.exit(0)

    additional_context = recall.format_prompt_context(context)
    if not additional_context:
        sys.exit(0)

    print(json.dumps({"additionalContext": additional_context}))

    n_facts = len(context.get("facts", []))
    n_rels = len(context.get("relationships", []))
    entities_hit = context.get("entities_hit", [])
    print(
        f"[memory] Recalled {n_facts} facts, {n_rels} relations"
        + (f" (entities: {', '.join(entities_hit[:5])})" if entities_hit else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        payload = {}
    main(payload)
