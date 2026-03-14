#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
# ]
# ///
"""
user_prompt_submit.py — Claude Code UserPromptSubmit hook.

Fires before Claude processes each user message.
Embeds the user's prompt via Ollama, queries the memory database for
semantically relevant facts, ideas, relationships, and open questions,
and injects them as `additionalContext`.

This hook is prompt-specific: it recalls narrower, more targeted knowledge
than the session-level systemMessage set by session_start.py.

Output (stdout) — JSON with an `additionalContext` field (or nothing).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))


def main(payload: dict) -> None:
    from memory.config import DB_PATH
    if not DB_PATH.exists():
        sys.exit(0)

    prompt_text = payload.get("prompt", "").strip()
    if not prompt_text or len(prompt_text) < 10:
        # Very short prompts (e.g. "ok", "yes") — not worth embedding
        sys.exit(0)

    # Embed the prompt
    from memory import embeddings
    query_embedding = embeddings.embed(prompt_text)
    if query_embedding is None:
        # Ollama down — skip gracefully; session context already covers things
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

    # Brief diagnostic
    n_facts = len(context.get("facts", []))
    n_rels  = len(context.get("relationships", []))
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
