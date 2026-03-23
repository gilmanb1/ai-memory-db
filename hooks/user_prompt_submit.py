#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.1.0",
#   "numpy>=1.24.0",
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
    """Parse and store a /remember command.

    Supports prefixes:
      /remember <text>                     → project-scoped long-term fact
      /remember global: <text>             → global fact
      /remember decision: <text>           → project-scoped decision
      /remember global decision: <text>    → global decision
      /remember guardrail: <text>          → guardrail (warning)
      /remember procedure: <text>          → procedure (how-to)
      /remember error: <pattern> -> <fix>  → error→solution pair
    """
    from memory import db, embeddings
    from memory.scope import resolve_scope
    from memory.config import GLOBAL_SCOPE
    from memory.decay import compute_decay_score

    # Strip the /remember prefix
    text = prompt_text[len("/remember"):].strip()
    if not text:
        print(json.dumps({"additionalContext":
            "## /remember\nUsage: `/remember <text>` to store a fact.\n"
            "Prefixes: `global:`, `decision:`, `guardrail:`, `procedure:`, `error:`\n"
            "Examples:\n"
            "- `/remember global: My name is Ben`\n"
            "- `/remember guardrail: Don't use ORM for reports — raw SQL is 10x faster`\n"
            "- `/remember procedure: Deploy: 1. Run tests 2. Build Docker 3. Push`\n"
            "- `/remember error: ImportError onnxruntime -> pip install onnxruntime-silicon`"
        }))
        return

    # Parse prefixes
    is_global = False
    is_decision = False
    is_guardrail = False
    is_procedure = False
    is_error = False

    prefix_pattern = re.compile(
        r'^((?:global|decision|guardrail|procedure|error)[:\s]+)+', re.IGNORECASE
    )
    match = prefix_pattern.match(text)
    if match:
        prefix = match.group(0).lower()
        is_global = "global" in prefix
        is_decision = "decision" in prefix
        is_guardrail = "guardrail" in prefix
        is_procedure = "procedure" in prefix
        is_error = "error" in prefix
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
    try:
        if is_guardrail:
            # Parse "warning — rationale" or just "warning"
            parts = re.split(r'\s*[—–-]{1,3}\s*', text, maxsplit=1)
            warning = parts[0].strip()
            rationale = parts[1].strip() if len(parts) > 1 else ""
            item_id, is_new = db.upsert_guardrail(
                conn, warning=warning, rationale=rationale,
                embedding=emb, session_id=session_id, scope=scope,
            )
            item_type = "guardrail"
        elif is_procedure:
            # Parse "task: steps" or "task — steps"
            parts = re.split(r'\s*[:\—–]{1,3}\s*', text, maxsplit=1)
            task_desc = parts[0].strip()
            steps = parts[1].strip() if len(parts) > 1 else text
            item_id, is_new = db.upsert_procedure(
                conn, task_description=task_desc, steps=steps,
                embedding=emb, session_id=session_id, scope=scope,
            )
            item_type = "procedure"
        elif is_error:
            # Parse "error pattern -> solution"
            parts = re.split(r'\s*->\s*', text, maxsplit=1)
            error_pattern = parts[0].strip()
            solution = parts[1].strip() if len(parts) > 1 else "See conversation for fix"
            item_id, is_new = db.upsert_error_solution(
                conn, error_pattern=error_pattern, solution=solution,
                embedding=emb, session_id=session_id, scope=scope,
            )
            item_type = "error_solution"
        elif is_decision:
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
    finally:
        conn.close()

    # Build confirmation
    action = "Stored" if is_new else "Reinforced"
    scope_label = "global" if scope == GLOBAL_SCOPE else Path(scope).name
    confirmation = (
        f"## Memory Stored\n"
        f"- **{action}** {item_type}: {text}\n"
        f"- **Scope:** {scope_label}\n"
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

    # Fast path: check pre-fetch cache from status_line
    additional_context = None
    session_id = payload.get("session_id", "")
    if session_id:
        try:
            from memory.extraction_state import load_prefetch
            cached = load_prefetch(session_id, max_age_s=60.0)
            if cached and cached.get("embedding") and cached.get("context"):
                # Check if prompt is similar to pre-fetched context
                from memory import embeddings
                query_embedding = embeddings.embed(prompt_text)
                if query_embedding:
                    import numpy as _np
                    sim = float(_np.dot(query_embedding, cached["embedding"]) / (
                        _np.linalg.norm(query_embedding) * _np.linalg.norm(cached["embedding"]) + 1e-9
                    ))
                    if sim > 0.75:
                        additional_context = cached["context"]
                        print(f"[memory] Pre-fetch cache hit (sim={sim:.2f})", file=sys.stderr)
        except Exception:
            pass

    if additional_context is None:
        from memory import embeddings
        query_embedding = embeddings.embed(prompt_text)
        if query_embedding is None:
            sys.exit(0)

        conn = None
        context = {}
        try:
            from memory import db, recall
            from memory.scope import resolve_scope
            cwd = payload.get("cwd", "")
            scope = resolve_scope(cwd) if cwd else None
            conn = db.get_connection(read_only=True)
            context = recall.prompt_recall(conn, query_embedding, prompt_text, scope=scope)
        except Exception as exc:
            print(f"[user_prompt] Recall failed: {exc}", file=sys.stderr)
            sys.exit(0)
        finally:
            if conn is not None:
                conn.close()

        # ── Track recalled items (outcome scoring) ─────────────────────
        try:
            recalled_ids: dict[str, list[str]] = {}
            for key in ("facts", "ideas", "observations", "guardrails", "procedures", "error_solutions"):
                ids = [item.get("id") for item in context.get(key, []) if isinstance(item, dict) and item.get("id")]
                if ids:
                    recalled_ids[key] = ids
            if recalled_ids:
                rw_conn = db.get_connection()
                try:
                    db.increment_recalled(rw_conn, recalled_ids)
                finally:
                    rw_conn.close()
        except Exception:
            pass  # Non-critical

        additional_context = recall.format_prompt_context(context)
    if not additional_context:
        sys.exit(0)

    print(json.dumps({"additionalContext": additional_context}))

    n_facts = len(context.get("facts", []))
    n_rels = len(context.get("relationships", []))
    n_guards = len(context.get("guardrails", []))
    entities_hit = context.get("entities_hit", [])
    parts = [f"[memory] Recalled {n_facts} facts, {n_rels} relations"]
    if n_guards:
        parts[0] += f", {n_guards} guardrails"
    if entities_hit:
        parts[0] += f" (entities: {', '.join(entities_hit[:5])})"
    print(parts[0], file=sys.stderr)


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        payload = {}
    main(payload)
