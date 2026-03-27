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
    truncation_stats = {"included": 0, "truncated": 0, "section_counts": {}}
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

        # ── Save recall log for /recalled command ──────────────────────
        try:
            from memory.config import MEMORY_DIR
            recall_log = {
                "session_id": session_id,
                "prompt": prompt_text[:200],
                "facts": [{"id": f.get("id","")[:12], "text": f.get("text","")[:120], "score": round(f.get("score",0), 3), "temporal_class": f.get("temporal_class",""), "scope": f.get("scope","")} for f in context.get("facts", []) if isinstance(f, dict)],
                "guardrails": [{"id": g.get("id","")[:12], "text": g.get("warning","")[:120], "scope": g.get("scope","")} for g in context.get("guardrails", []) if isinstance(g, dict)],
                "procedures": [{"id": p.get("id","")[:12], "text": p.get("task_description","")[:120]} for p in context.get("procedures", []) if isinstance(p, dict)],
                "error_solutions": [{"id": e.get("id","")[:12], "text": e.get("error_pattern","")[:120]} for e in context.get("error_solutions", []) if isinstance(e, dict)],
                "observations": [{"id": o.get("id","")[:12], "text": o.get("text","")[:120]} for o in context.get("observations", []) if isinstance(o, dict)],
                "relationships": [f"{r.get('from','')} --[{r.get('rel_type','')}]--> {r.get('to','')}" for r in context.get("relationships", []) if isinstance(r, dict)],
                "entities_hit": context.get("entities_hit", []),
                "code_context": [{"file": c.get("file_path",""), "symbols": len(c.get("symbols", []))} for c in context.get("code_context", []) if isinstance(c, dict)],
            }
            recall_log_path = MEMORY_DIR / "last_recall.json"
            import json as _json
            recall_log_path.write_text(_json.dumps(recall_log, indent=2))
        except Exception:
            pass  # Non-critical

        # ── Check for user corrections to previously recalled facts ───
        correction_msg = ""
        try:
            from memory.config import CORRECTION_DETECTION_ENABLED
            if CORRECTION_DETECTION_ENABLED:
                from memory.corrections import detect_correction, resolve_correction, apply_correction
                detection = detect_correction(prompt_text)
                if detection:
                    # Load previous recall items
                    from memory.config import MEMORY_DIR
                    prev_recall_path = MEMORY_DIR / "last_recall.json"
                    if prev_recall_path.exists():
                        import json as _json2
                        prev_data = _json2.loads(prev_recall_path.read_text())
                        prev_items = prev_data.get("facts", []) + prev_data.get("guardrails", [])
                        prev_items += prev_data.get("procedures", []) + prev_data.get("error_solutions", [])
                        if prev_items:
                            resolved = resolve_correction(prompt_text, prev_items)
                            if resolved:
                                rw_conn2 = db.get_connection()
                                try:
                                    from memory.scope import resolve_scope as _rs
                                    _scope = _rs(payload.get("cwd", "")) if payload.get("cwd") else "__global__"
                                    success = apply_correction(rw_conn2, resolved, session_id, _scope)
                                    if success:
                                        correction_msg = (
                                            f"\n\n## Memory Corrected\n"
                                            f"- Superseded: `{resolved['old_item_id'][:12]}...`\n"
                                            f"- New fact: {resolved['new_text']}\n"
                                        )
                                        print(f"[memory] Correction applied: superseded {resolved['old_item_id'][:12]}", file=sys.stderr)
                                finally:
                                    rw_conn2.close()
        except Exception as exc:
            print(f"[memory] Correction detection failed: {exc}", file=sys.stderr)

        additional_context, truncation_stats = recall.format_prompt_context(context)
    if not additional_context and not correction_msg:
        sys.exit(0)

    output_context = (additional_context or "") + correction_msg
    print(json.dumps({"additionalContext": output_context}))

    n_facts = len(context.get("facts", []))
    n_rels = len(context.get("relationships", []))
    n_guards = len(context.get("guardrails", []))
    entities_hit = context.get("entities_hit", [])
    parts = [f"[memory] Recalled {n_facts} facts, {n_rels} relations"]
    if n_guards:
        parts[0] += f", {n_guards} guardrails"
    if entities_hit:
        parts[0] += f" (entities: {', '.join(entities_hit[:5])})"
    if truncation_stats.get("truncated", 0) > 0:
        parts[0] += f" [truncated {truncation_stats['truncated']} items]"
    print(parts[0], file=sys.stderr)


if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        payload = {}
    main(payload)
