#!/usr/bin/env python3
"""
live_bench.py — Benchmark against the live ~/.claude/memory/knowledge.duckdb

Measures real-world performance with actual accumulated knowledge.
No API calls needed — reads existing data only.

Usage: python3 bench/live_bench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Use the DEV memory package (has latest code) but point at the LIVE database
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory import db, recall, embeddings
from memory.config import (
    DB_PATH, SESSION_TOKEN_BUDGET, PROMPT_TOKEN_BUDGET,
    CHARS_PER_TOKEN, GLOBAL_SCOPE,
)


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run():
    if not DB_PATH.exists():
        print(f"No database at {DB_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("  LIVE MEMORY BENCHMARK")
    print(f"  DB: {DB_PATH} ({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 60)

    # Open read-write first to apply any pending migrations, then read-only
    rw_conn = db.get_connection(read_only=False)
    rw_conn.close()
    conn = db.get_connection(read_only=True)
    stats = db.get_stats(conn)

    # ── Database overview ────────────────────────────────────────────
    section("DATABASE CONTENTS")
    for key in ("facts", "ideas", "entities", "relationships", "decisions", "questions", "sessions"):
        val = stats.get(key, {})
        total = val.get("total", val) if isinstance(val, dict) else val
        extra = ""
        if key == "facts" and isinstance(val, dict):
            extra = f" (long={val.get('long',0)}, medium={val.get('medium',0)}, short={val.get('short',0)}, inactive={val.get('inactive',0)})"
        print(f"  {key:20s} {total:>5}{extra}")

    # Check for new tables
    for key in ("observations", "guardrails", "procedures", "error_solutions"):
        val = stats.get(key, {})
        total = val.get("total", 0) if isinstance(val, dict) else 0
        print(f"  {key:20s} {total:>5}")

    # ── Schema version ───────────────────────────────────────────────
    section("SCHEMA VERSION")
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()]
    print(f"  Migrations applied: {versions}")
    tables = sorted(r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall())
    print(f"  Tables: {', '.join(tables)}")

    has_new_tables = "guardrails" in tables
    has_importance = False
    try:
        conn.execute("SELECT importance FROM facts LIMIT 1")
        has_importance = True
    except Exception:
        pass
    print(f"  Has guardrails table: {has_new_tables}")
    print(f"  Has importance column: {has_importance}")

    # ── Session recall latency ───────────────────────────────────────
    section("SESSION RECALL LATENCY")

    # Try different scopes
    scopes_to_test = [None]  # global
    try:
        scope_rows = conn.execute(
            "SELECT DISTINCT scope FROM facts WHERE scope != '__global__' LIMIT 3"
        ).fetchall()
        for r in scope_rows:
            scopes_to_test.append(r[0])
    except Exception:
        pass

    for scope in scopes_to_test:
        scope_label = "global" if scope is None else Path(scope).name if "/" in str(scope) else scope
        t0 = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            ctx = recall.session_recall(conn, scope=scope)
        elapsed = (time.perf_counter() - t0) / iterations * 1000

        text = recall.format_session_context(ctx)
        tokens = len(text) // CHARS_PER_TOKEN
        n_facts = len(ctx.get("long_facts", [])) + len(ctx.get("medium_facts", []))
        n_decisions = len(ctx.get("decisions", []))
        n_guardrails = len(ctx.get("guardrails", []))
        n_procedures = len(ctx.get("procedures", []))

        print(f"\n  Scope: {scope_label}")
        print(f"    Latency:     {elapsed:.1f}ms avg")
        print(f"    Facts:       {n_facts} (long+medium)")
        print(f"    Decisions:   {n_decisions}")
        print(f"    Guardrails:  {n_guardrails}")
        print(f"    Procedures:  {n_procedures}")
        print(f"    Tokens:      {tokens} / {SESSION_TOKEN_BUDGET} budget")
        within = "✓" if tokens <= SESSION_TOKEN_BUDGET else "✗ OVER BUDGET"
        print(f"    Budget:      {within}")

    # ── Prompt recall latency ────────────────────────────────────────
    section("PROMPT RECALL LATENCY")

    test_queries = [
        "How does the memory system work?",
        "What database is used for storage?",
        "How do I add a new extraction type?",
        "What are the guardrails for this codebase?",
        "Tell me about the embedding system",
    ]

    for query in test_queries:
        emb = embeddings.embed_query(query) if hasattr(embeddings, 'embed_query') else embeddings.embed(query)
        if not emb:
            print(f"\n  Query: {query[:50]}")
            print(f"    SKIPPED — embeddings not available")
            continue

        t0 = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            ctx = recall._legacy_prompt_recall(conn, emb, query)
        elapsed = (time.perf_counter() - t0) / iterations * 1000

        text = recall.format_prompt_context(ctx)
        tokens = len(text) // CHARS_PER_TOKEN
        n_facts = len(ctx.get("facts", []))
        n_guardrails = len(ctx.get("guardrails", []))

        top_fact = ctx["facts"][0]["text"][:60] if ctx.get("facts") else "(none)"

        print(f"\n  Query: \"{query}\"")
        print(f"    Latency:     {elapsed:.1f}ms")
        print(f"    Facts:       {n_facts}")
        print(f"    Guardrails:  {n_guardrails}")
        print(f"    Tokens:      {tokens} / {PROMPT_TOKEN_BUDGET}")
        print(f"    Top result:  {top_fact}")

    # ── Entity graph stats ───────────────────────────────────────────
    section("ENTITY GRAPH")
    try:
        top_entities = db.get_top_entities(conn, 10)
        print(f"  Top entities by session count:")
        for i, name in enumerate(top_entities[:10], 1):
            count = conn.execute(
                "SELECT session_count FROM entities WHERE name = ?", [name]
            ).fetchone()
            sc = count[0] if count else 0
            print(f"    {i:2d}. {name} (seen {sc}x)")
    except Exception as e:
        print(f"  Error: {e}")

    try:
        rel_count = conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE is_active = TRUE"
        ).fetchone()[0]
        print(f"\n  Active relationships: {rel_count}")

        # Most connected entities
        top_connected = conn.execute("""
            SELECT entity, COUNT(*) as connections FROM (
                SELECT from_entity as entity FROM relationships WHERE is_active = TRUE
                UNION ALL
                SELECT to_entity as entity FROM relationships WHERE is_active = TRUE
            )
            GROUP BY entity
            ORDER BY connections DESC
            LIMIT 5
        """).fetchall()
        if top_connected:
            print(f"  Most connected:")
            for name, count in top_connected:
                print(f"    {name}: {count} connections")
    except Exception as e:
        print(f"  Error: {e}")

    # ── Temporal class distribution ──────────────────────────────────
    section("TEMPORAL CLASS DISTRIBUTION")
    for tc in ("long", "medium", "short"):
        try:
            count = conn.execute(
                f"SELECT COUNT(*) FROM facts WHERE is_active = TRUE AND temporal_class = '{tc}'"
            ).fetchone()[0]
            avg_decay = conn.execute(
                f"SELECT AVG(decay_score) FROM facts WHERE is_active = TRUE AND temporal_class = '{tc}'"
            ).fetchone()[0]
            avg_sessions = conn.execute(
                f"SELECT AVG(session_count) FROM facts WHERE is_active = TRUE AND temporal_class = '{tc}'"
            ).fetchone()[0]
            print(f"  {tc:8s}: {count:>4} facts, avg decay={avg_decay:.3f}, avg sessions={avg_sessions:.1f}")
        except Exception:
            pass

    # ── Embedding coverage ───────────────────────────────────────────
    section("EMBEDDING COVERAGE")
    for table in ("facts", "ideas", "decisions", "entities"):
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE is_active = TRUE").fetchone()[0]
            with_emb = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE is_active = TRUE AND embedding IS NOT NULL"
            ).fetchone()[0]
            pct = (with_emb / total * 100) if total else 0
            print(f"  {table:15s}: {with_emb}/{total} ({pct:.0f}%)")
        except Exception:
            pass

    conn.close()

    print(f"\n{'=' * 60}")
    print("  LIVE BENCHMARK COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
