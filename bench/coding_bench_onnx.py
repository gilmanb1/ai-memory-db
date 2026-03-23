#!/usr/bin/env python3
"""
coding_bench_onnx.py — Coding memory benchmark with real ONNX embeddings.

Same tests as coding_bench.py but uses real semantic embeddings via
onnxruntime (nomic-embed-text-v1.5) for accurate retrieval quality measurement.

Usage: python3 bench/coding_bench_onnx.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import memory.config as _cfg
from memory import db, recall, embeddings
from memory.decay import compute_decay_score

# Verify ONNX works
_test = embeddings.embed("test")
if not _test:
    print("ERROR: ONNX embeddings not available. Install onnxruntime.")
    sys.exit(1)
print(f"Using ONNX embeddings ({len(_test)} dims)\n")


def _noop_decay(last_seen_at, session_count, temporal_class):
    return 1.0


# ── Import test data from the main benchmark ────────────────────────────
from bench.coding_bench import (
    GUARDRAILS, PROCEDURES, ERROR_SOLUTIONS, CODING_FACTS, RECALL_QUERIES,
)


def _setup_db(scale: int = 0):
    db_path = Path(tempfile.mktemp(suffix=".duckdb"))
    _cfg.DB_PATH = db_path
    conn = db.get_connection(db_path=str(db_path))
    db.upsert_session(conn, "bench-sess", "benchmark", "/tmp/project", "/tmp/t.jsonl", 100, "Benchmark")
    scope = "/tmp/project"

    t0 = time.perf_counter()

    for g in GUARDRAILS:
        emb = embeddings.embed(g["warning"])
        db.upsert_guardrail(
            conn, warning=g["warning"], rationale=g.get("rationale", ""),
            consequence=g.get("consequence", ""), file_paths=g.get("file_paths", []),
            line_range=g.get("line_range", ""), embedding=emb,
            session_id="bench-sess", scope=scope,
        )

    for p in PROCEDURES:
        emb = embeddings.embed(p["task_description"])
        db.upsert_procedure(
            conn, task_description=p["task_description"], steps=p["steps"],
            file_paths=p.get("file_paths", []), embedding=emb,
            session_id="bench-sess", scope=scope,
        )

    for e in ERROR_SOLUTIONS:
        emb = embeddings.embed(e["error_pattern"])
        db.upsert_error_solution(
            conn, error_pattern=e["error_pattern"], solution=e["solution"],
            error_context=e.get("error_context", ""), file_paths=e.get("file_paths", []),
            embedding=emb, session_id="bench-sess", scope=scope,
        )

    for text, cat, tc, conf, importance, file_paths in CODING_FACTS:
        emb = embeddings.embed(text)
        db.upsert_fact(
            conn, text=text, category=cat, temporal_class=tc, confidence=conf,
            embedding=emb, session_id="bench-sess", decay_fn=_noop_decay,
            scope=scope, importance=importance, file_paths=file_paths,
        )

    # Scale up
    for i in range(scale):
        text = f"Module {i % 20} uses pattern {i % 15} in file_{i % 30}.py for feature {i % 10}"
        emb = embeddings.embed(text)
        db.upsert_fact(
            conn, text=text, category="implementation", temporal_class="medium",
            confidence="medium", embedding=emb, session_id="bench-sess",
            decay_fn=_noop_decay, scope=scope,
            importance=(i % 10) + 1, file_paths=[f"module_{i % 20}/file_{i % 30}.py"],
        )

    entities = ["DuckDB", "Ollama", "Claude", "ONNX", "Python", "FastAPI", "Redis", "PostgreSQL"]
    for e in entities:
        db.upsert_entity(conn, e, scope=scope)
    db.upsert_relationship(conn, "DuckDB", "Python", "uses", "DuckDB has Python bindings", "bench-sess", scope=scope)
    db.upsert_relationship(conn, "FastAPI", "Redis", "uses", "FastAPI caches in Redis", "bench-sess", scope=scope)

    elapsed = time.perf_counter() - t0
    n = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active = TRUE").fetchone()[0]
    print(f"  Setup: {n} facts + guardrails/procedures/errors in {elapsed:.1f}s")

    return conn, db_path


def bench_guardrail_semantic(conn, scope):
    """Do guardrails surface via SEMANTIC search (not just file linking)?"""
    results = {"tests": 0, "passed": 0, "details": []}

    queries = [
        ("I'm going to refactor the polling loop to use exponential backoff", "polling loop"),
        ("Let me replace urllib3 with requests for cleaner code", "urllib3"),
        ("I'll remove that sleep call in the batch processor", "sleep"),
        ("Time to rewrite the revenue report queries with the ORM", "ORM"),
        ("Switching to orjson for faster JSON serialization", "orjson"),
    ]

    for query, keyword in queries:
        results["tests"] += 1
        emb = embeddings.embed_query(query)
        found = db.search_guardrails(conn, emb, limit=3, threshold=0.0, scope=scope)
        hit = any(keyword.lower() in f.get("warning", "").lower() for f in found)
        results["passed"] += int(hit)
        results["details"].append({
            "query": query[:60],
            "keyword": keyword,
            "found": hit,
            "top": found[0]["warning"][:60] if found else "(none)",
            "score": f"{found[0].get('score', 0):.3f}" if found else "n/a",
        })

    return results


def bench_procedure_semantic(conn, scope):
    """Do procedures surface via semantic search?"""
    results = {"tests": 0, "passed": 0, "details": []}

    queries = [
        ("How do I add a new database migration?", "migration"),
        ("How do I deploy to production?", "Deploy"),
        ("How to add an API endpoint?", "API endpoint"),
        ("How to debug a failing Celery task?", "Celery"),
        ("How to add a new extraction type to the memory system?", "extraction"),
    ]

    for query, keyword in queries:
        results["tests"] += 1
        emb = embeddings.embed_query(query)
        found = db.search_procedures(conn, emb, limit=3, threshold=0.0, scope=scope)
        hit = any(keyword.lower() in f.get("task_description", "").lower() for f in found)
        results["passed"] += int(hit)
        results["details"].append({
            "query": query[:60],
            "keyword": keyword,
            "found": hit,
            "top": found[0]["task_description"][:60] if found else "(none)",
            "score": f"{found[0].get('score', 0):.3f}" if found else "n/a",
        })

    return results


def bench_error_semantic(conn, scope):
    """Do error→solution pairs surface via semantic search?"""
    results = {"tests": 0, "passed": 0, "details": []}

    queries = [
        ("ImportError: No module named 'onnxruntime'", "onnxruntime"),
        ("DuckDB CatalogError: Table 'guardrails' does not exist", "CatalogError"),
        ("ConnectionRefusedError on port 11434", "ConnectionRefused"),
        ("anthropic.RateLimitError: 429 Too Many Requests", "429"),
        ("TypeError: list_cosine_similarity() unexpected keyword argument", "cosine_similarity"),
    ]

    for query, keyword in queries:
        results["tests"] += 1
        emb = embeddings.embed_query(query)
        found = db.search_error_solutions(conn, emb, limit=3, threshold=0.0, scope=scope)
        hit = any(keyword.lower() in f.get("error_pattern", "").lower() for f in found)
        results["passed"] += int(hit)
        results["details"].append({
            "query": query[:60],
            "keyword": keyword,
            "found": hit,
            "top": found[0]["error_pattern"][:60] if found else "(none)",
            "score": f"{found[0].get('score', 0):.3f}" if found else "n/a",
        })

    return results


def bench_cross_type_retrieval(conn, scope):
    """Can a single query surface guardrails + facts + procedures together?"""
    results = {"tests": 0, "passed": 0, "details": []}

    # A query that should hit guardrails, facts, AND procedures
    query = "I need to make changes to the database schema and add a migration"
    emb = embeddings.embed_query(query)

    # Should find: guardrail (none expected), procedure (migration), facts (db.py related)
    procedures = db.search_procedures(conn, emb, limit=3, threshold=0.0, scope=scope)
    facts = db.search_facts(conn, emb, limit=5, threshold=0.0, scope=scope)
    path_facts = db.get_items_by_file_paths(conn, ["memory/db.py"], item_table="facts", scope=scope)

    results["tests"] += 1
    has_proc = any("migration" in p.get("task_description", "").lower() for p in procedures)
    results["passed"] += int(has_proc)
    results["details"].append({"type": "procedure", "query": "migration-related", "found": has_proc,
                               "top": procedures[0]["task_description"][:50] if procedures else "(none)"})

    results["tests"] += 1
    has_fact = len(facts) > 0
    results["passed"] += int(has_fact)
    results["details"].append({"type": "fact", "query": "db-related", "found": has_fact,
                               "top": facts[0]["text"][:50] if facts else "(none)"})

    results["tests"] += 1
    has_path = len(path_facts) > 0
    results["passed"] += int(has_path)
    results["details"].append({"type": "path-linked fact", "query": "memory/db.py", "found": has_path,
                               "count": len(path_facts)})

    return results


def bench_decision_recall(conn, scope):
    """Do architectural decisions with rationale surface correctly?"""
    results = {"tests": 0, "passed": 0, "details": []}

    queries = [
        ("Why did we choose DuckDB over SQLite?", "DuckDB", "decision_rationale"),
        ("Why do we use tool_use for extraction instead of JSON?", "tool", "decision_rationale"),
        ("What's the dedup strategy?", "dedup", None),
    ]

    for query, keyword, expected_cat in queries:
        results["tests"] += 1
        emb = embeddings.embed_query(query)
        found = db.search_facts(conn, emb, limit=5, threshold=0.0, scope=scope)
        hit = any(keyword.lower() in f.get("text", "").lower() for f in found)
        results["passed"] += int(hit)
        top = found[0] if found else {}
        results["details"].append({
            "query": query[:60],
            "keyword": keyword,
            "found": hit,
            "top_text": top.get("text", "")[:60],
            "top_cat": top.get("category", ""),
            "top_importance": top.get("importance", "n/a"),
        })

    return results


def bench_latency_onnx(conn, db_path, scope, n_items):
    """Measure latency including ONNX embedding time."""
    results = {"n_items": n_items, "timings": {}}

    # Embedding latency
    t0 = time.perf_counter()
    for _ in range(50):
        embeddings.embed_query("How does the dedup threshold work?")
    results["timings"]["embed_query_avg_ms"] = round((time.perf_counter() - t0) * 20, 2)

    # Session recall
    t0 = time.perf_counter()
    for _ in range(10):
        recall.session_recall(conn, scope=scope)
    results["timings"]["session_recall_avg_ms"] = round((time.perf_counter() - t0) * 100, 1)

    # Prompt recall (legacy, avoids connection conflicts)
    query_emb = embeddings.embed_query("How does the dedup threshold work?")
    t0 = time.perf_counter()
    for _ in range(10):
        recall._legacy_prompt_recall(conn, query_emb, "How does the dedup threshold work?", scope=scope)
    results["timings"]["prompt_recall_avg_ms"] = round((time.perf_counter() - t0) * 100, 1)

    # Guardrail lookup
    t0 = time.perf_counter()
    for _ in range(100):
        db.get_guardrails_for_files(conn, ["sync_worker.py"], scope=scope)
    results["timings"]["guardrail_lookup_avg_ms"] = round((time.perf_counter() - t0) * 10, 2)

    # Path-scoped
    t0 = time.perf_counter()
    for _ in range(100):
        db.get_items_by_file_paths(conn, ["memory/db.py"], scope=scope)
    results["timings"]["path_lookup_avg_ms"] = round((time.perf_counter() - t0) * 10, 2)

    # Vector search
    t0 = time.perf_counter()
    for _ in range(10):
        db.search_facts(conn, query_emb, limit=20, scope=scope)
    results["timings"]["vector_search_avg_ms"] = round((time.perf_counter() - t0) * 100, 1)

    # Guardrail semantic search
    query_emb2 = embeddings.embed_query("refactor polling loop exponential backoff")
    t0 = time.perf_counter()
    for _ in range(10):
        db.search_guardrails(conn, query_emb2, limit=5, scope=scope)
    results["timings"]["guardrail_semantic_avg_ms"] = round((time.perf_counter() - t0) * 100, 1)

    return results


def run_all():
    print("=" * 70)
    print("  CODING MEMORY BENCHMARK (ONNX embeddings)")
    print("=" * 70)

    scope = "/tmp/project"
    total_pass = 0
    total_tests = 0

    conn, db_path = _setup_db(scale=0)

    benchmarks = [
        ("Guardrail Semantic Search", bench_guardrail_semantic),
        ("Procedure Semantic Search", bench_procedure_semantic),
        ("Error→Solution Semantic Search", bench_error_semantic),
        ("Cross-Type Retrieval", bench_cross_type_retrieval),
        ("Decision Rationale Recall", bench_decision_recall),
    ]

    # Also run the deterministic (non-semantic) benchmarks from the base module
    from bench.coding_bench import (
        bench_guardrail_recall, bench_importance_ranking,
        bench_path_scoped_recall, bench_bitemporal,
        bench_session_recall_format,
    )
    benchmarks += [
        ("Guardrail File-Scoped Recall", bench_guardrail_recall),
        ("Importance Ranking", bench_importance_ranking),
        ("Path-Scoped Fact Recall", bench_path_scoped_recall),
        ("Bi-Temporal Invalidation", bench_bitemporal),
        ("Session Recall Formatting", bench_session_recall_format),
    ]

    for name, bench_fn in benchmarks:
        result = bench_fn(conn, scope)
        passed = result["passed"]
        tests = result["tests"]
        total_pass += passed
        total_tests += tests
        status = "✓" if passed == tests else "✗"
        print(f"\n  {status} {name}: {passed}/{tests}")
        for d in result["details"]:
            hit = d.get("passed", d.get("found", d.get("hit", d.get("present", False))))
            marker = "  ✓" if hit else "  ✗"
            parts = []
            for k, v in d.items():
                if k in ("passed", "found", "hit", "present"):
                    continue
                parts.append(f"{k}={v}")
            print(f"    {marker} {', '.join(parts)}")

    conn.close()
    db_path.unlink(missing_ok=True)

    # ── Latency ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  LATENCY (with ONNX embedding)")
    print(f"{'─' * 70}")

    for scale in [0, 100, 500]:
        conn, db_path = _setup_db(scale=scale)
        n = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active = TRUE").fetchone()[0]
        latency = bench_latency_onnx(conn, db_path, scope, n)
        print(f"\n  Items: {n:,}")
        for metric, ms in latency["timings"].items():
            bar = "█" * max(1, int(ms / 2))
            print(f"    {metric:<30s} {ms:>8.1f}ms  {bar}")
        conn.close()
        db_path.unlink(missing_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {total_pass}/{total_tests} correctness tests passed")
    pct = (total_pass / total_tests * 100) if total_tests else 0
    print(f"  Score: {pct:.0f}%")
    print(f"{'=' * 70}")

    return total_pass == total_tests


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
