#!/usr/bin/env python3
"""
coding_bench.py — Coding-oriented memory benchmarks.

Measures what matters for a coding agent's memory system:
  1. Guardrail recall    — Does the right "don't touch this" surface when editing a file?
  2. Procedure recall    — Does "how to do X" surface when the agent needs it?
  3. Error→fix recall    — Does the known fix surface when the agent hits an error?
  4. Path-scoped recall  — Do file-associated facts surface when editing that file?
  5. Importance ranking  — Do critical facts outrank trivial ones in limited budgets?
  6. Contradiction handling — Does bi-temporal invalidation work correctly?
  7. Cross-type retrieval — Can a single query surface guardrails + facts + procedures?
  8. Latency at scale     — How fast is recall with 100/500/1000/5000 items?

Runs locally against in-memory DuckDB. No Ollama or API key required.
Usage: python3 bench/coding_bench.py
"""
from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import memory.config as _cfg
from memory import db, recall
from memory.decay import compute_decay_score, temporal_weight


# ── Deterministic mock embeddings (no Ollama needed) ─────────────────────

def _mock_embed(text: str) -> list[float]:
    raw: list[float] = []
    seed = hashlib.sha256(text.encode()).digest()
    while len(raw) < _cfg.EMBEDDING_DIM:
        seed = hashlib.sha256(seed).digest()
        for i in range(0, len(seed) - 3, 4):
            if len(raw) < _cfg.EMBEDDING_DIM:
                val = int.from_bytes(seed[i:i+4], "big") / (2**32) - 0.5
                raw.append(val)
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def _noop_decay(last_seen_at, session_count, temporal_class):
    return 1.0


# ── Synthetic coding data ────────────────────────────────────────────────

GUARDRAILS = [
    {
        "warning": "Do not refactor the polling loop in sync_worker.py to exponential backoff",
        "rationale": "Downstream API detects backoff patterns and penalizes exponential clients",
        "consequence": "Client gets blocked for 24h, breaking all sync operations",
        "file_paths": ["sync_worker.py"],
        "line_range": "L45-L78",
    },
    {
        "warning": "Do not replace urllib3 with requests in http_client.py",
        "rationale": "requests bundles certifi which conflicts with corporate CA trust store",
        "consequence": "All HTTPS connections fail in corporate network",
        "file_paths": ["http_client.py", "utils/networking.py"],
    },
    {
        "warning": "Do not remove the sleep(0.1) in batch_processor.py",
        "rationale": "Rate limiting needed to avoid overwhelming the upstream service",
        "consequence": "Upstream returns 429 and blacklists our IP for 1 hour",
        "file_paths": ["batch_processor.py"],
        "line_range": "L112",
    },
    {
        "warning": "Do not use ORM for queries in reports/revenue.py",
        "rationale": "Raw SQL is 10x faster for the aggregation queries in the revenue report",
        "consequence": "Report generation times out after 30 seconds",
        "file_paths": ["reports/revenue.py", "reports/base.py"],
    },
    {
        "warning": "Do not change the JSON serialization in api/serializers.py to use orjson",
        "rationale": "orjson doesn't support our custom Decimal encoder needed for financial amounts",
        "consequence": "Amounts lose precision, causing accounting discrepancies",
        "file_paths": ["api/serializers.py"],
    },
]

PROCEDURES = [
    {
        "task_description": "Add a new database migration",
        "steps": "1. Add entry to MIGRATIONS list in db.py with next version number. 2. Write SQL in the migration string. 3. Run python3 test_memory.py to verify. 4. Commit with message 'Add migration N: description'.",
        "file_paths": ["memory/db.py", "test_memory.py"],
    },
    {
        "task_description": "Add a new API endpoint",
        "steps": "1. Create handler function in routes/{resource}.py. 2. Add Pydantic model in schemas/{resource}.py. 3. Register route in app.py. 4. Add integration test in tests/test_{resource}.py. 5. Update OpenAPI spec.",
        "file_paths": ["routes/", "schemas/", "app.py"],
    },
    {
        "task_description": "Deploy to production",
        "steps": "1. Ensure all tests pass (python3 -m pytest). 2. Build Docker image (docker build -t app:latest .). 3. Push to ECR. 4. Update ECS task definition. 5. Monitor CloudWatch for errors for 15 minutes.",
        "file_paths": ["Dockerfile", "deploy.sh", "ecs-task-def.json"],
    },
    {
        "task_description": "Debug a failing Celery task",
        "steps": "1. Check Flower dashboard at localhost:5555 for task status. 2. Read the traceback in CloudWatch logs. 3. Reproduce locally with 'celery -A app.celery call task_name'. 4. Fix and add a regression test.",
        "file_paths": ["tasks/", "celery_config.py"],
    },
    {
        "task_description": "Add a new extraction type to the memory system",
        "steps": "1. Add table in migration (memory/db.py). 2. Add CRUD functions (upsert, search, get). 3. Add to extraction tool schema (memory/extract.py). 4. Add storage in ingest pipeline (memory/ingest.py). 5. Add recall formatting (memory/recall.py). 6. Write tests. 7. Update config limits.",
        "file_paths": ["memory/db.py", "memory/extract.py", "memory/ingest.py", "memory/recall.py"],
    },
]

ERROR_SOLUTIONS = [
    {
        "error_pattern": "ImportError: No module named 'onnxruntime'",
        "error_context": "On macOS ARM when running memory extraction",
        "solution": "pip install onnxruntime-silicon (ARM-specific build)",
        "file_paths": ["memory/embeddings.py"],
    },
    {
        "error_pattern": "DuckDB CatalogError: Table 'guardrails' does not exist",
        "error_context": "After updating to a new version without running migrations",
        "solution": "Delete ~/.claude/memory/knowledge.duckdb and restart — migrations will re-run",
        "file_paths": ["memory/db.py"],
    },
    {
        "error_pattern": "ConnectionRefusedError: [Errno 61] Connection refused on port 11434",
        "error_context": "When Ollama is not running locally",
        "solution": "Start Ollama with 'ollama serve' or install from ollama.ai. System degrades gracefully without it.",
        "file_paths": ["memory/embeddings.py"],
    },
    {
        "error_pattern": "anthropic.RateLimitError: 429 Too Many Requests",
        "error_context": "During bulk extraction with many sessions",
        "solution": "Add a 2-second delay between extraction calls or reduce EXTRACT_MAX_TOKENS",
        "file_paths": ["memory/ingest.py", "memory/extract.py"],
    },
    {
        "error_pattern": "TypeError: list_cosine_similarity() got unexpected keyword argument",
        "error_context": "DuckDB version mismatch with VSS extension",
        "solution": "Update DuckDB: pip install --upgrade duckdb. The fallback Python cosine will work meanwhile.",
        "file_paths": ["memory/db.py"],
    },
]

CODING_FACTS = [
    # Architecture facts with file associations
    ("memory/db.py contains all schema migrations and CRUD operations — it's the largest file at ~1800 lines", "architecture", "long", "high", 8, ["memory/db.py"]),
    ("recall.py has two modes: session_recall (broad, at start) and prompt_recall (narrow, per message)", "architecture", "long", "high", 7, ["memory/recall.py"]),
    ("The extraction pipeline has 3 triggers: status_line (90% context), pre_compact, and session_end", "architecture", "long", "high", 8, ["memory/ingest.py", "hooks/status_line.py"]),
    ("Embeddings use a dual-backend: ONNX Runtime (fast, 2ms) primary, Ollama (50ms) fallback", "implementation", "long", "high", 7, ["memory/embeddings.py"]),
    ("Dedup threshold is 0.92 cosine similarity — items above this are reinforced, not duplicated", "implementation", "long", "high", 9, ["memory/db.py", "memory/config.py"]),
    ("Token budgets: 3000 for session context, 4000 for per-prompt context, ~4 chars per token", "constraint", "long", "high", 8, ["memory/recall.py", "memory/config.py"]),
    ("Project scoping uses git repo root as scope key — items auto-promote to global after 3 projects", "architecture", "long", "high", 7, ["memory/scope.py"]),
    ("The consolidation engine synthesizes observations from raw facts using Claude", "architecture", "long", "high", 6, ["memory/consolidation.py"]),
    # Operational facts
    ("Run tests with: python3 test_memory.py", "operational", "long", "high", 9, ["test_memory.py"]),
    ("Install with: bash install.sh — copies to ~/.claude/ and configures hooks", "operational", "long", "high", 8, ["install.sh"]),
    ("The benchmark requires Ollama running: ollama pull nomic-embed-text", "operational", "medium", "high", 5, ["benchmark.py"]),
    # Decision rationale
    ("DuckDB was chosen over SQLite because SQLite's single-writer lock fails under concurrent hook execution", "decision_rationale", "long", "high", 9, []),
    ("Tool-use extraction was chosen over JSON parsing because it guarantees valid structured output", "decision_rationale", "long", "high", 8, ["memory/extract.py"]),
    ("Per-session lock files (O_CREAT|O_EXCL) prevent duplicate extraction — simpler than a DB lock", "decision_rationale", "long", "high", 7, ["memory/ingest.py"]),
    # Trivial facts (low importance, should be outranked)
    ("The config file is 102 lines long", "contextual", "short", "low", 1, ["memory/config.py"]),
    ("We discussed dark mode preferences", "user_preference", "short", "low", 2, []),
    ("The last benchmark took 45 seconds to run", "contextual", "short", "low", 1, ["benchmark.py"]),
    ("There are 7 migrations in the database schema", "contextual", "short", "low", 2, ["memory/db.py"]),
]

# Queries that should retrieve specific items
RECALL_QUERIES = [
    {
        "query": "I'm going to refactor sync_worker.py to use exponential backoff",
        "should_surface": ["guardrail:polling loop", "guardrail:backoff"],
        "file_context": ["sync_worker.py"],
        "description": "Agent about to violate a guardrail",
    },
    {
        "query": "How do I add a new database migration?",
        "should_surface": ["procedure:migration"],
        "file_context": ["memory/db.py"],
        "description": "Agent needs a procedure",
    },
    {
        "query": "ImportError: No module named 'onnxruntime'",
        "should_surface": ["error:onnxruntime"],
        "file_context": ["memory/embeddings.py"],
        "description": "Agent encounters a known error",
    },
    {
        "query": "I want to switch from urllib3 to requests for cleaner HTTP code",
        "should_surface": ["guardrail:urllib3", "guardrail:requests"],
        "file_context": ["http_client.py"],
        "description": "Agent about to violate a library guardrail",
    },
    {
        "query": "What's the dedup threshold and why was it chosen?",
        "should_surface": ["fact:dedup threshold", "fact:0.92"],
        "file_context": ["memory/db.py", "memory/config.py"],
        "description": "Agent needs implementation detail",
    },
    {
        "query": "The revenue report is too slow, I'll rewrite it with the ORM",
        "should_surface": ["guardrail:ORM", "guardrail:revenue"],
        "file_context": ["reports/revenue.py"],
        "description": "Agent about to violate performance guardrail",
    },
    {
        "query": "How do I deploy this to production?",
        "should_surface": ["procedure:deploy"],
        "file_context": [],
        "description": "Agent needs deployment procedure",
    },
    {
        "query": "DuckDB CatalogError: Table 'guardrails' does not exist",
        "should_surface": ["error:CatalogError", "error:guardrails"],
        "file_context": ["memory/db.py"],
        "description": "Agent encounters a known DuckDB error",
    },
]


# ══════════════════════════════════════════════════════════════════════════
# Benchmark runners
# ══════════════════════════════════════════════════════════════════════════

def _setup_db(scale: int = 1) -> tuple:
    """Create and populate an in-memory test database. Returns (conn, db_path)."""
    db_path = Path(tempfile.mktemp(suffix=".duckdb"))
    _cfg.DB_PATH = db_path
    conn = db.get_connection(db_path=str(db_path))
    db.upsert_session(conn, "bench-sess", "benchmark", "/tmp/project", "/tmp/t.jsonl", 100, "Benchmark session")

    scope = "/tmp/project"

    # Insert guardrails
    for g in GUARDRAILS:
        emb = _mock_embed(g["warning"])
        db.upsert_guardrail(
            conn, warning=g["warning"], rationale=g.get("rationale", ""),
            consequence=g.get("consequence", ""), file_paths=g.get("file_paths", []),
            line_range=g.get("line_range", ""), embedding=emb,
            session_id="bench-sess", scope=scope,
        )

    # Insert procedures
    for p in PROCEDURES:
        emb = _mock_embed(p["task_description"])
        db.upsert_procedure(
            conn, task_description=p["task_description"], steps=p["steps"],
            file_paths=p.get("file_paths", []), embedding=emb,
            session_id="bench-sess", scope=scope,
        )

    # Insert error solutions
    for e in ERROR_SOLUTIONS:
        emb = _mock_embed(e["error_pattern"])
        db.upsert_error_solution(
            conn, error_pattern=e["error_pattern"], solution=e["solution"],
            error_context=e.get("error_context", ""), file_paths=e.get("file_paths", []),
            embedding=emb, session_id="bench-sess", scope=scope,
        )

    # Insert coding facts
    for text, cat, tc, conf, importance, file_paths in CODING_FACTS:
        emb = _mock_embed(text)
        db.upsert_fact(
            conn, text=text, category=cat, temporal_class=tc, confidence=conf,
            embedding=emb, session_id="bench-sess", decay_fn=_noop_decay,
            scope=scope, importance=importance, file_paths=file_paths,
        )

    # Scale up with synthetic facts for latency testing
    if scale > 1:
        for i in range(scale):
            text = f"Synthetic fact #{i}: module_{i % 20} uses pattern_{i % 15} in file_{i % 30}.py for feature_{i % 10}"
            emb = _mock_embed(text)
            fp = [f"module_{i % 20}/file_{i % 30}.py"]
            db.upsert_fact(
                conn, text=text, category="implementation", temporal_class="medium",
                confidence="medium", embedding=emb, session_id="bench-sess",
                decay_fn=_noop_decay, scope=scope,
                importance=(i % 10) + 1, file_paths=fp,
            )

    # Add entities and relationships for graph traversal
    entities = ["DuckDB", "Ollama", "Claude", "ONNX", "Python", "FastAPI", "Redis", "PostgreSQL"]
    for e in entities:
        db.upsert_entity(conn, e, scope=scope)
    db.upsert_relationship(conn, "DuckDB", "Python", "uses", "DuckDB has Python bindings", "bench-sess", scope=scope)
    db.upsert_relationship(conn, "Claude", "ONNX", "uses", "Claude extraction uses ONNX embeddings", "bench-sess", scope=scope)
    db.upsert_relationship(conn, "FastAPI", "Redis", "uses", "FastAPI caches in Redis", "bench-sess", scope=scope)

    return conn, db_path


def bench_guardrail_recall(conn, scope: str) -> dict:
    """Test: Do guardrails surface when working on associated files?"""
    results = {"tests": 0, "passed": 0, "details": []}

    for g in GUARDRAILS:
        for fp in g.get("file_paths", [])[:1]:  # test first file path
            results["tests"] += 1
            found = db.get_guardrails_for_files(conn, [fp], scope=scope)
            hit = any(g["warning"][:30] in f.get("warning", "") for f in found)
            results["passed"] += int(hit)
            results["details"].append({
                "file": fp,
                "expected": g["warning"][:50],
                "found": hit,
                "n_results": len(found),
            })

    return results


def bench_procedure_recall(conn, scope: str) -> dict:
    """Test: Do procedures surface via vector search?"""
    results = {"tests": 0, "passed": 0, "details": []}

    queries = [
        ("How do I add a migration?", "migration"),
        ("How do I deploy?", "Deploy"),
        ("How to add an API endpoint?", "API endpoint"),
        ("How to debug a Celery task?", "Celery"),
        ("How to add a new extraction type?", "extraction"),
    ]

    for query, keyword in queries:
        results["tests"] += 1
        emb = _mock_embed(query)
        found = db.search_procedures(conn, emb, limit=3, threshold=0.0, scope=scope)
        hit = any(keyword.lower() in f.get("task_description", "").lower() for f in found)
        results["passed"] += int(hit)
        results["details"].append({
            "query": query,
            "keyword": keyword,
            "found": hit,
            "top_result": found[0]["task_description"][:60] if found else "(none)",
        })

    return results


def bench_error_recall(conn, scope: str) -> dict:
    """Test: Do error→solution pairs surface for matching errors?"""
    results = {"tests": 0, "passed": 0, "details": []}

    queries = [
        ("ImportError onnxruntime", "onnxruntime"),
        ("CatalogError table not found", "CatalogError"),
        ("Connection refused port 11434", "Connection"),
        ("429 Too Many Requests", "429"),
        ("list_cosine_similarity unexpected", "cosine"),
    ]

    for query, keyword in queries:
        results["tests"] += 1
        emb = _mock_embed(query)
        found = db.search_error_solutions(conn, emb, limit=3, threshold=0.0, scope=scope)
        hit = any(keyword.lower() in f.get("error_pattern", "").lower() for f in found)
        results["passed"] += int(hit)
        results["details"].append({
            "query": query,
            "keyword": keyword,
            "found": hit,
            "top_result": found[0]["error_pattern"][:60] if found else "(none)",
        })

    return results


def bench_importance_ranking(conn, scope: str) -> dict:
    """Test: Do high-importance facts outrank low-importance ones?"""
    results = {"tests": 0, "passed": 0, "details": []}

    facts = db.get_facts_by_temporal(conn, "long", 50, scope=scope)
    if len(facts) < 2:
        return results

    results["tests"] += 1
    # Top facts should have higher importance than bottom facts
    top_importance = sum(f.get("importance", 5) for f in facts[:3]) / 3
    bot_importance = sum(f.get("importance", 5) for f in facts[-3:]) / 3
    passed = top_importance >= bot_importance
    results["passed"] += int(passed)
    results["details"].append({
        "test": "top-3 avg importance >= bottom-3",
        "top_avg": round(top_importance, 1),
        "bot_avg": round(bot_importance, 1),
        "passed": passed,
    })

    return results


def bench_path_scoped_recall(conn, scope: str) -> dict:
    """Test: Do facts with file_paths surface when querying by path?"""
    results = {"tests": 0, "passed": 0, "details": []}

    test_paths = [
        ("memory/db.py", ["migrations", "CRUD", "schema"]),
        ("memory/recall.py", ["session_recall", "prompt_recall", "context"]),
        ("memory/embeddings.py", ["ONNX", "Ollama", "embed"]),
        ("memory/ingest.py", ["trigger", "extraction", "lock"]),
    ]

    for file_path, keywords in test_paths:
        results["tests"] += 1
        found = db.get_items_by_file_paths(conn, [file_path], item_table="facts", scope=scope)
        hit = len(found) > 0
        results["passed"] += int(hit)
        results["details"].append({
            "file": file_path,
            "found_count": len(found),
            "hit": hit,
            "first_text": found[0]["text"][:60] if found else "(none)",
        })

    return results


def bench_bitemporal(conn, scope: str) -> dict:
    """Test: Does bi-temporal invalidation exclude old facts correctly?"""
    results = {"tests": 0, "passed": 0, "details": []}

    # Insert two facts about the same topic
    old_emb = _mock_embed("project database is PostgreSQL for sure unique bench")
    old_id, _ = db.upsert_fact(
        conn, "Project uses PostgreSQL for primary storage",
        "technical", "long", "high", old_emb, "bench-sess", _noop_decay,
        scope=scope, importance=7,
    )
    new_emb = _mock_embed("project database is DuckDB for sure unique bench")
    new_id, _ = db.upsert_fact(
        conn, "Project migrated from PostgreSQL to DuckDB",
        "technical", "long", "high", new_emb, "bench-sess", _noop_decay,
        scope=scope, importance=8,
    )

    # Invalidate old fact
    db.invalidate_fact(conn, old_id)

    results["tests"] += 1
    current = db.get_current_facts(conn, scope=scope)
    old_in_current = any(f["id"] == old_id for f in current)
    new_in_current = any(f["id"] == new_id for f in current)
    passed = (not old_in_current) and new_in_current
    results["passed"] += int(passed)
    results["details"].append({
        "test": "invalidated fact excluded, new fact included",
        "old_excluded": not old_in_current,
        "new_included": new_in_current,
        "passed": passed,
    })

    return results


def bench_session_recall_format(conn, scope: str) -> dict:
    """Test: Does session recall format include all new sections?"""
    results = {"tests": 0, "passed": 0, "details": []}

    ctx = recall.session_recall(conn, scope=scope)
    text = recall.format_session_context(ctx)

    for section in ["Guardrails", "Procedures", "Established Knowledge"]:
        results["tests"] += 1
        hit = section in text
        results["passed"] += int(hit)
        results["details"].append({"section": section, "present": hit})

    # Check token budget
    results["tests"] += 1
    est_tokens = len(text) // _cfg.CHARS_PER_TOKEN
    within_budget = est_tokens <= _cfg.SESSION_TOKEN_BUDGET * 1.1  # 10% margin
    results["passed"] += int(within_budget)
    results["details"].append({
        "test": "within token budget",
        "tokens": est_tokens,
        "budget": _cfg.SESSION_TOKEN_BUDGET,
        "passed": within_budget,
    })

    return results


def bench_latency(conn, db_path: Path, scope: str, n_items: int) -> dict:
    """Measure recall latency at the current scale."""
    results = {"n_items": n_items, "timings": {}}

    # Session recall
    t0 = time.perf_counter()
    for _ in range(10):
        recall.session_recall(conn, scope=scope)
    results["timings"]["session_recall_avg_ms"] = round((time.perf_counter() - t0) * 100, 1)  # per call

    # Prompt recall (legacy path — parallel retrieval needs separate connections)
    query_emb = _mock_embed("How does the dedup threshold work?")
    t0 = time.perf_counter()
    for _ in range(10):
        recall._legacy_prompt_recall(conn, query_emb, "How does the dedup threshold work?", scope=scope)
    results["timings"]["prompt_recall_avg_ms"] = round((time.perf_counter() - t0) * 100, 1)

    # Guardrail file lookup
    t0 = time.perf_counter()
    for _ in range(100):
        db.get_guardrails_for_files(conn, ["sync_worker.py"], scope=scope)
    results["timings"]["guardrail_lookup_avg_ms"] = round((time.perf_counter() - t0) * 10, 2)

    # Path-scoped fact lookup
    t0 = time.perf_counter()
    for _ in range(100):
        db.get_items_by_file_paths(conn, ["memory/db.py"], scope=scope)
    results["timings"]["path_lookup_avg_ms"] = round((time.perf_counter() - t0) * 10, 2)

    # Vector search facts
    t0 = time.perf_counter()
    for _ in range(10):
        db.search_facts(conn, query_emb, limit=20, scope=scope)
    results["timings"]["vector_search_avg_ms"] = round((time.perf_counter() - t0) * 100, 1)

    return results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def bench_outcome_scoring(conn, scope: str) -> dict:
    """Test: Does recall_utility multiplier affect ranking?"""
    results = {"tests": 0, "passed": 0, "details": []}

    # Insert two facts, give one high utility
    emb_a = _mock_embed("outcome bench fact alpha unique")
    emb_b = _mock_embed("outcome bench fact beta unique")
    fid_a, _ = db.upsert_fact(
        conn, "Outcome bench alpha", "contextual", "long", "high",
        emb_a, "bench-sess", _noop_decay, scope=scope, importance=5,
    )
    fid_b, _ = db.upsert_fact(
        conn, "Outcome bench beta", "contextual", "long", "high",
        emb_b, "bench-sess", _noop_decay, scope=scope, importance=5,
    )
    # Simulate: fact_a recalled 10 times, applied 8 times (useful)
    for _ in range(10):
        db.increment_recalled(conn, {"facts": [fid_a]})
    for _ in range(8):
        db.mark_applied(conn, fid_a, "facts")
    # fact_b recalled 10 times, never applied (noise)
    for _ in range(10):
        db.increment_recalled(conn, {"facts": [fid_b]})

    results["tests"] += 1
    row_a = conn.execute("SELECT recall_utility FROM facts WHERE id=?", [fid_a]).fetchone()
    row_b = conn.execute("SELECT recall_utility FROM facts WHERE id=?", [fid_b]).fetchone()
    passed = row_a[0] > row_b[0]
    results["passed"] += int(passed)
    results["details"].append({
        "test": "applied fact has higher utility than ignored fact",
        "applied_utility": round(row_a[0], 3),
        "ignored_utility": round(row_b[0], 3),
        "passed": passed,
    })

    return results


def bench_failure_priority(conn, scope: str) -> dict:
    """Test: Does failure_probability boost retrieval priority?"""
    results = {"tests": 0, "passed": 0, "details": []}

    emb_safe = _mock_embed("safe fact low failure unique bench")
    emb_risky = _mock_embed("risky fact high failure unique bench")
    db.upsert_fact(
        conn, "Safe fact — easy to derive", "contextual", "long", "high",
        emb_safe, "bench-sess", _noop_decay, scope=scope,
        importance=7, failure_probability=0.1,
    )
    db.upsert_fact(
        conn, "Risky fact — agent will get wrong", "constraint", "long", "high",
        emb_risky, "bench-sess", _noop_decay, scope=scope,
        importance=7, failure_probability=0.9,
    )

    facts = db.get_facts_by_temporal(conn, "long", 50, scope=scope)
    # Find our test facts
    risky = [f for f in facts if "Risky" in f.get("text", "")]
    safe = [f for f in facts if "Safe" in f.get("text", "")]

    results["tests"] += 1
    if risky and safe:
        risky_idx = facts.index(risky[0])
        safe_idx = facts.index(safe[0])
        passed = risky_idx < safe_idx
        results["passed"] += int(passed)
        results["details"].append({
            "test": "high failure_prob fact ranks above low",
            "risky_rank": risky_idx,
            "safe_rank": safe_idx,
            "passed": passed,
        })
    else:
        results["details"].append({"test": "facts not found", "passed": False})

    return results


def bench_community_summaries(conn, scope: str) -> dict:
    """Test: Do community summaries surface in session recall?"""
    results = {"tests": 0, "passed": 0, "details": []}

    emb = _mock_embed("auth module community summary unique bench")
    db.upsert_community_summary(
        conn, level=1,
        summary="The auth module uses JWT tokens with 24h expiry and argon2id hashing",
        entity_ids=["JWT", "argon2id"],
        source_item_ids=["f1", "f2"],
        embedding=emb, scope=scope,
    )

    results["tests"] += 1
    ctx = recall.session_recall(conn, scope=scope)
    has_summaries = len(ctx.get("community_summaries", [])) > 0
    results["passed"] += int(has_summaries)
    results["details"].append({
        "test": "community summary in session recall",
        "found": has_summaries,
    })

    results["tests"] += 1
    text = recall.format_session_context(ctx)
    has_section = "Summaries" in text
    results["passed"] += int(has_section)
    results["details"].append({
        "test": "Summaries section in formatted output",
        "present": has_section,
    })

    return results


def bench_coherence_check(conn, scope: str) -> dict:
    """Test: Does coherence check detect near-contradictions?"""
    results = {"tests": 0, "passed": 0, "details": []}

    # The find_potential_contradictions function needs embeddings in the right similarity range
    # With mock embeddings, we can test the function runs without error
    from memory.consolidation import run_coherence_check
    stats = run_coherence_check(conn, scope=scope, quiet=True)

    results["tests"] += 1
    passed = isinstance(stats, dict) and "pairs_checked" in stats
    results["passed"] += int(passed)
    results["details"].append({
        "test": "coherence check runs without error",
        "stats": stats,
        "passed": passed,
    })

    return results


def bench_code_graph(scope: str) -> dict:
    """Test: Does the code graph parse this repo and enable impact analysis?"""
    results = {"tests": 0, "passed": 0, "details": []}

    try:
        from memory.code_graph import parse_python_file, parse_repo, ensure_code_graph_tables
        from memory.code_graph import get_dependents, get_file_symbols, get_impact_analysis, search_symbol
    except ImportError:
        results["tests"] += 1
        results["details"].append({"test": "import code_graph", "passed": False, "error": "not found"})
        return results

    db_path_cg = Path(tempfile.mktemp(suffix=".duckdb"))
    conn_cg = db.get_connection(db_path=str(db_path_cg))
    ensure_code_graph_tables(conn_cg)

    # Parse this project's memory/ directory
    repo_root = str(PROJECT_ROOT)
    t0 = time.perf_counter()
    stats = parse_repo(repo_root, conn_cg, scope, max_files=50)
    parse_time = time.perf_counter() - t0

    results["tests"] += 1
    passed = stats["files_parsed"] > 0
    results["passed"] += int(passed)
    results["details"].append({
        "test": f"parsed {stats['files_parsed']} files, {stats['symbols_found']} symbols in {parse_time:.1f}s",
        "passed": passed,
    })

    # Test symbol search
    results["tests"] += 1
    symbol_results = search_symbol(conn_cg, "upsert_fact")
    passed = len(symbol_results) > 0
    results["passed"] += int(passed)
    results["details"].append({
        "test": f"search_symbol('upsert_fact') found {len(symbol_results)} results",
        "passed": passed,
    })

    # Test dependency tracking
    results["tests"] += 1
    deps = get_dependents(conn_cg, "memory/db.py")
    passed = len(deps) > 0
    results["passed"] += int(passed)
    results["details"].append({
        "test": f"get_dependents('memory/db.py') found {len(deps)} files",
        "passed": passed,
    })

    # Test impact analysis
    results["tests"] += 1
    impact = get_impact_analysis(conn_cg, "memory/db.py")
    passed = len(impact.get("dependents", [])) > 0 and len(impact.get("symbols", [])) > 0
    results["passed"] += int(passed)
    results["details"].append({
        "test": f"impact_analysis: {len(impact.get('dependents', []))} dependents, {len(impact.get('symbols', []))} symbols",
        "passed": passed,
    })

    # Test re-parse skips unchanged
    results["tests"] += 1
    stats2 = parse_repo(repo_root, conn_cg, scope, max_files=50)
    passed = stats2["skipped_unchanged"] == stats["files_parsed"]
    results["passed"] += int(passed)
    results["details"].append({
        "test": f"re-parse skipped {stats2['skipped_unchanged']} unchanged files",
        "passed": passed,
    })

    conn_cg.close()
    db_path_cg.unlink(missing_ok=True)

    return results


def run_all():
    # Suppress noisy stderr from parallel retrieval connection errors
    import os
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    print("=" * 70)
    print("  CODING MEMORY BENCHMARK")
    print("=" * 70)
    sys.stdout.flush()

    scope = "/tmp/project"
    total_pass = 0
    total_tests = 0

    # ── Correctness benchmarks (base data only) ──────────────────────────
    conn, db_path = _setup_db(scale=1)

    benchmarks = [
        ("Guardrail Recall (file-scoped)", bench_guardrail_recall),
        ("Procedure Recall (semantic)", bench_procedure_recall),
        ("Error→Solution Recall", bench_error_recall),
        ("Importance Ranking", bench_importance_ranking),
        ("Path-Scoped Fact Recall", bench_path_scoped_recall),
        ("Bi-Temporal Invalidation", bench_bitemporal),
        ("Session Recall Formatting", bench_session_recall_format),
        ("Outcome-Based Scoring", bench_outcome_scoring),
        ("Failure Probability Ranking", bench_failure_priority),
        ("Community Summaries", bench_community_summaries),
        ("Coherence Check", bench_coherence_check),
    ]

    # Code graph benchmark (uses its own DB, not conn)
    cg_result = bench_code_graph(scope)
    passed = cg_result["passed"]
    tests = cg_result["tests"]
    total_pass += passed
    total_tests += tests
    status = "✓" if passed == tests else "✗"
    print(f"\n  {status} Code Graph (AST parsing): {passed}/{tests}")
    for d in cg_result["details"]:
        marker = "  ✓" if d.get("passed", False) else "  ✗"
        parts = [f"{k}={v}" for k, v in d.items() if k not in ("passed",)]
        print(f"    {marker} {', '.join(parts)}")

    for name, bench_fn in benchmarks:
        result = bench_fn(conn, scope)
        passed = result["passed"]
        tests = result["tests"]
        total_pass += passed
        total_tests += tests
        status = "✓" if passed == tests else "✗"
        print(f"\n  {status} {name}: {passed}/{tests}")
        for d in result["details"]:
            marker = "  ✓" if d.get("passed", d.get("found", d.get("hit", d.get("present", False)))) else "  ✗"
            # Format detail line
            detail_parts = []
            for k, v in d.items():
                if k in ("passed", "found", "hit", "present"):
                    continue
                detail_parts.append(f"{k}={v}")
            print(f"    {marker} {', '.join(detail_parts)}")

    conn.close()
    db_path.unlink(missing_ok=True)

    # ── Latency benchmarks at different scales ───────────────────────────
    print(f"\n{'─' * 70}")
    print("  LATENCY BENCHMARKS")
    print(f"{'─' * 70}")

    for scale in [0, 100, 500, 1000, 2000]:
        conn, db_path = _setup_db(scale=scale)
        n = conn.execute("SELECT COUNT(*) FROM facts WHERE is_active = TRUE").fetchone()[0]
        latency = bench_latency(conn, db_path, scope, n)
        print(f"\n  Items: {n:,}")
        for metric, ms in latency["timings"].items():
            bar = "█" * max(1, int(ms / 2))
            print(f"    {metric:<30s} {ms:>8.1f}ms  {bar}")
        conn.close()
        db_path.unlink(missing_ok=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TOTAL: {total_pass}/{total_tests} correctness tests passed")
    pct = (total_pass / total_tests * 100) if total_tests else 0
    print(f"  Score: {pct:.0f}%")
    print(f"{'=' * 70}")

    return total_pass == total_tests


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
