"""
test_corpus.py — Large realistic corpus for continuous integration testing.

Simulates ~6 weeks of a senior engineer using Claude Code across 3 projects:
  - backend-api: FastAPI + DuckDB service
  - web-dashboard: Next.js + Cytoscape dashboard
  - infra: Terraform + Kubernetes deployment

Contains:
  - 80+ facts (long/medium/short, various scopes, some decayed)
  - 15 decisions
  - 8 guardrails linked to specific files
  - 6 procedures
  - 10 error solutions
  - 30+ entities with relationships
  - 10 sessions with narratives
  - 5 observations
  - Cross-scope facts that should auto-promote
  - Contradictory pairs for coherence testing
  - Deactivated items that should never surface
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

# These must be supplied by the caller
_mock_embed = None
_noop_decay = None


def set_helpers(mock_embed_fn, noop_decay_fn):
    """Set the mock embedding and decay functions (avoids circular imports)."""
    global _mock_embed, _noop_decay
    _mock_embed = mock_embed_fn
    _noop_decay = noop_decay_fn


SCOPE_BACKEND = "/home/user/projects/backend-api"
SCOPE_FRONTEND = "/home/user/projects/web-dashboard"
SCOPE_INFRA = "/home/user/projects/infra"
GLOBAL = "__global__"


def build_corpus(conn, db) -> dict:
    """Populate DB with a large realistic corpus. Returns metadata dict."""
    assert _mock_embed is not None, "Call set_helpers() first"

    now = datetime.now(timezone.utc)
    meta = {
        "scopes": [SCOPE_BACKEND, SCOPE_FRONTEND, SCOPE_INFRA],
        "fact_ids": {},
        "guardrail_ids": {},
        "session_ids": [],
        "entity_names": [],
    }

    # ── Sessions (10 sessions over 6 weeks) ────────────────────────────
    sessions = [
        ("sess-01", "PreCompact/pass1", SCOPE_BACKEND, 40,
         "Set up FastAPI backend with DuckDB, JWT auth, and CORS"),
        ("sess-02", "SessionEnd", SCOPE_BACKEND, 25,
         "Debugged DuckDB lock contention with multiple Claude sessions"),
        ("sess-03", "PreCompact/pass1", SCOPE_FRONTEND, 50,
         "Built Next.js dashboard with Cytoscape graph and Tailwind styling"),
        ("sess-04", "SessionEnd", SCOPE_FRONTEND, 20,
         "Fixed hydration errors and CORS between frontend and backend"),
        ("sess-05", "PreCompact/pass2", SCOPE_BACKEND, 65,
         "Implemented retry logic, connection backoff, and read-only optimization"),
        ("sess-06", "SessionEnd", SCOPE_INFRA, 30,
         "Set up Terraform for AWS infra and Kubernetes deployment manifests"),
        ("sess-07", "PreCompact/pass1", SCOPE_BACKEND, 35,
         "Added extraction validation, review queue, and correction detection"),
        ("sess-08", "SessionEnd", SCOPE_FRONTEND, 15,
         "Added heatmap mode and resizable panels to knowledge graph"),
        ("sess-09", "PreCompact/pass1", SCOPE_INFRA, 45,
         "Configured monitoring with Grafana dashboards and PagerDuty alerts"),
        ("sess-10", "SessionEnd", SCOPE_BACKEND, 20,
         "Performance tuning: added BM25 indexes, HNSW for embeddings"),
    ]
    for i, (sid, trigger, cwd, msgs, summary) in enumerate(sessions):
        ts = now - timedelta(days=42 - i * 4)
        db.upsert_session(conn, sid, trigger, cwd, f"/tmp/{sid}.jsonl", msgs, summary, scope=cwd)
        conn.execute("UPDATE sessions SET created_at = ? WHERE id = ?", [ts, sid])
        meta["session_ids"].append(sid)

    # ── Entities (30+) ──────────────────────────────────────────────────
    entities = [
        ("DuckDB", "technology"), ("FastAPI", "technology"), ("Python", "technology"),
        ("React", "technology"), ("Next.js", "technology"), ("TypeScript", "technology"),
        ("Cytoscape.js", "technology"), ("Tailwind CSS", "technology"),
        ("PostgreSQL", "technology"), ("Redis", "technology"), ("SQLite", "technology"),
        ("Terraform", "technology"), ("Kubernetes", "technology"), ("Docker", "technology"),
        ("AWS", "technology"), ("Grafana", "technology"), ("PagerDuty", "technology"),
        ("WAL", "technology"), ("CORS", "concept"), ("JWT", "concept"),
        ("Connection Pool", "concept"), ("BM25", "concept"), ("HNSW", "concept"),
        ("Ben", "person"), ("Sarah", "person"),
        ("Authentication", "concept"), ("Rate Limiting", "concept"),
        ("CI/CD", "concept"), ("Observability", "concept"),
        ("Ollama", "technology"), ("nomic-embed-text", "technology"),
    ]
    for name, etype in entities:
        db.upsert_entity(conn, name, entity_type=etype, embedding=_mock_embed(name))
        meta["entity_names"].append(name)

    # ── Relationships (25+) ─────────────────────────────────────────────
    rels = [
        ("FastAPI", "DuckDB", "uses", "FastAPI stores data in DuckDB"),
        ("FastAPI", "Python", "implemented_in", "FastAPI is Python-based"),
        ("FastAPI", "JWT", "uses", "JWT for authentication tokens"),
        ("React", "TypeScript", "implemented_in", "Frontend in TypeScript"),
        ("React", "Cytoscape.js", "uses", "Graph rendering via Cytoscape"),
        ("React", "Next.js", "built_with", "Dashboard uses Next.js 16"),
        ("Next.js", "Tailwind CSS", "uses", "Styling with Tailwind"),
        ("DuckDB", "WAL", "uses", "Write-ahead logging for safety"),
        ("DuckDB", "Connection Pool", "requires", "Single-writer needs pooling"),
        ("DuckDB", "BM25", "supports", "Full-text search via FTS extension"),
        ("DuckDB", "HNSW", "supports", "Vector search via VSS extension"),
        ("Terraform", "AWS", "provisions", "Terraform manages AWS resources"),
        ("Kubernetes", "Docker", "orchestrates", "K8s runs Docker containers"),
        ("Grafana", "Observability", "provides", "Dashboards for monitoring"),
        ("PagerDuty", "Observability", "provides", "Alerting for incidents"),
        ("Ben", "DuckDB", "uses", "Ben works with DuckDB daily"),
        ("Ben", "React", "uses", "Ben builds the dashboard"),
        ("Ben", "Terraform", "uses", "Ben manages infra"),
        ("Sarah", "FastAPI", "uses", "Sarah works on the API"),
        ("Ollama", "nomic-embed-text", "serves", "Local embedding model"),
        ("Authentication", "JWT", "uses", "Auth uses JWT tokens"),
        ("Authentication", "Rate Limiting", "requires", "Auth endpoints are rate limited"),
        ("CI/CD", "Docker", "uses", "CI builds Docker images"),
        ("CI/CD", "Kubernetes", "deploys_to", "CD deploys to K8s"),
    ]
    for fr, to, rt, desc in rels:
        db.upsert_relationship(conn, fr, to, rt, desc, "sess-01")

    # ── Facts (80+) ─────────────────────────────────────────────────────
    facts = [
        # Backend architecture (scope: backend, long-term)
        ("be_duckdb_writer", "DuckDB enforces single-writer concurrency: only one write connection at a time", "architecture", "long", "high", "sess-02", SCOPE_BACKEND, 9),
        ("be_fastapi_port", "FastAPI backend runs on port 8000 with uvicorn and auto-reload in dev", "operational", "long", "high", "sess-01", SCOPE_BACKEND, 7),
        ("be_jwt_auth", "Authentication uses JWT tokens in httpOnly cookies with 24h expiry", "architecture", "long", "high", "sess-01", SCOPE_BACKEND, 8),
        ("be_retry_backoff", "Connection retry uses exponential backoff: 150ms base delay, 5 max retries, doubling each time", "implementation", "long", "high", "sess-05", SCOPE_BACKEND, 9),
        ("be_read_only", "All read operations use read-only DB connections to avoid blocking writers", "architecture", "long", "high", "sess-05", SCOPE_BACKEND, 8),
        ("be_db_path", "Database file lives at ~/.claude/memory/knowledge.duckdb, shared across all projects", "operational", "long", "high", "sess-01", SCOPE_BACKEND, 6),
        ("be_fts_install", "FTS extension must be installed per-connection via INSTALL fts; LOAD fts", "operational", "medium", "medium", "sess-02", SCOPE_BACKEND, 5),
        ("be_vss_hnsw", "VSS extension provides HNSW indexes for fast vector similarity search", "architecture", "long", "high", "sess-10", SCOPE_BACKEND, 7),
        ("be_scope_filter", "Every query filters by scope: project-local items first, then __global__", "architecture", "long", "high", "sess-01", SCOPE_BACKEND, 7),
        ("be_migration_system", "Schema migrations are versioned and applied sequentially on first write connection per process", "implementation", "long", "high", "sess-05", SCOPE_BACKEND, 6),
        ("be_embedding_dim", "Embeddings are 768-dimensional float vectors from nomic-embed-text via Ollama", "architecture", "long", "high", "sess-01", SCOPE_BACKEND, 7),
        ("be_dedup_threshold", "Deduplication uses cosine similarity threshold of 0.92 — above means same item", "implementation", "long", "high", "sess-01", SCOPE_BACKEND, 8),
        ("be_recall_threshold", "Recall threshold is 0.60 cosine similarity — below is not relevant enough", "implementation", "long", "medium", "sess-01", SCOPE_BACKEND, 6),
        ("be_decay_rates", "Decay rates: short=0.18/day, medium=0.04/day, long=0.007/day exponential", "implementation", "long", "high", "sess-01", SCOPE_BACKEND, 7),
        ("be_token_budget_session", "Session context token budget is 3000 tokens (~12000 chars)", "operational", "medium", "high", "sess-05", SCOPE_BACKEND, 5),
        ("be_token_budget_prompt", "Per-prompt context token budget is 4000 tokens (~16000 chars)", "operational", "medium", "high", "sess-05", SCOPE_BACKEND, 5),
        ("be_api_model", "Extraction uses Claude Sonnet 4.6 with tool_use for structured output", "architecture", "long", "high", "sess-01", SCOPE_BACKEND, 7),
        ("be_extraction_thresholds", "Extraction triggers at 40%, 70%, 90% context window usage", "operational", "long", "high", "sess-05", SCOPE_BACKEND, 8),
        ("be_parallel_retrieve", "Prompt recall uses 4-way parallel retrieval: semantic + BM25 + graph + temporal", "architecture", "long", "high", "sess-10", SCOPE_BACKEND, 9),
        ("be_community_detection", "Community detection clusters facts by shared entities using union-find algorithm", "architecture", "long", "high", "sess-07", SCOPE_BACKEND, 6),

        # Frontend facts (scope: frontend)
        ("fe_nextjs_stack", "Dashboard uses Next.js 16 with React 19, Tailwind CSS 4, and shadcn/ui", "architecture", "long", "high", "sess-03", SCOPE_FRONTEND, 8),
        ("fe_cytoscape", "Graph visualization uses Cytoscape.js v3.33 with fcose layout engine", "architecture", "long", "high", "sess-03", SCOPE_FRONTEND, 7),
        ("fe_cors_config", "CORS configured in FastAPI for localhost:3000 and localhost:3001 origins", "operational", "medium", "high", "sess-04", SCOPE_FRONTEND, 6),
        ("fe_dark_mode", "Dark mode via CSS custom properties on html element, default theme is dark", "implementation", "medium", "medium", "sess-08", SCOPE_FRONTEND, 4),
        ("fe_ssr_false", "Cytoscape components use dynamic import with ssr: false to avoid server rendering", "implementation", "long", "high", "sess-03", SCOPE_FRONTEND, 7),
        ("fe_polling", "Dashboard uses custom usePolling hook for real-time data (3-10s intervals)", "implementation", "long", "high", "sess-03", SCOPE_FRONTEND, 6),
        ("fe_heatmap", "Knowledge graph has heatmap mode coloring nodes by degree/importance/recency", "implementation", "long", "high", "sess-08", SCOPE_FRONTEND, 5),
        ("fe_markdown", "Chat panel uses react-markdown with remark-gfm for rich text rendering", "implementation", "medium", "high", "sess-08", SCOPE_FRONTEND, 5),
        ("fe_resizable", "Knowledge page has resizable graph+chat split layout with drag handle", "implementation", "medium", "medium", "sess-08", SCOPE_FRONTEND, 4),
        ("fe_ndjson", "Chat streams responses via NDJSON: sources event, text chunks, done event", "implementation", "long", "high", "sess-08", SCOPE_FRONTEND, 7),

        # Infra facts (scope: infra)
        ("infra_terraform", "Infrastructure managed with Terraform v1.7, state in S3 backend", "architecture", "long", "high", "sess-06", SCOPE_INFRA, 8),
        ("infra_k8s_cluster", "Production runs on EKS cluster with 3 node groups in us-east-1", "architecture", "long", "high", "sess-06", SCOPE_INFRA, 8),
        ("infra_docker_base", "All services use python:3.13-slim as base Docker image", "operational", "long", "high", "sess-06", SCOPE_INFRA, 6),
        ("infra_grafana_url", "Grafana dashboard at grafana.internal/d/memory-system for monitoring", "operational", "long", "high", "sess-09", SCOPE_INFRA, 7),
        ("infra_pagerduty", "PagerDuty escalation: p1 pages oncall in 5min, p2 in 30min", "operational", "long", "high", "sess-09", SCOPE_INFRA, 8),
        ("infra_ci_github", "CI runs on GitHub Actions: lint → test → build → push → deploy", "operational", "long", "high", "sess-06", SCOPE_INFRA, 7),
        ("infra_secrets", "Secrets managed via AWS Secrets Manager, injected as env vars in K8s pods", "architecture", "long", "high", "sess-06", SCOPE_INFRA, 8),
        ("infra_rds_backup", "RDS snapshots taken daily at 03:00 UTC, retained for 7 days", "operational", "long", "high", "sess-09", SCOPE_INFRA, 7),
        ("infra_scaling", "Backend API auto-scales 2-10 pods based on CPU >70% for 2 minutes", "operational", "long", "high", "sess-09", SCOPE_INFRA, 7),
        ("infra_networking", "Services communicate via internal ALB, no public internet for DB traffic", "architecture", "long", "high", "sess-06", SCOPE_INFRA, 8),

        # Global facts (user preferences, cross-project)
        ("gl_terse_code", "Ben prefers short, direct code without unnecessary abstractions or over-engineering", "user_preference", "long", "high", "sess-01", GLOBAL, 10),
        ("gl_tdd", "Always write tests before implementing new features using red/green methodology", "user_preference", "long", "high", "sess-03", GLOBAL, 10),
        ("gl_uv_runner", "All Python projects use uv as the package runner and dependency manager", "operational", "long", "high", "sess-01", GLOBAL, 7),
        ("gl_git_convention", "Commit messages: imperative mood, 72-char subject, blank line, body with context", "user_preference", "long", "high", "sess-01", GLOBAL, 8),
        ("gl_no_emoji", "Do not use emojis in code, commits, or documentation unless explicitly requested", "user_preference", "long", "high", "sess-01", GLOBAL, 9),
        ("gl_security_first", "Always consider OWASP top 10 when writing code that handles user input", "user_preference", "long", "high", "sess-01", GLOBAL, 9),
        ("gl_ollama_model", "Embedding model is nomic-embed-text served locally via Ollama on port 11434", "operational", "long", "high", "sess-01", GLOBAL, 6),
        ("gl_anthropic_key", "ANTHROPIC_API_KEY must be set in environment for extraction to work", "operational", "long", "high", "sess-01", GLOBAL, 6),
        ("gl_ben_role", "Ben is a senior distributed systems engineer focused on reliability and performance", "user_preference", "long", "high", "sess-01", GLOBAL, 8),
        ("gl_sarah_role", "Sarah is a backend engineer who handles the API layer and database migrations", "user_preference", "long", "high", "sess-05", GLOBAL, 6),

        # Contradictory pairs (outdated facts that conflict with correct ones)
        ("contra_postgres", "PostgreSQL is the primary database for the backend API", "architecture", "medium", "medium", "sess-01", SCOPE_BACKEND, 3),
        ("contra_port", "The API server runs on port 3000 with express", "operational", "short", "low", "sess-01", SCOPE_BACKEND, 2),

        # Medium-term facts (should appear in session context but lower priority)
        ("med_pr_review", "All PRs require at least one approval before merge to main", "operational", "medium", "high", "sess-06", GLOBAL, 6),
        ("med_branch_naming", "Branch naming convention: feature/JIRA-123-short-description", "operational", "medium", "high", "sess-06", GLOBAL, 5),
        ("med_lock_issue", "DuckDB lock contention was observed when multiple Claude sessions run simultaneously", "bug_pattern", "medium", "high", "sess-02", SCOPE_BACKEND, 7),

        # Short-term facts (should decay and eventually disappear)
        ("short_debug_flag", "DEBUG=1 is currently set in the dev environment for verbose logging", "operational", "short", "low", "sess-05", SCOPE_BACKEND, 3),
        ("short_temp_workaround", "Temporary: skip FTS rebuild on read-only connections to avoid lock issues", "operational", "short", "medium", "sess-05", SCOPE_BACKEND, 4),

        # Deactivated facts (should never surface)
        ("dead_sqlite", "The backend initially used SQLite before switching to DuckDB", "architecture", "short", "low", "sess-01", SCOPE_BACKEND, 2),
        ("dead_express", "The API was originally built with Express.js before migrating to FastAPI", "architecture", "short", "low", "sess-01", SCOPE_BACKEND, 2),
    ]
    for label, text, cat, tc, conf, sid, scope, imp in facts:
        emb = _mock_embed(text)
        fid, _ = db.upsert_fact(
            conn, text, cat, tc, conf, emb, sid, _noop_decay,
            scope=scope, importance=imp,
        )
        meta["fact_ids"][label] = fid

    # Deactivate dead facts
    for label in ("dead_sqlite", "dead_express"):
        conn.execute("UPDATE facts SET is_active = FALSE, deactivated_at = ? WHERE id = ?",
                     [now, meta["fact_ids"][label]])

    # ── Fact-entity links ───────────────────────────────────────────────
    entity_links = {
        "be_duckdb_writer": ["DuckDB", "Connection Pool"],
        "be_fastapi_port": ["FastAPI"],
        "be_jwt_auth": ["Authentication", "JWT", "FastAPI"],
        "be_retry_backoff": ["DuckDB", "Connection Pool"],
        "be_parallel_retrieve": ["BM25", "HNSW"],
        "fe_nextjs_stack": ["React", "Next.js", "TypeScript", "Tailwind CSS"],
        "fe_cytoscape": ["Cytoscape.js", "React"],
        "infra_terraform": ["Terraform", "AWS"],
        "infra_k8s_cluster": ["Kubernetes", "AWS"],
        "infra_grafana_url": ["Grafana", "Observability"],
        "infra_pagerduty": ["PagerDuty", "Observability"],
        "infra_ci_github": ["CI/CD", "Docker"],
        "gl_ollama_model": ["Ollama", "nomic-embed-text"],
    }
    for fact_label, entity_names in entity_links.items():
        if fact_label in meta["fact_ids"]:
            db.link_fact_entities(conn, meta["fact_ids"][fact_label], entity_names)

    # ── Decisions ───────────────────────────────────────────────────────
    decisions = [
        ("Use DuckDB instead of PostgreSQL: local, embedded, no server dependency", "long", "sess-01", SCOPE_BACKEND),
        ("Use read-only connections for all reads to avoid writer blocking", "long", "sess-05", SCOPE_BACKEND),
        ("Bundle graph and chat into single knowledge page rather than separate apps", "long", "sess-08", SCOPE_FRONTEND),
        ("Use Cytoscape.js over D3 for graph: better API, fcose layout, compound nodes", "long", "sess-03", SCOPE_FRONTEND),
        ("Use Terraform over CloudFormation: multi-cloud, better state management", "long", "sess-06", SCOPE_INFRA),
        ("Deploy as Docker containers on EKS rather than Lambda: long-running connections", "long", "sess-06", SCOPE_INFRA),
        ("Use nomic-embed-text locally via Ollama rather than API embeddings: privacy + speed", "long", "sess-01", GLOBAL),
        ("Use exponential backoff with jitter for all retry logic", "long", "sess-05", GLOBAL),
        ("Prefer BM25 keyword search as fallback when embeddings unavailable", "long", "sess-10", SCOPE_BACKEND),
        ("Use NDJSON streaming over WebSockets for chat: simpler, stateless, cacheable", "long", "sess-08", SCOPE_FRONTEND),
        ("Auto-snapshot DB on every session end for safety", "long", "sess-07", GLOBAL),
        ("Use git stash for guardrail enforcement: reversible, non-destructive", "long", "sess-07", GLOBAL),
        ("Extraction validation rejects bare URLs and meta-commentary facts", "long", "sess-07", SCOPE_BACKEND),
        ("Token budget prioritizes guardrails > procedures > long facts > decisions", "long", "sess-05", SCOPE_BACKEND),
        ("Community detection uses union-find on shared entities with min overlap 2", "long", "sess-07", SCOPE_BACKEND),
    ]
    for text, tc, sid, scope in decisions:
        db.upsert_decision(conn, text, tc, _mock_embed(text), sid, _noop_decay, scope=scope)

    # ── Guardrails ──────────────────────────────────────────────────────
    guardrails = [
        ("Never modify the retry logic in db.py without running concurrency tests",
         "Fragile concurrent code debugged extensively over 3 sessions",
         "Connection failures in production, data corruption risk",
         ["memory/db.py"], "sess-05", SCOPE_BACKEND),
        ("Do not store session tokens in localStorage: use httpOnly cookies only",
         "XSS vulnerability if tokens in JS-accessible storage",
         "Security breach: token theft via cross-site scripting",
         ["backend/auth.py", "backend/middleware.py"], "sess-01", SCOPE_BACKEND),
        ("Always run next build before deploying: never deploy dev mode",
         "Dev mode exposes source maps, debug endpoints, and verbose errors",
         "Source code and internal API structure exposed to users",
         ["dashboard/frontend/package.json"], "sess-03", SCOPE_FRONTEND),
        ("Do not modify Terraform state files manually: always use terraform apply",
         "Manual state edits cause drift and can destroy resources on next apply",
         "Accidental deletion of production infrastructure",
         ["infra/main.tf", "infra/state.tf"], "sess-06", SCOPE_INFRA),
        ("Never skip CI checks with --no-verify: fix the failing check instead",
         "Skipped checks have caused broken deploys twice in the past month",
         "Broken code in production, rollback required",
         [], "sess-06", GLOBAL),
        ("Do not increase PROMPT_TOKEN_BUDGET above 5000 without testing recall quality",
         "Higher budgets risk injecting too much low-relevance context that confuses Claude",
         "Degraded response quality from context pollution",
         ["memory/config.py"], "sess-05", SCOPE_BACKEND),
        ("Never delete production DuckDB file without creating a snapshot first",
         "The DB contains months of accumulated knowledge with no other backup",
         "Total knowledge loss: all facts, decisions, entities, relationships gone",
         ["memory/db.py", "memory/backup.py"], "sess-07", SCOPE_BACKEND),
        ("Do not modify the extraction prompt without running the fidelity benchmark",
         "Small prompt changes can dramatically change extraction quality",
         "Silent quality degradation in fact extraction",
         ["memory/extract.py"], "sess-07", SCOPE_BACKEND),
    ]
    for warning, rationale, consequence, files, sid, scope in guardrails:
        gid, _ = db.upsert_guardrail(
            conn, warning=warning, rationale=rationale, consequence=consequence,
            file_paths=files, embedding=_mock_embed(warning), session_id=sid, scope=scope,
        )
        meta["guardrail_ids"][warning[:40]] = gid

    # ── Procedures ──────────────────────────────────────────────────────
    procedures = [
        ("Deploy backend to production",
         "1. Run python3 test_memory.py (all green) 2. docker build -t api . 3. docker push 4. kubectl rollout restart deployment/api",
         ["Dockerfile", "k8s/deployment.yaml"], "sess-06", SCOPE_BACKEND),
        ("Run the full test suite",
         "1. Ensure Ollama is running (ollama serve) 2. python3 test_memory.py 3. Verify 0 failures 4. Check bench/live_bench.py for regressions",
         ["test_memory.py"], "sess-05", GLOBAL),
        ("Add a new database migration",
         "1. Find max version in MIGRATIONS list in db.py 2. Add (N+1, description, sql) tuple 3. Run tests 4. Verify with python -m memory stats",
         ["memory/db.py"], "sess-07", SCOPE_BACKEND),
        ("Build and deploy the frontend",
         "1. cd dashboard/frontend 2. npm run build 3. Copy out/ to static hosting 4. Verify at dashboard.internal",
         ["dashboard/frontend/package.json"], "sess-03", SCOPE_FRONTEND),
        ("Rotate AWS secrets",
         "1. Generate new credentials in AWS Console 2. Update in Secrets Manager 3. kubectl rollout restart 4. Verify in Grafana",
         ["infra/secrets.tf"], "sess-09", SCOPE_INFRA),
        ("Respond to a production incident",
         "1. Check PagerDuty alert details 2. Open Grafana dashboard 3. Check K8s pod logs 4. Identify root cause 5. Apply fix 6. Write postmortem",
         [], "sess-09", GLOBAL),
    ]
    for task, steps, files, sid, scope in procedures:
        db.upsert_procedure(
            conn, task_description=task, steps=steps, file_paths=files,
            embedding=_mock_embed(task), session_id=sid, scope=scope,
        )

    # ── Error Solutions ─────────────────────────────────────────────────
    error_solutions = [
        ("DuckDB IOException: Could not set lock on file knowledge.duckdb",
         "Find blocking process with ps -p <PID>. Either wait for it to finish or kill it. The lock is held by another Claude Code session.",
         ["memory/db.py"], "sess-02", SCOPE_BACKEND),
        ("CORS error: No Access-Control-Allow-Origin header present",
         "Add the frontend origin to allow_origins list in server.py. Check if the origin includes the port number.",
         ["dashboard/backend/server.py"], "sess-04", SCOPE_FRONTEND),
        ("next build fails: pre cannot be a descendant of p",
         "Fix react-markdown code component: use separate pre and code overrides instead of checking inline prop.",
         ["dashboard/frontend/src/components/chat-panel.tsx"], "sess-08", SCOPE_FRONTEND),
        ("ImportError: No module named 'memory'",
         "Add sys.path.insert(0, str(Path.home() / '.claude')) at the top of the script before importing memory.",
         ["hooks/session_start.py"], "sess-01", SCOPE_BACKEND),
        ("DuckDB CatalogError: table not found after migration",
         "Delete knowledge.duckdb and restart to re-run all migrations from scratch. Or run get_connection(read_only=False) to trigger migration.",
         ["memory/db.py"], "sess-02", SCOPE_BACKEND),
        ("Terraform state lock: ConditionalCheckFailedException",
         "Run terraform force-unlock <LOCK_ID> after confirming no other apply is running.",
         ["infra/main.tf"], "sess-06", SCOPE_INFRA),
        ("kubectl: pod CrashLoopBackOff",
         "Check logs with kubectl logs <pod> --previous. Common causes: missing env var, OOM, failing health check.",
         [], "sess-09", SCOPE_INFRA),
        ("Embedding dimension mismatch: expected 768 got 384",
         "Ensure Ollama is serving nomic-embed-text (768-dim), not all-minilm (384-dim). Run: ollama pull nomic-embed-text",
         ["memory/embeddings.py", "memory/config.py"], "sess-01", SCOPE_BACKEND),
        ("React hydration mismatch: server/client content differs",
         "Wrap browser-only code in useEffect or use dynamic import with ssr: false. Common with Cytoscape and other canvas libs.",
         ["dashboard/frontend/src/components/entity-graph.tsx"], "sess-04", SCOPE_FRONTEND),
        ("BM25 search returns 'match_bm25 does not exist'",
         "FTS extension not loaded. Call db.rebuild_fts_indexes(conn) after inserting data, or ensure LOAD fts runs on the connection.",
         ["memory/db.py"], "sess-10", SCOPE_BACKEND),
    ]
    for pattern, solution, files, sid, scope in error_solutions:
        db.upsert_error_solution(
            conn, error_pattern=pattern, solution=solution,
            file_paths=files, embedding=_mock_embed(pattern), session_id=sid, scope=scope,
        )

    # ── Observations ────────────────────────────────────────────────────
    obs_data = [
        ("DuckDB's single-writer model requires careful connection management with retry logic, read-only connections for reads, and per-process init caching to avoid lock contention",
         ["be_duckdb_writer", "be_retry_backoff", "be_read_only"], SCOPE_BACKEND),
        ("The web dashboard is a Next.js 16 app using Cytoscape.js for graph rendering, Tailwind for styling, and NDJSON streaming for the chat interface",
         ["fe_nextjs_stack", "fe_cytoscape", "fe_ndjson"], SCOPE_FRONTEND),
        ("Infrastructure follows IaC principles with Terraform on AWS, EKS for orchestration, and Grafana+PagerDuty for observability",
         ["infra_terraform", "infra_k8s_cluster", "infra_grafana_url"], SCOPE_INFRA),
        ("The memory system extracts knowledge at 40/70/90% context thresholds using Claude Sonnet, validates inline, and stores with temporal decay and scope management",
         ["be_extraction_thresholds", "be_api_model", "be_scope_filter"], SCOPE_BACKEND),
        ("The retrieval pipeline combines 4 strategies (semantic, BM25, graph, temporal) fused via reciprocal rank fusion with a 250ms timeout budget",
         ["be_parallel_retrieve"], SCOPE_BACKEND),
    ]
    for text, fact_labels, scope in obs_data:
        fact_ids = [meta["fact_ids"][l] for l in fact_labels if l in meta["fact_ids"]]
        db.upsert_observation(conn, text, fact_ids, _mock_embed(text), scope=scope)

    # ── Session Narratives ──────────────────────────────────────────────
    narratives = [
        ("sess-05", 1, False,
         "Identified DuckDB lock contention caused by multiple Claude Code sessions. Added retry with exponential backoff."),
        ("sess-05", 2, True,
         "Completed DuckDB concurrency fixes: retry logic, read-only optimization, per-process init caching. Added 5 concurrency tests."),
        ("sess-07", 1, True,
         "Added extraction validation (rejects bare URLs, meta-commentary, low-confidence items), review queue table, and correction detection heuristics."),
        ("sess-09", 1, True,
         "Configured Grafana dashboards for memory system metrics and PagerDuty alerting for production incidents."),
    ]
    for sid, pass_num, is_final, text in narratives:
        db.upsert_narrative(conn, sid, pass_num, text, embedding=_mock_embed(text),
                            is_final=is_final, scope=SCOPE_BACKEND)

    # Build FTS indexes for BM25 search
    db.rebuild_fts_indexes(conn)

    return meta
