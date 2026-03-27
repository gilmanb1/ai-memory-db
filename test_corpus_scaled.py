"""
test_corpus_scaled.py — Scaled corpus generators for 1 day, 1 week, 1 month, 1 year.

Each corpus simulates realistic engineering usage at that timescale.
All generators share the same structure: build_corpus_Xd/Xw/Xm/Xy(conn, db) -> meta dict.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

_mock_embed: Callable | None = None
_noop_decay: Callable | None = None


def set_helpers(mock_embed_fn, noop_decay_fn):
    global _mock_embed, _noop_decay
    _mock_embed = mock_embed_fn
    _noop_decay = noop_decay_fn


# ── Vocabulary pools for procedural generation ──────────────────────────

SCOPES = [
    "/home/user/projects/backend-api",
    "/home/user/projects/web-dashboard",
    "/home/user/projects/infra",
    "/home/user/projects/mobile-app",
    "/home/user/projects/data-pipeline",
]

TECHNOLOGIES = [
    ("DuckDB", "technology"), ("FastAPI", "technology"), ("Python", "technology"),
    ("React", "technology"), ("Next.js", "technology"), ("TypeScript", "technology"),
    ("PostgreSQL", "technology"), ("Redis", "technology"), ("Docker", "technology"),
    ("Kubernetes", "technology"), ("Terraform", "technology"), ("AWS", "technology"),
    ("Grafana", "technology"), ("Kafka", "technology"), ("Elasticsearch", "technology"),
    ("GraphQL", "technology"), ("gRPC", "technology"), ("Nginx", "technology"),
    ("Celery", "technology"), ("RabbitMQ", "technology"), ("MongoDB", "technology"),
    ("Prometheus", "technology"), ("Jaeger", "technology"), ("Vault", "technology"),
    ("Go", "technology"), ("Rust", "technology"), ("Swift", "technology"),
    ("SQLAlchemy", "technology"), ("Pydantic", "technology"), ("pytest", "technology"),
]

PEOPLE = [
    ("Ben", "person"), ("Sarah", "person"), ("Alex", "person"),
    ("Jordan", "person"), ("Casey", "person"), ("Morgan", "person"),
]

CONCEPTS = [
    ("Authentication", "concept"), ("Rate Limiting", "concept"),
    ("Caching", "concept"), ("CI/CD", "concept"), ("Observability", "concept"),
    ("Connection Pool", "concept"), ("Load Balancing", "concept"),
    ("Circuit Breaker", "concept"), ("Event Sourcing", "concept"),
    ("CQRS", "concept"), ("Microservices", "concept"), ("API Gateway", "concept"),
    ("Service Mesh", "concept"), ("Blue-Green Deploy", "concept"),
    ("Feature Flags", "concept"), ("A/B Testing", "concept"),
]

CATEGORIES = ["architecture", "implementation", "operational", "dependency",
              "decision_rationale", "constraint", "bug_pattern", "user_preference"]

FACT_TEMPLATES = [
    "{tech} uses {concept} for {purpose}",
    "The {scope_name} service runs on port {port} with {tech}",
    "{tech} is configured with {setting} = {value} in production",
    "When {tech} fails, the system falls back to {fallback}",
    "{person} prefers {preference} when working with {tech}",
    "The {scope_name} team uses {tech} for {purpose}",
    "{tech} requires {dependency} to be running before startup",
    "Latency threshold for {scope_name} is {latency}ms at p99",
    "The {tech} cluster has {count} nodes in {region}",
    "{tech} logs are shipped to {destination} via {shipper}",
    "Database {tech} is backed up every {interval} to {storage}",
    "{scope_name} API rate limit is {rate} requests per {period}",
    "The {tech} configuration lives in {config_file}",
    "{person} owns the {scope_name} {component} component",
    "All {tech} connections use TLS with {cert_type} certificates",
]

GUARDRAIL_TEMPLATES = [
    ("Never modify {file} without running the full test suite",
     "Changes here have caused production incidents twice",
     "Service outage affecting all users"),
    ("Do not store secrets in {file} — use environment variables",
     "Hardcoded secrets are a critical security vulnerability",
     "Credential exposure and potential data breach"),
    ("Always run {tool} before deploying {scope_name}",
     "Skipping this step has caused broken deploys",
     "Broken production deployment requiring rollback"),
    ("Do not increase {setting} above {threshold} without load testing",
     "Higher values cause memory pressure and OOM kills",
     "Pod restarts and degraded service availability"),
]

ERROR_TEMPLATES = [
    ("{tech} IOException: Connection refused on port {port}",
     "Check if {tech} is running. Restart with: {restart_cmd}"),
    ("{tech} TimeoutError: Request exceeded {timeout}ms deadline",
     "Increase timeout in config or optimize the query. Check {dashboard} for slow queries."),
    ("ImportError: No module named '{module}'",
     "Install with: pip install {module}. Or add to requirements.txt and run pip install -r requirements.txt."),
    ("{tech} OOMKilled: Container exceeded memory limit",
     "Increase memory limit in {config_file} or optimize memory usage. Check for memory leaks with {profiler}."),
]

DECISION_TEMPLATES = [
    "Use {tech_a} instead of {tech_b} for {purpose}: {reason}",
    "Deploy {scope_name} as {deploy_type} rather than {alt_deploy}: {reason}",
    "Store {data_type} in {tech} rather than {alt_tech}: {reason}",
]

REL_TYPES = ["uses", "depends_on", "implemented_in", "provides", "consumes",
             "deploys_to", "monitors", "authenticates", "caches", "queries"]


def _rand_choice(lst):
    return lst[hash(str(lst) + str(len(lst))) % len(lst)]


def _build_corpus(
    conn, db,
    num_facts: int,
    num_entities: int,
    num_rels: int,
    num_sessions: int,
    num_guardrails: int,
    num_decisions: int,
    num_errors: int,
    num_procedures: int,
    num_scopes: int,
    time_span_days: int,
    label: str,
) -> dict:
    """Generic corpus builder parameterized by size."""
    assert _mock_embed is not None, "Call set_helpers() first"

    now = datetime.now(timezone.utc)
    rng = random.Random(42)  # deterministic for reproducibility
    scopes = SCOPES[:num_scopes]
    meta = {"scopes": scopes, "fact_ids": {}, "label": label,
            "counts": {"facts": 0, "entities": 0, "relationships": 0,
                       "sessions": 0, "guardrails": 0, "decisions": 0,
                       "error_solutions": 0, "procedures": 0}}

    # ── Sessions ────────────────────────────────────────────────────────
    for i in range(num_sessions):
        sid = f"sess-{label}-{i:04d}"
        scope = scopes[i % len(scopes)]
        ts = now - timedelta(days=time_span_days - (i * time_span_days // max(num_sessions, 1)))
        trigger = rng.choice(["PreCompact/pass1", "SessionEnd", "PreCompact/pass2"])
        msgs = rng.randint(10, 80)
        summary = f"Session {i} in {Path(scope).name}: worked on various tasks"
        db.upsert_session(conn, sid, trigger, scope, f"/tmp/{sid}.jsonl", msgs, summary, scope=scope)
        conn.execute("UPDATE sessions SET created_at = ? WHERE id = ?", [ts, sid])
        meta["counts"]["sessions"] += 1

    # ── Entities ────────────────────────────────────────────────────────
    all_entities = TECHNOLOGIES[:min(num_entities, len(TECHNOLOGIES))]
    remaining = num_entities - len(all_entities)
    if remaining > 0:
        all_entities += PEOPLE[:min(remaining, len(PEOPLE))]
        remaining -= min(remaining, len(PEOPLE))
    if remaining > 0:
        all_entities += CONCEPTS[:min(remaining, len(CONCEPTS))]

    entity_names = []
    for name, etype in all_entities:
        db.upsert_entity(conn, name, entity_type=etype, embedding=_mock_embed(name))
        entity_names.append(name)
        meta["counts"]["entities"] += 1

    # ── Relationships ───────────────────────────────────────────────────
    rel_pairs = set()
    for i in range(num_rels):
        # Pick two different entities
        a = entity_names[i % len(entity_names)]
        b = entity_names[(i * 7 + 3) % len(entity_names)]
        if a == b:
            b = entity_names[(i * 7 + 5) % len(entity_names)]
        if (a, b) in rel_pairs:
            continue
        rel_pairs.add((a, b))
        rt = REL_TYPES[i % len(REL_TYPES)]
        desc = f"{a} {rt} {b}"
        sid = f"sess-{label}-{i % max(num_sessions, 1):04d}"
        db.upsert_relationship(conn, a, b, rt, desc, sid)
        meta["counts"]["relationships"] += 1

    # ── Facts ───────────────────────────────────────────────────────────
    scope_names = {s: Path(s).name for s in scopes}
    deactivated = []

    for i in range(num_facts):
        scope = scopes[i % len(scopes)] if rng.random() > 0.15 else "__global__"
        scope_name = scope_names.get(scope, "global")
        tc = rng.choices(["long", "medium", "short"], weights=[0.5, 0.35, 0.15])[0]
        conf = rng.choices(["high", "medium", "low"], weights=[0.6, 0.3, 0.1])[0]
        imp = rng.randint(3, 10)
        cat = rng.choice(CATEGORIES)
        sid = f"sess-{label}-{i % max(num_sessions, 1):04d}"

        # Generate fact text from templates
        tmpl = FACT_TEMPLATES[i % len(FACT_TEMPLATES)]
        tech = entity_names[i % len(entity_names)] if entity_names else "System"
        text = tmpl.format(
            tech=tech, concept=rng.choice(["caching", "retry", "pooling", "encryption", "compression"]),
            purpose=rng.choice(["performance", "reliability", "security", "scalability"]),
            scope_name=scope_name, port=rng.randint(3000, 9999),
            setting=rng.choice(["max_connections", "timeout_ms", "batch_size", "pool_size"]),
            value=rng.randint(10, 1000), fallback=rng.choice(["Redis", "local cache", "default config"]),
            person=rng.choice(["Ben", "Sarah", "Alex"]),
            preference=rng.choice(["type annotations", "minimal dependencies", "explicit error handling"]),
            dependency=rng.choice(["Ollama", "Redis", "PostgreSQL"]),
            latency=rng.choice([100, 200, 500, 1000]),
            count=rng.randint(2, 10), region=rng.choice(["us-east-1", "eu-west-1", "ap-southeast-1"]),
            destination=rng.choice(["Elasticsearch", "S3", "CloudWatch"]),
            shipper=rng.choice(["Fluentd", "Filebeat", "Vector"]),
            interval=rng.choice(["1 hour", "6 hours", "daily"]),
            storage=rng.choice(["S3", "GCS", "Azure Blob"]),
            rate=rng.randint(100, 10000), period=rng.choice(["minute", "second", "hour"]),
            config_file=rng.choice(["config.yaml", "settings.json", ".env"]),
            component=rng.choice(["auth", "API", "database", "frontend"]),
            cert_type=rng.choice(["Let's Encrypt", "internal CA", "self-signed"]),
        )

        fid, _ = db.upsert_fact(
            conn, text, cat, tc, conf, _mock_embed(text), sid, _noop_decay,
            scope=scope, importance=imp,
        )
        meta["fact_ids"][f"fact-{label}-{i:04d}"] = fid

        # Link some facts to entities
        if i % 3 == 0 and entity_names:
            link_entities = [entity_names[i % len(entity_names)]]
            if len(entity_names) > 1:
                link_entities.append(entity_names[(i + 1) % len(entity_names)])
            db.link_fact_entities(conn, fid, link_entities)

        # Mark some short-term facts as deactivated (simulating decay)
        if tc == "short" and rng.random() < 0.3:
            deactivated.append(fid)

        meta["counts"]["facts"] += 1

    # Deactivate some facts
    for fid in deactivated:
        conn.execute("UPDATE facts SET is_active = FALSE, deactivated_at = ? WHERE id = ?", [now, fid])

    # ── Guardrails ──────────────────────────────────────────────────────
    files_pool = [
        "memory/db.py", "memory/config.py", "memory/ingest.py", "memory/recall.py",
        "backend/auth.py", "backend/server.py", "frontend/package.json",
        "infra/main.tf", "infra/secrets.tf", "k8s/deployment.yaml",
        "Dockerfile", "docker-compose.yaml", ".github/workflows/ci.yaml",
    ]
    for i in range(num_guardrails):
        tmpl_idx = i % len(GUARDRAIL_TEMPLATES)
        warning_t, rationale_t, consequence_t = GUARDRAIL_TEMPLATES[tmpl_idx]
        scope = scopes[i % len(scopes)]
        file = files_pool[i % len(files_pool)]
        warning = warning_t.format(
            file=file, tool="pytest", scope_name=Path(scope).name,
            setting="max_connections", threshold=100,
        )
        sid = f"sess-{label}-{i % max(num_sessions, 1):04d}"
        db.upsert_guardrail(
            conn, warning=warning, rationale=rationale_t,
            consequence=consequence_t, file_paths=[file],
            embedding=_mock_embed(warning), session_id=sid, scope=scope,
        )
        meta["counts"]["guardrails"] += 1

    # ── Decisions ───────────────────────────────────────────────────────
    for i in range(num_decisions):
        scope = scopes[i % len(scopes)] if rng.random() > 0.2 else "__global__"
        tech_a = entity_names[i % len(entity_names)] if entity_names else "Tool A"
        tech_b = entity_names[(i + 5) % len(entity_names)] if entity_names else "Tool B"
        text = f"Use {tech_a} over {tech_b} for the {Path(scope).name if scope != '__global__' else 'all'} project: better {rng.choice(['performance', 'reliability', 'developer experience', 'cost efficiency'])}"
        sid = f"sess-{label}-{i % max(num_sessions, 1):04d}"
        db.upsert_decision(conn, text, "long", _mock_embed(text), sid, _noop_decay, scope=scope)
        meta["counts"]["decisions"] += 1

    # ── Error Solutions ─────────────────────────────────────────────────
    for i in range(num_errors):
        tmpl_idx = i % len(ERROR_TEMPLATES)
        pattern_t, solution_t = ERROR_TEMPLATES[tmpl_idx]
        scope = scopes[i % len(scopes)]
        tech = entity_names[i % len(entity_names)] if entity_names else "Service"
        pattern = pattern_t.format(
            tech=tech, port=rng.randint(3000, 9999), timeout=rng.randint(100, 5000),
            module=rng.choice(["memory", "fastapi", "duckdb", "numpy"]),
        )
        module = rng.choice(["memory", "fastapi", "duckdb", "numpy"])
        solution = solution_t.format(
            tech=tech, restart_cmd=f"systemctl restart {tech.lower()}",
            dashboard="Grafana", config_file="config.yaml", profiler="py-spy",
            module=module,
        )
        sid = f"sess-{label}-{i % max(num_sessions, 1):04d}"
        db.upsert_error_solution(
            conn, error_pattern=pattern, solution=solution,
            embedding=_mock_embed(pattern), session_id=sid, scope=scope,
        )
        meta["counts"]["error_solutions"] += 1

    # ── Procedures ──────────────────────────────────────────────────────
    for i in range(num_procedures):
        scope = scopes[i % len(scopes)] if rng.random() > 0.3 else "__global__"
        task = f"Procedure {i}: {rng.choice(['deploy', 'test', 'migrate', 'rotate secrets', 'scale', 'debug'])} {Path(scope).name if scope != '__global__' else 'system'}"
        steps = f"1. Check prerequisites 2. Run {rng.choice(['tests', 'linter', 'build'])} 3. Execute 4. Verify"
        sid = f"sess-{label}-{i % max(num_sessions, 1):04d}"
        db.upsert_procedure(
            conn, task_description=task, steps=steps,
            embedding=_mock_embed(task), session_id=sid, scope=scope,
        )
        meta["counts"]["procedures"] += 1

    # Build FTS indexes
    db.rebuild_fts_indexes(conn)

    return meta


# ── Public corpus builders ────────────────────────────────────────────────

def build_corpus_1d(conn, db) -> dict:
    """1 day of usage: light exploration of one project."""
    return _build_corpus(conn, db,
        num_facts=8, num_entities=6, num_rels=4, num_sessions=1,
        num_guardrails=1, num_decisions=2, num_errors=1, num_procedures=1,
        num_scopes=1, time_span_days=1, label="1d")


def build_corpus_1w(conn, db) -> dict:
    """1 week of usage: active development across 2 projects."""
    return _build_corpus(conn, db,
        num_facts=40, num_entities=20, num_rels=15, num_sessions=5,
        num_guardrails=3, num_decisions=5, num_errors=3, num_procedures=2,
        num_scopes=2, time_span_days=7, label="1w")


def build_corpus_1m(conn, db) -> dict:
    """1 month of usage: sustained development across 3 projects."""
    return _build_corpus(conn, db,
        num_facts=200, num_entities=60, num_rels=50, num_sessions=22,
        num_guardrails=8, num_decisions=15, num_errors=10, num_procedures=6,
        num_scopes=3, time_span_days=30, label="1m")


def build_corpus_1y(conn, db) -> dict:
    """1 year of usage: long-running engineering practice across 5 projects."""
    return _build_corpus(conn, db,
        num_facts=1500, num_entities=250, num_rels=400, num_sessions=200,
        num_guardrails=30, num_decisions=80, num_errors=50, num_procedures=25,
        num_scopes=5, time_span_days=365, label="1y")
