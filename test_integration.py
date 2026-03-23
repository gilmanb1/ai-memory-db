#!/usr/bin/env python3
"""
test_integration.py — Integration tests that make real Claude API calls.

Requires ANTHROPIC_API_KEY set in environment.
Skipped automatically if the key is not available.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 test_integration.py
    python3 test_integration.py  # skips all tests if no key

These tests cost ~$0.05-0.15 total and take ~30-60 seconds.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import memory.config as _cfg

# Use a temp DB for all integration tests
_INTEGRATION_DB = Path(tempfile.mktemp(suffix=".integration.duckdb"))
_cfg.DB_PATH = _INTEGRATION_DB

from memory import db, embeddings, extract, recall
from memory.decay import compute_decay_score

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SKIP_REASON = "ANTHROPIC_API_KEY not set — skipping integration tests"


def _noop_decay(last_seen_at, session_count, temporal_class):
    return 1.0


# ── Synthetic conversations for extraction testing ───────────────────────

CODING_CONVERSATION = """
--- USER ---
I'm working on the auth module in src/auth.py. We use JWT tokens with 24-hour
expiry and argon2id for password hashing. The auth middleware is in
src/middleware.py.

--- ASSISTANT ---
I see the auth setup. A few things I notice about the current implementation:

1. The retry loop in src/auth.py lines 45-78 uses linear retry (1s, 2s, 3s)
   instead of exponential backoff. This looks unusual but is actually deliberate —
   the OAuth provider rate-limits based on backoff patterns and penalizes
   exponential clients with 24-hour blocks.

2. We use urllib3 directly instead of the requests library. This is because
   requests bundles certifi which conflicts with the corporate CA trust store
   deployed via the system certificate chain.

--- USER ---
Good catches. I was about to refactor that retry loop to exponential backoff.
Glad you caught it. How do I add a new API endpoint to this project?

--- ASSISTANT ---
Here's the process for adding a new endpoint:

1. Create the handler function in routes/{resource}.py
2. Define the Pydantic model in schemas/{resource}.py
3. Register the route in app.py
4. Add an integration test in tests/test_{resource}.py
5. Run the test suite with: python3 -m pytest tests/

Also, I noticed an error that comes up sometimes: if you get
"ImportError: No module named 'onnxruntime'" when running on macOS ARM,
the fix is to install the ARM-specific build: pip install onnxruntime-silicon

--- USER ---
We chose DuckDB over SQLite for the knowledge store because SQLite's
single-writer lock causes data corruption under concurrent hook execution.
That was a hard-learned lesson from the production incident in January.
"""

SIMPLE_CONVERSATION = """
--- USER ---
What's the project structure?

--- ASSISTANT ---
The project uses Python 3.12 with FastAPI for the REST API. PostgreSQL is
the primary database, Redis handles session caching, and Celery processes
async tasks. Tests run with pytest and we have 89% code coverage.

--- USER ---
We decided to store all monetary values as integer cents to avoid floating
point rounding errors. This is a strict rule — never use float for money.
"""


# ══════════════════════════════════════════════════════════════════════════
# Extraction Quality Tests
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(API_KEY, SKIP_REASON)
class TestExtractionQuality(unittest.TestCase):
    """
    GIVEN a synthetic coding conversation
    WHEN Claude extracts knowledge
    THEN guardrails, procedures, error_solutions, importance, and file_paths are present
    """

    @classmethod
    def setUpClass(cls):
        """Extract once, verify many times (saves API calls)."""
        cls.knowledge = extract.extract_knowledge(CODING_CONVERSATION, API_KEY)

    def test_extraction_returns_facts(self):
        facts = self.knowledge.get("facts", [])
        self.assertGreater(len(facts), 0, "Should extract at least 1 fact")

    def test_extraction_returns_entities(self):
        entities = self.knowledge.get("entities", [])
        self.assertGreater(len(entities), 0, "Should extract at least 1 entity")

    def test_extraction_returns_guardrails(self):
        """Claude should extract the 'don't use exponential backoff' guardrail."""
        guardrails = self.knowledge.get("guardrails", [])
        self.assertGreater(len(guardrails), 0,
                           "Should extract at least 1 guardrail from the retry loop discussion")
        # At least one guardrail should mention backoff or retry
        texts = " ".join(g.get("warning", "") + " " + g.get("rationale", "") for g in guardrails).lower()
        self.assertTrue(
            "backoff" in texts or "retry" in texts or "exponential" in texts or "linear" in texts,
            f"Guardrail should mention retry/backoff pattern. Got: {texts[:200]}"
        )

    def test_guardrail_has_rationale(self):
        guardrails = self.knowledge.get("guardrails", [])
        if guardrails:
            g = guardrails[0]
            self.assertTrue(
                g.get("rationale", ""),
                "Guardrail should have a rationale explaining why"
            )

    def test_guardrail_has_file_paths(self):
        guardrails = self.knowledge.get("guardrails", [])
        has_paths = any(g.get("file_paths") for g in guardrails)
        self.assertTrue(has_paths,
                        "At least one guardrail should have file_paths (src/auth.py)")

    def test_extraction_returns_procedures(self):
        """Claude should extract the 'how to add an API endpoint' procedure."""
        procedures = self.knowledge.get("procedures", [])
        self.assertGreater(len(procedures), 0,
                           "Should extract at least 1 procedure")
        texts = " ".join(p.get("task_description", "") + " " + p.get("steps", "") for p in procedures).lower()
        self.assertTrue(
            "endpoint" in texts or "route" in texts or "handler" in texts,
            f"Procedure should mention API endpoint creation. Got: {texts[:200]}"
        )

    def test_procedure_has_steps(self):
        procedures = self.knowledge.get("procedures", [])
        if procedures:
            self.assertTrue(
                procedures[0].get("steps", ""),
                "Procedure should have steps"
            )

    def test_extraction_returns_error_solutions(self):
        """Claude should extract the onnxruntime error→solution pair."""
        errors = self.knowledge.get("error_solutions", [])
        self.assertGreater(len(errors), 0,
                           "Should extract the onnxruntime error→solution")
        texts = " ".join(e.get("error_pattern", "") + " " + e.get("solution", "") for e in errors).lower()
        self.assertTrue(
            "onnxruntime" in texts or "importerror" in texts,
            f"Error solution should mention onnxruntime. Got: {texts[:200]}"
        )

    def test_error_solution_has_solution(self):
        errors = self.knowledge.get("error_solutions", [])
        if errors:
            self.assertTrue(
                errors[0].get("solution", ""),
                "Error solution should have a solution field"
            )

    def test_facts_have_importance(self):
        """At least some facts should have importance scores."""
        facts = self.knowledge.get("facts", [])
        scored = [f for f in facts if f.get("importance") is not None]
        self.assertGreater(len(scored), 0,
                           "At least some facts should have importance scores")

    def test_high_importance_for_critical_knowledge(self):
        """The DuckDB-over-SQLite decision should have high importance."""
        facts = self.knowledge.get("facts", [])
        decisions = self.knowledge.get("key_decisions", [])
        all_items = facts + decisions
        duckdb_items = [
            i for i in all_items
            if "duckdb" in i.get("text", "").lower() or "sqlite" in i.get("text", "").lower()
        ]
        if duckdb_items:
            importances = [i.get("importance", 5) for i in duckdb_items]
            max_imp = max(importances)
            self.assertGreaterEqual(max_imp, 7,
                                    f"DuckDB-over-SQLite decision should have importance >= 7, got {max_imp}")

    def test_facts_have_file_paths(self):
        """Facts about specific files should include file_paths."""
        facts = self.knowledge.get("facts", [])
        with_paths = [f for f in facts if f.get("file_paths")]
        self.assertGreater(len(with_paths), 0,
                           "At least some facts should have file_paths (src/auth.py, etc.)")

    def test_facts_have_failure_probability(self):
        """At least some facts should have failure_probability scores."""
        facts = self.knowledge.get("facts", [])
        scored = [f for f in facts if f.get("failure_probability") is not None]
        # This field is optional — may or may not be present
        if scored:
            for f in scored:
                self.assertGreaterEqual(f["failure_probability"], 0.0)
                self.assertLessEqual(f["failure_probability"], 1.0)

    def test_coding_categories_used(self):
        """Facts should use code-oriented categories."""
        facts = self.knowledge.get("facts", [])
        categories = {f.get("category", "") for f in facts}
        coding_cats = {"architecture", "implementation", "operational",
                       "dependency", "decision_rationale", "constraint", "bug_pattern"}
        overlap = categories & coding_cats
        # At least one coding-specific category should be used
        self.assertGreater(len(overlap), 0,
                           f"Should use coding categories. Got: {categories}")

    def test_relationships_extracted(self):
        rels = self.knowledge.get("relationships", [])
        self.assertGreater(len(rels), 0, "Should extract relationships")

    def test_decisions_extracted(self):
        decisions = self.knowledge.get("key_decisions", [])
        self.assertGreater(len(decisions), 0, "Should extract decisions")


# ══════════════════════════════════════════════════════════════════════════
# Consolidation Quality Tests
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(API_KEY, SKIP_REASON)
class TestConsolidationQuality(unittest.TestCase):
    """
    GIVEN a set of related facts in the database
    WHEN Claude runs consolidation
    THEN coherent observations are produced
    """

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        _cfg.DB_PATH = self.db_path
        self.conn = db.get_connection()
        db.upsert_session(self.conn, "int-sess", "test", "/tmp", "/t.jsonl", 10, "Integration test")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_consolidation_produces_observations(self):
        """Given 5+ related facts, consolidation should synthesize an observation."""
        # Insert related facts about the auth module
        auth_facts = [
            "The auth module uses JWT tokens with 24-hour expiry",
            "Password hashing uses argon2id with 64MB memory cost",
            "Two-factor auth is mandatory for transfers over $10,000",
            "OAuth2 scopes restrict third-party API access to read-only",
            "Session cookies use SameSite=Strict and HttpOnly flags",
        ]
        for text in auth_facts:
            emb = embeddings.embed(text)
            db.upsert_fact(
                self.conn, text=text, category="architecture",
                temporal_class="long", confidence="high",
                embedding=emb, session_id="int-sess",
                decay_fn=_noop_decay, importance=7,
            )

        from memory.consolidation import run_consolidation
        stats = run_consolidation(self.conn, API_KEY, scope="__global__", quiet=True)

        self.assertGreater(stats.get("created", 0) + stats.get("updated", 0), 0,
                           "Consolidation should create at least 1 observation")

        # Verify the observation is stored
        from memory.config import RECALL_THRESHOLD
        query_emb = embeddings.embed_query("auth security")
        obs = db.search_observations(self.conn, query_emb, limit=5, threshold=0.0)
        self.assertGreater(len(obs), 0, "Observation should be searchable")


# ══════════════════════════════════════════════════════════════════════════
# Community Summary Quality Tests
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(API_KEY, SKIP_REASON)
class TestCommunitySummaryQuality(unittest.TestCase):
    """
    GIVEN facts clustered by shared entities
    WHEN Claude builds community summaries
    THEN the summary captures key relationships
    """

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        _cfg.DB_PATH = self.db_path
        self.conn = db.get_connection()
        db.upsert_session(self.conn, "int-sess", "test", "/tmp", "/t.jsonl", 10, "Integration test")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_community_summary_captures_entities(self):
        """A cluster about FastAPI+PostgreSQL should produce a summary mentioning both."""
        # Create entities and linked facts
        db.upsert_entity(self.conn, "FastAPI")
        db.upsert_entity(self.conn, "PostgreSQL")

        # Each fact text must contain entity names for link_fact_entities to match
        facts_texts = [
            "FastAPI serves REST endpoints connecting to PostgreSQL",
            "FastAPI queries PostgreSQL via SQLAlchemy ORM layer",
            "FastAPI and PostgreSQL handle financial transactions together",
        ]

        for text in facts_texts:
            emb = embeddings.embed(text)
            fid, _ = db.upsert_fact(
                self.conn, text=text, category="architecture",
                temporal_class="long", confidence="high",
                embedding=emb, session_id="int-sess",
                decay_fn=_noop_decay, importance=7,
            )
            # Directly link both entities (they appear in the text)
            db.link_fact_entities(self.conn, fid, ["FastAPI", "PostgreSQL"])

        from memory.communities import build_community_summaries
        stats = build_community_summaries(self.conn, API_KEY, quiet=True)

        self.assertGreater(stats["clusters_found"], 0, "Should find at least 1 cluster")
        self.assertGreater(stats["summaries_created"], 0, "Should create at least 1 summary")

        summaries = db.get_community_summaries(self.conn, level=1)
        self.assertGreater(len(summaries), 0)
        summary_text = summaries[0]["summary"].lower()
        self.assertTrue(
            "fastapi" in summary_text or "postgresql" in summary_text,
            f"Summary should mention key entities. Got: {summary_text[:200]}"
        )


# ══════════════════════════════════════════════════════════════════════════
# End-to-End Pipeline Tests
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(API_KEY, SKIP_REASON)
class TestEndToEndPipeline(unittest.TestCase):
    """
    GIVEN a synthetic conversation
    WHEN the full pipeline runs (extract → store → recall)
    THEN guardrails, procedures, and error_solutions surface correctly
    """

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        _cfg.DB_PATH = self.db_path
        # Create a JSONL transcript
        self.transcript = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        # Write the conversation as JSONL entries
        lines = CODING_CONVERSATION.strip().split("\n--- ")
        role = "user"
        for line in lines:
            line = line.strip()
            if line.startswith("USER ---"):
                role = "user"
                text = line[len("USER ---"):].strip()
            elif line.startswith("ASSISTANT ---"):
                role = "assistant"
                text = line[len("ASSISTANT ---"):].strip()
            else:
                continue
            if not text:
                continue
            content = text if role == "user" else [{"type": "text", "text": text}]
            self.transcript.write(json.dumps({
                "type": role,
                "message": {"role": role, "content": content},
                "timestamp": "2025-01-01T00:00:00Z",
            }) + "\n")
        self.transcript.close()

    def tearDown(self):
        try:
            self.db_path.unlink()
        except Exception:
            pass
        try:
            Path(self.transcript.name).unlink()
        except Exception:
            pass

    def test_full_pipeline_extracts_and_recalls(self):
        """Run extraction, then verify recall surfaces the right knowledge."""
        from memory.ingest import run_extraction

        result = run_extraction(
            session_id="e2e-test",
            transcript_path=self.transcript.name,
            trigger="integration_test",
            cwd="/tmp/test-project",
            api_key=API_KEY,
            quiet=True,
        )

        self.assertIsNotNone(result, "Extraction should succeed")
        counters = result["counters"]
        self.assertGreater(counters["facts"], 0, "Should extract facts")

        # Now verify recall
        conn = db.get_connection()
        try:
            # Session recall should have content
            ctx = recall.session_recall(conn)
            text = recall.format_session_context(ctx)
            self.assertGreater(len(text), 0, "Session recall should produce output")

            # Check for guardrails
            guardrails = db.get_all_guardrails(conn)
            if guardrails:
                self.assertIn("Guardrails", text)

            # Check for procedures
            procedures = db.get_procedures(conn)
            if procedures:
                # Procedures should be in session context
                self.assertGreater(len(procedures), 0)

            # Prompt recall for a specific query
            query = "I want to refactor the retry loop in auth.py"
            query_emb = embeddings.embed_query(query)
            if query_emb:
                prompt_ctx = recall.prompt_recall(conn, query_emb, query)
                # Should find guardrails about retry/backoff
                found_guardrails = prompt_ctx.get("guardrails", [])
                if guardrails:  # if extraction found guardrails, recall should too
                    self.assertGreater(len(found_guardrails), 0,
                                       "Guardrail about retry should surface when querying about retry refactor")
        finally:
            conn.close()

    def test_pipeline_stores_error_solutions(self):
        """The extraction should capture the onnxruntime error→solution."""
        from memory.ingest import run_extraction

        run_extraction(
            session_id="e2e-error-test",
            transcript_path=self.transcript.name,
            trigger="integration_test",
            cwd="/tmp/test-project",
            api_key=API_KEY,
            quiet=True,
        )

        conn = db.get_connection()
        try:
            results = db.search_all_by_text(conn, "onnxruntime")
            tables = {r["table"] for r in results}
            # Should be in error_solutions or facts
            self.assertTrue(
                "error_solutions" in tables or "facts" in tables,
                f"onnxruntime should be stored. Tables found: {tables}"
            )
        finally:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════
# Incremental Extraction Tests
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(API_KEY, SKIP_REASON)
class TestIncrementalExtraction(unittest.TestCase):
    """
    GIVEN a conversation processed incrementally
    WHEN new segments are added
    THEN superseding works and no duplicates are created
    """

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        _cfg.DB_PATH = self.db_path

    def tearDown(self):
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_incremental_extraction_produces_narrative(self):
        """Incremental extraction should produce a cumulative narrative."""
        knowledge = extract.extract_knowledge_incremental(
            delta_text=SIMPLE_CONVERSATION,
            api_key=API_KEY,
            prior_narrative=None,
        )
        self.assertIn("narrative_summary", knowledge)
        self.assertGreater(len(knowledge["narrative_summary"]), 0,
                           "Should produce a narrative summary")
        self.assertIn("facts", knowledge)
        self.assertGreater(len(knowledge["facts"]), 0)


# ══════════════════════════════════════════════════════════════════════════
# Extraction Schema Compliance Tests
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(API_KEY, SKIP_REASON)
class TestExtractionSchemaCompliance(unittest.TestCase):
    """
    GIVEN the extraction tool schema
    WHEN Claude returns structured output
    THEN all fields conform to the schema
    """

    @classmethod
    def setUpClass(cls):
        cls.knowledge = extract.extract_knowledge(SIMPLE_CONVERSATION, API_KEY)

    def test_facts_have_required_fields(self):
        for fact in self.knowledge.get("facts", []):
            self.assertIn("text", fact)
            self.assertIn("category", fact)
            self.assertIn("confidence", fact)
            self.assertIn("temporal_class", fact)

    def test_temporal_class_values_valid(self):
        valid = {"short", "medium", "long"}
        for fact in self.knowledge.get("facts", []):
            self.assertIn(fact["temporal_class"], valid,
                          f"Invalid temporal_class: {fact['temporal_class']}")

    def test_confidence_values_valid(self):
        valid = {"high", "medium", "low"}
        for fact in self.knowledge.get("facts", []):
            self.assertIn(fact["confidence"], valid)

    def test_importance_in_range(self):
        for fact in self.knowledge.get("facts", []):
            imp = fact.get("importance")
            if imp is not None:
                self.assertGreaterEqual(imp, 1)
                self.assertLessEqual(imp, 10)

    def test_decisions_have_text(self):
        for dec in self.knowledge.get("key_decisions", []):
            self.assertTrue(dec.get("text", ""), "Decision should have text")

    def test_relationships_have_required_fields(self):
        for rel in self.knowledge.get("relationships", []):
            self.assertIn("from", rel)
            self.assertIn("to", rel)
            self.assertIn("type", rel)

    def test_session_summary_present(self):
        self.assertIn("session_summary", self.knowledge)
        self.assertGreater(len(self.knowledge["session_summary"]), 0)


if __name__ == "__main__":
    if not API_KEY:
        print("ANTHROPIC_API_KEY not set — all integration tests will be skipped.")
        print("Set it to run: ANTHROPIC_API_KEY=sk-... python3 test_integration.py")
        print()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
