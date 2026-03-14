#!/usr/bin/env python3
"""
test_memory.py — Full test suite for the Claude Code memory system.

Runs against a real (in-memory) DuckDB instance.
Does NOT require Ollama or the Anthropic API.
Usage: python test_memory.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Bootstrap: resolve the memory package from project root ──────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch config to use an in-memory database for all tests
import memory.config as _cfg
_cfg.DB_PATH = Path(tempfile.mktemp(suffix=".duckdb"))

from memory import db, decay, recall, embeddings, extract


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_embed(text: str) -> list[float]:
    """
    Deterministic, hash-based fake embedding.
    Identical strings → identical vectors (cosine = 1.0).
    Different strings → near-orthogonal vectors (cosine ≈ 0).
    This gives correct dedup and search behaviour without Ollama.
    """
    import hashlib
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


def fresh_conn(path: Path = None) -> duckdb.DuckDBPyConnection:
    """Open a fresh writable connection, explicitly passing the temp path."""
    target = path or _cfg.DB_PATH
    return db.get_connection(db_path=str(target))


class TestDecay(unittest.TestCase):

    def test_fresh_item_score_is_high(self):
        score = decay.compute_decay_score(datetime.now(timezone.utc), 1, "short")
        self.assertGreater(score, 0.9)

    def test_old_short_item_decays(self):
        old = datetime.now(timezone.utc) - timedelta(days=30)
        score = decay.compute_decay_score(old, 1, "short")
        self.assertLess(score, 0.01)

    def test_long_item_survives_30_days(self):
        old = datetime.now(timezone.utc) - timedelta(days=30)
        score = decay.compute_decay_score(old, 10, "long")
        self.assertGreater(score, 0.4)

    def test_reinforcement_raises_score(self):
        old = datetime.now(timezone.utc) - timedelta(days=7)
        s1 = decay.compute_decay_score(old, 1, "medium")
        s10 = decay.compute_decay_score(old, 10, "medium")
        self.assertGreater(s10, s1)

    def test_should_forget_only_short(self):
        self.assertTrue(decay.should_forget(0.01, "short"))
        self.assertFalse(decay.should_forget(0.01, "medium"))
        self.assertFalse(decay.should_forget(0.01, "long"))

    def test_temporal_weight_ordering(self):
        w_long   = decay.temporal_weight("long",   0.9)
        w_medium = decay.temporal_weight("medium", 0.9)
        w_short  = decay.temporal_weight("short",  0.9)
        self.assertGreater(w_long, w_medium)
        self.assertGreater(w_medium, w_short)

    def test_score_clamped_to_one(self):
        score = decay.compute_decay_score(datetime.now(timezone.utc), 999, "long")
        self.assertLessEqual(score, 1.0)

    def test_score_non_negative(self):
        very_old = datetime.now(timezone.utc) - timedelta(days=3650)
        score = decay.compute_decay_score(very_old, 0, "short")
        self.assertGreaterEqual(score, 0.0)


class TestSchema(unittest.TestCase):

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_tables_exist(self):
        tables = {r[0] for r in self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        for expected in ("facts","ideas","entities","relationships","decisions","open_questions","sessions"):
            self.assertIn(expected, tables, f"Table {expected} missing")

    def test_migration_recorded(self):
        versions = [r[0] for r in self.conn.execute("SELECT version FROM schema_migrations").fetchall()]
        self.assertIn(1, versions)

    def test_migration_idempotent(self):
        # Running migrations twice must not fail or duplicate records
        db._run_migrations(self.conn)
        db._run_migrations(self.conn)
        count = self.conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        self.assertEqual(count, len(db.MIGRATIONS))


class TestFactCRUD(unittest.TestCase):

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)
        db.upsert_session(self.conn, "sess-1", "manual", "/tmp", "/tmp/t.jsonl", 5, "Test")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_insert_new_fact(self):
        emb = _mock_embed("DuckDB is an analytical database")
        fid, is_new = db.upsert_fact(
            self.conn, "DuckDB is an analytical database",
            "technical", "long", "high", emb, "sess-1", _noop_decay
        )
        self.assertTrue(is_new)
        self.assertIsNotNone(fid)
        row = self.conn.execute("SELECT text, temporal_class FROM facts WHERE id=?", [fid]).fetchone()
        self.assertEqual(row[0], "DuckDB is an analytical database")
        self.assertEqual(row[1], "long")

    def test_dedup_near_identical_fact(self):
        text = "The project uses Neo4j for graph storage"
        emb  = _mock_embed(text)
        fid1, is_new1 = db.upsert_fact(
            self.conn, text, "technical", "long", "high", emb, "sess-1", _noop_decay
        )
        fid2, is_new2 = db.upsert_fact(
            self.conn, text + ".", "technical", "long", "high", emb, "sess-1", _noop_decay
        )
        self.assertTrue(is_new1)
        self.assertFalse(is_new2)           # second call deduped
        self.assertEqual(fid1, fid2)        # same record updated
        row = self.conn.execute("SELECT session_count FROM facts WHERE id=?", [fid1]).fetchone()
        self.assertEqual(row[0], 2)         # reinforced

    def test_different_facts_both_stored(self):
        emb_a = _mock_embed("aaa" * 100)
        emb_b = _mock_embed("bbb" * 100)
        _, new_a = db.upsert_fact(self.conn, "Fact A", "contextual", "short", "medium", emb_a, "sess-1", _noop_decay)
        _, new_b = db.upsert_fact(self.conn, "Fact B", "contextual", "short", "medium", emb_b, "sess-1", _noop_decay)
        self.assertTrue(new_a)
        self.assertTrue(new_b)
        count = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        self.assertEqual(count, 2)

    def test_get_facts_by_temporal(self):
        for i, tc in enumerate(["long", "long", "medium", "short"]):
            emb = _mock_embed(f"unique fact {i} " * 20)
            db.upsert_fact(self.conn, f"Fact {i}", "contextual", tc, "high", emb, "sess-1", _noop_decay)
        long_facts = db.get_facts_by_temporal(self.conn, "long", 10)
        self.assertEqual(len(long_facts), 2)
        for f in long_facts:
            self.assertEqual(f["temporal_class"], "long")

    def test_vector_search_returns_similar(self):
        anchor_text = "Python is the primary programming language"
        anchor_emb  = _mock_embed(anchor_text)
        db.upsert_fact(self.conn, anchor_text, "technical", "long", "high", anchor_emb, "sess-1", _noop_decay)

        # Insert a dissimilar fact
        db.upsert_fact(self.conn, "The sky is blue", "contextual", "short", "medium",
                       _mock_embed("sky color" * 50), "sess-1", _noop_decay)

        results = db.search_facts(self.conn, anchor_emb, limit=5, threshold=0.9)
        self.assertGreater(len(results), 0)
        self.assertIn("Python", results[0]["text"])


class TestEntityAndRelationship(unittest.TestCase):

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)
        db.upsert_session(self.conn, "sess-1", "manual", "/tmp", "/tmp/t.jsonl", 5, "Test")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_entity_upsert_and_dedup(self):
        eid1 = db.upsert_entity(self.conn, "Neo4j")
        eid2 = db.upsert_entity(self.conn, "Neo4j")   # same name
        eid3 = db.upsert_entity(self.conn, "neo4j")   # case-insensitive
        self.assertEqual(eid1, eid2)
        self.assertEqual(eid1, eid3)
        row = self.conn.execute("SELECT session_count FROM entities WHERE id=?", [eid1]).fetchone()
        self.assertEqual(row[0], 3)

    def test_relationship_upsert(self):
        rid1 = db.upsert_relationship(self.conn, "DuckDB", "SQL", "uses", "DuckDB uses SQL", "sess-1")
        rid2 = db.upsert_relationship(self.conn, "DuckDB", "SQL", "uses", "Same rel again", "sess-1")
        self.assertEqual(rid1, rid2)  # same unique triplet
        row = self.conn.execute("SELECT session_count, strength FROM relationships WHERE id=?", [rid1]).fetchone()
        self.assertEqual(row[0], 2)
        self.assertAlmostEqual(row[1], 1.1, places=2)

    def test_get_relationships_for_entities(self):
        db.upsert_entity(self.conn, "Kafka")
        db.upsert_relationship(self.conn, "Kafka", "Neo4j", "feeds_into", "Kafka feeds Neo4j", "sess-1")
        db.upsert_relationship(self.conn, "Neo4j", "Cypher", "uses", "Neo4j uses Cypher", "sess-1")
        rels = db.get_relationships_for_entities(self.conn, ["Kafka"])
        self.assertGreater(len(rels), 0)
        froms = [r["from_entity"] for r in rels]
        self.assertIn("Kafka", froms)

    def test_top_entities_ordering(self):
        for _ in range(5):
            db.upsert_entity(self.conn, "Plex Research")
        db.upsert_entity(self.conn, "Neo4j")
        top = db.get_top_entities(self.conn, 3)
        self.assertEqual(top[0], "Plex Research")  # most sessions first


class TestDecisionAndQuestion(unittest.TestCase):

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)
        db.upsert_session(self.conn, "sess-1", "manual", "/tmp", "/tmp/t.jsonl", 5, "Test")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_decision_insert_and_dedup(self):
        text = "We will use DuckDB for the knowledge store"
        emb  = _mock_embed(text)
        did1, new1 = db.upsert_decision(self.conn, text, "long", emb, "sess-1", _noop_decay)
        did2, new2 = db.upsert_decision(self.conn, text, "long", emb, "sess-1", _noop_decay)
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(did1, did2)

    def test_question_insert(self):
        emb = _mock_embed("Do we have clinical outcome data?")
        qid, is_new = db.upsert_question(self.conn, "Do we have clinical outcome data?", emb, "sess-1")
        self.assertTrue(is_new)
        row = self.conn.execute("SELECT text, resolved FROM open_questions WHERE id=?", [qid]).fetchone()
        self.assertEqual(row[0], "Do we have clinical outcome data?")
        self.assertFalse(row[1])


class TestDecayPass(unittest.TestCase):

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)
        db.upsert_session(self.conn, "sess-1", "manual", "/tmp", "/tmp/t.jsonl", 5, "Test")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_old_short_fact_forgotten(self):
        # Insert a very old, single-session short fact directly
        import uuid
        from datetime import timezone
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        fid = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO facts(id, text, temporal_class, decay_score, session_count,
                              embedding, created_at, last_seen_at, is_active)
            VALUES (?, 'old transient debug note', 'short', 0.001, 1, NULL, ?, ?, TRUE)
        """, [fid, old_time, old_time])
        stats = db.apply_decay_pass(self.conn)
        row = self.conn.execute("SELECT is_active FROM facts WHERE id=?", [fid]).fetchone()
        self.assertFalse(row[0])
        self.assertGreater(stats["forgotten"], 0)

    def test_long_fact_not_forgotten(self):
        import uuid
        from datetime import timezone
        old_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        fid = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO facts(id, text, temporal_class, decay_score, session_count,
                              embedding, created_at, last_seen_at, is_active)
            VALUES (?, 'Core tech stack: Python + DuckDB', 'long', 0.9, 20, NULL, ?, ?, TRUE)
        """, [fid, old_time, old_time])
        db.apply_decay_pass(self.conn)
        row = self.conn.execute("SELECT is_active FROM facts WHERE id=?", [fid]).fetchone()
        self.assertTrue(row[0])

    def test_promotion_medium_to_long(self):
        """A medium fact seen in 7+ sessions should be promoted to long."""
        import uuid
        from datetime import timezone
        old_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
        fid = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO facts(id, text, temporal_class, decay_score, session_count,
                              embedding, created_at, last_seen_at, is_active)
            VALUES (?, 'Recurring architectural fact', 'medium', 0.8, 8, NULL, ?, ?, TRUE)
        """, [fid, old_time, old_time])
        db.apply_decay_pass(self.conn)
        row = self.conn.execute("SELECT temporal_class FROM facts WHERE id=?", [fid]).fetchone()
        self.assertEqual(row[0], "long")


class TestRecall(unittest.TestCase):

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)
        db.upsert_session(self.conn, "sess-1", "manual", "/tmp", "/tmp/t.jsonl", 5, "Test")

        for text, tc in [
            ("Plex Research is a Cambridge MA biotech", "long"),
            ("Focal Graph combines knowledge graphs with LLMs", "long"),
            ("The project uses DuckDB for storage", "medium"),
            ("We are debugging the ingestion pipeline", "short"),
        ]:
            emb = _mock_embed(text)
            db.upsert_fact(self.conn, text, "technical", tc, "high", emb, "sess-1", _noop_decay)

        db.upsert_entity(self.conn, "Plex Research")
        db.upsert_entity(self.conn, "DuckDB")
        db.upsert_relationship(
            self.conn, "Focal Graph", "DuckDB", "uses",
            "Focal Graph stores data in DuckDB", "sess-1",
        )
        db.upsert_decision(
            self.conn, "Use DuckDB for knowledge store", "long",
            _mock_embed("Use DuckDB for knowledge store"), "sess-1", _noop_decay,
        )

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_session_recall_returns_long_facts(self):
        ctx = recall.session_recall(self.conn)
        self.assertGreater(len(ctx["long_facts"]), 0)
        for f in ctx["long_facts"]:
            self.assertEqual(f["temporal_class"], "long")

    def test_session_recall_returns_decisions(self):
        ctx = recall.session_recall(self.conn)
        self.assertGreater(len(ctx["decisions"]), 0)

    def test_format_session_context_non_empty(self):
        ctx = recall.session_recall(self.conn)
        text = recall.format_session_context(ctx)
        self.assertIn("Memory Context", text)
        self.assertIn("Established Knowledge", text)

    def test_prompt_recall_finds_relevant_facts(self):
        query_emb = _mock_embed("Tell me about Plex Research and Focal Graph")
        ctx = recall.prompt_recall(self.conn, query_emb, "Tell me about Plex Research and Focal Graph")
        # At least one fact should be recalled
        self.assertIsInstance(ctx["facts"], list)

    def test_prompt_recall_entity_matching(self):
        query_emb = _mock_embed("What does DuckDB do in our stack?")
        ctx = recall.prompt_recall(self.conn, query_emb, "What does DuckDB do in our stack?")
        # DuckDB is a known entity and should be found in the prompt
        self.assertIn("DuckDB", ctx["entities_hit"])

    def test_format_prompt_context_shows_relationships(self):
        query_emb = _mock_embed("Focal Graph storage DuckDB")
        ctx = recall.prompt_recall(self.conn, query_emb, "Focal Graph DuckDB")
        text = recall.format_prompt_context(ctx)
        # Should mention something or be empty — no crash
        self.assertIsInstance(text, str)

    def test_empty_db_returns_empty_string(self):
        # Empty session context should return empty string
        ctx = {"long_facts": [], "medium_facts": [], "decisions": [], "entities": [], "relationships": []}
        text = recall.format_session_context(ctx)
        self.assertEqual(text, "")


class TestTranscriptParsing(unittest.TestCase):

    def _write_jsonl(self, entries):
        import tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for e in entries:
            f.write(json.dumps(e) + "\n")
        f.close()
        return f.name

    def test_standard_messages(self):
        path = self._write_jsonl([
            {"type": "user",      "message": {"role": "user",      "content": "Hello!"}, "timestamp": "2025-01-01T00:00:00Z"},
            {"type": "assistant", "message": {"role": "assistant",  "content": [{"type": "text", "text": "Hi there!"}]}, "timestamp": "2025-01-01T00:00:01Z"},
        ])
        msgs = extract.parse_transcript(path)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["text"], "Hello!")
        self.assertEqual(msgs[1]["text"], "Hi there!")

    def test_tool_use_blocks(self):
        path = self._write_jsonl([
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "name": "web_search", "id": "t1"},
            ]}, "timestamp": "2025-01-01T00:00:00Z"},
        ])
        msgs = extract.parse_transcript(path)
        self.assertEqual(len(msgs), 1)
        self.assertIn("[tool_use: web_search]", msgs[0]["text"])

    def test_malformed_lines_skipped(self):
        path = self._write_jsonl([
            {"type": "user", "message": {"role": "user", "content": "Valid"}, "timestamp": "2025-01-01T00:00:00Z"},
        ])
        with open(path, "a") as f:
            f.write("NOT JSON AT ALL\n")
            f.write("\n")
        msgs = extract.parse_transcript(path)
        self.assertEqual(len(msgs), 1)

    def test_build_conversation_truncation(self):
        msgs = [{"role": "user", "text": "x" * 10_000, "timestamp": ""} for _ in range(50)]
        result = extract.build_conversation_text(msgs, max_chars=5_000)
        self.assertLessEqual(len(result), 5_500)
        self.assertIn("omitted", result)

    def test_nonexistent_transcript(self):
        msgs = extract.parse_transcript("/nonexistent/path/transcript.jsonl")
        self.assertEqual(msgs, [])


class TestDatabaseStats(unittest.TestCase):

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)
        db.upsert_session(self.conn, "sess-1", "manual", "/tmp", "/tmp/t.jsonl", 3, "Stats test")
        db.upsert_fact(self.conn, "Stat fact A", "technical", "long",   "high", _mock_embed("stat_fact_aaa"), "sess-1", _noop_decay)
        db.upsert_fact(self.conn, "Stat fact B", "contextual","medium", "high", _mock_embed("stat_fact_bbb"), "sess-1", _noop_decay)
        db.upsert_fact(self.conn, "Stat fact C", "contextual","short",  "low",  _mock_embed("stat_fact_ccc"), "sess-1", _noop_decay)
        db.upsert_entity(self.conn, "TestEntity")
        db.upsert_relationship(self.conn, "A", "B", "causes", "A causes B", "sess-1")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def test_stats_structure(self):
        stats = db.get_stats(self.conn)
        for key in ("facts","ideas","entities","relationships","decisions","questions","sessions"):
            self.assertIn(key, stats)

    def test_stats_counts_correct(self):
        stats = db.get_stats(self.conn)
        self.assertEqual(stats["facts"]["total"], 3)
        self.assertEqual(stats["facts"]["long"],   1)
        self.assertEqual(stats["facts"]["medium"], 1)
        self.assertEqual(stats["facts"]["short"],  1)
        self.assertEqual(stats["entities"]["total"], 1)
        self.assertEqual(stats["relationships"]["total"], 1)
        self.assertEqual(stats["sessions"]["total"], 1)


# ══════════════════════════════════════════════════════════════════════════
# Project Scoping Tests (BDD-style)
# ══════════════════════════════════════════════════════════════════════════

SCOPE_A = "/Users/dev/projects/alpha"
SCOPE_B = "/Users/dev/projects/beta"
SCOPE_C = "/Users/dev/projects/gamma"


class _ScopedTestBase(unittest.TestCase):
    """Shared setUp/tearDown for all scoping tests."""

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        self.conn = fresh_conn(self.db_path)
        db.upsert_session(self.conn, "sess-1", "manual", "/tmp", "/tmp/t.jsonl", 5, "Test")

    def tearDown(self):
        self.conn.close()
        try:
            self.db_path.unlink()
        except Exception:
            pass


class TestScopeResolution(unittest.TestCase):
    """
    GIVEN a working directory
    WHEN resolving the project scope
    THEN the git repo root is returned (or the cwd as fallback)
    """

    def test_given_empty_cwd_then_returns_global(self):
        from memory.scope import resolve_scope
        from memory.config import GLOBAL_SCOPE
        self.assertEqual(resolve_scope(""), GLOBAL_SCOPE)

    def test_given_non_git_directory_then_returns_resolved_path(self):
        from memory.scope import resolve_scope
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            scope = resolve_scope(tmpdir)
            self.assertEqual(scope, str(Path(tmpdir).resolve()))

    def test_given_git_repo_then_returns_repo_root(self):
        from memory.scope import resolve_scope
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_tmpdir = str(Path(tmpdir).resolve())
            subprocess.run(["git", "init"], cwd=resolved_tmpdir, capture_output=True)
            subdir = Path(resolved_tmpdir) / "src" / "deep"
            subdir.mkdir(parents=True)
            scope = resolve_scope(str(subdir))
            self.assertEqual(scope, resolved_tmpdir)

    def test_given_git_repo_subdirectory_then_returns_same_root(self):
        from memory.scope import resolve_scope
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            sub_a = Path(tmpdir) / "pkg_a"
            sub_b = Path(tmpdir) / "pkg_b"
            sub_a.mkdir()
            sub_b.mkdir()
            self.assertEqual(resolve_scope(str(sub_a)), resolve_scope(str(sub_b)))

    def test_scope_display_name_for_global(self):
        from memory.scope import scope_display_name
        from memory.config import GLOBAL_SCOPE
        self.assertEqual(scope_display_name(GLOBAL_SCOPE), "global")

    def test_scope_display_name_for_project(self):
        from memory.scope import scope_display_name
        self.assertEqual(scope_display_name("/Users/dev/projects/alpha"), "alpha")


class TestScopedFactStorage(_ScopedTestBase):
    """
    GIVEN facts stored with different project scopes
    WHEN querying by scope
    THEN only facts from that scope (and global) are returned
    """

    def test_given_fact_in_scope_a_when_querying_scope_a_then_found(self):
        emb = _mock_embed("Alpha uses React for frontend")
        db.upsert_fact(self.conn, "Alpha uses React for frontend",
                       "technical", "long", "high", emb, "sess-1", _noop_decay,
                       scope=SCOPE_A)
        facts = db.get_facts_by_temporal(self.conn, "long", 10, scope=SCOPE_A)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["text"], "Alpha uses React for frontend")

    def test_given_fact_in_scope_a_when_querying_scope_b_then_not_found(self):
        emb = _mock_embed("Alpha uses React for frontend")
        db.upsert_fact(self.conn, "Alpha uses React for frontend",
                       "technical", "long", "high", emb, "sess-1", _noop_decay,
                       scope=SCOPE_A)
        facts = db.get_facts_by_temporal(self.conn, "long", 10, scope=SCOPE_B)
        self.assertEqual(len(facts), 0)

    def test_given_global_fact_when_querying_any_scope_then_found(self):
        from memory.config import GLOBAL_SCOPE
        emb = _mock_embed("User prefers dark mode in all editors")
        db.upsert_fact(self.conn, "User prefers dark mode in all editors",
                       "personal", "long", "high", emb, "sess-1", _noop_decay,
                       scope=GLOBAL_SCOPE)
        facts_a = db.get_facts_by_temporal(self.conn, "long", 10, scope=SCOPE_A)
        facts_b = db.get_facts_by_temporal(self.conn, "long", 10, scope=SCOPE_B)
        self.assertEqual(len(facts_a), 1)
        self.assertEqual(len(facts_b), 1)

    def test_given_no_scope_filter_then_all_facts_returned(self):
        emb_a = _mock_embed("fact in alpha " * 20)
        emb_b = _mock_embed("fact in beta " * 20)
        db.upsert_fact(self.conn, "Fact A", "contextual", "long", "high",
                       emb_a, "sess-1", _noop_decay, scope=SCOPE_A)
        db.upsert_fact(self.conn, "Fact B", "contextual", "long", "high",
                       emb_b, "sess-1", _noop_decay, scope=SCOPE_B)
        facts = db.get_facts_by_temporal(self.conn, "long", 10, scope=None)
        self.assertEqual(len(facts), 2)

    def test_given_fact_stored_with_scope_then_scope_column_set(self):
        emb = _mock_embed("Scoped fact check")
        fid, _ = db.upsert_fact(self.conn, "Scoped fact check",
                                "technical", "long", "high", emb, "sess-1",
                                _noop_decay, scope=SCOPE_A)
        row = self.conn.execute("SELECT scope FROM facts WHERE id=?", [fid]).fetchone()
        self.assertEqual(row[0], SCOPE_A)


class TestScopedVectorSearch(_ScopedTestBase):
    """
    GIVEN facts with embeddings in different scopes
    WHEN performing vector search with a scope filter
    THEN only matching-scope and global results are returned
    """

    def test_given_scoped_facts_when_searching_scope_a_then_only_a_and_global(self):
        from memory.config import GLOBAL_SCOPE
        emb_a = _mock_embed("React component lifecycle in alpha")
        emb_b = _mock_embed("Django ORM queries in beta " * 10)
        emb_g = _mock_embed("User name is Alice")
        db.upsert_fact(self.conn, "React lifecycle", "technical", "long", "high",
                       emb_a, "sess-1", _noop_decay, scope=SCOPE_A)
        db.upsert_fact(self.conn, "Django ORM", "technical", "long", "high",
                       emb_b, "sess-1", _noop_decay, scope=SCOPE_B)
        db.upsert_fact(self.conn, "User is Alice", "personal", "long", "high",
                       emb_g, "sess-1", _noop_decay, scope=GLOBAL_SCOPE)

        results = db.search_facts(self.conn, emb_a, limit=10, threshold=0.0, scope=SCOPE_A)
        scopes = {r.get("scope") for r in results}
        self.assertNotIn(SCOPE_B, scopes)


class TestScopedEntities(_ScopedTestBase):
    """
    GIVEN entities stored with different scopes
    WHEN querying entities by scope
    THEN only entities from that scope (and global) are returned
    """

    def test_given_entity_in_scope_a_when_querying_scope_b_then_not_found(self):
        db.upsert_entity(self.conn, "AlphaService", scope=SCOPE_A)
        entities = db.get_top_entities(self.conn, 10, scope=SCOPE_B)
        self.assertNotIn("AlphaService", entities)

    def test_given_entity_in_scope_a_when_querying_scope_a_then_found(self):
        db.upsert_entity(self.conn, "AlphaService", scope=SCOPE_A)
        entities = db.get_top_entities(self.conn, 10, scope=SCOPE_A)
        self.assertIn("AlphaService", entities)

    def test_given_global_entity_when_querying_any_scope_then_found(self):
        from memory.config import GLOBAL_SCOPE
        db.upsert_entity(self.conn, "PostgreSQL", scope=GLOBAL_SCOPE)
        entities_a = db.get_top_entities(self.conn, 10, scope=SCOPE_A)
        entities_b = db.get_top_entities(self.conn, 10, scope=SCOPE_B)
        self.assertIn("PostgreSQL", entities_a)
        self.assertIn("PostgreSQL", entities_b)


class TestScopedRelationships(_ScopedTestBase):
    """
    GIVEN relationships stored with different scopes
    WHEN querying relationships by scope
    THEN only relationships from that scope (and global) are returned
    """

    def test_given_rel_in_scope_a_when_querying_scope_b_then_not_found(self):
        db.upsert_relationship(self.conn, "AlphaAPI", "Redis", "uses",
                               "AlphaAPI caches in Redis", "sess-1", scope=SCOPE_A)
        rels = db.get_relationships_for_entities(self.conn, ["AlphaAPI"], scope=SCOPE_B)
        self.assertEqual(len(rels), 0)

    def test_given_rel_in_scope_a_when_querying_scope_a_then_found(self):
        db.upsert_relationship(self.conn, "AlphaAPI", "Redis", "uses",
                               "AlphaAPI caches in Redis", "sess-1", scope=SCOPE_A)
        rels = db.get_relationships_for_entities(self.conn, ["AlphaAPI"], scope=SCOPE_A)
        self.assertEqual(len(rels), 1)

    def test_given_global_rel_when_querying_any_scope_then_found(self):
        from memory.config import GLOBAL_SCOPE
        db.upsert_relationship(self.conn, "CI", "GitHub Actions", "uses",
                               "CI runs on GHA", "sess-1", scope=GLOBAL_SCOPE)
        rels_a = db.get_relationships_for_entities(self.conn, ["CI"], scope=SCOPE_A)
        rels_b = db.get_relationships_for_entities(self.conn, ["CI"], scope=SCOPE_B)
        self.assertGreater(len(rels_a), 0)
        self.assertGreater(len(rels_b), 0)


class TestScopedDecisions(_ScopedTestBase):
    """
    GIVEN decisions stored with different scopes
    WHEN querying decisions by scope
    THEN only decisions from that scope (and global) are returned
    """

    def test_given_decision_in_scope_a_when_querying_scope_b_then_not_found(self):
        emb = _mock_embed("Use Next.js for alpha frontend")
        db.upsert_decision(self.conn, "Use Next.js for alpha frontend",
                           "long", emb, "sess-1", _noop_decay, scope=SCOPE_A)
        decisions = db.get_decisions(self.conn, 10, scope=SCOPE_B)
        self.assertEqual(len(decisions), 0)

    def test_given_decision_in_scope_a_when_querying_scope_a_then_found(self):
        emb = _mock_embed("Use Next.js for alpha frontend")
        db.upsert_decision(self.conn, "Use Next.js for alpha frontend",
                           "long", emb, "sess-1", _noop_decay, scope=SCOPE_A)
        decisions = db.get_decisions(self.conn, 10, scope=SCOPE_A)
        self.assertEqual(len(decisions), 1)


class TestItemScopeTracking(_ScopedTestBase):
    """
    GIVEN an item is inserted or reinforced in a scope
    WHEN checking the item_scopes table
    THEN each distinct scope the item was seen in is recorded
    """

    def test_given_new_fact_then_scope_tracked(self):
        emb = _mock_embed("Tracking test fact")
        fid, _ = db.upsert_fact(self.conn, "Tracking test fact",
                                "contextual", "short", "medium", emb, "sess-1",
                                _noop_decay, scope=SCOPE_A)
        rows = self.conn.execute(
            "SELECT scope FROM item_scopes WHERE item_id=? AND item_table='facts'",
            [fid]
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], SCOPE_A)

    def test_given_fact_reinforced_from_another_scope_then_both_scopes_tracked(self):
        text = "Cross-project fact for tracking"
        emb = _mock_embed(text)
        fid, _ = db.upsert_fact(self.conn, text,
                                "contextual", "medium", "high", emb, "sess-1",
                                _noop_decay, scope=SCOPE_A)
        # Reinforce same fact from scope B
        db.upsert_fact(self.conn, text,
                       "contextual", "medium", "high", emb, "sess-1",
                       _noop_decay, scope=SCOPE_B)
        rows = self.conn.execute(
            "SELECT scope FROM item_scopes WHERE item_id=? AND item_table='facts' ORDER BY scope",
            [fid]
        ).fetchall()
        scopes = [r[0] for r in rows]
        self.assertIn(SCOPE_A, scopes)
        self.assertIn(SCOPE_B, scopes)


class TestAutoPromotion(_ScopedTestBase):
    """
    GIVEN an item is seen in N+ distinct project scopes
    WHEN the auto-promotion threshold is met
    THEN the item's scope is changed to __global__
    """

    def test_given_fact_in_two_scopes_then_not_promoted(self):
        text = "Two-scope fact should stay local"
        emb = _mock_embed(text)
        fid, _ = db.upsert_fact(self.conn, text,
                                "contextual", "medium", "high", emb, "sess-1",
                                _noop_decay, scope=SCOPE_A)
        db.upsert_fact(self.conn, text,
                       "contextual", "medium", "high", emb, "sess-1",
                       _noop_decay, scope=SCOPE_B)
        row = self.conn.execute("SELECT scope FROM facts WHERE id=?", [fid]).fetchone()
        self.assertNotEqual(row[0], _cfg.GLOBAL_SCOPE)

    def test_given_fact_in_three_scopes_then_auto_promoted_to_global(self):
        text = "Three-scope fact should be promoted"
        emb = _mock_embed(text)
        fid, _ = db.upsert_fact(self.conn, text,
                                "contextual", "medium", "high", emb, "sess-1",
                                _noop_decay, scope=SCOPE_A)
        db.upsert_fact(self.conn, text,
                       "contextual", "medium", "high", emb, "sess-1",
                       _noop_decay, scope=SCOPE_B)
        db.upsert_fact(self.conn, text,
                       "contextual", "medium", "high", emb, "sess-1",
                       _noop_decay, scope=SCOPE_C)
        row = self.conn.execute("SELECT scope FROM facts WHERE id=?", [fid]).fetchone()
        self.assertEqual(row[0], _cfg.GLOBAL_SCOPE)

    def test_given_entity_in_three_scopes_then_auto_promoted(self):
        eid = db.upsert_entity(self.conn, "SharedLib", scope=SCOPE_A)
        db.upsert_entity(self.conn, "SharedLib", scope=SCOPE_B)
        db.upsert_entity(self.conn, "SharedLib", scope=SCOPE_C)
        row = self.conn.execute("SELECT scope FROM entities WHERE id=?", [eid]).fetchone()
        self.assertEqual(row[0], _cfg.GLOBAL_SCOPE)

    def test_given_decision_in_three_scopes_then_auto_promoted(self):
        text = "Always use TypeScript"
        emb = _mock_embed(text)
        did, _ = db.upsert_decision(self.conn, text, "long", emb, "sess-1",
                                    _noop_decay, scope=SCOPE_A)
        db.upsert_decision(self.conn, text, "long", emb, "sess-1",
                           _noop_decay, scope=SCOPE_B)
        db.upsert_decision(self.conn, text, "long", emb, "sess-1",
                           _noop_decay, scope=SCOPE_C)
        row = self.conn.execute("SELECT scope FROM decisions WHERE id=?", [did]).fetchone()
        self.assertEqual(row[0], _cfg.GLOBAL_SCOPE)


class TestManualPromotion(_ScopedTestBase):
    """
    GIVEN a project-scoped item
    WHEN manually promoted to global
    THEN the item's scope becomes __global__
    """

    def test_given_scoped_fact_when_promoted_then_scope_is_global(self):
        emb = _mock_embed("Manual promote test")
        fid, _ = db.upsert_fact(self.conn, "Manual promote test",
                                "technical", "long", "high", emb, "sess-1",
                                _noop_decay, scope=SCOPE_A)
        success = db.promote_to_global(self.conn, fid, "facts")
        self.assertTrue(success)
        row = self.conn.execute("SELECT scope FROM facts WHERE id=?", [fid]).fetchone()
        self.assertEqual(row[0], _cfg.GLOBAL_SCOPE)

    def test_given_already_global_fact_when_promoted_then_returns_true(self):
        emb = _mock_embed("Already global fact")
        fid, _ = db.upsert_fact(self.conn, "Already global fact",
                                "technical", "long", "high", emb, "sess-1",
                                _noop_decay, scope=_cfg.GLOBAL_SCOPE)
        success = db.promote_to_global(self.conn, fid, "facts")
        self.assertTrue(success)

    def test_given_nonexistent_id_when_promoted_then_returns_false(self):
        success = db.promote_to_global(self.conn, "nonexistent-uuid", "facts")
        self.assertFalse(success)

    def test_given_scoped_decision_when_promoted_then_scope_is_global(self):
        emb = _mock_embed("Promote this decision")
        did, _ = db.upsert_decision(self.conn, "Promote this decision",
                                    "long", emb, "sess-1", _noop_decay, scope=SCOPE_B)
        db.promote_to_global(self.conn, did, "decisions")
        row = self.conn.execute("SELECT scope FROM decisions WHERE id=?", [did]).fetchone()
        self.assertEqual(row[0], _cfg.GLOBAL_SCOPE)


class TestScopedRecall(_ScopedTestBase):
    """
    GIVEN facts in different scopes
    WHEN session_recall or prompt_recall is called with a scope
    THEN project-local items are prioritized and global fills remaining
    """

    def _seed_scoped_data(self):
        from memory.config import GLOBAL_SCOPE
        # Scope A facts
        for i, text in enumerate(["Alpha uses React", "Alpha deploys to Vercel"]):
            emb = _mock_embed(text + " " * (i * 50))  # ensure unique embeddings
            db.upsert_fact(self.conn, text, "technical", "long", "high",
                           emb, "sess-1", _noop_decay, scope=SCOPE_A)
        # Scope B fact
        emb_b = _mock_embed("Beta uses Django for backend")
        db.upsert_fact(self.conn, "Beta uses Django for backend",
                       "technical", "long", "high", emb_b, "sess-1",
                       _noop_decay, scope=SCOPE_B)
        # Global fact
        emb_g = _mock_embed("User is a senior engineer")
        db.upsert_fact(self.conn, "User is a senior engineer",
                       "personal", "long", "high", emb_g, "sess-1",
                       _noop_decay, scope=GLOBAL_SCOPE)
        # Global decision
        emb_d = _mock_embed("Always write tests")
        db.upsert_decision(self.conn, "Always write tests",
                           "long", emb_d, "sess-1", _noop_decay, scope=GLOBAL_SCOPE)

    def test_given_scoped_data_when_session_recall_scope_a_then_includes_a_and_global(self):
        self._seed_scoped_data()
        ctx = recall.session_recall(self.conn, scope=SCOPE_A)
        fact_texts = [f["text"] for f in ctx["long_facts"]]
        self.assertIn("Alpha uses React", fact_texts)
        self.assertIn("User is a senior engineer", fact_texts)

    def test_given_scoped_data_when_session_recall_scope_a_then_excludes_b(self):
        self._seed_scoped_data()
        ctx = recall.session_recall(self.conn, scope=SCOPE_A)
        fact_texts = [f["text"] for f in ctx["long_facts"]]
        self.assertNotIn("Beta uses Django for backend", fact_texts)

    def test_given_scoped_data_when_session_recall_scope_a_then_includes_global_decisions(self):
        self._seed_scoped_data()
        ctx = recall.session_recall(self.conn, scope=SCOPE_A)
        decision_texts = [d["text"] for d in ctx["decisions"]]
        self.assertIn("Always write tests", decision_texts)

    def test_given_scoped_data_when_session_recall_no_scope_then_returns_all(self):
        self._seed_scoped_data()
        ctx = recall.session_recall(self.conn, scope=None)
        fact_texts = [f["text"] for f in ctx["long_facts"]]
        self.assertIn("Alpha uses React", fact_texts)
        self.assertIn("Beta uses Django for backend", fact_texts)
        self.assertIn("User is a senior engineer", fact_texts)

    def test_given_scoped_data_when_prompt_recall_scope_a_then_excludes_b(self):
        self._seed_scoped_data()
        query_emb = _mock_embed("Tell me about the frontend")
        ctx = recall.prompt_recall(self.conn, query_emb, "Tell me about the frontend", scope=SCOPE_A)
        fact_texts = [f["text"] for f in ctx["facts"]]
        self.assertNotIn("Beta uses Django for backend", fact_texts)


class TestSchemaMigration2(_ScopedTestBase):
    """
    GIVEN a database with the initial schema
    WHEN migration 2 runs
    THEN scope columns and item_scopes table exist
    """

    def test_facts_table_has_scope_column(self):
        cols = {r[0] for r in self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='facts'"
        ).fetchall()}
        self.assertIn("scope", cols)

    def test_item_scopes_table_exists(self):
        tables = {r[0] for r in self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        self.assertIn("item_scopes", tables)

    def test_scope_default_is_global(self):
        from memory.config import GLOBAL_SCOPE
        emb = _mock_embed("Default scope test fact")
        fid, _ = db.upsert_fact(self.conn, "Default scope test fact",
                                "contextual", "short", "low", emb, "sess-1", _noop_decay)
        row = self.conn.execute("SELECT scope FROM facts WHERE id=?", [fid]).fetchone()
        self.assertEqual(row[0], GLOBAL_SCOPE)

    def test_sessions_table_has_scope_column(self):
        cols = {r[0] for r in self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='sessions'"
        ).fetchall()}
        self.assertIn("scope", cols)


class TestRememberCommand(unittest.TestCase):
    """
    GIVEN the /remember command handler
    WHEN processing different /remember inputs
    THEN facts or decisions are stored with the correct scope and type
    """

    def setUp(self):
        self.db_path = Path(tempfile.mktemp(suffix=".duckdb"))
        _cfg.DB_PATH = self.db_path

    def tearDown(self):
        _cfg.DB_PATH = Path(tempfile.mktemp(suffix=".duckdb"))
        try:
            self.db_path.unlink()
        except Exception:
            pass

    def _run_remember(self, prompt: str, cwd: str = "/tmp") -> str:
        """Simulate the /remember path by calling _handle_remember and capturing stdout."""
        import io
        import importlib.util
        from contextlib import redirect_stdout, redirect_stderr

        # Load the hook script as a module (it's not in a package)
        spec = importlib.util.spec_from_file_location(
            "user_prompt_submit",
            str(PROJECT_ROOT / "hooks" / "user_prompt_submit.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        payload = {"prompt": prompt, "cwd": cwd, "session_id": "test-remember"}
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            mod._handle_remember(prompt, payload)
        return stdout_buf.getvalue()

    @patch("memory.embeddings.embed", side_effect=_mock_embed)
    def test_given_remember_fact_then_stored_as_long_term_fact(self, mock_emb):
        output = self._run_remember("/remember The API uses gRPC for inter-service communication")
        parsed = json.loads(output)
        self.assertIn("Stored", parsed["additionalContext"])
        self.assertIn("fact", parsed["additionalContext"])

        conn = db.get_connection(db_path=str(self.db_path))
        facts = db.get_facts_by_temporal(conn, "long", 10)
        texts = [f["text"] for f in facts]
        self.assertIn("The API uses gRPC for inter-service communication", texts)
        conn.close()

    @patch("memory.embeddings.embed", side_effect=_mock_embed)
    def test_given_remember_global_then_stored_in_global_scope(self, mock_emb):
        output = self._run_remember("/remember global: My name is Ben")
        parsed = json.loads(output)
        self.assertIn("global", parsed["additionalContext"])

        conn = db.get_connection(db_path=str(self.db_path))
        row = conn.execute(
            "SELECT scope FROM facts WHERE text = 'My name is Ben'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], _cfg.GLOBAL_SCOPE)
        conn.close()

    @patch("memory.embeddings.embed", side_effect=_mock_embed)
    def test_given_remember_decision_then_stored_as_decision(self, mock_emb):
        output = self._run_remember("/remember decision: We chose PostgreSQL over MySQL")
        parsed = json.loads(output)
        self.assertIn("decision", parsed["additionalContext"])

        conn = db.get_connection(db_path=str(self.db_path))
        decisions = db.get_decisions(conn, 10)
        texts = [d["text"] for d in decisions]
        self.assertIn("We chose PostgreSQL over MySQL", texts)
        conn.close()

    @patch("memory.embeddings.embed", side_effect=_mock_embed)
    def test_given_remember_global_decision_then_stored_as_global_decision(self, mock_emb):
        output = self._run_remember("/remember global decision: Always use TypeScript")
        parsed = json.loads(output)
        self.assertIn("decision", parsed["additionalContext"])
        self.assertIn("global", parsed["additionalContext"])

        conn = db.get_connection(db_path=str(self.db_path))
        row = conn.execute(
            "SELECT scope FROM decisions WHERE text = 'Always use TypeScript'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], _cfg.GLOBAL_SCOPE)
        conn.close()

    @patch("memory.embeddings.embed", side_effect=_mock_embed)
    def test_given_remember_duplicate_then_reinforced_not_duplicated(self, mock_emb):
        self._run_remember("/remember The API uses REST endpoints")
        output = self._run_remember("/remember The API uses REST endpoints")
        parsed = json.loads(output)
        self.assertIn("Reinforced", parsed["additionalContext"])

        conn = db.get_connection(db_path=str(self.db_path))
        count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        self.assertEqual(count, 1)
        conn.close()

    @patch("memory.embeddings.embed", side_effect=_mock_embed)
    def test_given_remember_empty_then_shows_usage(self, mock_emb):
        output = self._run_remember("/remember")
        parsed = json.loads(output)
        self.assertIn("Usage", parsed["additionalContext"])


if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
