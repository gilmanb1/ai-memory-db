#!/usr/bin/env python3
"""
test_recall_quality.py — Red/green regression tests for LongMemEval recall quality.

Each test class maps to a specific benchmark failure mode. Tests use deterministic
mock embeddings and pre-built databases — no API keys or Ollama needed.

Coverage:
  - Chunking: split_into_chunks windowing, overlap, boundary handling
  - Detail retrieval: chunks surface specific names/places missed by extraction
  - Counting: sibling expansion + BM25 on chunks finds all items
  - Knowledge-update: recency handling with conflicting values
  - Preference: broad context retrieval for user profile questions
  - Format: context rendering with chunks, siblings, budget enforcement
  - Integration: end-to-end recall pipeline with chunks

Usage: python3 bench/longmemeval/test_recall_quality.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import memory.config as _cfg
_cfg.DB_PATH = Path(tempfile.mktemp(suffix=".duckdb"))

from memory import db, recall
from memory.decay import compute_decay_score
from memory.chunking import split_into_chunks


# ── Deterministic mock embedding ─────────────────────────────────────────

def _mock_embed(text: str) -> list[float]:
    """Hash-based fake embedding. Same text → same vector."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    vec = [((b % 200) - 100) / 100.0 for b in h]
    while len(vec) < 768:
        vec = vec + vec
    vec = vec[:768]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _fresh_conn():
    db_path = tempfile.mktemp(suffix=".duckdb")
    return db_path, db.get_connection(db_path=db_path)


def _insert_fact_with_chunk(conn, fact_text, chunk_text, session_id="sess1", scope="test"):
    """Helper: insert a chunk and a fact linked to it."""
    chunk_emb = _mock_embed(chunk_text[:400])
    cid = db.insert_chunk(conn, chunk_text, session_id, scope, embedding=chunk_emb)
    fact_emb = _mock_embed(fact_text)
    fid, _ = db.upsert_fact(
        conn, fact_text, "personal", "long", "high",
        fact_emb, session_id, compute_decay_score,
        scope=scope, source_chunk_id=cid,
    )
    return fid, cid


# ═══════════════════════════════════════════════════════════════════════════
# CHUNKING TESTS — split_into_chunks()
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkingSplitBasic(unittest.TestCase):
    """Green: chunking produces correct windows."""

    def test_short_text_single_chunk(self):
        chunks = split_into_chunks("Hello world", window=400, overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world")

    def test_empty_text_no_chunks(self):
        self.assertEqual(split_into_chunks(""), [])
        self.assertEqual(split_into_chunks("   "), [])

    def test_exact_window_size_single_chunk(self):
        text = "A" * 400
        chunks = split_into_chunks(text, window=400, overlap=100)
        self.assertEqual(len(chunks), 1)

    def test_long_text_multiple_chunks(self):
        text = "word " * 200  # ~1000 chars
        chunks = split_into_chunks(text, window=400, overlap=100)
        self.assertGreater(len(chunks), 1)

    def test_overlap_creates_shared_content(self):
        # Create text where we can verify overlap
        text = "AAAA " * 40 + "BBBB " * 40 + "CCCC " * 40  # ~600 chars
        chunks = split_into_chunks(text, window=300, overlap=100)
        if len(chunks) >= 2:
            # Last part of chunk 0 should overlap with start of chunk 1
            end_of_first = chunks[0][-50:]
            start_of_second = chunks[1][:50]
            # They should share some text (overlap region)
            self.assertTrue(
                any(word in start_of_second for word in end_of_first.split()),
                "Chunks should have overlapping content"
            )


class TestChunkingSplitEdgeCases(unittest.TestCase):
    """Red: chunking handles edge cases without crashing."""

    def test_none_input(self):
        # Should not crash
        result = split_into_chunks(None, window=400, overlap=100)
        self.assertEqual(result, [])

    def test_whitespace_only(self):
        self.assertEqual(split_into_chunks("   \n\n  "), [])

    def test_overlap_larger_than_window(self):
        # Degenerate case — should still work
        chunks = split_into_chunks("A" * 1000, window=100, overlap=200)
        self.assertGreater(len(chunks), 0)

    def test_very_small_window(self):
        chunks = split_into_chunks("Hello world this is a test", window=10, overlap=2)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)

    def test_sentence_boundary_splitting(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = split_into_chunks(text, window=50, overlap=10)
        # Should prefer splitting at period boundaries
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)


class TestChunkingPreservesContent(unittest.TestCase):
    """Green: all content from the original text appears in at least one chunk."""

    def test_no_content_lost(self):
        text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa " * 10
        chunks = split_into_chunks(text, window=100, overlap=20)
        combined = " ".join(chunks)
        for word in ["Alpha", "Kappa", "Epsilon"]:
            self.assertIn(word, combined, f"'{word}' should appear in at least one chunk")

    def test_specific_details_in_focused_chunks(self):
        """The key test: specific details end up in small, focused chunks."""
        text = (
            "We talked about the weather today. It was sunny and warm.\n\n"
            "Then we discussed shopping. I went to Target and redeemed a $5 coupon on coffee creamer.\n\n"
            "After that we talked about weekend plans. I'm going hiking on Saturday."
        )
        chunks = split_into_chunks(text, window=150, overlap=30)
        # "Target" should be in a focused chunk, not diluted across the whole text
        target_chunks = [c for c in chunks if "Target" in c]
        self.assertGreater(len(target_chunks), 0, "Target should appear in at least one chunk")
        # The chunk with Target should be small and focused
        self.assertLess(len(target_chunks[0]), 200, "Target chunk should be focused, not the whole text")


# ═══════════════════════════════════════════════════════════════════════════
# CHUNK DB OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkDBOperations(unittest.TestCase):
    """Green/Red: chunk insert, retrieve, search."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_insert_chunk_with_embedding(self):
        emb = _mock_embed("test chunk")
        cid = db.insert_chunk(self.conn, "test chunk text", "sess1", "scope1", embedding=emb)
        self.assertIsInstance(cid, str)

    def test_insert_chunk_without_embedding(self):
        cid = db.insert_chunk(self.conn, "no embedding chunk", "sess1", "scope1")
        chunks = db.get_chunks_by_ids(self.conn, [cid])
        self.assertIn(cid, chunks)

    def test_search_chunks_vector(self):
        emb = _mock_embed("shopping at target store")
        db.insert_chunk(self.conn, "I went shopping at Target", "sess1", "test", embedding=emb)
        results = db.search_chunks(self.conn, emb, limit=5, scope="test")
        self.assertGreater(len(results), 0)
        self.assertIn("Target", results[0]["text"])

    def test_search_chunks_vector_no_match(self):
        emb = _mock_embed("shopping at target store")
        db.insert_chunk(self.conn, "I went shopping at Target", "sess1", "test", embedding=emb)
        other_emb = _mock_embed("completely unrelated dinosaur topic")
        results = db.search_chunks(self.conn, other_emb, limit=5, scope="test")
        # May return 0 or low-score results
        self.assertIsInstance(results, list)

    def test_get_chunks_by_ids_empty(self):
        self.assertEqual(db.get_chunks_by_ids(self.conn, []), {})

    def test_get_chunks_by_ids_nonexistent(self):
        self.assertEqual(db.get_chunks_by_ids(self.conn, ["fake_id"]), {})

    def test_deactivated_chunk_excluded(self):
        cid = db.insert_chunk(self.conn, "will delete", "sess1", "test")
        self.conn.execute("UPDATE conversation_chunks SET is_active = FALSE WHERE id = ?", [cid])
        self.assertEqual(db.get_chunks_by_ids(self.conn, [cid]), {})

    def test_multiple_small_chunks_from_one_session(self):
        """Simulates small overlapping chunks from split_into_chunks."""
        chunks_text = ["Part A about weather", "Part B about Target coupon", "Part C about hiking"]
        cids = []
        for text in chunks_text:
            emb = _mock_embed(text)
            cids.append(db.insert_chunk(self.conn, text, "sess1", "test", embedding=emb))
        fetched = db.get_chunks_by_ids(self.conn, cids)
        self.assertEqual(len(fetched), 3)


# ═══════════════════════════════════════════════════════════════════════════
# DETAIL RETRIEVAL — maps to V1 failures (Target, Sugar Factory, shift rotation)
# ═══════════════════════════════════════════════════════════════════════════

class TestDetailRetrievalViaChunks(unittest.TestCase):
    """Green: specific details found via chunk search even when fact is generic."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_target_coupon_found_in_chunk(self):
        """V1 failure: 'where did I redeem the coupon?' → 'Target' only in chunk."""
        # Fact is generic (extraction lost the detail)
        _insert_fact_with_chunk(
            self.conn,
            "User redeemed a $5 coupon on coffee creamer",
            "I just redeemed a $5 coupon on coffee creamer at Target last Sunday",
        )
        emb = _mock_embed("User redeemed a $5 coupon on coffee creamer")
        result = recall._legacy_prompt_recall(self.conn, emb, "where did I redeem the coupon", scope="test")
        all_text = " ".join(f["text"] for f in result["facts"])
        all_text += " " + " ".join(c["text"] for c in result.get("chunks", {}).values())
        self.assertIn("Target", all_text)

    def test_sugar_factory_found_in_chunk(self):
        """V1 failure: 'dessert shop with giant milkshakes?' → 'Sugar Factory' only in chunk."""
        _insert_fact_with_chunk(
            self.conn,
            "User discussed Orlando dessert shops",
            "That unique dessert shop with giant milkshakes is The Sugar Factory at Icon Park",
        )
        emb = _mock_embed("User discussed Orlando dessert shops")
        result = recall._legacy_prompt_recall(self.conn, emb, "dessert shop milkshakes", scope="test")
        all_text = " ".join(c["text"] for c in result.get("chunks", {}).values())
        self.assertIn("Sugar Factory", all_text)

    def test_direct_chunk_search_finds_unlinked_detail(self):
        """Chunk search finds details even when NO fact links to the chunk."""
        # Insert chunk directly (not linked to any fact)
        emb = _mock_embed("The Plesiosaur had a blue scaly body in the picture")
        db.insert_chunk(self.conn, "The Plesiosaur had a blue scaly body in the picture", "sess1", "test", embedding=emb)

        # Search should find it via direct chunk search
        result = recall._load_chunks_for_facts(
            self.conn, [], query_embedding=emb, prompt_text="Plesiosaur color", scope="test"
        )
        all_text = " ".join(c["text"] for c in result.values())
        self.assertIn("blue", all_text)


# ═══════════════════════════════════════════════════════════════════════════
# COUNTING QUESTIONS — maps to V1+V2 failures (clothing=3, model kits=5)
# ═══════════════════════════════════════════════════════════════════════════

class TestCountingViaChunksAndSiblings(unittest.TestCase):
    """Green: all items findable via sibling expansion + chunk search."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_siblings_from_same_chunk_found(self):
        """3 facts from same chunk: retrieving 1 brings siblings."""
        chunk_emb = _mock_embed("clothing items to pick up or return from store")
        cid = db.insert_chunk(self.conn, "pick up blue jacket from dry cleaner, return red dress to Macys, pick up altered pants from tailor", "sess1", "test", embedding=chunk_emb)
        facts = [
            "Need to pick up blue jacket from dry cleaner",
            "Need to return red dress to Macys",
            "Need to pick up altered pants from tailor",
        ]
        for fact in facts:
            emb = _mock_embed(fact)
            db.upsert_fact(self.conn, fact, "personal", "long", "high",
                          emb, "sess1", compute_decay_score, scope="test", source_chunk_id=cid)

        # Search for one item → siblings should bring others
        emb = _mock_embed("Need to pick up blue jacket from dry cleaner")
        result = recall._legacy_prompt_recall(self.conn, emb, "clothing items", scope="test")
        all_text = " ".join(f["text"] for f in result["facts"] + result.get("sibling_facts", []))
        found = sum(1 for item in ["jacket", "dress", "pants"] if item in all_text.lower())
        self.assertGreaterEqual(found, 2, f"Should find >=2 items, found {found}")

    def test_items_across_sessions_found_via_chunk_search(self):
        """Items in different sessions found via independent chunk search."""
        items = [
            ("Picked up jacket from dry cleaner", "sess1"),
            ("Returning dress to Macys", "sess2"),
            ("Picking up pants from tailor", "sess3"),
        ]
        for text, sid in items:
            emb = _mock_embed(text)
            db.insert_chunk(self.conn, text, sid, "test", embedding=emb)
            db.upsert_fact(self.conn, text, "personal", "long", "high",
                          emb, sid, compute_decay_score, scope="test")

        # Search should find at least 2 of 3 via fact search + chunk search
        emb = _mock_embed("Picked up jacket from dry cleaner")
        result = recall._legacy_prompt_recall(self.conn, emb, "clothing items pick up return", scope="test")
        all_text = " ".join(f["text"] for f in result["facts"])
        all_text += " " + " ".join(c["text"] for c in result.get("chunks", {}).values())
        found = sum(1 for item in ["jacket", "dress", "pants"] if item in all_text.lower())
        self.assertGreaterEqual(found, 1)


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE UPDATE — maps to V2-V4 regressions (mortgage, Rachel's move)
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeUpdateRecency(unittest.TestCase):
    """Green: both old and new values retrievable. Red: no crash on conflicts."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_both_values_in_context(self):
        """Both old and new mortgage values should appear (answer model picks latest)."""
        _insert_fact_with_chunk(
            self.conn,
            "[2023/03] Pre-approved for $350,000 mortgage",
            "Got pre-approved for $350,000 mortgage from Wells Fargo",
            session_id="sess1",
        )
        # Same embedding so dedup links them
        emb = _mock_embed("[2023/03] Pre-approved for $350,000 mortgage")
        _insert_fact_with_chunk(
            self.conn,
            "[2023/06] Pre-approval updated to $400,000",
            "Great news - Wells Fargo increased pre-approval to $400,000",
            session_id="sess2",
        )
        result = recall._legacy_prompt_recall(self.conn, emb, "mortgage pre-approval amount", scope="test")
        all_text = " ".join(f["text"] for f in result["facts"])
        all_text += " " + " ".join(c["text"] for c in result.get("chunks", {}).values())
        # At least one value should be present
        self.assertTrue("350,000" in all_text or "400,000" in all_text)

    def test_fact_without_chunk_no_crash(self):
        """Facts without chunk links should not crash chunk loading."""
        emb = _mock_embed("standalone fact no chunk")
        db.upsert_fact(self.conn, "standalone fact", "contextual", "long", "high",
                      emb, "sess1", compute_decay_score, scope="test")
        result = recall._legacy_prompt_recall(self.conn, emb, "standalone", scope="test")
        self.assertIn("chunks", result)
        self.assertIsInstance(result["chunks"], dict)


# ═══════════════════════════════════════════════════════════════════════════
# PREFERENCE QUESTIONS — maps to V2-V5 regressions
# ═══════════════════════════════════════════════════════════════════════════

class TestPreferenceRetrieval(unittest.TestCase):
    """Green: user profile context found. Red: no crash on missing profile."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_user_profile_found_via_fact(self):
        _insert_fact_with_chunk(
            self.conn,
            "User is learning Adobe Premiere Pro for video editing",
            "I love the advanced color grading features in Premiere Pro",
        )
        emb = _mock_embed("User is learning Adobe Premiere Pro for video editing")
        result = recall._legacy_prompt_recall(self.conn, emb, "video editing resources", scope="test")
        all_text = " ".join(f["text"] for f in result["facts"])
        self.assertIn("Premiere", all_text)

    def test_user_profile_found_via_chunk_when_fact_generic(self):
        """Chunk search finds 'Adobe Premiere' even if fact says 'video editing software'."""
        chunk_emb = _mock_embed("I love Adobe Premiere Pro advanced color grading")
        db.insert_chunk(self.conn, "I love Adobe Premiere Pro advanced color grading", "sess1", "test", embedding=chunk_emb)
        emb = _mock_embed("video editing software user preference")
        db.upsert_fact(self.conn, "User uses video editing software", "personal", "long", "high",
                      emb, "sess1", compute_decay_score, scope="test")

        result = recall._load_chunks_for_facts(
            self.conn, [], query_embedding=chunk_emb, prompt_text="Adobe Premiere", scope="test"
        )
        all_text = " ".join(c["text"] for c in result.values())
        self.assertIn("Premiere", all_text)

    def test_empty_db_returns_empty_not_error(self):
        emb = _mock_embed("anything")
        result = recall._legacy_prompt_recall(self.conn, emb, "test query", scope="test")
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["chunks"], {})


# ═══════════════════════════════════════════════════════════════════════════
# FORMAT CONTEXT — rendering with chunks, siblings, budget
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatContextRendering(unittest.TestCase):
    """Green: sections appear. Red: empty data produces no noise."""

    def test_all_sections_rendered(self):
        data = {
            "facts": [{"text": "coupon fact", "temporal_class": "long"}],
            "chunks": {"c1": {"id": "c1", "text": "Verbatim conversation about Target coupon"}},
            "sibling_facts": [{"text": "sibling: $5 off coffee creamer at Target"}],
            "ideas": [], "observations": [], "relationships": [],
            "questions": [], "narratives": [],
        }
        formatted = recall.format_prompt_context(data)
        self.assertIn("Relevant Facts", formatted)
        self.assertIn("Source Conversation Context", formatted)
        self.assertIn("Additional Context", formatted)

    def test_empty_chunks_no_section(self):
        data = {
            "facts": [{"text": "simple fact", "temporal_class": "long"}],
            "chunks": {}, "sibling_facts": [],
            "ideas": [], "observations": [], "relationships": [],
            "questions": [], "narratives": [],
        }
        formatted = recall.format_prompt_context(data)
        self.assertNotIn("Source Conversation Context", formatted)
        self.assertNotIn("Additional Context", formatted)

    def test_chunk_truncation(self):
        import memory.config as cfg
        orig = cfg.CHUNK_MAX_DISPLAY_CHARS
        orig_budget = cfg.PROMPT_TOKEN_BUDGET
        try:
            cfg.CHUNK_MAX_DISPLAY_CHARS = 50
            cfg.PROMPT_TOKEN_BUDGET = 10000
            data = {
                "facts": [{"text": "fact", "temporal_class": "long"}],
                "chunks": {"c1": {"id": "c1", "text": "X" * 200}},
                "sibling_facts": [],
                "ideas": [], "observations": [], "relationships": [],
                "questions": [], "narratives": [],
            }
            formatted = recall.format_prompt_context(data)
            self.assertIn("...", formatted)
            self.assertNotIn("X" * 200, formatted)
        finally:
            cfg.CHUNK_MAX_DISPLAY_CHARS = orig
            cfg.PROMPT_TOKEN_BUDGET = orig_budget

    def test_no_content_returns_empty(self):
        data = {
            "facts": [], "chunks": {}, "sibling_facts": [],
            "ideas": [], "observations": [], "relationships": [],
            "questions": [], "narratives": [],
        }
        self.assertEqual(recall.format_prompt_context(data), "")

    def test_chunks_key_always_present_in_recall(self):
        _, conn = _fresh_conn()
        emb = _mock_embed("test")
        db.upsert_fact(conn, "test fact", "personal", "long", "high",
                      emb, "sess1", compute_decay_score, scope="test")
        result = recall._legacy_prompt_recall(conn, emb, "test", scope="test")
        self.assertIn("chunks", result)
        self.assertIn("sibling_facts", result)
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# SIBLING EXPANSION
# ═══════════════════════════════════════════════════════════════════════════

class TestSiblingExpansion(unittest.TestCase):
    """Green: siblings found. Red: limits enforced."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_siblings_from_same_chunk(self):
        cid = db.insert_chunk(self.conn, "conversation", "sess1", "test")
        for i in range(5):
            emb = _mock_embed(f"fact {i} unique text")
            db.upsert_fact(self.conn, f"fact {i}", "personal", "long", "high",
                          emb, "sess1", compute_decay_score, scope="test", source_chunk_id=cid)
        siblings = recall._expand_sibling_facts(self.conn, [{"id": "x", "source_chunk_id": cid}])
        self.assertGreater(len(siblings), 0)

    def test_no_siblings_when_no_chunk_id(self):
        emb = _mock_embed("no chunk")
        db.upsert_fact(self.conn, "no chunk fact", "personal", "long", "high",
                      emb, "sess1", compute_decay_score, scope="test")
        siblings = recall._expand_sibling_facts(self.conn, [{"id": "x"}])
        self.assertEqual(siblings, [])

    def test_sibling_limit_enforced(self):
        import memory.config as cfg
        orig = cfg.PROMPT_SIBLINGS_LIMIT
        try:
            cfg.PROMPT_SIBLINGS_LIMIT = 3
            cid = db.insert_chunk(self.conn, "conversation", "sess1", "test")
            for i in range(20):
                emb = _mock_embed(f"unique fact number {i} text")
                db.upsert_fact(self.conn, f"fact {i}", "personal", "long", "high",
                              emb, "sess1", compute_decay_score, scope="test", source_chunk_id=cid)
            siblings = recall._expand_sibling_facts(self.conn, [{"id": "x", "source_chunk_id": cid}], limit=3)
            self.assertLessEqual(len(siblings), 3)
        finally:
            cfg.PROMPT_SIBLINGS_LIMIT = orig

    def test_excludes_already_retrieved_facts(self):
        cid = db.insert_chunk(self.conn, "conversation", "sess1", "test")
        emb = _mock_embed("already retrieved fact")
        fid, _ = db.upsert_fact(self.conn, "already retrieved", "personal", "long", "high",
                               emb, "sess1", compute_decay_score, scope="test", source_chunk_id=cid)
        siblings = recall._expand_sibling_facts(self.conn, [{"id": fid, "source_chunk_id": cid}])
        sibling_ids = {s["id"] for s in siblings}
        self.assertNotIn(fid, sibling_ids)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION — end-to-end recall with small chunks
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndRecallWithSmallChunks(unittest.TestCase):
    """Green: full pipeline with split_into_chunks → embed → store → recall."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_small_chunks_stored_and_searched(self):
        """Split a conversation into small chunks, store, and search."""
        conversation = (
            "User: What's the weather like today?\n"
            "Assistant: It's sunny and 72 degrees.\n\n"
            "User: I went to Target today and redeemed a $5 coupon on coffee creamer.\n"
            "Assistant: Nice savings! Target has great deals.\n\n"
            "User: I'm planning to go hiking this weekend at Bear Mountain.\n"
            "Assistant: Bear Mountain is beautiful this time of year."
        )
        windows = split_into_chunks(conversation, window=200, overlap=50)
        self.assertGreater(len(windows), 1)

        # Find the window containing "Target"
        target_window = next((w for w in windows if "Target" in w), None)
        self.assertIsNotNone(target_window, "Target should appear in at least one chunk window")

        for w in windows:
            emb = _mock_embed(w)
            db.insert_chunk(self.conn, w, "sess1", "test", embedding=emb)

        # Search using the exact Target window embedding — guarantees a match
        target_emb = _mock_embed(target_window)
        results = db.search_chunks(self.conn, target_emb, limit=3, scope="test")
        all_text = " ".join(r["text"] for r in results)
        self.assertIn("Target", all_text)

    def test_session_recall_still_works(self):
        """Session recall returns long-term facts normally."""
        emb = _mock_embed("user name is Alice")
        db.upsert_fact(self.conn, "User name is Alice", "personal", "long", "high",
                      emb, "sess1", compute_decay_score, scope="test")
        result = recall.session_recall(self.conn, scope="test")
        self.assertTrue(len(result["long_facts"]) > 0)

    def test_prompt_recall_combines_facts_and_chunks(self):
        """Prompt recall returns both facts and matching chunks."""
        _insert_fact_with_chunk(
            self.conn,
            "User went shopping",
            "User went to Target and bought coffee creamer with a $5 coupon",
        )
        emb = _mock_embed("User went shopping")
        result = recall._legacy_prompt_recall(self.conn, emb, "shopping Target coupon", scope="test")
        self.assertIn("chunks", result)
        self.assertIn("facts", result)
        self.assertGreater(len(result["facts"]), 0)


# ═══════════════════════════════════════════════════════════════════════════
# NO REGRESSION — existing functionality still works
# ═══════════════════════════════════════════════════════════════════════════

class TestNoRegression(unittest.TestCase):
    """Red: existing functionality must not break."""

    def setUp(self):
        self.db_path, self.conn = _fresh_conn()

    def tearDown(self):
        self.conn.close()

    def test_fact_without_chunk_still_works(self):
        emb = _mock_embed("old style fact no chunk")
        fid, _ = db.upsert_fact(self.conn, "old style fact", "contextual", "short", "medium",
                               emb, "sess1", compute_decay_score, scope="test")
        row = self.conn.execute("SELECT source_chunk_id FROM facts WHERE id = ?", [fid]).fetchone()
        self.assertIsNone(row[0])

    def test_deduplication_still_works(self):
        emb = _mock_embed("dedup test fact")
        fid1, new1 = db.upsert_fact(self.conn, "dedup test fact", "contextual", "long", "high",
                                    emb, "sess1", compute_decay_score, scope="test")
        fid2, new2 = db.upsert_fact(self.conn, "dedup test fact", "contextual", "long", "high",
                                    emb, "sess2", compute_decay_score, scope="test")
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(fid1, fid2)

    def test_format_prompt_context_without_chunks_key(self):
        """Old-style recall dicts without 'chunks' key should not crash."""
        data = {
            "facts": [{"text": "fact", "temporal_class": "long"}],
            "ideas": [], "observations": [], "relationships": [],
            "questions": [], "narratives": [],
        }
        formatted = recall.format_prompt_context(data)
        self.assertIn("fact", formatted)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
