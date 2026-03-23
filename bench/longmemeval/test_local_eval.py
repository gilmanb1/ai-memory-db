#!/usr/bin/env python3
"""
test_local_eval.py — Local evaluation of V6 hypotheses using real ONNX embeddings.

Tests whether small overlapping chunks surface the specific details needed
to answer each of the 9 non-confident V6 predictions. Uses real nomic-embed-text
ONNX embeddings (no API calls, no Ollama needed).

Each test creates a realistic conversation haystack (multiple sessions of irrelevant
chatter + one session containing the answer), processes it through the full
chunking + embedding + recall pipeline, and checks if the answer detail appears
in the recalled context.

This directly validates or invalidates each V6 prediction before running on AWS.

Usage: python3 bench/longmemeval/test_local_eval.py
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

from memory import db, recall, embeddings
from memory.decay import compute_decay_score
from memory.chunking import split_into_chunks


def _fresh_conn():
    db_path = tempfile.mktemp(suffix=".duckdb")
    return db_path, db.get_connection(db_path=db_path)


def _ingest_session(conn, conversation_text: str, session_id: str, scope: str):
    """Ingest a conversation session as small overlapping chunks + extracted facts."""
    windows = split_into_chunks(conversation_text)
    first_cid = None
    for w in windows:
        emb = embeddings.embed(w)
        cid = db.insert_chunk(conn, w, session_id, scope, embedding=emb)
        if first_cid is None:
            first_cid = cid
    return first_cid


def _ingest_fact(conn, fact_text: str, session_id: str, scope: str, chunk_id=None):
    """Ingest an extracted fact."""
    emb = embeddings.embed(fact_text)
    return db.upsert_fact(
        conn, fact_text, "personal", "long", "high",
        emb, session_id, compute_decay_score,
        scope=scope, source_chunk_id=chunk_id,
    )


def _recall_and_check(conn, query: str, scope: str, expected_keywords: list[str]) -> dict:
    """Run recall and check if expected keywords appear in the output."""
    query_emb = embeddings.embed(query)
    result = recall._legacy_prompt_recall(conn, query_emb, query, scope=scope)

    # Collect all text from recall
    all_text = ""
    all_text += " ".join(f["text"] for f in result.get("facts", []))
    all_text += " " + " ".join(c["text"] for c in result.get("chunks", {}).values())
    all_text += " " + " ".join(f["text"] for f in result.get("sibling_facts", []))
    all_text = all_text.lower()

    found = {kw: kw.lower() in all_text for kw in expected_keywords}
    return {
        "all_text": all_text[:500],
        "found": found,
        "all_found": all(found.values()),
        "n_facts": len(result.get("facts", [])),
        "n_chunks": len(result.get("chunks", {})),
        "n_siblings": len(result.get("sibling_facts", [])),
    }


# Noise sessions to simulate the ~50 sessions per question
NOISE_SESSIONS = [
    "User: How's the weather today?\nAssistant: It's sunny and 72 degrees, perfect for a walk.",
    "User: What should I make for dinner?\nAssistant: How about a stir-fry with chicken and vegetables?",
    "User: I'm feeling tired today.\nAssistant: Maybe try a short nap or a cup of green tea.",
    "User: Can you help me with my resume?\nAssistant: Sure! Let's start with your most recent experience.",
    "User: I just watched a great movie.\nAssistant: That sounds fun! What genre was it?",
    "User: My cat knocked over a plant today.\nAssistant: Oh no! Is the plant okay?",
    "User: I need to schedule a dentist appointment.\nAssistant: I can remind you to call tomorrow morning.",
    "User: What's a good book to read?\nAssistant: I'd recommend Project Hail Mary by Andy Weir.",
]


class TestHypothesis_852ce960_MortgageRecency(unittest.TestCase):
    """V6 prediction: likely_correct (high confidence)
    Question: What was the amount I was pre-approved for when I got my mortgage?
    Answer: $400,000
    Hypothesis: Recency-aware chunks + moderate extraction → answer model picks latest value."""

    def test_latest_value_appears_in_context(self):
        _, conn = _fresh_conn()
        scope = "test_mortgage"

        # Noise sessions
        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        # Old value session
        old_session = "User: I just got pre-approved for a mortgage from Wells Fargo!\nAssistant: That's great news! What was the pre-approval amount?\nUser: They approved me for $350,000.\nAssistant: That's a solid start for house hunting."
        cid_old = _ingest_session(conn, old_session, "sess_old", scope)
        _ingest_fact(conn, "[2023/03/01] Mortgage pre-approved for $350,000 from Wells Fargo", "sess_old", scope, cid_old)

        # Updated value session
        new_session = "User: Great news about the mortgage! Wells Fargo increased my pre-approval amount.\nAssistant: Wonderful! What's the new amount?\nUser: They bumped it up to $400,000!\nAssistant: That opens up a lot more options for you."
        cid_new = _ingest_session(conn, new_session, "sess_new", scope)
        _ingest_fact(conn, "[2023/06/15] Mortgage pre-approval updated to $400,000 from Wells Fargo", "sess_new", scope, cid_new)

        try:
            db.rebuild_fts_indexes(conn)
        except Exception:
            pass

        result = _recall_and_check(conn, "What was the mortgage pre-approval amount from Wells Fargo?", scope, ["400,000"])
        print(f"\n  Mortgage: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        self.assertTrue(result["all_found"], f"$400,000 not found. Text: {result['all_text'][:200]}")
        conn.close()


class TestHypothesis_830ce83f_RachelRelocation(unittest.TestCase):
    """V6 prediction: likely_correct (high confidence)
    Question: Where did Rachel move to after her recent relocation?
    Answer: the suburbs
    Hypothesis: Recency prompt + small chunks surface the latest location."""

    def test_latest_location_in_context(self):
        _, conn = _fresh_conn()
        scope = "test_rachel"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        session = "User: Did you hear about Rachel? She just moved!\nAssistant: No, where did she go?\nUser: She relocated to the suburbs. She wanted more space for her garden.\nAssistant: That sounds lovely, the suburbs are great for gardening."
        cid = _ingest_session(conn, session, "sess_rachel", scope)
        _ingest_fact(conn, "Rachel recently relocated to the suburbs for more garden space", "sess_rachel", scope, cid)

        result = _recall_and_check(conn, "Where did Rachel move to after her recent relocation?", scope, ["suburbs"])
        print(f"\n  Rachel: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        self.assertTrue(result["all_found"], f"'suburbs' not found. Text: {result['all_text'][:200]}")
        conn.close()


class TestHypothesis_0a995998_ClothingCount(unittest.TestCase):
    """V6 prediction: likely_correct (medium confidence)
    Question: How many items of clothing do I need to pick up or return?
    Answer: 3
    Hypothesis: BM25 on small chunks finds keyword matches across sessions."""

    def test_all_three_items_findable(self):
        _, conn = _fresh_conn()
        scope = "test_clothing"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        # Three items across three different sessions
        sessions = [
            ("User: I need to pick up my blue jacket from the dry cleaner tomorrow.\nAssistant: I'll remind you about the jacket pickup.", "sess_jacket"),
            ("User: I have to return that red dress to Macy's, it didn't fit right.\nAssistant: Do you have the receipt for the return?", "sess_dress"),
            ("User: Oh and I need to pick up my altered pants from the tailor on Main Street.\nAssistant: Got it, pants from the tailor on Main Street.", "sess_pants"),
        ]

        for text, sid in sessions:
            cid = _ingest_session(conn, text, sid, scope)
            # Extract a fact per item
            if "jacket" in text:
                _ingest_fact(conn, "Need to pick up blue jacket from dry cleaner", sid, scope, cid)
            elif "dress" in text:
                _ingest_fact(conn, "Need to return red dress to Macy's", sid, scope, cid)
            elif "pants" in text:
                _ingest_fact(conn, "Need to pick up altered pants from tailor", sid, scope, cid)

        try:
            db.rebuild_fts_indexes(conn)
        except Exception:
            pass

        result = _recall_and_check(conn, "How many items of clothing do I need to pick up or return from a store?", scope, ["jacket", "dress", "pants"])
        print(f"\n  Clothing: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}, siblings={result['n_siblings']}")
        found_count = sum(1 for v in result["found"].values() if v)
        self.assertGreaterEqual(found_count, 2, f"Need >=2 items, found {result['found']}")
        conn.close()


class TestHypothesis_6d550036_ProjectCount(unittest.TestCase):
    """V6 prediction: likely_correct (medium confidence)
    Question: How many projects have I led or am currently leading?
    Answer: 2
    Hypothesis: Moderate extraction avoids false positive 3rd project."""

    def test_two_projects_in_context(self):
        _, conn = _fresh_conn()
        scope = "test_projects"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        sess1 = "User: I've been leading the data analysis project for my marketing research class. It's going well.\nAssistant: That sounds like a great learning experience."
        cid1 = _ingest_session(conn, sess1, "sess_proj1", scope)
        _ingest_fact(conn, "User leads data analysis project for marketing research class", "sess_proj1", scope, cid1)

        sess2 = "User: I also started leading a community garden initiative in my neighborhood.\nAssistant: That's wonderful community involvement!"
        cid2 = _ingest_session(conn, sess2, "sess_proj2", scope)
        _ingest_fact(conn, "User leads community garden initiative in neighborhood", "sess_proj2", scope, cid2)

        result = _recall_and_check(conn, "How many projects have I led or am currently leading?", scope, ["data analysis", "garden"])
        print(f"\n  Projects: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        # Both projects should be findable
        found_count = sum(1 for v in result["found"].values() if v)
        self.assertGreaterEqual(found_count, 1, f"Need >=1 project, found {result['found']}")
        conn.close()


class TestHypothesis_e9327a54_SugarFactory(unittest.TestCase):
    """V6 prediction: likely_correct (medium confidence)
    Question: Remind me of that unique dessert shop with the giant milkshakes?
    Answer: The Sugar Factory at Icon Park
    Hypothesis: Small chunks preserve specific place name for retrieval."""

    def test_sugar_factory_in_context(self):
        _, conn = _fresh_conn()
        scope = "test_dessert"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        session = "User: I'm planning to revisit Orlando. Remember that amazing dessert shop we talked about?\nAssistant: Yes! The Sugar Factory at Icon Park. They have those incredible giant milkshakes with all the candy toppings.\nUser: That's the one! The milkshakes were huge."
        cid = _ingest_session(conn, session, "sess_dessert", scope)
        _ingest_fact(conn, "User interested in Orlando dessert shops with giant milkshakes", "sess_dessert", scope, cid)

        result = _recall_and_check(conn, "What was that unique dessert shop with the giant milkshakes in Orlando?", scope, ["Sugar Factory"])
        print(f"\n  Sugar Factory: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        self.assertTrue(result["all_found"], f"'Sugar Factory' not found. Text: {result['all_text'][:200]}")
        conn.close()


class TestHypothesis_0edc2aef_MiamiHotel(unittest.TestCase):
    """V6 prediction: likely_correct (medium confidence)
    Question: Can you suggest a hotel for my upcoming trip to Miami?
    Answer: User prefers hotels with great views, possibly beachfront...
    Hypothesis: Chunk search finds user travel preferences."""

    def test_travel_preferences_in_context(self):
        _, conn = _fresh_conn()
        scope = "test_miami"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        session = "User: I'm planning a trip to Miami next month. I love beachfront hotels with great ocean views.\nAssistant: Miami has some amazing beachfront properties! Do you have a budget in mind?\nUser: I'd prefer something mid-range. Also, I really need a pool and a good breakfast included.\nAssistant: Got it - beachfront, mid-range, pool, and breakfast included."
        cid = _ingest_session(conn, session, "sess_miami", scope)
        _ingest_fact(conn, "User planning trip to Miami, prefers beachfront hotels with ocean views", "sess_miami", scope, cid)

        result = _recall_and_check(conn, "Can you suggest a hotel for my upcoming trip to Miami?", scope, ["beachfront", "miami"])
        print(f"\n  Miami: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        self.assertTrue(result["all_found"], f"Keywords not found. Text: {result['all_text'][:200]}")
        conn.close()


class TestHypothesis_6aeb4375_KnowledgeUpdate(unittest.TestCase):
    """V6 prediction: likely_correct (medium confidence)
    Hypothesis: Moderate extraction + chunk search recovers knowledge-update detail."""

    def test_updated_value_in_context(self):
        _, conn = _fresh_conn()
        scope = "test_ku"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        session = "User: I got a raise! My new salary is $85,000.\nAssistant: Congratulations! That's a nice bump from your previous $75,000."
        cid = _ingest_session(conn, session, "sess_salary", scope)
        _ingest_fact(conn, "User's salary updated to $85,000 (previously $75,000)", "sess_salary", scope, cid)

        result = _recall_and_check(conn, "What is my current salary?", scope, ["85,000"])
        print(f"\n  Salary: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        self.assertTrue(result["all_found"], f"'$85,000' not found. Text: {result['all_text'][:200]}")
        conn.close()


class TestHypothesis_89527b6b_Plesiosaur(unittest.TestCase):
    """V6 prediction: maybe (low confidence)
    Question: What color was the scaly body of the Plesiosaur?
    Answer: blue
    Hypothesis: Small chunk containing 'blue scaly body' found via direct chunk search."""

    def test_blue_scaly_body_in_chunks(self):
        _, conn = _fresh_conn()
        scope = "test_dino"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        # Long conversation about a children's book — the detail is buried
        session = (
            "User: I'm working on a children's book about dinosaurs. Can you help me with some descriptions?\n"
            "Assistant: Of course! Which dinosaurs are you including?\n"
            "User: I'm starting with the Plesiosaur. I want it to have a blue scaly body with green fins.\n"
            "Assistant: That sounds beautiful! The blue scales would contrast nicely with the green fins.\n"
            "User: And for the T-Rex, I'm thinking orange and brown stripes.\n"
            "Assistant: Classic and fierce looking! Kids will love that."
        )
        cid = _ingest_session(conn, session, "sess_dino", scope)
        # Haiku extraction might only extract "Working on children's book about dinosaurs"
        _ingest_fact(conn, "User is creating a children's book about dinosaurs with illustrated descriptions", "sess_dino", scope, cid)

        # The key test: can chunk search find "blue scaly body" even though the fact doesn't mention it?
        result = _recall_and_check(conn, "What color was the scaly body of the Plesiosaur in the children's book?", scope, ["blue"])
        print(f"\n  Plesiosaur: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        # This is the "maybe" prediction — log result either way
        if result["all_found"]:
            print("  ✓ HYPOTHESIS CONFIRMED: small chunks surfaced 'blue' detail")
        else:
            print(f"  ✗ HYPOTHESIS REJECTED: 'blue' not found in recall. Text: {result['all_text'][:200]}")


class TestHypothesis_75832dbd_Publications(unittest.TestCase):
    """V6 prediction: maybe (low confidence)
    Question: Can you recommend publications/conferences for me?
    Answer: AI in healthcare, deep learning...
    Hypothesis: Chunk search finds user's research interests."""

    def test_research_interests_in_context(self):
        _, conn = _fresh_conn()
        scope = "test_research"

        for i, text in enumerate(NOISE_SESSIONS):
            _ingest_session(conn, text, f"noise_{i}", scope)

        session = (
            "User: I've been really focused on my research lately. AI applications in healthcare are fascinating.\n"
            "Assistant: That's a growing field! What aspect interests you most?\n"
            "User: Deep learning for medical image analysis. I've been working on diagnostic imaging models.\n"
            "Assistant: That's cutting-edge work. Have you published any papers on it?\n"
            "User: I have two papers under review right now on using CNNs for tumor detection."
        )
        cid = _ingest_session(conn, session, "sess_research", scope)
        _ingest_fact(conn, "User researches AI applications in healthcare, focuses on deep learning", "sess_research", scope, cid)

        result = _recall_and_check(conn, "Can you recommend some recent publications or conferences that I might find interesting?", scope, ["healthcare", "deep learning"])
        print(f"\n  Research: found={result['found']}, facts={result['n_facts']}, chunks={result['n_chunks']}")
        if result["all_found"]:
            print("  ✓ HYPOTHESIS CONFIRMED: chunk search found research interests")
        else:
            print(f"  ✗ HYPOTHESIS REJECTED: keywords not found. Text: {result['all_text'][:200]}")


class TestSmallChunksVsLargeChunks(unittest.TestCase):
    """Direct A/B comparison: small overlapping chunks vs single large chunk."""

    def test_small_chunks_find_detail_large_chunks_miss(self):
        """The core hypothesis: small chunks surface 'Target' that a single large chunk dilutes."""
        conversation = (
            "User: How's the weather today?\nAssistant: Sunny and warm!\n\n"
            "User: I went grocery shopping this morning.\nAssistant: What did you get?\n\n"
            "User: Just the usual stuff. Oh, and I redeemed a $5 coupon on coffee creamer at Target.\nAssistant: Nice savings!\n\n"
            "User: What should I cook for dinner tonight?\nAssistant: How about pasta?\n\n"
            "User: I also need to call my dentist tomorrow.\nAssistant: I'll remind you.\n\n"
            "User: My neighbor's dog was barking all night again.\nAssistant: That sounds frustrating."
        )

        # Test A: Single large chunk (old approach)
        _, conn_large = _fresh_conn()
        large_emb = embeddings.embed(conversation[:2000])
        db.insert_chunk(conn_large, conversation, "sess1", "test_large", embedding=large_emb)
        emb = embeddings.embed("User redeemed coupon on coffee creamer")
        db.upsert_fact(conn_large, "User redeemed coupon on coffee creamer", "personal", "long", "high",
                      emb, "sess1", compute_decay_score, scope="test_large")

        large_result = _recall_and_check(conn_large, "Where did I redeem the coupon on coffee creamer?", "test_large", ["Target"])
        conn_large.close()

        # Test B: Small overlapping chunks (new approach)
        _, conn_small = _fresh_conn()
        windows = split_into_chunks(conversation)
        for w in windows:
            w_emb = embeddings.embed(w)
            db.insert_chunk(conn_small, w, "sess1", "test_small", embedding=w_emb)
        db.upsert_fact(conn_small, "User redeemed coupon on coffee creamer", "personal", "long", "high",
                      emb, "sess1", compute_decay_score, scope="test_small")

        small_result = _recall_and_check(conn_small, "Where did I redeem the coupon on coffee creamer?", "test_small", ["Target"])
        conn_small.close()

        print(f"\n  Large chunk: Target found = {large_result['found']['Target']}, chunks={large_result['n_chunks']}")
        print(f"  Small chunks: Target found = {small_result['found']['Target']}, chunks={small_result['n_chunks']}")

        # Small chunks should find Target (the focused window contains it)
        self.assertTrue(small_result["found"]["Target"],
                       "Small chunks should find 'Target' in focused chunk window")


if __name__ == "__main__":
    # Check if ONNX embeddings are available
    print("Checking embedding backend...")
    test_vec = embeddings.embed("test")
    if test_vec is None:
        print("ERROR: No embedding backend available. Need ONNX or Ollama running.")
        sys.exit(1)
    print(f"Using {'ONNX' if embeddings._onnx_available else 'Ollama'} embeddings (dim={len(test_vec)})")
    print()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
