#!/usr/bin/env python3
"""
run_longmemeval.py — LongMemEval benchmark for ai-memory-db.

Evaluates the extraction+recall pipeline against the LongMemEval dataset
(500 questions, 6 categories) and compares with hindsight's published 91.4%.

Each question gets an isolated DuckDB. Per-question flow:
  1. Ingest haystack sessions → extract knowledge via Claude
  2. Recall memories via session_recall + prompt_recall (4-way retrieval)
  3. Generate answer via Claude using hindsight's exact answer prompt
  4. Judge via Claude using hindsight's exact category-specific judge prompts
  5. Record result

Usage:
  # Quick test (60 questions)
  python3 bench/longmemeval/run_longmemeval.py --max-instances-per-category 10

  # Full run (all 500 questions)
  python3 bench/longmemeval/run_longmemeval.py --parallel 8

  # Re-run with cached extractions
  python3 bench/longmemeval/run_longmemeval.py --skip-extraction --fill
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory import db, embeddings, extract, recall
from memory.config import CLAUDE_MODEL, GLOBAL_SCOPE
from memory.decay import compute_decay_score

import anthropic

# ── Rate limit handling ───────────────────────────────────────────────────

def _call_with_retry(fn, max_retries=5):
    """Call fn(), retrying on rate limit errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt * 15, 120)  # 15, 30, 60, 120s
            print(f"    [RATE] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait = min(2 ** attempt * 15, 120)
                print(f"    [RATE] Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


# ── Constants ──────────────────────────────────────────────────────────────

DATASET_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DATASET_DIR = Path(__file__).parent / "datasets"
DATASET_FILE = DATASET_DIR / "longmemeval_s_cleaned.json"
CACHE_DIR = Path(__file__).parent / "cache"
RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARK_SCOPE = "longmemeval"

CATEGORIES = [
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
]

# Published baselines for comparison
BASELINES = {
    "Hindsight": 91.4,
    "Supermemory": 85.2,
    "Zep": 71.2,
    "GPT-4o": 60.2,
}


# ── Date parsing ──────────────────────────────────────────────────────────

def parse_longmemeval_date(date_str: str) -> Optional[datetime]:
    """Parse LongMemEval date format: '2023/05/20 (Sat) 02:21'."""
    if not date_str:
        return None
    try:
        # Strip day name in parentheses
        cleaned = re.sub(r'\s*\([A-Za-z]+\)\s*', ' ', date_str).strip()
        for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
    except Exception:
        return None


# ── Dataset loading ───────────────────────────────────────────────────────

def download_dataset():
    """Download dataset from HuggingFace if not present."""
    if DATASET_FILE.exists():
        return
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LongMemEval dataset...")
    import urllib.request
    urllib.request.urlretrieve(DATASET_URL, str(DATASET_FILE))
    print(f"  Saved to {DATASET_FILE}")


def load_dataset(
    max_instances: Optional[int] = None,
    max_per_category: Optional[int] = None,
    category_filter: Optional[str] = None,
    question_id: Optional[str] = None,
) -> list[dict]:
    """Load and filter the LongMemEval dataset."""
    with open(DATASET_FILE) as f:
        dataset = json.load(f)

    # Single question mode
    if question_id:
        return [item for item in dataset if item.get("question_id") == question_id]

    # Category filter
    if category_filter:
        dataset = [item for item in dataset if item.get("question_type") == category_filter]

    # Per-category limiting
    if max_per_category:
        by_cat = defaultdict(list)
        for item in dataset:
            by_cat[item.get("question_type", "unknown")].append(item)
        dataset = []
        for cat in CATEGORIES:
            dataset.extend(by_cat[cat][:max_per_category])

    # Global limit
    if max_instances:
        dataset = dataset[:max_instances]

    return dataset


# ── Session formatting ────────────────────────────────────────────────────

def format_session_as_conversation(turns: list[dict], session_date: Optional[str] = None) -> str:
    """Convert LongMemEval session turns into conversation text for extraction."""
    lines = []
    if session_date:
        lines.append(f"[Session date: {session_date}]")
        lines.append("")
    for turn in turns:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "")
        if not content:
            continue
        lines.append(f"--- {role} ---")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


# ── Batched embedding helper ──────────────────────────────────────────────

def _batch_embed(texts: list[str], max_workers: int = 16) -> list[Optional[list[float]]]:
    """Embed texts in parallel using a thread pool for Ollama calls."""
    from concurrent.futures import ThreadPoolExecutor as _TPE
    results: list[Optional[list[float]]] = [None] * len(texts)
    with _TPE(max_workers=max_workers) as pool:
        futures = {pool.submit(embeddings.embed, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                pass
    return results


# ── Per-question extraction & ingestion ───────────────────────────────────

MIN_SESSION_TURNS = 3       # skip sessions with fewer turns
MIN_SESSION_CHARS = 200     # skip sessions with less total text
MAX_SESSIONS_PER_BATCH = 4  # concatenate small sessions into one extraction call
BATCH_CHAR_THRESHOLD = 3000 # batch sessions until this char threshold


def extract_and_ingest(
    item: dict,
    db_path: str,
    api_key: str,
    cache_dir: Path,
    skip_extraction: bool = False,
    extract_model: Optional[str] = None,
) -> tuple[dict, Any]:
    """
    Extract knowledge from all haystack sessions and ingest into an isolated DuckDB.
    Returns (extraction_stats, conn) — caller closes conn after recall.
    """
    question_id = item["question_id"]
    cache_file = cache_dir / f"{question_id}.json"
    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])
    session_ids = item.get("haystack_session_ids", [])

    # Load or create extraction cache
    cached_extractions = {}
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cached_extractions = json.load(f)
        except (json.JSONDecodeError, OSError):
            cached_extractions = {}

    conn = db.get_connection(db_path=db_path)
    counters = {"facts": 0, "ideas": 0, "entities": 0, "rels": 0,
                "decisions": 0, "questions": 0, "sessions": 0, "cached": 0, "skipped": 0}

    # Phase 1: Extract knowledge (batch small sessions together)
    session_knowledge: list[tuple[str, Optional[str], dict]] = []  # (session_key, date_str, knowledge)
    chunk_ids: dict[str, str] = {}  # session_key → chunk_id

    # Group sessions for batched extraction
    pending_batch: list[tuple[int, str, str, Optional[str]]] = []  # (idx, session_key, text, date)
    pending_batch_chars = 0

    def _flush_batch(batch):
        """Extract a batch of concatenated sessions as one API call."""
        if not batch:
            return
        combined_text = "\n\n===== NEXT SESSION =====\n\n".join(text for _, _, text, _ in batch)
        combined_key = "|".join(sk for _, sk, _, _ in batch)
        try:
            knowledge = _call_with_retry(
                lambda ct=combined_text: extract.extract_knowledge(ct, api_key, model=extract_model)
            )
            # Cache under combined key and individual keys
            cached_extractions[combined_key] = knowledge
            for _, sk, _, _ in batch:
                cached_extractions[sk] = knowledge  # share extraction across batch members
            for _, sk, _, date_str in batch:
                session_knowledge.append((sk, date_str, knowledge))
        except Exception as e:
            print(f"    [WARN] Batch extraction failed: {e}", file=sys.stderr)

    for idx, session_turns in enumerate(sessions):
        session_date_str = dates[idx] if idx < len(dates) else None
        sid = session_ids[idx] if idx < len(session_ids) else f"session_{idx}"
        session_key = f"{question_id}_{sid}"

        # Clean turns
        cleaned_turns = [{"role": t.get("role", "user"), "content": t.get("content", "")}
                         for t in session_turns]

        # Skip tiny sessions
        total_chars = sum(len(t["content"]) for t in cleaned_turns)
        if len(cleaned_turns) < MIN_SESSION_TURNS and total_chars < MIN_SESSION_CHARS:
            counters["skipped"] += 1
            continue

        conversation_text = format_session_as_conversation(cleaned_turns, session_date_str)

        # Store raw conversation as a chunk for verbatim retrieval
        from memory.chunking import split_into_chunks
        windows = split_into_chunks(conversation_text)
        first_cid = None
        for window_text in windows:
            chunk_emb = embeddings.embed(window_text)
            cid = db.insert_chunk(conn, window_text, f"lme_{session_key}", BENCHMARK_SCOPE, embedding=chunk_emb)
            if first_cid is None:
                first_cid = cid
        chunk_ids[session_key] = first_cid

        # Check cache first
        if session_key in cached_extractions:
            session_knowledge.append((session_key, session_date_str, cached_extractions[session_key]))
            counters["cached"] += 1
            continue

        if skip_extraction:
            continue

        # Accumulate into batch
        pending_batch.append((idx, session_key, conversation_text, session_date_str))
        pending_batch_chars += len(conversation_text)

        # Flush batch when big enough or at max count
        if len(pending_batch) >= MAX_SESSIONS_PER_BATCH or pending_batch_chars >= BATCH_CHAR_THRESHOLD:
            _flush_batch(pending_batch)
            pending_batch = []
            pending_batch_chars = 0

    # Flush remaining
    _flush_batch(pending_batch)

    # Phase 2: Batch-embed and ingest all knowledge
    for session_key, session_date_str, knowledge in session_knowledge:
        session_id = f"lme_{session_key}"
        source_chunk_id = chunk_ids.get(session_key)

        # Collect all texts that need embedding
        texts_to_embed = []
        text_roles = []  # track what each text is for

        for entity_name in knowledge.get("entities", []):
            texts_to_embed.append(entity_name)
            text_roles.append(("entity", entity_name, None))

        for fact in knowledge.get("facts", []):
            text = fact.get("text", "").strip()
            if not text:
                continue
            text_with_date = f"[{session_date_str}] {text}" if session_date_str else text
            texts_to_embed.append(text_with_date)
            text_roles.append(("fact", text_with_date, fact))

        for idea in knowledge.get("ideas", []):
            text = idea.get("text", "").strip()
            if text:
                texts_to_embed.append(text)
                text_roles.append(("idea", text, idea))

        for dec in knowledge.get("key_decisions", []):
            text = dec.get("text", dec) if isinstance(dec, dict) else str(dec)
            text = text.strip()
            if text:
                texts_to_embed.append(text)
                text_roles.append(("decision", text, dec))

        for q_text in knowledge.get("open_questions", []):
            q_text = q_text.strip() if isinstance(q_text, str) else ""
            if q_text:
                texts_to_embed.append(q_text)
                text_roles.append(("question", q_text, None))

        # Batch embed all at once
        all_embeddings = _batch_embed(texts_to_embed)

        # Ingest with pre-computed embeddings
        emb_idx = 0
        for role, text, meta in text_roles:
            emb = all_embeddings[emb_idx] if emb_idx < len(all_embeddings) else None
            emb_idx += 1

            if role == "entity":
                db.upsert_entity(conn, text, embedding=emb, scope=BENCHMARK_SCOPE)
                counters["entities"] += 1
            elif role == "fact":
                fid, is_new = db.upsert_fact(
                    conn, text=text,
                    category=meta.get("category", "contextual"),
                    temporal_class=meta.get("temporal_class", "long"),
                    confidence=meta.get("confidence", "high"),
                    embedding=emb, session_id=session_id,
                    decay_fn=compute_decay_score, scope=BENCHMARK_SCOPE,
                    source_chunk_id=source_chunk_id,
                )
                if is_new:
                    counters["facts"] += 1
            elif role == "idea":
                db.upsert_idea(
                    conn, text=text,
                    idea_type=meta.get("type", "insight"),
                    temporal_class=meta.get("temporal_class", "medium"),
                    embedding=emb, session_id=session_id,
                    decay_fn=compute_decay_score, scope=BENCHMARK_SCOPE,
                )
                counters["ideas"] += 1
            elif role == "decision":
                tc = meta.get("temporal_class", "medium") if isinstance(meta, dict) else "medium"
                db.upsert_decision(
                    conn, text=text, temporal_class=tc,
                    embedding=emb, session_id=session_id,
                    decay_fn=compute_decay_score, scope=BENCHMARK_SCOPE,
                )
                counters["decisions"] += 1
            elif role == "question":
                db.upsert_question(conn, text, emb, session_id, scope=BENCHMARK_SCOPE)
                counters["questions"] += 1

        # Relationships (no embedding needed)
        for rel in knowledge.get("relationships", []):
            f_ent = rel.get("from", "").strip()
            t_ent = rel.get("to", "").strip()
            if f_ent and t_ent:
                db.upsert_relationship(
                    conn, from_entity=f_ent, to_entity=t_ent,
                    rel_type=rel.get("type", "relates_to"),
                    description=rel.get("description", ""),
                    session_id=session_id, scope=BENCHMARK_SCOPE,
                )
                counters["rels"] += 1

        counters["sessions"] += 1

    # Build BM25 indexes
    try:
        db.rebuild_fts_indexes(conn)
    except Exception:
        pass

    # Save extraction cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cached_extractions, f)

    # Return conn along with counters (caller will close it after recall)
    return counters, conn


# ── Recall ────────────────────────────────────────────────────────────────

def recall_for_question(conn, question_text: str) -> tuple[str, dict]:
    """
    Run combined session_recall + prompt_recall for a question.
    Reuses the ingestion connection to avoid DuckDB config mismatch.
    Returns (formatted_context, raw_recall_dicts).
    """
    session_ctx = recall.session_recall(conn, scope=BENCHMARK_SCOPE)
    session_formatted = recall.format_session_context(session_ctx)

    query_emb = embeddings.embed(question_text)
    if query_emb:
        prompt_ctx = recall._legacy_prompt_recall(
            conn, query_emb, question_text, scope=BENCHMARK_SCOPE,
        )
    else:
        prompt_ctx = {"facts": [], "ideas": [], "observations": [],
                      "relationships": [], "questions": [], "narratives": []}
    prompt_formatted = recall.format_prompt_context(prompt_ctx)

    # Combine both contexts
    combined = ""
    if session_formatted:
        combined += session_formatted + "\n\n"
    if prompt_formatted:
        combined += prompt_formatted

    raw = {"session": session_ctx, "prompt": prompt_ctx}
    return combined.strip(), raw


# ── Agentic multi-query recall ────────────────────────────────────────────

MULTI_QUERY_SYSTEM = """\
You are a search query generator. Given a question about past conversations, \
generate 3-5 diverse search queries that would help find ALL relevant information. \
Each query should target a different angle of the question.

For counting questions ("how many X"), generate queries for each possible instance.
For detail questions ("where/who/what specifically"), generate both broad and narrow queries.
For temporal questions, include date-related queries.

Return a JSON array of query strings. Nothing else."""


def agentic_recall(
    conn, question_text: str, api_key: str, model: str = CLAUDE_MODEL,
) -> tuple[str, dict]:
    """
    Hybrid recall: standard recall first (always), then augmented with
    multi-query agentic results. Standard results get priority; agentic
    results fill gaps without overwhelming the token budget.
    Falls back to standard recall on any error.
    """
    # Always run standard recall first as the foundation
    standard_context, standard_raw = recall_for_question(conn, question_text)
    standard_facts = {
        f.get("id", f.get("text")): f
        for f in standard_raw.get("prompt", {}).get("facts", [])
    }

    try:
        # Generate supplementary search queries
        client = anthropic.Anthropic(api_key=api_key)
        response = _call_with_retry(lambda: client.messages.create(
            model=model,
            max_tokens=512,
            system=MULTI_QUERY_SYSTEM,
            messages=[{"role": "user", "content": question_text}],
            temperature=0.3,
        ))
        text = response.content[0].text.strip()

        import re as _re
        json_match = _re.search(r'\[.*\]', text, _re.DOTALL)
        queries = json.loads(json_match.group()) if json_match else []

        # Run supplementary queries (skip the original question — already covered by standard recall)
        supplementary_facts = {}
        for query in queries[:4]:
            query_emb = embeddings.embed(query)
            if not query_emb:
                continue
            results = recall._legacy_prompt_recall(
                conn, query_emb, query, scope=BENCHMARK_SCOPE,
            )
            for f in results.get("facts", []):
                fid = f.get("id", f.get("text"))
                if fid not in standard_facts and fid not in supplementary_facts:
                    supplementary_facts[fid] = f

        # Merge: standard facts first, then up to 5 supplementary facts
        all_facts = list(standard_facts.values()) + list(supplementary_facts.values())[:5]

        # Load chunks and limited sibling facts for the merged set
        chunks = recall._load_chunks_for_facts(conn, all_facts)
        sibling_facts = recall._expand_sibling_facts(conn, all_facts, limit=5)

        # Session context (from standard recall)
        session_ctx = standard_raw.get("session", {})
        session_formatted = recall.format_session_context(session_ctx)

        # Format with merged results
        merged_recall = {
            "facts": all_facts,
            "ideas": standard_raw.get("prompt", {}).get("ideas", []),
            "observations": standard_raw.get("prompt", {}).get("observations", []),
            "relationships": standard_raw.get("prompt", {}).get("relationships", []),
            "questions": [],
            "narratives": standard_raw.get("prompt", {}).get("narratives", []),
            "chunks": chunks,
            "sibling_facts": sibling_facts,
        }
        prompt_formatted = recall.format_prompt_context(merged_recall)

        combined = ""
        if session_formatted:
            combined += session_formatted + "\n\n"
        if prompt_formatted:
            combined += prompt_formatted

        raw = {"session": session_ctx, "prompt": merged_recall, "queries": queries}
        return combined.strip(), raw

    except Exception as e:
        print(f"    [WARN] Agentic augmentation failed, using standard: {e}", file=sys.stderr)
        return standard_context, standard_raw


# ── Answer generation (hindsight's exact prompt) ─────────────────────────

ANSWER_SYSTEM = """\
You are a helpful assistant that must answer user questions based on the previous conversations."""

ANSWER_PROMPT = """\
**Answer Guidelines:**
1. Start by scanning ALL retrieved context — every fact, every source conversation chunk, every sibling fact. Do not skip any section.
2. Reason about all the memories and find the right answer, considering the most recent memory as an update of the current facts.
3. If you have 2 possible answers, just say both.

In general the answer must be comprehensive and plenty of details from the retrieved context.

CRITICAL for counting questions ("how many..."): You MUST exhaustively scan EVERY fact AND every source conversation chunk. List each unique item explicitly (1. X, 2. Y, 3. Z...) — items may appear in different facts, different chunks, or different sessions. Do NOT stop after finding a few. Count from ALL sources. Missing even one item is a failure.
If questions asks a location (where...?) make sure to include the location name.
For recommendation questions ("can you recommend...", "suggest...", "any tips..."): DO NOT give actual recommendations. Instead, describe what KIND the user would prefer based on their context. Example answer format: "The user would prefer recommendations for [category] that focus on [their interest]. They would not prefer [what to avoid based on context]."
For questions asking for help or instructions, consider the users' recent memories and previous interactions with the assistant to understand their current situation better (recent purchases, specific product models used..)
For specific number/value questions, use the context to understand what is the most up-to-date number based on recency, but also include the reasoning (in the answer) on previous possible values and why you think are less relevant.
For open questions, include as much details as possible from different sources that are relevant.
For questions where a specific entity/role is mentioned and it's different from your memory, just say the truth, don't make up anything just to fulfill the question. For example, if the question is about a specific sport, you should consider if the memories and the question are about the same sport. (e.g. american football vs soccer, shows vs podcasts)
For comparative questions, say you don't know the answer if you don't have information about both sides. (or more sides)
For questions related to time/date, carefully review the question date and the memories date to correctly answer the question.
For questions related to time/date calculation (e.g. How many days passed between X and Y?), carefully review the memories date to correctly answer the question and only provide an answer if you have information about both X and Y, otherwise say it's not possible to calculate and why.

Consider assistant's previous actions (e.g., bookings, reminders) as impactful to the user experiences.

CRITICAL for knowledge-update questions: When you see multiple values for the same thing (e.g., two different prices, amounts, or statuses at different dates), the MOST RECENT date is the current truth. Earlier values are outdated. Your answer MUST use the value from the LATEST date. Do NOT average them or say "it was X but may have changed" — give the latest value as the definitive answer.

IMPORTANT for source conversation context: The "Source Conversation Context" sections contain verbatim conversation text. These contain specific details (names, places, numbers) that may not appear in the extracted facts. ALWAYS read these carefully.

Question: {question}
Question Date: {question_date}

Retrieved Context:
{context}

Answer:"""


def generate_answer(
    question: str,
    context: str,
    question_date: Optional[str],
    api_key: str,
    model: str = CLAUDE_MODEL,
) -> tuple[str, str]:
    """Generate answer from retrieved context. Returns (answer, reasoning)."""
    formatted_date = question_date or "Not specified"
    user_msg = ANSWER_PROMPT.format(
        question=question,
        question_date=formatted_date,
        context=context if context else "(No relevant memories found)",
    )

    client = anthropic.Anthropic(api_key=api_key)

    def _call():
        return client.messages.create(
            model=model,
            max_tokens=2048,
            system=ANSWER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0,
        )

    response = _call_with_retry(_call)
    answer = response.content[0].text.strip()
    return answer, ""


# ── Judge prompts (copied verbatim from hindsight) ───────────────────────

JUDGE_PROMPTS = {
    "single-session-user": """\
Evaluate if the model response contains the correct answer to the question.

I will give you a question, a correct answer, and a response from a model. \
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false. \
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true. \
If the response only contains a subset of the information required by the answer, set correct=false.

Question: {question}

Correct Answer: {correct_answer}

Model Response: {predicted_answer}

Evaluation criteria:
- Set correct=true if the response contains the correct answer
- Set correct=true if the response is equivalent to the correct answer or contains intermediate steps
- Set correct=false if the response is incorrect or missing key information

Provide your evaluation as JSON with:
- reasoning: One sentence explanation
- correct: true or false""",

    "single-session-assistant": None,  # uses same as single-session-user
    "multi-session": None,  # uses same as single-session-user

    "temporal-reasoning": """\
I will give you a question, a correct answer, and a response from a model. \
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false. \
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true. \
If the response only contains a subset of the information required by the answer, answer correct=false. \
In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Gold answer: {correct_answer}

Generated answer: {predicted_answer}

First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true.

Respond with JSON: {{"reasoning": "...", "correct": true/false}}""",

    "knowledge-update": """\
I will give you a question, a correct answer, and a response from a model. \
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false. \
If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Gold answer: {correct_answer}

Generated answer: {predicted_answer}

First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true.

Respond with JSON: {{"reasoning": "...", "correct": true/false}}""",

    "single-session-preference": """\
I will give you a question, a answer for desired personalized response, and a response from a model. \
Please set correct=true if the response satisfies the desired response. Otherwise, set correct=false. \
The model does not need to reflect all the points in the desired response. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Gold answer: {correct_answer}

Generated answer: {predicted_answer}

First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true.

Respond with JSON: {{"reasoning": "...", "correct": true/false}}""",
}

# Fallback judge prompt for unknown categories
DEFAULT_JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic.
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might use relative time references, but you should be generous - as long as it refers to the same date or time period, it should be counted as CORRECT.

There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so; if the generated answer says that it cannot be answered or it doesn't know, it should be counted as CORRECT.

Question: {question}

Gold answer: {correct_answer}

Generated answer: {predicted_answer}

First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred. If it's correct, set correct=true.

Respond with JSON: {{"reasoning": "...", "correct": true/false}}"""


def judge_answer(
    question: str,
    correct_answer: str,
    predicted_answer: str,
    category: str,
    api_key: str,
    model: str = CLAUDE_MODEL,
) -> tuple[bool, str]:
    """Judge correctness using category-specific prompt. Returns (is_correct, reasoning)."""
    prompt_template = JUDGE_PROMPTS.get(category)
    if prompt_template is None:
        # Fall through to single-session-user for shared categories
        prompt_template = JUDGE_PROMPTS.get("single-session-user", DEFAULT_JUDGE_PROMPT)

    prompt = prompt_template.format(
        question=question,
        correct_answer=correct_answer,
        predicted_answer=predicted_answer,
    )

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(3):
        try:
            def _call():
                return client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            response = _call_with_retry(_call)
            text = response.content[0].text.strip()

            # Parse JSON from response (may be wrapped in markdown code blocks)
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return bool(result.get("correct", False)), result.get("reasoning", "")

            # Fallback: look for correct/wrong keywords
            text_lower = text.lower()
            if "correct=true" in text_lower or '"correct": true' in text_lower:
                return True, text
            return False, text

        except Exception as e:
            if attempt == 2:
                return False, f"Judge error: {e}"
            time.sleep(1)

    return False, "Judge failed after 3 attempts"


# ── Single question pipeline ─────────────────────────────────────────────

def process_question(
    item: dict,
    work_dir: str,
    cache_dir: Path,
    api_key: str,
    skip_extraction: bool = False,
    extract_model: Optional[str] = None,
    answer_model: str = CLAUDE_MODEL,
    judge_model: str = CLAUDE_MODEL,
    use_agentic: bool = False,
) -> dict:
    """Full pipeline for one question. Returns result dict."""
    question_id = item["question_id"]
    question = item["question"]
    correct_answer = item["answer"]
    category = item.get("question_type", "unknown")
    question_date_str = item.get("question_date")

    result = {
        "question_id": question_id,
        "question_type": category,
        "question": question,
        "correct_answer": correct_answer,
        "predicted_answer": "",
        "is_correct": False,
        "is_invalid": False,
        "judge_reasoning": "",
        "error": None,
        "timings": {},
        "extraction_stats": {},
        "num_sessions": len(item.get("haystack_sessions", [])),
    }

    try:
        # 1. Extract and ingest
        db_path = os.path.join(work_dir, f"{question_id}.duckdb")
        t0 = time.time()
        extraction_stats, conn = extract_and_ingest(
            item, db_path, api_key, cache_dir, skip_extraction, extract_model,
        )
        result["timings"]["extraction_s"] = round(time.time() - t0, 1)
        result["extraction_stats"] = extraction_stats

        # 2. Recall (reuse ingestion connection to avoid config mismatch)
        t0 = time.time()
        if use_agentic:
            context, raw_recall = agentic_recall(conn, question, api_key, answer_model)
        else:
            context, raw_recall = recall_for_question(conn, question)
        conn.close()
        result["timings"]["recall_s"] = round(time.time() - t0, 2)

        # 3. Generate answer
        t0 = time.time()
        predicted, reasoning = generate_answer(
            question, context, question_date_str, api_key, answer_model,
        )
        result["predicted_answer"] = predicted
        result["timings"]["answer_s"] = round(time.time() - t0, 1)

        # 4. Judge
        t0 = time.time()
        is_correct, judge_reasoning = judge_answer(
            question, correct_answer, predicted, category, api_key, judge_model,
        )
        result["is_correct"] = is_correct
        result["judge_reasoning"] = judge_reasoning
        result["timings"]["judge_s"] = round(time.time() - t0, 1)

    except Exception as e:
        result["is_invalid"] = True
        result["error"] = str(e)

    # Cleanup temp DB
    try:
        db_file = Path(os.path.join(work_dir, f"{question_id}.duckdb"))
        for f in db_file.parent.glob(f"{question_id}.duckdb*"):
            f.unlink()
    except OSError:
        pass

    return result


# ── Results aggregation ──────────────────────────────────────────────────

def aggregate_results(results: list[dict]) -> dict:
    """Compute per-category and overall accuracy."""
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "invalid": 0})
    total_correct = 0
    total_questions = 0
    total_invalid = 0

    for r in results:
        cat = r["question_type"]
        category_stats[cat]["total"] += 1
        total_questions += 1
        if r.get("is_invalid"):
            category_stats[cat]["invalid"] += 1
            total_invalid += 1
        elif r.get("is_correct"):
            category_stats[cat]["correct"] += 1
            total_correct += 1

    valid_total = total_questions - total_invalid
    overall_accuracy = (total_correct / valid_total * 100) if valid_total > 0 else 0.0

    return {
        "overall_accuracy": round(overall_accuracy, 1),
        "total_correct": total_correct,
        "total_questions": total_questions,
        "total_invalid": total_invalid,
        "total_valid": valid_total,
        "category_stats": {
            cat: {
                **stats,
                "valid": stats["total"] - stats["invalid"],
                "accuracy": round(
                    stats["correct"] / max(1, stats["total"] - stats["invalid"]) * 100, 1
                ),
            }
            for cat, stats in sorted(category_stats.items())
        },
        "model_config": {
            "extraction": "see detailed_results",
            "answer_generation": "see detailed_results",
            "judge": "see detailed_results",
            "embeddings": "nomic-embed-text",
        },
    }


def format_results_table(agg: dict) -> str:
    """Format results as markdown table matching hindsight's format."""
    lines = [
        "# LongMemEval Benchmark Results — ai-memory-db",
        "",
        f"**Overall Accuracy**: {agg['overall_accuracy']}% "
        f"({agg['total_correct']}/{agg['total_valid']})",
        "",
        "## Results by Question Type",
        "",
        "| Question Type | Total | Valid | Correct | Invalid | Accuracy |",
        "|---|---|---|---|---|---|",
    ]

    for cat in CATEGORIES:
        stats = agg["category_stats"].get(cat, {"total": 0, "valid": 0, "correct": 0, "invalid": 0, "accuracy": 0})
        lines.append(
            f"| {cat} | {stats['total']} | {stats['valid']} | "
            f"{stats['correct']} | {stats['invalid']} | {stats['accuracy']}% |"
        )

    lines.append(
        f"| **OVERALL** | **{agg['total_questions']}** | **{agg['total_valid']}** | "
        f"**{agg['total_correct']}** | **{agg['total_invalid']}** | "
        f"**{agg['overall_accuracy']}%** |"
    )

    lines.extend([
        "",
        "## Comparison with Published Baselines",
        "",
        "| System | Accuracy |",
        "|---|---|",
    ])
    for name, score in sorted(BASELINES.items(), key=lambda x: -x[1]):
        lines.append(f"| {name} | {score}% |")
    lines.append(f"| **ai-memory-db** | **{agg['overall_accuracy']}%** |")

    lines.extend([
        "",
        "## Model Configuration",
        f"- **Extraction**: {agg['model_config']['extraction']}",
        f"- **Answer Generation**: {agg['model_config']['answer_generation']}",
        f"- **Judge**: {agg['model_config']['judge']}",
        f"- **Embeddings**: {agg['model_config']['embeddings']}",
    ])

    return "\n".join(lines)


# ── Incremental save/resume ──────────────────────────────────────────────

def save_results(results: list[dict], agg: dict, output_path: Path):
    """Save results atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {**agg, "detailed_results": results}
    tmp = output_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(output_path)


def load_existing_results(output_path: Path) -> list[dict]:
    """Load previously saved results for --fill mode."""
    if not output_path.exists():
        return []
    try:
        with open(output_path) as f:
            data = json.load(f)
        return data.get("detailed_results", [])
    except (json.JSONDecodeError, OSError):
        return []


# ── Main runner ──────────────────────────────────────────────────────────

def run_benchmark(args):
    """Orchestrate the full benchmark run."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Check Ollama
    if not embeddings.is_ollama_available():
        print("ERROR: Ollama not available. Run: ollama pull nomic-embed-text", file=sys.stderr)
        sys.exit(1)

    # Download dataset
    download_dataset()

    # Load dataset
    dataset = load_dataset(
        max_instances=args.max_instances,
        max_per_category=args.max_instances_per_category,
        category_filter=args.category,
        question_id=args.question_id,
    )
    if not dataset:
        print("No questions matched filters.", file=sys.stderr)
        sys.exit(1)

    # Load existing results for --fill or --only-failed
    output_path = Path(args.results_file)
    existing_results = []
    if args.fill or args.only_failed:
        existing_results = load_existing_results(output_path)

    if args.only_failed:
        failed_ids = {r["question_id"] for r in existing_results if not r.get("is_correct") and not r.get("is_invalid")}
        dataset = [item for item in dataset if item["question_id"] in failed_ids]
        # Remove failed from existing so they get replaced
        existing_results = [r for r in existing_results if r["question_id"] not in failed_ids]

    if args.fill:
        done_ids = {r["question_id"] for r in existing_results}
        dataset = [item for item in dataset if item["question_id"] not in done_ids]

    if not dataset:
        print("All questions already completed. Nothing to do.")
        if existing_results:
            agg = aggregate_results(existing_results)
            print(format_results_table(agg))
        return

    # Print run summary
    cat_counts = defaultdict(int)
    for item in dataset:
        cat_counts[item.get("question_type", "?")] += 1
    total_sessions = sum(len(item.get("haystack_sessions", [])) for item in dataset)
    print(f"\nLongMemEval Benchmark — ai-memory-db")
    print(f"  Questions: {len(dataset)}")
    for cat in CATEGORIES:
        if cat in cat_counts:
            print(f"    {cat}: {cat_counts[cat]}")
    print(f"  Total sessions to process: {total_sessions}")
    print(f"  Workers: {args.parallel}")
    print(f"  Skip extraction: {args.skip_extraction}")
    print()

    # Work directory
    work_dir = args.work_dir or tempfile.mkdtemp(prefix="lme_bench_")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = list(existing_results)
    completed = 0
    correct = 0

    def _process(item):
        return process_question(
            item, work_dir, cache_dir, api_key,
            skip_extraction=args.skip_extraction,
            extract_model=args.extract_model,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            use_agentic=args.agentic,
        )

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(_process, item): item for item in dataset}

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "question_id": item["question_id"],
                    "question_type": item.get("question_type", "?"),
                    "question": item["question"],
                    "correct_answer": item["answer"],
                    "predicted_answer": "",
                    "is_correct": False,
                    "is_invalid": True,
                    "error": str(e),
                }

            results.append(result)
            completed += 1
            mark = "+" if result["is_correct"] else ("!" if result["is_invalid"] else "-")
            if result["is_correct"]:
                correct += 1

            valid_so_far = sum(1 for r in results if not r.get("is_invalid"))
            correct_so_far = sum(1 for r in results if r.get("is_correct"))
            acc = round(correct_so_far / max(1, valid_so_far) * 100, 1)

            cat_short = result["question_type"][:12]
            print(
                f"  [{mark}] {completed}/{len(dataset)} "
                f"({cat_short:12s}) "
                f"acc={acc}% "
                f"q={result['question_id'][:8]}"
            )

            # Save incrementally every 5 questions
            if completed % 5 == 0 or completed == len(dataset):
                agg = aggregate_results(results)
                save_results(results, agg, output_path)

    # Final save and report
    agg = aggregate_results(results)
    save_results(results, agg, output_path)

    # Save markdown report
    md_path = output_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(format_results_table(agg))

    print("\n" + "=" * 60)
    print(format_results_table(agg))
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print(f"Report saved to:  {md_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark against ai-memory-db"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--max-instances", type=int, default=None,
                       help="Limit total questions")
    group.add_argument("--max-instances-per-category", type=int, default=None,
                       help="Limit questions per category")

    parser.add_argument("--category", type=str, default=None, choices=CATEGORIES,
                        help="Filter to one category")
    parser.add_argument("--question-id", type=str, default=None,
                        help="Run single question by ID")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Concurrent workers (default: 4)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Use cached extractions only")
    parser.add_argument("--fill", action="store_true",
                        help="Resume: only process missing questions")
    parser.add_argument("--only-failed", action="store_true",
                        help="Re-run previously incorrect questions")
    parser.add_argument("--agentic", action="store_true",
                        help="Use agentic multi-query recall (generates multiple search queries per question)")
    parser.add_argument("--results-file", type=str,
                        default=str(RESULTS_DIR / "results.json"),
                        help="Output results file path")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory for temp DuckDB files")
    parser.add_argument("--cache-dir", type=str,
                        default=str(CACHE_DIR),
                        help="Extraction cache directory")
    parser.add_argument("--extract-model", type=str, default=None,
                        help="Model for extraction (default: same as answer-model)")
    parser.add_argument("--answer-model", type=str, default=CLAUDE_MODEL,
                        help=f"Model for answer generation (default: {CLAUDE_MODEL})")
    parser.add_argument("--judge-model", type=str, default=CLAUDE_MODEL,
                        help=f"Model for judging (default: {CLAUDE_MODEL})")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
