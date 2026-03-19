# Incremental Chained Extraction with Superseding & Narrative Summaries

## Context

The memory system currently does a single extraction pass per session — triggered at 90% context, pre-compact, or session end (whichever fires first). Long conversations get their middle truncated (120k char limit). This causes three problems:

1. **Multi-turn themes are missed** — the truncated middle often contains key discussions
2. **Stale information persists** — no mechanism to replace outdated facts/decisions when new information contradicts them (within or across sessions)
3. **Token budgets are too tight** — atomic bullet-point facts waste tokens on overhead and lose relational context

Inspired by Cursor's self-summarization approach, we add: incremental multi-pass extraction, a superseding mechanism, and narrative summaries. The system degrades gracefully to current single-pass behavior for short sessions or when state files are missing.

---

## Phase 1: Foundation (no behavioral changes)

### 1A. New file: `memory/extraction_state.py`

Manages per-session extraction state, replacing the one-shot lock for multi-pass sessions.

**State dir:** `~/.claude/memory/extraction_state/`

**State file schema** (`{session_id}.json`):
```json
{
  "session_id": "abc123",
  "pass_count": 2,
  "last_byte_offset": 48210,
  "last_narrative": "User is building a REST API with Express...",
  "prior_item_ids": {
    "facts": ["uuid1", "uuid2"],
    "ideas": ["uuid3"],
    "decisions": ["uuid4"]
  },
  "recalled_item_ids": ["uuid5", "uuid6"],
  "last_pass_at": "2026-03-18T14:22:00Z"
}
```

**Functions:**
- `load_state(session_id) -> Optional[dict]` — read JSON, return None if missing/corrupt
- `save_state(session_id, state)` — atomic write (write `.tmp`, rename)
- `delete_state(session_id)` — remove state file
- `acquire_running_lock(session_id) -> bool` — atomic `O_CREAT|O_EXCL` on `locks/{session_id}.running` (prevents concurrent passes)
- `release_running_lock(session_id)` — remove `.running` file
- `mark_extraction_complete(session_id)` — create old-style `.lock` file for backward compat with hooks that check it
- `is_extraction_complete(session_id) -> bool` — check `.lock` exists
- `cleanup_old_state(max_age_hours=48)` — remove stale state files

### 1B. Config additions: `memory/config.py`

```python
# ── Incremental extraction ──────────────────────────────────────────
EXTRACTION_THRESHOLDS = [40, 70, 90]   # context % triggers
CROSS_PASS_DEDUP_THRESHOLD = 0.85      # tighter dedup between passes and against recalled items
DELTA_MIN_USER_MESSAGES = 3            # skip extraction if delta has fewer user messages
DELTA_MIN_USER_CHARS = 500             # skip extraction if delta has less user text
NARRATIVE_SEARCH_LIMIT = 3             # narratives returned in prompt recall
NARRATIVE_TOKEN_BUDGET = 400           # tokens reserved for narratives in prompt recall
```

### 1C. DB migration 4: `memory/db.py`

Add to `MIGRATIONS` list:

```sql
CREATE TABLE IF NOT EXISTS session_narratives (
    id              VARCHAR PRIMARY KEY,
    session_id      VARCHAR NOT NULL,
    pass_number     INTEGER NOT NULL,
    narrative       TEXT NOT NULL,
    embedding       DOUBLE[],
    is_final        BOOLEAN DEFAULT FALSE,
    scope           VARCHAR DEFAULT '__global__',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS narratives_session ON session_narratives(session_id);
CREATE INDEX IF NOT EXISTS narratives_final ON session_narratives(is_final);

ALTER TABLE facts ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
ALTER TABLE ideas ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
ALTER TABLE open_questions ADD COLUMN IF NOT EXISTS superseded_by VARCHAR;
```

**New DB functions** (in `db.py`):
- `upsert_narrative(conn, session_id, pass_number, narrative, embedding, is_final, scope) -> str`
- `finalize_narratives(conn, session_id)` — set highest-pass narrative to `is_final=TRUE`, delete non-final rows for that session
- `search_narratives(conn, query_embedding, limit, threshold, scope) -> list[dict]` — vector search on final narratives only
- `supersede_item(conn, old_id, old_table, new_id, reason) -> bool` — set `is_active=FALSE`, `deactivated_at=now`, `superseded_by=new_id`
- `get_items_by_ids(conn, item_ids: list[str], tables: list[str]) -> list[dict]` — fetch items by ID for inclusion in extraction prompts (returns id, text, table)

### 1D. Tests for Phase 1

Add to `test_memory.py`:
- `TestExtractionState` (~8 tests): save/load round-trip, missing/corrupt returns None, delete, running lock acquire/release, sanitize session_id
- `TestSchemaMigration4` (~3 tests): narratives table exists, superseded_by column on facts/ideas/decisions, idempotent
- `TestSuperseding` (~6 tests): supersede sets inactive + superseded_by, nonexistent returns False, superseded items excluded from search, works on facts/ideas/decisions
- `TestNarrativeDB` (~5 tests): upsert creates row, finalize keeps only final, search returns only final, respects scope

**Verify:** All 188 existing tests still pass + ~22 new tests pass.

---

## Phase 2: Extraction Changes

### 2A. Delta parsing: `memory/extract.py`

New functions alongside existing ones (existing functions unchanged):

- `parse_transcript_delta(path, byte_offset) -> tuple[list[dict], int]` — seek to `byte_offset`, read remaining JSONL lines, return `(messages, new_byte_offset)`. When `byte_offset=0`, equivalent to `parse_transcript`.
- `is_delta_substantial(messages, min_user_msgs=DELTA_MIN_USER_MESSAGES, min_user_chars=DELTA_MIN_USER_CHARS) -> bool` — count user messages and total user text. Return False if below thresholds. **(Gotcha #5: wasted API calls on tool-heavy deltas)**

### 2B. Incremental extraction tool schema: `memory/extract.py`

New tool `INCREMENTAL_EXTRACTION_TOOL` alongside existing `EXTRACTION_TOOL` (which stays unchanged):

Same fields as `EXTRACTION_TOOL` plus:
- `narrative_summary` (string) — replaces `session_summary`. "Cumulative 3-6 sentence narrative of the ENTIRE conversation so far. Must be self-contained."
- `supersedes` (array of objects) — each object has:
  - `old_id` (string) — UUID of item being replaced
  - `old_table` (string, enum: facts/ideas/decisions/open_questions) — which table
  - `reason` (string) — why outdated, 1 sentence

**(Gotcha #2: structured IDs prevent free-text matching failures)**

### 2C. Incremental extraction prompts: `memory/extract.py`

New function: `extract_knowledge_incremental(delta_text, prior_narrative, existing_items, prior_items, api_key) -> dict`

**System prompt** (extends existing `SYSTEM_PROMPT`):
```
You are processing a SEGMENT of an ongoing conversation, not the full transcript.

RULES:
- Extract only NEW knowledge from the segment below.
- Do NOT re-extract items listed under PRIOR PASS ITEMS — they are already stored.
- If the conversation contradicts or updates any EXISTING DATABASE ITEMS, include their
  IDs in the supersedes array. When superseding, the replacement item must capture the
  FULL updated state, not just the changed part.
- Only mark items as superseded if the conversation EXPLICITLY contradicts or replaces them.
  Do not supersede items merely because they are related.
- Your narrative_summary must cover everything in the conversation so far, not just this segment.
```

**(Gotcha #3: partial superseding — prompt requires full updated state)**
**(Gotcha #6: "do not re-extract" instruction prevents near-duplicates across passes)**
**(Gotcha #9: "only supersede if EXPLICITLY contradicted" prevents hallucinated supersedes from recall-then-extract)**

**User message structure:**
```
EXISTING DATABASE ITEMS (reference by ID if any are now outdated):
[fact-{uuid1}] The user prefers tabs over spaces
[dec-{uuid2}] We decided to use PostgreSQL for storage

PRIOR PASS ITEMS (already stored this session — do NOT re-extract):
[fact-{uuid3}] DuckDB is used for storage
[idea-{uuid4}] Consider adding HNSW indexes

{if prior_narrative:}
PRIOR NARRATIVE (from earlier in this session — update, don't restart):
{prior_narrative}

--- NEW CONVERSATION SEGMENT ---
{delta_text}
```

**For pass 1:** PRIOR PASS ITEMS and PRIOR NARRATIVE sections are empty/omitted. EXISTING DATABASE ITEMS come from session-start recall cache.

**For pass 2+:** PRIOR NARRATIVE is `state.last_narrative`. EXISTING DATABASE ITEMS come from embedding search using the prior narrative as query. PRIOR PASS ITEMS come from `state.prior_item_ids`.

### 2D. Tests for Phase 2

- `TestDeltaParsing` (~5 tests): delta from offset, from zero equals full, byte offset correctness, substantial above/below threshold
- `TestIncrementalTool` (~4 tests): schema has supersedes/narrative fields, prompt construction for pass 1 vs pass 2+

---

## Phase 3: Pipeline Integration

### 3A. Incremental pipeline: `memory/ingest.py`

New function `run_incremental_extraction(session_id, transcript_path, trigger, cwd, api_key, quiet, is_final, session_recall_items)`:

**Flow:**

1. Load state: `extraction_state.load_state(session_id)`. If None → pass 1 (byte_offset=0).
2. Parse delta: `extract.parse_transcript_delta(transcript_path, state.last_byte_offset or 0)`.
3. **Quality gate (Gotcha #5):** If `not is_delta_substantial(delta_messages)` AND not `is_final`:
   - Save state with updated byte offset only (carry delta forward to next pass).
   - Return early (no API call).
   - If `is_final`, process anyway — last chance to capture knowledge.
4. Build delta text: `extract.build_conversation_text(delta_messages)`.
5. Gather **existing items** for superseding context:
   - Pass 1: use `session_recall_items` if provided. Otherwise, embed first 1000 chars of delta, search DB.
   - Pass 2+: embed `state.last_narrative`, search DB. Use `RECALL_THRESHOLD` (0.60), limit 10 items.
6. Gather **prior pass items**: fetch from DB using `state.prior_item_ids` via `db.get_items_by_ids()`.
7. Call `extract.extract_knowledge_incremental(...)`.
8. **Process supersedes:** For each entry in `result["supersedes"]`:
   - Find which new item replaces it (by semantic similarity between supersede reason and new items).
   - Call `db.supersede_item(old_id, old_table, new_id, reason)`.
9. **Embed and upsert structured items** — reuse existing upsert logic, BUT:
   - **(Gotcha #6 + #9):** Before each upsert, check cosine similarity against items in `prior_item_ids` AND `recalled_item_ids`. If >= `CROSS_PASS_DEDUP_THRESHOLD` (0.85), treat as reinforcement (just call existing upsert which will dedup), don't create new entry.
   - Collect new item IDs for state tracking.
10. **Upsert narrative:** `db.upsert_narrative(session_id, pass_number, narrative, embedding, is_final=is_final, scope)`.
11. **Update and save state:** increment pass_count, update byte offset, narrative, prior_item_ids, recalled_item_ids.
12. **If `is_final`:**
    - `db.finalize_narratives(conn, session_id)` — **(Gotcha #7: only keep final narrative)**
    - `extraction_state.mark_extraction_complete(session_id)` — create `.lock` for backward compat
    - `extraction_state.delete_state(session_id)` — clean up state file
    - Run decay pass + purge (as current system does)
13. Return summary dict.

**Existing `run_extraction` becomes a wrapper:**
```python
def run_extraction(session_id, transcript_path, trigger, cwd, api_key, quiet=False):
    return run_incremental_extraction(
        session_id, transcript_path, trigger, cwd, api_key, quiet,
        is_final=True, session_recall_items=None,
    )
```
This preserves the existing API — any caller that uses `run_extraction` gets single-pass behavior (processes full transcript, finalizes immediately).

### 3B. Narrative recall: `memory/recall.py`

**Modify `prompt_recall`:** Add narrative search after existing queries:
```python
narratives = db.search_narratives(conn, query_embedding, NARRATIVE_SEARCH_LIMIT, RECALL_THRESHOLD, scope)
# Add to return dict: "narratives": narratives
```

**Modify `format_prompt_context`:** Add narratives as priority 2 (after facts, before ideas). **(Gotcha #8: narratives at prompt-recall only, not session start)**

Carve narrative space from within the existing 1500-token budget:
```python
# Priority 1: Facts (existing)
# Priority 2: Related Session Context (NEW — narratives)
if recall.get("narratives") and budget > 0:
    section = ["### Related Session Context"]
    for n in recall["narratives"]:
        section.append(f"- {n['narrative']}")
    section.append("")
    budget = _budget_append(lines, section, budget)
# Priority 3: Ideas (existing, was priority 2)
# Priority 4: Relationships (existing, was priority 3)
# Priority 5: Open questions (existing, was priority 4)
```

**`session_recall` and `format_session_context` are unchanged.** Session start continues to inject structured items (long facts, decisions, entities). No narratives at session start since there's no query signal.

### 3C. Tests for Phase 3

- `TestIncrementalPipeline` (~8 tests, mocking Claude API):
  - pass 1 processes full transcript from offset 0
  - pass 2 processes only delta from stored offset
  - low-knowledge delta is skipped (non-final)
  - low-knowledge delta is NOT skipped when final
  - final pass creates `.lock` and finalizes narratives
  - fallback to single-pass on missing state
  - cross-pass dedup at 0.85 prevents near-duplicates
  - supersede flow deactivates old item and sets superseded_by
- `TestNarrativeRecall` (~4 tests):
  - narratives included in prompt_recall return
  - narratives formatted in prompt context
  - narratives NOT included in session_recall
  - narrative budget respected

---

## Phase 4: Hook Updates

### 4A. `hooks/status_line.py` — multi-threshold triggers

**Current:** Triggers at 90% if `.lock` doesn't exist.

**Change to:**
1. Import `extraction_state.load_state` and `is_extraction_complete`.
2. If `is_extraction_complete(session_id)` → no trigger (all done).
3. Load state → get `pass_count` (default 0).
4. Look up next threshold: `EXTRACTION_THRESHOLDS[min(pass_count, len(EXTRACTION_THRESHOLDS)-1)]`.
5. If `used_pct >= threshold` → spawn `_extract_worker.py` (same subprocess pattern).
6. Status line display: `mem: {used_pct}% ctx [pass {pass_count+1}]` when extracting.

### 4B. `hooks/_extract_worker.py` — incremental + final flag

**Current:** Takes `session_id, transcript_path, cwd` as CLI args. Acquires lock, calls `run_extraction`, releases lock on failure.

**Change to:**
1. Accept optional `--final` CLI flag.
2. Use `extraction_state.acquire_running_lock(session_id)` instead of the old `acquire_lock`. Release on exit (both success and failure).
3. Call `run_incremental_extraction(is_final=("--final" in sys.argv))`.
4. For pass 1: try to load session-start recall cache from `extraction_state/{session_id}_recall.json`.

### 4C. `hooks/session_end.py` — mark as final

**Current:** Checks `.lock`, spawns worker.

**Change to:**
1. Check `is_extraction_complete(session_id)` instead of raw `.lock` check.
2. Pass `--final` flag to worker.

### 4D. `hooks/pre_compact.py` — mark as final

**Current:** Acquires lock, calls `run_extraction` inline.

**Change to:**
1. Check `is_extraction_complete(session_id)`.
2. Acquire running lock.
3. Call `run_incremental_extraction(is_final=True)` inline.
4. Release running lock.

### 4E. `hooks/session_start.py` — cache recall IDs

**Current:** Computes session_recall, formats context, outputs systemMessage.

**Add:** After computing recall, write item IDs to `extraction_state/{session_id}_recall.json`:
```json
{
  "fact_ids": ["uuid1", "uuid2", ...],
  "decision_ids": ["uuid3", ...],
  "items": [
    {"id": "uuid1", "text": "...", "table": "facts"},
    ...
  ]
}
```
This is a small file (~2KB). The first extraction pass reads it to know which existing items to include in the superseding prompt, avoiding a redundant DB query. **(Gotcha #4: pass 1 recall query)**

---

## File-by-File Summary

| File | Change | Lines est. |
|------|--------|-----------|
| `memory/extraction_state.py` | **NEW** — state file + running lock management | ~120 |
| `memory/config.py` | Add 6 constants | ~10 |
| `memory/db.py` | Migration 4 + 5 new functions (upsert_narrative, finalize_narratives, search_narratives, supersede_item, get_items_by_ids) | ~120 |
| `memory/extract.py` | Add parse_transcript_delta, is_delta_substantial, INCREMENTAL_EXTRACTION_TOOL, extract_knowledge_incremental | ~150 |
| `memory/ingest.py` | Add run_incremental_extraction, refactor run_extraction as wrapper | ~130 |
| `memory/recall.py` | Add narratives to prompt_recall + format_prompt_context | ~25 |
| `hooks/status_line.py` | Multi-threshold trigger logic | ~20 |
| `hooks/_extract_worker.py` | --final flag, running lock, incremental call | ~20 |
| `hooks/session_end.py` | Updated lock check, --final flag | ~5 |
| `hooks/pre_compact.py` | Updated to use incremental + running lock | ~10 |
| `hooks/session_start.py` | Cache recall IDs to state dir | ~15 |
| `test_memory.py` | ~39 new tests across 7 test classes | ~400 |

---

## Gotcha Mitigation Checklist

| # | Gotcha | Where Handled |
|---|--------|--------------|
| 1 | Delta tracking (byte offset) | `extraction_state.py` state file + `extract.py:parse_transcript_delta` |
| 2 | Structured supersede IDs | `INCREMENTAL_EXTRACTION_TOOL` schema: typed array with old_id + old_table |
| 3 | Partial superseding loses context | Extraction prompt: "replacement must capture FULL updated state" |
| 4 | Pass 1 has no recall query | `session_start.py` caches recall IDs → pass 1 reads them |
| 5 | Wasted API calls on thin deltas | `is_delta_substantial()` gate; skipped unless `is_final` |
| 6 | Near-duplicates across passes | Prior items included in prompt + cross-pass dedup at 0.85 threshold |
| 7 | Intermediate narratives waste storage | `finalize_narratives()` deletes non-final on completion |
| 8 | No query signal for narratives at session start | Narratives in prompt_recall only, not session_recall |
| 9 | Recall-then-extract feedback loop | Dedup new items against recalled items at 0.85 + "only supersede if EXPLICITLY contradicted" |
| 10 | Supersede chains lose history | Narratives naturally preserve the "why"; no code fix needed |
| 11 | Concurrent session conflicts | Accepted as edge case; DuckDB single-writer serializes writes |

---

## Verification

1. **Unit tests:** `python3 test_memory.py` — all 188 existing + ~39 new tests pass
2. **Migration test:** Start with existing DB, verify migration 4 applies cleanly and idempotently
3. **Single-pass fallback:** Delete state file mid-session → verify system falls back to single-pass extraction at session end (same as current behavior)
4. **Short session:** Run a <20% context session → verify single extraction at session end, no intermediate passes
5. **Long session simulation:** Create a synthetic transcript >120k chars, verify multiple passes fire, no middle truncation, narratives accumulate, final narrative persists
6. **Supersede test:** Session 1: store "using PostgreSQL". Session 2: discuss switching to DuckDB → verify PostgreSQL fact is deactivated with `superseded_by` set
7. **Narrative recall:** After several sessions, verify `prompt_recall` returns relevant narratives and they appear in `format_prompt_context` output
8. **Backward compat:** Verify `.lock` file is created after final pass, `session_end.py` and `pre_compact.py` skip correctly when `.lock` exists

---

## Baseline Measurements (from test_fidelity.py)

Measured 2026-03-19 using current single-pass extraction system.

| Scale | Recall | Precision | F1 | Items Extracted | Truncation | Extract Time |
|-------|--------|-----------|-----|-----------------|------------|-------------|
| 1k (3 facts) | 100% | 11% | 20% | 27 | 0% | 27s |
| 10k (8 facts) | 100% | 21% | 34% | 39 | 0% | 40s |
| 100k (15 facts) | 87-93% | 36% | 51% | 36 | ~0% | 43s |
| 1M (20 facts) | 95% | 41% | 58% | 46 | **95.9%** | 44s |

**Key findings:**
- Recall is high even at 1M because ground truth facts are near transcript edges (beginning/end), which survive truncation
- Real-world conversations would have important info scattered throughout → expected worse recall
- **95.9% of a 1M transcript is discarded** by middle truncation → incremental extraction eliminates this
- Precision is low because Claude over-extracts code-level details from tool_use blocks
- **Multi-session superseding is broken**: stale facts ("The API is built with Flask", "Redis handles caching") remain active even after later sessions explicitly replace them
- Near-duplicate accumulation: 3 sessions produced 51 active facts, many semantically overlapping

**Targets for incremental system:**
- 1M recall: >= 95% (maintain or improve)
- Multi-session: stale facts must be superseded (0 false actives)
- Near-duplicates: reduce total active facts by 30-50% through tighter dedup
- Precision improvement via narratives: compress 51 facts worth of info into fewer, denser items

---

## Implementation Order

Build and test in this order, running `python3 test_memory.py` after each step:

1. `memory/extraction_state.py` + its tests
2. `memory/config.py` additions
3. `memory/db.py` migration 4 + new functions + their tests
4. `memory/extract.py` delta parsing + incremental tool/prompt + their tests
5. `memory/ingest.py` incremental pipeline + its tests
6. `memory/recall.py` narrative recall + its tests
7. `hooks/status_line.py` multi-threshold
8. `hooks/_extract_worker.py` incremental + final flag
9. `hooks/session_end.py` + `hooks/pre_compact.py` updates
10. `hooks/session_start.py` recall caching
11. Full integration verification
