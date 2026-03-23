# Architecture

Persistent knowledge base for Claude Code. Hooks extract facts, ideas, relationships, and decisions from conversations, store them in DuckDB with Ollama embeddings, and inject relevant context into future sessions. Knowledge is scoped per git repo with automatic cross-project promotion.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Claude Code Session                         │
│                                                                     │
│  SessionStart ──→ Inject long/medium-term context (systemMessage)   │
│  UserPromptSubmit ──→ Semantic recall per prompt (additionalContext) │
│  StatusLine ──→ Monitor context %, trigger extraction at 90%        │
│  PreCompact ──→ Extract knowledge before compaction (inline)        │
│  SessionEnd ──→ Extract knowledge on exit (background)              │
│                                                                     │
│  /remember ──→ Route to DuckDB + auto-memory markdown               │
│  /forget ──→ Search, soft-delete, cleanup auto-memory               │
└───────────┬──────────────┬──────────────┬───────────────────────────┘
            │              │              │
     ┌──────▼──────┐ ┌────▼────┐ ┌───────▼────────┐
     │   DuckDB    │ │ Ollama  │ │  Claude Sonnet  │
     │  (storage)  │ │ (embed) │ │  (extraction)   │
     └─────────────┘ └─────────┘ └────────────────┘
```

---

## Storage

### DuckDB Schema

All data lives in `~/.claude/memory/knowledge.duckdb`. Three versioned migrations manage the schema.

**Migration 1 — Base tables:**

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `facts` | text, category, temporal_class, confidence, decay_score, session_count, embedding, is_active | Core knowledge items |
| `ideas` | text, idea_type, temporal_class, decay_score, session_count, embedding, is_active | Hypotheses, proposals, insights |
| `decisions` | text, temporal_class, decay_score, session_count, embedding, is_active | Architectural/design decisions |
| `entities` | name, name_lower, entity_type, session_count, embedding | People, tools, projects, technologies |
| `relationships` | from_entity, to_entity, rel_type, description, strength, is_active | Entity graph edges |
| `open_questions` | text, resolved, embedding, is_active | Unresolved questions |
| `sessions` | trigger, cwd, transcript_path, message_count, summary | Extraction session history |
| `fact_entity_links` | fact_id, entity_name | Links facts to entities mentioned in them |

**Migration 2 — Project scoping:**
- `scope` column added to all tables (default: `__global__`)
- `item_scopes` table tracks which projects have seen each item

**Migration 3 — Soft-delete support:**
- `deactivated_at` column on all tables
- `is_active` + `deactivated_at` on entities (previously lacked `is_active`)

### Deduplication

Two-tier strategy, checked in order:

1. **Embedding similarity** — cosine >= 0.92 across all scopes (via `list_cosine_similarity` or Python fallback)
2. **Exact text match** — within same scope + global, used when either side lacks an embedding (Ollama was down)

When a duplicate is found, the existing item is reinforced: `session_count++`, `decay_score` recalculated, `temporal_class` promoted if thresholds met, embedding backfilled if previously null.

### Scope System

- **Project scope:** git repo root from `git rev-parse --show-toplevel`, or resolved cwd as fallback
- **Global scope:** `__global__`, shared across all projects
- **Scope filtering:** queries always match `(scope = project OR scope = __global__)` via parameterized SQL
- **Auto-promotion:** when an item appears in 3+ distinct project scopes, it's promoted to global

---

## Temporal Decay

Three classes with exponential decay:

| Class | Decay Rate | Auto-forget | Promotion Rule |
|-------|-----------|-------------|----------------|
| `short` | 0.18/day | Yes, at score < 0.05 | → medium at 3 sessions or 7 days |
| `medium` | 0.04/day | No | → long at 7 sessions or 30 days |
| `long` | 0.007/day | No | Terminal class |

**Decay formula:**
```
score = exp(-rate / reinforcement * days_since_last_seen)
reinforcement = min(10.0, 1.0 + 0.5 * (session_count - 1))
```

The LLM assigns the initial class at extraction time. The system only promotes upward, never demotes. Frequently-seen items decay slower due to the reinforcement factor.

**Recall weight** combines class and score:
```
weight = decay_score * class_boost    (long=1.5, medium=1.1, short=0.8)
```

---

## Knowledge Extraction

### Triggers

Three triggers share a per-session lock to prevent duplicate work:

| Trigger | When | How | Lock behavior |
|---------|------|-----|---------------|
| **StatusLine** | Context window >= 90% | Background worker (`_extract_worker.py`) | First to acquire wins |
| **PreCompact** | Before compaction | Inline (blocking, prints summary) | First to acquire wins |
| **SessionEnd** | Session exit | Background worker (`_extract_worker.py`) | First to acquire wins |

Lock files live at `~/.claude/memory/locks/{session_id}.lock`, created atomically with `O_CREAT | O_EXCL`. Stale locks (>24h) are cleaned up after each extraction.

### Pipeline (`ingest.py:run_extraction`)

```
Parse transcript (JSONL → messages)
  ↓
Build conversation text (truncate to 120k chars)
  ↓
Claude Sonnet extraction (tool_use with structured output, 1 retry)
  ↓
Resolve project scope
  ↓
Embed & upsert to DuckDB:
  Entities → Facts (+ entity links) → Ideas → Relationships → Decisions → Questions
  ↓
Decay pass (update scores, promote classes, forget short items)
  ↓
Purge hard-deleted items (soft-deleted > 30 days)
  ↓
Cleanup old locks (> 24 hours)
```

### Extraction Schema

Claude extracts via `store_knowledge` tool:
- **Facts** (5–25): text, category (`technical`/`decision`/`personal`/`contextual`/`numerical`), confidence, temporal_class
- **Ideas** (2–10): text, type (`hypothesis`/`proposal`/`insight`/`concern`/`analogy`), temporal_class
- **Relationships** (4–15): from, to, type (`depends_on`/`causes`/`part_of`/`uses`/etc.), description
- **Decisions** (0–10): text, temporal_class
- **Open questions** (0–8): text
- **Entities** (4–20): names of people, projects, tools, technologies
- **Session summary**: 2–4 sentences

---

## Context Recall

### Session Recall (SessionStart)

Fires at the beginning of every session. Retrieves broad, temporally-weighted context.

```
Resolve scope → Query DB → Format as systemMessage → Inject into session
```

**Retrieved items (project-local first, fill with global):**
- Long-term facts (limit 20)
- Medium-term facts (limit 8)
- Decisions (limit 10)
- Top entities by session_count (limit 25)
- Relationships for those entities

**Formatting priority** within 3,000-token budget:
1. Established Knowledge (long facts)
2. Key Decisions
3. Working Context (medium facts)
4. Known Entities
5. Relationship Graph

### Prompt Recall (UserPromptSubmit)

Fires on every user prompt (>= 10 chars). Retrieves narrow, embedding-focused context.

```
Embed prompt → Semantic search → Entity graph → Re-rank → Format as additionalContext
```

**Search targets:**
- Facts (limit 8, cosine >= 0.60)
- Ideas (limit 4)
- Open questions (limit 3)
- Relationships for entities mentioned in prompt text (limit 6)

**Re-ranking:** `temporal_weight(class, decay_score) * embedding_similarity`

**Budget:** 1,500 tokens.

---

## /remember Command

### Data Flow

```
/remember <text>
  ↓
Strip prefixes: global:, decision:, global decision:
  ↓
Classify via routing.classify_memory()
  ↓
Store to DuckDB (always):
  Facts: category=personal, temporal_class=long, confidence=high
  Decisions: temporal_class=long
  ↓
If route == "both", also store to auto-memory:
  Write ~/.claude/projects/<project>/memory/{type}_{slug}.md
  Update MEMORY.md index
  ↓
Output: confirmation + routing explanation (if MEMORY_DEBUG=1)
```

### Routing Classification

Text is matched against pattern categories in priority order:

| Type | Triggers | Example |
|------|----------|---------|
| **feedback** | "always", "never", "don't", "avoid", "I prefer", "stop", "before you..." | "never mock the database in tests" |
| **user** | "I am a", "my role", "my background", "I'm new to" | "I'm a senior backend engineer" |
| **reference** | URLs, "tracked in", "Linear", "Jira", "dashboard at" | "bugs are tracked in Linear project INGEST" |
| **project** | "deadline", "freeze", "sprint", "milestone", "release" | "merge freeze starts Thursday" |
| *(no match)* | — | "the API uses gRPC" |

- **Match → route "both"**: DuckDB + auto-memory markdown file
- **No match → route "duckdb"**: DuckDB only

### Auto-Memory Files

Written to `~/.claude/projects/{cwd-slugified}/memory/`:
- File: `{type}_{slug}_{hash}.md` with YAML frontmatter (name, description, type)
- Index: `MEMORY.md` with links (max 180 entries, opinionated budget cap)
- MEMORY.md is loaded into every conversation's system prompt — guaranteed visibility
- Set `MEMORY_DEBUG=0` to suppress routing explanations

---

## /forget Command

### Two-Stage Flow

**Stage 1 — Search** (`/forget <query>`):
- Case-insensitive substring search across all 6 tables
- Also tries exact ID match
- Displays numbered list with table, text preview, ID

**Stage 2 — Confirm** (`/forget-confirm <id> <table>`):
- Soft-delete: `is_active=FALSE`, `deactivated_at=now()`
- Cleanup: removes auto-memory markdown file + MEMORY.md entry if present
- Hard purge after 30 days (runs at end of each extraction)

---

## Graceful Degradation

| Condition | Impact | Behavior |
|-----------|--------|----------|
| **Ollama down** | No embeddings | Dedup falls back to text match; per-prompt recall disabled; session recall still works; warn once to stderr |
| **Anthropic API fails** | No extraction | Retry once (2s delay); skip if still fails; lock released for next trigger |
| **No API key** | No extraction | Skip silently; recall from existing DB still works |
| **No DB yet** | Nothing to recall | All hooks exit cleanly; DB created on first extraction |
| **Large transcript** | Exceeds 120k chars | Middle truncated; per-message capped at 6k chars |

---

## File Map

### Memory Package (`memory/`)

| Module | Purpose |
|--------|---------|
| `config.py` | All tunable constants (paths, thresholds, limits, budgets) |
| `db.py` | DuckDB schema, migrations, CRUD, vector search, decay pass, soft-delete, purge |
| `decay.py` | Decay formula, should_forget, temporal_weight |
| `embeddings.py` | Ollama client with cache, batch support, graceful fallback |
| `extract.py` | Claude API extraction via tool_use, transcript parsing |
| `ingest.py` | Full extraction pipeline, lock system |
| `recall.py` | Session + prompt recall, token-budgeted formatting |
| `scope.py` | Git-based scope resolution |
| `routing.py` | /remember classification, auto-memory file read/write/delete |
| `cli.py` | CLI inspector (stats, search, promote, etc.) |

### Hooks (`hooks/`)

| Script | Trigger | Mode |
|--------|---------|------|
| `session_start.py` | SessionStart | Inject systemMessage (broad recall) |
| `user_prompt_submit.py` | UserPromptSubmit | /remember handling + prompt recall |
| `session_end.py` | SessionEnd | Spawn background extraction |
| `pre_compact.py` | PreCompact | Inline extraction |
| `status_line.py` | StatusLine | Display context %, trigger extraction at 90% |
| `_extract_worker.py` | (spawned) | Background extraction with lock |
| `remember_cmd.py` | /remember slash command | Route to DuckDB + auto-memory |
| `forget_cmd.py` | /forget slash command | Search, soft-delete, cleanup |

### Commands (`commands/`)

| File | Slash Command | Action |
|------|---------------|--------|
| `remember.md` | `/remember <text>` | Invokes `remember_cmd.py` via `MEMORY_TEXT` env var |
| `forget.md` | `/forget <query>` | Invokes `forget_cmd.py` via `MEMORY_TEXT` env var |
| `forget-confirm.md` | `/forget-confirm <id> <table>` | Invokes `forget_cmd.py` via `MEMORY_FORGET_ID` env var |

### Other

| File | Purpose |
|------|---------|
| `install.sh` | Copies package + hooks + commands to `~/.claude/`, merges settings.json |
| `test_memory.py` | 188 BDD-style tests against real DuckDB (no Ollama/API needed) |
| `benchmark.py` | Synthetic data benchmark (100–10k items, 3 repos) |

---

## Installation

```bash
bash install.sh            # global install (default)
bash install.sh --project  # project-local install
```

Steps:
1. Copy `memory/` package to `~/.claude/memory/`
2. Copy hook scripts to `~/.claude/hooks/`
3. Copy slash commands to `~/.claude/commands/`
4. Create `~/.claude/memory/locks/`
5. Merge hook + status line config into `settings.json`
6. Run test suite

Prerequisites: `uv`, `python3`, `ollama pull nomic-embed-text`, `ANTHROPIC_API_KEY`

---

## Key Design Decisions

1. **Dedup via cosine similarity (0.92)** — reinforce, don't duplicate
2. **Token budgets** cap context injection (3,000 session, 1,500 per-prompt)
3. **Only short-term items auto-forget** — medium/long persist unless explicitly deleted
4. **Hooks degrade gracefully** when Ollama is down (skip embedding-dependent features)
5. **Three extraction triggers** with per-session lock — whichever fires first wins
6. **Parameterized SQL** for all scope filtering — no inline string escaping
7. **Text-based dedup fallback** within scope when embeddings unavailable
8. **Auto-memory routing** stores behavioral/profile/reference memories as markdown for guaranteed visibility
9. **Soft-delete + 30-day purge** for /forget — recoverable window before permanent deletion
10. **Scope-aware dedup** — project A's facts don't cross-pollinate with project B's
