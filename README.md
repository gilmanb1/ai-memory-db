# Claude Code Memory System

Persistent, evolving knowledge base for Claude Code. Conversations are automatically mined for facts, ideas, relationships, and decisions. That knowledge lives in a local DuckDB database, enriched with Ollama embeddings, and injected back into future sessions — scoped per project with automatic cross-project promotion.

## How It Works

Three extraction triggers ensure knowledge is captured before it's lost:

| Trigger | When | How |
|---------|------|-----|
| **Status line** | Context window hits 90% | Background process, non-blocking |
| **PreCompact** | Just before compaction | Inline, prints summary to transcript |
| **SessionEnd** | Session terminates | Background process, exits within 1.5s |

A per-session lock file prevents duplicate extraction. Whichever trigger fires first wins.

Two injection points make stored knowledge available:

| Hook | What's Injected |
|------|-----------------|
| **SessionStart** | Long-term facts, decisions, entities, relationships as `systemMessage` |
| **UserPromptSubmit** | Per-prompt semantic recall as `additionalContext` |

Both enforce token budgets to avoid context window bloat.

## /remember Command

Type `/remember` in any Claude Code prompt to store something to long-term memory immediately:

```
/remember The API uses gRPC for inter-service communication
/remember global: My name is Ben
/remember decision: We chose PostgreSQL over MySQL
/remember global decision: Always use TypeScript for frontend projects
```

| Prefix | Effect |
|--------|--------|
| *(none)* | Store as fact in current project scope |
| `global:` | Store as fact in global scope (all projects) |
| `decision:` | Store as decision in current project scope |
| `global decision:` | Store as decision in global scope |

All `/remember` items are stored with `temporal_class=long` and `confidence=high`. Duplicate text is reinforced rather than duplicated.

## Project Scoping

Knowledge is scoped per git repository. Each repo's facts, decisions, and entities are isolated from other projects.

**Scope resolution:** The system runs `git rev-parse --show-toplevel` to identify the project. Non-git directories use the resolved path.

**Recall priority:** Project-local items fill the token budget first. Global items fill whatever remains.

**Promotion to global** (shared across all projects):

- **Automatic:** When an item is seen in 3+ distinct projects, it's promoted to `__global__` scope
- **Manual:** `python -m memory promote facts <uuid>`

## Temporal Memory Model

Each stored item has a `temporal_class` assigned by the LLM at extraction time, then adjusted upward by reinforcement:

| Class | Decay Rate | Auto-forgotten? | Examples |
|-------|------------|-----------------|----------|
| `short` | 0.18/day | Yes, when score < 0.05 | Current error, transient debug state |
| `medium` | 0.04/day | No | Active project phase, current bug |
| `long` | 0.007/day | No | User's name, tech stack, architecture decisions |

**Promotion rules** (only upward):
- short -> medium: seen in 3+ sessions, or age > 7 days
- medium -> long: seen in 7+ sessions, or age > 30 days

**Decay formula:** `score = exp(-base_rate / reinforcement * days_old)` where `reinforcement = min(10, 1 + 0.5 * (sessions - 1))`

## What Gets Stored

| Type | Volume | Examples |
|------|--------|----------|
| **Facts** | 5-25/session | "The project uses DuckDB for storage" |
| **Ideas** | 2-10/session | "FOXP3 could be a long-horizon target" |
| **Relationships** | 4-15/session | Focal Graph --[uses]--> DuckDB |
| **Decisions** | 0-10/session | "We will use Neo4j for graph queries" |
| **Open questions** | 0-8/session | "Do we have clinical outcome data?" |
| **Entities** | 4-20/session | "Plex Research", "DuckDB", "FOXP3" |

Extraction uses Claude Sonnet via `tool_use` for guaranteed structured output (no JSON parsing failures).

## Install

```bash
# Prerequisites
curl -LsSf https://astral.sh/uv/install.sh | sh
ollama pull nomic-embed-text

# Ensure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY="sk-ant-..."

# Install globally (all Claude Code sessions)
bash install.sh

# Or install for current project only
bash install.sh --project
```

The installer copies `memory/` to `~/.claude/memory/`, hooks to `~/.claude/hooks/`, configures `settings.json`, and runs the test suite.

Restart Claude Code to activate.

## CLI Inspector

Inspect what's stored in the knowledge base:

```bash
python -m memory stats              # Database statistics
python -m memory facts              # List all active facts
python -m memory facts --class long # Filter by temporal class
python -m memory search "query"     # Semantic search (requires Ollama)
python -m memory entities           # List known entities
python -m memory decisions          # List decisions
python -m memory relationships      # List relationship graph
python -m memory sessions           # List stored sessions
python -m memory scopes             # List project scopes and item counts
python -m memory promote facts <id> # Promote an item to global scope
```

## Configuration

All constants are in `memory/config.py`:

```python
# Embedding model
OLLAMA_MODEL      = "nomic-embed-text"   # swap to mxbai-embed-large for higher accuracy
EMBEDDING_DIM     = 768                  # must match the model

# Extraction
CLAUDE_MODEL      = "claude-sonnet-4-6"

# Similarity thresholds
DEDUP_THRESHOLD   = 0.92   # cosine >= this -> reinforce, don't insert
RECALL_THRESHOLD  = 0.60   # cosine >= this -> relevant for recall

# Token budgets (prevent context window bloat)
SESSION_TOKEN_BUDGET  = 3000
PROMPT_TOKEN_BUDGET   = 1500

# Temporal decay
DECAY_RATES = {"short": 0.18, "medium": 0.04, "long": 0.007}

# Project scoping
AUTO_PROMOTE_PROJECT_COUNT = 3   # seen in N+ projects -> auto-promote to global
```

## Project Structure

```
memory/                     Python package (installed to ~/.claude/memory/)
  config.py                   All tuneable constants
  db.py                       DuckDB schema, migrations, CRUD, vector search
  decay.py                    Temporal scoring and forgetting
  embeddings.py               Ollama embedding client with graceful fallback
  extract.py                  Claude API extraction via tool_use
  ingest.py                   Shared extraction + storage pipeline
  recall.py                   Session and prompt recall with token budgets
  scope.py                    Git-based project scope resolution
  cli.py                      CLI inspector (python -m memory)

hooks/                      Claude Code hook scripts (installed to ~/.claude/hooks/)
  pre_compact.py              PreCompact: extract before compaction
  session_start.py            SessionStart: inject broad context
  session_end.py              SessionEnd: extract on exit (background)
  user_prompt_submit.py       UserPromptSubmit: per-prompt recall
  status_line.py              Status line: trigger extraction at 90% context
  _extract_worker.py          Background extraction worker

test_memory.py              100 tests (BDD-style, red/green methodology)
install.sh                  Installer script
```

## Schema

Single DuckDB file at `~/.claude/memory/knowledge.duckdb` with versioned migrations:

```
facts              ideas              entities           relationships
  text               text               name               from_entity
  category           idea_type          entity_type        to_entity
  temporal_class     temporal_class     embedding          rel_type
  confidence         decay_score        session_count      description
  decay_score        embedding          scope              strength
  session_count      scope                                 scope
  embedding
  scope

decisions          open_questions     sessions           item_scopes
  text               text               summary            item_id
  temporal_class     resolved           transcript_path    item_table
  decay_score        embedding          scope              scope
  embedding
  scope
```

The `scope` column defaults to `__global__`. Project-scoped items use the git repo root path. The `item_scopes` table tracks which projects have seen each item for auto-promotion.

## Schema Migrations

To add a column in a future version, append to the `MIGRATIONS` list in `db.py`:

```python
(3, "Add fact source_url column", """
    ALTER TABLE facts ADD COLUMN IF NOT EXISTS source_url VARCHAR;
"""),
```

Migrations run once on the next `get_connection()` call and are recorded in `schema_migrations`.

## Changing the Embedding Model

1. Pull the new model: `ollama pull mxbai-embed-large`
2. Update `config.py`: set `OLLAMA_MODEL` and `EMBEDDING_DIM`
3. Delete `~/.claude/memory/knowledge.duckdb` (old embeddings are incompatible)
4. The DB is rebuilt from scratch on the next extraction

## Tests

```bash
python3 test_memory.py
```

100 tests against real DuckDB instances. No Ollama or Anthropic API required — embeddings are mocked with deterministic hash-based vectors. Tests use BDD-style `given_when_then` naming and red/green methodology.

## Graceful Degradation

| Condition | Behavior |
|-----------|----------|
| Ollama down | Embedding-based dedup and per-prompt recall disabled. Session-level recall still works (DB queries, not embeddings). Warn-once message on stderr. |
| Anthropic API fails | Single retry with 2s delay. If still fails, extraction is skipped — no crash. |
| No ANTHROPIC_API_KEY | Extraction skipped silently. Recall still works from existing DB. |
| No database yet | All hooks exit cleanly. DB is created on first successful extraction. |
