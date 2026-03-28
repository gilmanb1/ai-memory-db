# Claude Code Memory System

[![Tests](https://github.com/gilmanb1/ai-memory-db/actions/workflows/test.yml/badge.svg)](https://github.com/gilmanb1/ai-memory-db/actions/workflows/test.yml)

Persistent knowledge base for Claude Code. Hooks extract facts, decisions, relationships, and more from conversations, store them in DuckDB with local embeddings, and inject relevant context into future sessions. Knowledge is scoped per git repo with automatic cross-project promotion.

## Quick Start

**One-command install:**

```bash
curl -fsSL https://raw.githubusercontent.com/gilmanb1/ai-memory-db/master/install-remote.sh | bash
```

**Or clone and install:**

```bash
git clone https://github.com/gilmanb1/ai-memory-db.git
cd ai-memory-db
bash install.sh
```

**Prerequisites** (install these first):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Python script runner
ollama pull nomic-embed-text                        # Local embedding model (768-dim)
export ANTHROPIC_API_KEY="sk-ant-..."               # For knowledge extraction
```

Restart Claude Code after installing — memory is now active. No manual steps during normal use.

**Platform support:** Linux, macOS, Windows (WSL2). Requires `bash`, `python3 >= 3.11`, `uv`, and `ollama`.

## How It Works

```
Session Start ──→ Inject long-term facts, guardrails, decisions as system context
Every Prompt  ──→ 6-way semantic recall: relevant facts, error solutions, guardrails
Background    ──→ Extract knowledge at 40%/70%/90% context usage + session end
Session End   ──→ Auto-snapshot DB, final extraction, consolidation, decay pass
```

The system is fully automatic. Knowledge builds up over time. Short-term facts decay and are forgotten. Items seen across 3+ sessions promote from short → medium → long term. Items seen across 3+ projects auto-promote to global scope.

## Slash Commands

### Knowledge Storage
| Command | Description |
|---------|-------------|
| `/remember <text>` | Store a fact in current project scope |
| `/remember global: <text>` | Store a fact visible to all projects |
| `/remember decision: <text>` | Store a decision |
| `/remember guardrail: <text>` | Store a guardrail (highest recall priority) |
| `/remember procedure: <text>` | Store a how-to procedure |
| `/remember error: <pattern> -> <fix>` | Store an error→solution pair |

### Knowledge Retrieval
| Command | Description |
|---------|-------------|
| `/knowledge <topic>` | Cross-type search: facts, decisions, guardrails, entities, relationships |
| `/search-memory <query>` | Semantic search with `--type` and `--scope` filters |
| `/reflect <question>` | Agentic Q&A — iterates search/synthesis up to 6 rounds |
| `/recalled` | Show what context was injected for the last prompt |
| `/session-learned [id]` | Show what was extracted from a session |

### Knowledge Management
| Command | Description |
|---------|-------------|
| `/forget <text>` | Search and soft-delete a memory |
| `/review` | List/approve/reject flagged extraction items |
| `/audit-memory` | Quality report: stale facts, contradictions, orphaned entities |
| `/memory-health` | System health: Ollama, DB locks, snapshots, embeddings, disk |

### Inspection
| Command | Description |
|---------|-------------|
| `/memories` | Database statistics |
| `/facts` | List facts (`--class long`, `--limit 20`) |
| `/decisions` | List decisions |
| `/entities` | List known entities |
| `/relationships` | Show entity graph |
| `/sessions` | List sessions with summaries |
| `/scopes` | List project scopes and item counts |

### Backup & Recovery
| Command | Description |
|---------|-------------|
| `/snapshots` | List available DB snapshots |
| `/export-memory` | Export to portable JSON (`--output path --scope X`) |
| `/import-memory <path>` | Import from JSON export |
| `/restore-memory <snapshot>` | Roll back to a snapshot |

## What Gets Stored

| Type | Description | Recall Priority |
|------|-------------|-----------------|
| **Guardrails** | "Don't modify X without Y" — protective rules | Highest (always surfaced) |
| **Error Solutions** | Known error→fix pairs | High |
| **Procedures** | Step-by-step how-to instructions | High |
| **Facts** | Technical, architectural, operational knowledge | Medium |
| **Decisions** | Why X was chosen over Y | Medium |
| **Observations** | Consolidated insights from multiple facts | Medium |
| **Entities** | Named concepts, people, technologies | Low |
| **Relationships** | Entity connections (uses, depends_on, etc.) | Low |

## Project Scoping

Knowledge is isolated per git repository. Facts from project A never leak into project B's context.

- **Scope resolution:** `git rev-parse --show-toplevel` identifies the project
- **Recall priority:** Project-local items first, global items fill remaining budget
- **Auto-promotion:** Items seen in 3+ projects promote to `__global__` scope
- **Multi-scope:** Set `MEMORY_ADDITIONAL_SCOPES=/other/repo` for cross-repo sessions (e.g., `--add-dir`)

## Retrieval Pipeline

Every prompt triggers 6-way parallel retrieval, fused via Reciprocal Rank Fusion:

| Strategy | What It Finds | Speed |
|----------|---------------|-------|
| **Semantic** | Embedding cosine similarity across 9 tables | ~50ms |
| **BM25** | Keyword/full-text search | ~10ms |
| **Graph** | Entity traversal (2-hop BFS through relationships) | ~20ms |
| **Temporal** | Date-range filtering ("last week", "yesterday") | ~5ms |
| **Path** | File-path matching via fact_file_links | ~10ms |
| **Code** | Symbol/dependency graph matching | ~15ms |

Results are fused, ranked, and capped at the token budget (4000 tokens per prompt, 3000 per session).

## System Hardening

| Feature | Description |
|---------|-------------|
| **Extraction validation** | Rejects bare URLs, meta-commentary, low-confidence items. Flags borderline items for `/review` |
| **Correction detection** | Detects "that's wrong" / "actually it's..." and auto-supersedes bad facts |
| **Guardrail enforcement** | PostToolUse hook detects edits to guardrailed files, auto-stashes via git |
| **Guardrail promotion** | Facts with directive language (always/never) + high reinforcement are proposed as guardrails |
| **Auto-snapshots** | DB snapshot on every session end (rotating last 5) |
| **Truncation visibility** | Stderr reports when items are truncated; footer in injected context |
| **DuckDB concurrency** | Retry with exponential backoff, per-process init caching, read-only optimization |

## Web Dashboard

A full-featured Next.js dashboard for exploring and managing the knowledge base:

```bash
# Start backend
cd dashboard/backend && uvicorn server:app --port 9111

# Start frontend
cd dashboard/frontend && npm run dev
```

Features:
- **Per-scope filtering** — dropdown in sidebar filters all pages by project
- **Knowledge graph** — unified multi-type graph with clustering, heatmap mode, and chat
- **Review queue** — approve/reject flagged extraction items
- **CRUD** on all item types — facts, decisions, guardrails, procedures, error solutions
- **Entity/relationship graph** — colored by type, sized by degree, clustered by edge type
- **Code graph** — file dependency visualization with symbol details

## Install

### Option A: One-command install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/gilmanb1/ai-memory-db/master/install-remote.sh | bash
```

This clones the repo to a temp directory, runs the installer, and cleans up. To install a specific version:

```bash
AI_MEMORY_DB_VERSION=v1.0.0 curl -fsSL https://raw.githubusercontent.com/gilmanb1/ai-memory-db/master/install-remote.sh | bash
```

### Option B: Clone and install

```bash
git clone https://github.com/gilmanb1/ai-memory-db.git
cd ai-memory-db
bash install.sh              # Install globally (all Claude Code sessions)
bash install.sh --project    # Or install for current project only
```

### Prerequisites

| Dependency | Install | Purpose |
|-----------|---------|---------|
| Python 3.11+ | System package manager | Runtime |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Inline script deps |
| [Ollama](https://ollama.com) | `ollama pull nomic-embed-text` | Local embeddings (768-dim) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Knowledge extraction |

**Optional:** Install `onnxruntime` + `transformers` for in-process embeddings (~3ms, no Ollama needed).

### What the installer does

1. Copies `memory/` package to `~/.claude/memory/`
2. Copies 15 hook scripts to `~/.claude/hooks/`
3. Copies 22 slash commands to `~/.claude/commands/`
4. Configures hooks, PostToolUse, and status line in `settings.json`
5. Runs the test suite to verify installation

### Updating

```bash
cd ai-memory-db && git pull && bash install.sh
```

Or re-run the one-liner — it always fetches the latest.

### Platform notes

- **macOS/Linux:** Works out of the box
- **WSL2:** Works as-is (`$HOME/.claude/` maps correctly)
- **Windows (native):** Not supported — use WSL2

## Configuration

Key settings in `memory/config.py` (83 configurable parameters):

```python
OLLAMA_MODEL       = "nomic-embed-text"    # Embedding model (768-dim)
CLAUDE_MODEL       = "claude-sonnet-4-6"   # Extraction LLM
DEDUP_THRESHOLD    = 0.92                  # Cosine >= this → reinforce, don't duplicate
RECALL_THRESHOLD   = 0.60                  # Cosine >= this → relevant for recall
SESSION_TOKEN_BUDGET = 3000                # Max tokens for session context
PROMPT_TOKEN_BUDGET  = 4000                # Max tokens per-prompt context
DECAY_RATES = {"short": 0.18, "medium": 0.04, "long": 0.007}  # Per-day exponential
AUTO_PROMOTE_PROJECT_COUNT = 3             # Seen in N+ projects → global
EXTRACTION_THRESHOLDS = [40, 70, 90]       # Context % triggers for extraction
GUARDRAIL_ENFORCEMENT_ENABLED = True       # PostToolUse guardrail checking
SNAPSHOT_ON_SESSION_END = True             # Auto-snapshot DB
```

## Tests

```bash
python3 test_memory.py        # Full suite
python3 -m pytest test_memory.py -k "TestCorpus" -v  # Corpus tests only
```

809 tests covering:
- Unit tests with mock embeddings (fast, no external deps)
- Integration tests exercising full hook pipelines
- Realistic corpus tests (hand-crafted 6-week corpus + scaled 1d/1w/1m/1y corpuses)
- Real ONNX embedding tests (semantic similarity verification)
- Cross-repo scope isolation tests (22 tests proving no contamination)
- Concurrency tests (multi-process DuckDB lock handling)

## Project Structure

```
memory/                     Python package
  config.py                   83 configurable constants
  db.py                       DuckDB schema (11 migrations), CRUD, vector search
  embeddings.py               ONNX (primary) + Ollama (fallback) embeddings
  extract.py                  Claude API extraction via tool_use
  ingest.py                   Multi-pass extraction pipeline with validation
  recall.py                   Session + prompt recall with token budgets
  retrieval.py                6-way parallel retrieval with RRF fusion
  scope.py                    Git-based scope resolution + multi-scope
  decay.py                    Temporal scoring and forgetting
  consolidation.py            Observation synthesis + coherence checking
  communities.py              Entity clustering via union-find
  validation.py               Extraction quality gates + review queue
  corrections.py              User correction detection + auto-supersede
  backup.py                   Snapshots, export/import, restore
  guardrail_check.py          File-edit guardrail enforcement
  guardrail_promotion.py      Auto-detect facts that should be guardrails
  cli.py                      CLI (python -m memory)
  code_graph.py               Tree-sitter code parsing + symbol graph
  routing.py                  /remember classification + routing

hooks/                      Claude Code hook scripts (15 hooks)
commands/                   Slash command definitions (22 commands)
dashboard/                  Web dashboard (FastAPI backend + Next.js frontend)
test_memory.py              809 tests
test_corpus.py              Hand-crafted 6-week corpus fixture
test_corpus_scaled.py       Scaled corpus generator (1d/1w/1m/1y)
test_embeddings_cache.py    ONNX embedding cache for tests
install.sh                  Installer script
```

## Graceful Degradation

| Condition | Behavior |
|-----------|----------|
| Ollama down | Falls back to ONNX embeddings (if installed). If both unavailable, BM25 keyword search only. |
| ONNX unavailable | Falls back to Ollama HTTP. If both down, embedding features disabled. |
| Anthropic API fails | Single retry with 2s delay. Extraction skipped — recall still works. |
| No ANTHROPIC_API_KEY | Extraction disabled. Recall works from existing DB. |
| No database yet | All hooks exit cleanly. DB created on first extraction. |
| DuckDB locked | Retry with exponential backoff (5 attempts, 150ms-2.4s). |
| Token budget exceeded | Lower-priority items truncated. Stderr reports what was dropped. |
