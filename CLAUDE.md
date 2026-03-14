# Claude Code Memory System

Persistent knowledge base for Claude Code. Hooks extract facts, ideas, relationships, and decisions from conversations, store them in DuckDB with Ollama embeddings, and inject relevant context into future sessions. Knowledge is scoped per git repo with automatic cross-project promotion.

## Test
```bash
python3 test_memory.py
```

## Prerequisites
- `uv` (inline script runner)
- `ollama pull nomic-embed-text`
- `ANTHROPIC_API_KEY` environment variable

## Structure
- `memory/` — Python package: config, db, decay, embeddings, extract, ingest, recall, scope, cli
- `hooks/` — Claude Code hook scripts: pre_compact, session_start, session_end, user_prompt_submit, status_line, _extract_worker
- `install.sh` — Copies to `~/.claude/`, configures hooks and status line in settings.json
- `benchmark.py` — Synthetic data benchmark (100-10k items, 3 repos)

## Architecture
- **DuckDB** for storage (`~/.claude/memory/knowledge.duckdb`)
- **Ollama nomic-embed-text** for local embeddings (768-dim)
- **Claude Sonnet** for knowledge extraction via tool_use
- **Temporal decay**: short/medium/long classes with exponential decay and session-based promotion
- **Project scoping**: git repo root, with auto-promotion to global when seen in 3+ projects

## Key Design Decisions
- Dedup via cosine similarity (threshold 0.92) — reinforce, don't duplicate
- Token budgets cap context injection (3000 session, 1500 per-prompt)
- Only short-term items can be auto-forgotten; medium/long persist
- Hooks degrade gracefully when Ollama is down (skip embedding-based features)
- Three extraction triggers (status line at 90%, PreCompact, SessionEnd) with per-session lock
- `/remember` command stores facts/decisions immediately from any prompt

## /remember Command
```
/remember <text>                    → project-scoped long-term fact
/remember global: <text>            → global fact (all projects)
/remember decision: <text>          → project-scoped decision
/remember global decision: <text>   → global decision
```
