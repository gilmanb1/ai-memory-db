# Claude Code Memory System

Persistent knowledge base for Claude Code. Hooks extract facts, ideas, relationships, and decisions from conversations, store them in DuckDB with Ollama embeddings, and inject relevant context into future sessions.

## Test
```bash
python3 test_memory.py
```

## Prerequisites
- `uv` (inline script runner)
- `ollama pull nomic-embed-text`
- `ANTHROPIC_API_KEY` environment variable

## Structure
- `memory/` — Python package: config, db, decay, embeddings, extract, recall, cli
- `hooks/` — Claude Code hook scripts: pre_compact, session_start, user_prompt_submit
- `install.sh` — Copies to `~/.claude/`, configures hooks in settings.json

## Architecture
- **DuckDB** for storage (`~/.claude/memory/knowledge.duckdb`)
- **Ollama nomic-embed-text** for local embeddings (768-dim)
- **Claude Sonnet** for knowledge extraction via tool_use
- **Temporal decay**: short/medium/long classes with exponential decay and session-based promotion

## Key Design Decisions
- Dedup via cosine similarity (threshold 0.92) — reinforce, don't duplicate
- Token budgets cap context injection to prevent context window bloat
- Only short-term items can be auto-forgotten; medium/long persist
- Hooks degrade gracefully when Ollama is down (skip embedding-based features)
