# Claude Code Memory System

Persistent knowledge base for Claude Code. Hooks extract facts, ideas, relationships, and decisions from conversations, store them in DuckDB with Ollama/ONNX embeddings, and inject relevant context into future sessions. Knowledge is scoped per git repo with automatic cross-project promotion.

## Test
```bash
python3 test_memory.py
```

809 tests. No Ollama or Anthropic API required — embeddings are mocked. Tests use red/green TDD methodology.

## Prerequisites
- `uv` (inline script runner)
- `ollama pull nomic-embed-text` (or ONNX runtime for local embeddings)
- `ANTHROPIC_API_KEY` environment variable

## Structure
- `memory/` — Python package: config, db, decay, embeddings, extract, ingest, recall, retrieval, scope, validation, corrections, backup, guardrail_check, guardrail_promotion, consolidation, communities, code_graph, cli
- `hooks/` — 15 Claude Code hook scripts: session lifecycle, extraction, slash commands, guardrail enforcement
- `commands/` — 22 slash command definitions
- `dashboard/` — Web dashboard: FastAPI backend + Next.js frontend with per-scope filtering
- `install.sh` — Copies to `~/.claude/`, configures hooks and status line in settings.json

## Architecture
- **DuckDB** for storage (`~/.claude/memory/knowledge.duckdb`), 11 schema migrations
- **ONNX nomic-embed-text** for local embeddings (768-dim, ~3ms), Ollama fallback
- **Claude Sonnet** for knowledge extraction via tool_use
- **6-way parallel retrieval**: semantic, BM25, graph, temporal, path, code — fused via RRF
- **Temporal decay**: short/medium/long classes with exponential decay and session-based promotion
- **Project scoping**: git repo root, with auto-promotion when seen in 3+ projects

## Key Design Decisions
- Dedup via cosine similarity (threshold 0.92) — reinforce, don't duplicate
- Token budgets cap context injection (3000 session, 4000 per-prompt)
- Only short-term items can be auto-forgotten; medium/long persist
- Hooks degrade gracefully when Ollama is down (skip embedding-based features)
- Three extraction triggers (status line at 40/70/90%, PreCompact, SessionEnd) with per-session lock
- Extraction validation rejects bare URLs, meta-commentary, low-confidence items
- Correction detection auto-supersedes facts when user says "that's wrong"
- Guardrail enforcement via PostToolUse hook with git stash safety net
- Auto-snapshots on every session end (rotating last 5)

## /remember Prefixes
```
/remember <text>                    → project-scoped long-term fact
/remember global: <text>            → global fact (all projects)
/remember decision: <text>          → project-scoped decision
/remember global decision: <text>   → global decision
/remember guardrail: <text>         → guardrail (highest priority)
/remember procedure: <text>         → how-to procedure
/remember error: <err> -> <fix>     → error→solution pair
```
