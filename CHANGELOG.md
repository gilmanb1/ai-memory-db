# Changelog

## v1.0.0 (2026-03-27)

First public release.

### Core System
- Automatic knowledge extraction from Claude Code conversations via hooks
- 8 knowledge types: facts, decisions, guardrails, procedures, error solutions, observations, entities, relationships
- DuckDB storage with HNSW vector indexes and 11 schema migrations
- ONNX Runtime (primary) and Ollama (fallback) for local 768-dim embeddings
- Temporal decay model: short (0.18/day), medium (0.04/day), long (0.007/day)
- Session-based promotion: short → medium at 3 sessions, medium → long at 7
- Project scoping via git repo root with auto-promotion to global at 3+ projects

### Retrieval Pipeline
- 6-way parallel retrieval: semantic, BM25, graph traversal, temporal, file-path, code graph
- Reciprocal Rank Fusion (RRF) for result merging
- Token-budgeted context injection: 3000 tokens session, 4000 per-prompt
- Embedding-based entity matching fallback for graph retrieval

### Extraction Pipeline
- Incremental chained extraction at 40%, 70%, 90% context usage
- Claude Sonnet extraction via tool_use for structured output
- Cross-pass dedup (0.85 threshold) and recall-aware dedup
- Narrative summaries stored per extraction pass
- Extraction validation: rejects bare URLs, meta-commentary, low-confidence items
- Correction detection: auto-supersedes facts on user correction

### Code Graph
- Multi-language AST parsing: Python (stdlib ast), TypeScript, JavaScript, Go, Rust (tree-sitter)
- Symbol extraction: functions, classes, methods, interfaces, structs, traits, enums
- Dependency tracking: imports, use statements, from...import
- Incremental mtime-based re-parsing
- PostToolUse hook for per-file updates on edit

### Web Dashboard
- FastAPI backend with 50+ REST endpoints
- Next.js + shadcn/ui frontend with 14 pages
- Full CRUD on all knowledge types
- Entity/relationship graph with Cytoscape.js (6 layout algorithms, focus mode, search, export)
- Code dependency graph with symbol detail panel
- Live polling (3-second refresh)
- Per-scope filtering

### Slash Commands (22)
- Storage: /remember (with global:, decision:, guardrail:, procedure:, error: prefixes)
- Retrieval: /knowledge, /search-memory, /reflect, /recalled, /session-learned
- Management: /forget, /review, /audit-memory, /memory-health
- Inspection: /memories, /facts, /decisions, /entities, /relationships, /sessions, /scopes
- Backup: /snapshots, /export-memory, /import-memory, /restore-memory

### System Hardening
- Guardrail enforcement via PostToolUse hook with git stash safety net
- Auto-snapshots on session end (rotating last 5)
- DuckDB concurrency: retry with exponential backoff
- Graceful degradation when Ollama/ONNX/API unavailable

### Tests
- 819 tests: unit, integration, corpus, concurrency
- Mock embeddings (no external deps required)
- Scaled corpus tests: 1-day, 1-week, 1-month, 1-year simulations
