# Contributing to ai-memory-db

Thanks for your interest in contributing. This document covers the basics.

## Getting Started

```bash
git clone https://github.com/gilmanb1/ai-memory-db.git
cd ai-memory-db
pip install duckdb numpy tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript tree-sitter-go tree-sitter-rust
python3 test_memory.py   # Should pass 800+ tests
```

## Running Tests

```bash
python3 test_memory.py                          # Full suite
python3 -m unittest test_memory.TestDecay -v    # Single test class
```

Tests use mock embeddings (hash-based vectors) — no Ollama or Anthropic API required. All tests run against real in-memory DuckDB instances.

**Test methodology:** Red/green TDD. Write the test first, watch it fail, then implement. All new tests should follow the BDD naming pattern:

```python
def test_given_X_when_Y_then_Z(self):
```

## Project Structure

| Directory | What's there |
|-----------|-------------|
| `memory/` | Core Python package — this is where most logic lives |
| `hooks/` | Claude Code hook scripts (SessionStart, UserPromptSubmit, etc.) |
| `commands/` | Slash command `.md` definitions |
| `dashboard/backend/` | FastAPI API server |
| `dashboard/frontend/` | Next.js + shadcn/ui web dashboard |
| `bench/` | Benchmarking scripts and results |
| `docs/` | GitHub Pages site |

## Making Changes

1. **Create a branch** from `master`
2. **Write tests first** — add to `test_memory.py`
3. **Make your changes** — keep them focused
4. **Run the full test suite** — `python3 test_memory.py`
5. **Open a PR** with a clear description of what and why

## What Needs Help

- **More language parsers** — `memory/parsers/` has TypeScript, Go, Rust. Adding Java, C#, C++, Swift would be great.
- **Retrieval quality** — improving recall precision, especially for the graph traversal strategy.
- **Dashboard features** — the web dashboard at `dashboard/` has room for visualization improvements.
- **Documentation** — API docs, architecture deep-dives, tutorials.
- **Cross-platform testing** — Windows (WSL2), different Python versions, edge cases.

## Code Style

- Python: no strict linter enforced, but keep it readable. Type hints appreciated.
- TypeScript: follow the existing shadcn/ui patterns in the dashboard.
- Commit messages: imperative mood, concise. "Add X" not "Added X".

## Database Migrations

If you need to change the schema, add a new migration to the `MIGRATIONS` list in `memory/db.py`:

```python
(N, "Description of change", """
    ALTER TABLE ... ;
"""),
```

Migrations are versioned and run once automatically. Never modify existing migrations.

## Questions?

Open an issue or start a discussion on GitHub. There's no Discord/Slack yet — GitHub Issues is the right place.
