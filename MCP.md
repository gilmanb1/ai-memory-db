# MCP Server Setup

The memory system includes an MCP (Model Context Protocol) server that exposes memory tools directly to Claude Code, enabling it to search, store, and check memories without going through hooks.

## Server File

`hooks/memory_mcp_server.py` — A stdio-based JSON-RPC 2.0 server implementing the MCP protocol.

## Tools Exposed

| Tool | Description |
|------|-------------|
| `memory_search` | Semantic search across facts, guardrails, procedures, error_solutions, and decisions |
| `memory_store` | Store a fact, decision, guardrail, procedure, or error_solution |
| `memory_guardrail` | Create a guardrail (convenience wrapper with warning/rationale/consequence fields) |
| `memory_check_file` | Get all memory items associated with a specific file path |

## Configuration

Add the MCP server to your Claude Code settings (`~/.claude/settings.json` for global, or `.claude/settings.json` for project-level):

```json
{
  "mcpServers": {
    "memory": {
      "command": "python3",
      "args": ["/Users/YOUR_USERNAME/.claude/hooks/memory_mcp_server.py"],
      "env": {}
    }
  }
}
```

After running `install.sh`, the server script is copied to `~/.claude/hooks/memory_mcp_server.py`. Update the path above to match your home directory.

## Prerequisites

Same as the main memory system:

- **Python 3** with `duckdb` package (installed via `uv` inline deps in the memory package)
- **Ollama** running locally with `nomic-embed-text` model pulled (`ollama pull nomic-embed-text`)
- **DuckDB database** initialized at `~/.claude/memory/knowledge.duckdb` (created automatically on first use)

## How It Works

1. Claude Code launches the server as a subprocess communicating over stdin/stdout
2. The server responds to `initialize`, `tools/list`, and `tools/call` JSON-RPC methods
3. Tool calls use the same `memory.db`, `memory.embeddings`, and `memory.config` modules as the hook system
4. Project scope is resolved from the `CWD` environment variable (falls back to `__global__`)
5. All logging goes to stderr (visible in Claude Code's MCP server logs)

## Install

The `install.sh` script copies the MCP server file alongside the other hooks. To enable it, you must manually add the `mcpServers` configuration to your settings file (see Configuration above).

## Verifying

After adding the configuration, restart Claude Code. The memory tools (`memory_search`, `memory_store`, `memory_guardrail`, `memory_check_file`) should appear in the available tools list.
