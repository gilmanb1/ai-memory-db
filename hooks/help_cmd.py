#!/usr/bin/env python3
"""
help_cmd.py — Show all memory system commands with usage and examples.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if (_project_root / "memory" / "__init__.py").exists():
    sys.path.insert(0, str(_project_root))
else:
    sys.path.insert(0, str(Path.home() / ".claude"))


COMMANDS = [
    {
        "category": "Store Knowledge",
        "commands": [
            {
                "name": "/remember <text>",
                "desc": "Store a fact in the current project scope",
                "examples": [
                    '/remember The API uses gRPC for inter-service communication',
                    '/remember global: My name is Ben',
                    '/remember decision: Use DuckDB over PostgreSQL for local storage',
                    '/remember guardrail: Never log raw credit card numbers — PCI violation',
                    '/remember procedure: Deploy: 1. Run tests 2. Build image 3. Push 4. Rollout',
                    '/remember error: ImportError onnxruntime -> pip install onnxruntime-silicon',
                ],
            },
            {
                "name": "/forget <search text>",
                "desc": "Search memories and soft-delete one",
                "examples": ['/forget old database config', '/forget-confirm abc123def456 facts'],
            },
        ],
    },
    {
        "category": "Search & Retrieve",
        "commands": [
            {
                "name": "/knowledge <topic>",
                "desc": "Cross-type search: facts, decisions, guardrails, entities, relationships, error solutions",
                "examples": ['/knowledge DuckDB concurrency', '/knowledge authentication'],
            },
            {
                "name": "/search-memory <query>",
                "desc": "Semantic search with type and scope filters",
                "examples": [
                    '/search-memory retry logic',
                    '/search-memory --type guardrails deployment',
                    '/search-memory --scope /home/user/project auth',
                ],
            },
            {
                "name": "/reflect <question>",
                "desc": "Agentic Q&A — searches observations then facts, synthesizes an answer (up to 6 iterations)",
                "examples": ['/reflect How does the memory system handle concurrent writes?'],
            },
            {
                "name": "/recalled",
                "desc": "Show what was injected as context for the last prompt (facts, guardrails, procedures, etc.)",
                "examples": ['/recalled'],
            },
            {
                "name": "/session-learned [session_id]",
                "desc": "Show what was extracted from a session. Without ID, shows the most recent",
                "examples": ['/session-learned', '/session-learned abc123'],
            },
        ],
    },
    {
        "category": "Quality & Management",
        "commands": [
            {
                "name": "/review",
                "desc": "List flagged extraction items. Approve or reject questionable facts",
                "examples": ['/review', '/review approve abc123def456', '/review reject abc123def456'],
            },
            {
                "name": "/audit-memory",
                "desc": "Quality report: stale facts, orphaned entities, contradictions, review backlog, scope imbalances",
                "examples": ['/audit-memory'],
            },
            {
                "name": "/memory-health",
                "desc": "System health: Ollama, ONNX, DB locks, snapshots, API key, embeddings, disk usage",
                "examples": ['/memory-health'],
            },
        ],
    },
    {
        "category": "Inspect",
        "commands": [
            {"name": "/memories", "desc": "Database statistics (counts by type)", "examples": ['/memories']},
            {"name": "/facts [--class long] [--limit 20]", "desc": "List stored facts, optionally filter by temporal class", "examples": ['/facts', '/facts --class long --limit 10']},
            {"name": "/decisions [--limit N]", "desc": "List stored decisions", "examples": ['/decisions']},
            {"name": "/entities [--limit N]", "desc": "List known entities by mention count", "examples": ['/entities']},
            {"name": "/relationships", "desc": "Show entity relationship graph", "examples": ['/relationships']},
            {"name": "/sessions [--limit N]", "desc": "List recorded sessions with summaries", "examples": ['/sessions', '/sessions --limit 5']},
            {"name": "/scopes", "desc": "List all project scopes and item counts", "examples": ['/scopes']},
        ],
    },
    {
        "category": "Backup & Recovery",
        "commands": [
            {"name": "/snapshots", "desc": "List available DB snapshots with size and date", "examples": ['/snapshots']},
            {"name": "/export-memory [--output path] [--scope X]", "desc": "Export memory to portable JSON file", "examples": ['/export-memory', '/export-memory --output ~/backup.json --scope /home/user/project']},
            {"name": "/import-memory <path>", "desc": "Import memory from a JSON export file", "examples": ['/import-memory ~/backup.json']},
            {"name": "/restore-memory <snapshot>", "desc": "Restore DB from a snapshot (auto-creates safety backup first)", "examples": ['/restore-memory knowledge_20260327_143000_session_end.duckdb']},
        ],
    },
]


def main() -> None:
    topic = os.environ.get("MEMORY_TEXT", "").strip().lower()

    # If a specific command was requested, show detailed help for it
    if topic:
        for cat in COMMANDS:
            for cmd in cat["commands"]:
                cmd_name = cmd["name"].split()[0].lstrip("/")
                if topic.lstrip("/") in cmd_name or cmd_name in topic:
                    print(f"## /{cmd_name}\n")
                    print(f"{cmd['desc']}\n")
                    if cmd.get("examples"):
                        print("**Examples:**")
                        for ex in cmd["examples"]:
                            print(f"  {ex}")
                    return
        print(f'No command matching "{topic}". Run /memory-help for full list.')
        return

    # Show full help
    print("## Memory System Commands\n")

    for cat in COMMANDS:
        print(f"### {cat['category']}\n")
        for cmd in cat["commands"]:
            print(f"**`{cmd['name']}`**")
            print(f"  {cmd['desc']}")
            if cmd.get("examples"):
                print(f"  Example: `{cmd['examples'][0]}`")
            print()

    print("---")
    print("For detailed help on a specific command: `/memory-help <command>`")
    print(f"Total: {sum(len(c['commands']) for c in COMMANDS)} commands across {len(COMMANDS)} categories")


if __name__ == "__main__":
    main()
