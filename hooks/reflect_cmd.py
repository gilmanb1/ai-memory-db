#!/usr/bin/env python3
"""
reflect_cmd.py — Entry point for the /reflect custom slash command.

Reads the question from MEMORY_TEXT, runs the agentic reflect loop,
and prints the synthesized answer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root so we can import the memory package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    question = os.environ.get("MEMORY_TEXT", "").strip()
    if not question:
        print(
            "Usage: /reflect <question>\n\n"
            "Ask a question and get a synthesized answer from memory.\n"
            "The reflect agent searches observations, then raw facts,\n"
            "then synthesizes an answer with source citations."
        )
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from memory.reflect import run_reflect
    from memory.scope import resolve_scope

    scope = resolve_scope(os.getcwd())

    print(f"Reflecting on: {question}", file=sys.stderr)
    result = run_reflect(question=question, api_key=api_key, scope=scope)

    if result.error and not result.answer:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    # Print answer
    print(f"## Reflect Answer\n")
    print(result.answer)

    # Print sources if available
    if result.sources:
        print(f"\n### Sources ({len(result.sources)} items)")
        for s in result.sources[:10]:
            sid = s.get("id", "?")[:8]
            text = s.get("text", "")[:100]
            print(f"- [{sid}...] {text}")

    # Print trace to stderr
    if result.tool_trace:
        print(f"\n[memory] Reflect: {result.iterations_used} iterations, "
              f"{len(result.tool_trace)} tool calls", file=sys.stderr)


if __name__ == "__main__":
    main()
