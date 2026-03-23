#!/usr/bin/env python3
"""
remember_cmd.py — Entry point for the /remember custom slash command.

Reads the text from the MEMORY_TEXT environment variable, delegates to
user_prompt_submit.py for DuckDB storage, and routes to auto-memory
(markdown files) when the text matches behavioral/profile/reference/project
patterns.

Set MEMORY_DEBUG=0 to suppress routing explanations.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Add project root so we can import the memory package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_auto_memory_dir(cwd: str) -> Path:
    """Derive the auto-memory directory from the cwd, matching Claude Code's convention."""
    # Claude Code uses: ~/.claude/projects/<cwd-with-slashes-as-dashes>/memory/
    slug = cwd.replace("/", "-")
    return Path.home() / ".claude" / "projects" / slug / "memory"


def _strip_remember_prefixes(text: str) -> tuple[str, bool, bool]:
    """Strip global:/decision: prefixes, return (clean_text, is_global, is_decision)."""
    is_global = False
    is_decision = False
    prefix_pattern = re.compile(r'^((?:global|decision)[:\s]+)+', re.IGNORECASE)
    match = prefix_pattern.match(text)
    if match:
        prefix = match.group(0).lower()
        is_global = "global" in prefix
        is_decision = "decision" in prefix
        text = text[match.end():].strip()
    return text, is_global, is_decision


def main() -> None:
    text = os.environ.get("MEMORY_TEXT", "").strip()
    if not text:
        print(
            "Usage: /remember <text>\n"
            "Prefixes: global:, decision:, 'global decision:'\n"
            "\n"
            "Routes automatically to:\n"
            "  - Auto-memory (always visible) for preferences, profile, references, project context\n"
            "  - DuckDB (semantic search) for technical facts and decisions\n"
            "  - Both systems when appropriate\n"
            "\n"
            "Set MEMORY_DEBUG=0 to hide routing info."
        )
        return

    debug = os.environ.get("MEMORY_DEBUG", "1") != "0"

    # ── Route the memory ──────────────────────────────────────────────────
    from memory.routing import classify_memory, write_auto_memory

    # Strip prefixes before classifying (so "global: I prefer X" classifies on "I prefer X")
    clean_text, is_global, is_decision = _strip_remember_prefixes(text)
    classification = classify_memory(clean_text)

    # ── Store in DuckDB (always) ──────────────────────────────────────────
    hook = Path.home() / ".claude" / "hooks" / "user_prompt_submit.py"
    if not hook.exists():
        print(f"Error: hook not found at {hook}", file=sys.stderr)
        sys.exit(1)

    payload = json.dumps({
        "prompt": f"/remember {text}",
        "cwd": os.getcwd(),
        "session_id": "manual",
    })

    result = subprocess.run(
        ["uv", "run", "--script", str(hook)],
        input=payload,
        capture_output=True,
        text=True,
    )

    duckdb_output = ""
    if result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            duckdb_output = data.get("additionalContext", result.stdout)
        except json.JSONDecodeError:
            duckdb_output = result.stdout

    # Surface the [memory] status line from stderr
    for line in result.stderr.splitlines():
        if line.startswith("[memory]"):
            print(line, file=sys.stderr)
            break

    if result.returncode != 0 and not result.stdout.strip():
        print(f"Failed (exit {result.returncode}): {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)

    # ── Store in auto-memory if routed there ──────────────────────────────
    auto_output = ""
    if classification["route"] == "both" and classification["auto_type"]:
        try:
            cwd = os.getcwd()
            memory_dir = _resolve_auto_memory_dir(cwd)
            memory_dir.mkdir(parents=True, exist_ok=True)
            # Ensure MEMORY.md exists
            index_path = memory_dir / "MEMORY.md"
            if not index_path.exists():
                index_path.write_text("# Memory Index\n\n")
            filepath = write_auto_memory(
                text=clean_text,
                auto_type=classification["auto_type"],
                memory_dir=memory_dir,
            )
            auto_output = f"- **Auto-memory:** {classification['auto_type']} → `{filepath.name}`"
        except Exception as exc:
            auto_output = f"- **Auto-memory:** failed ({exc})"

    # ── Print combined output ─────────────────────────────────────────────
    print(duckdb_output)
    if auto_output:
        print(auto_output)
    if debug:
        print(f"\n**Routing:** {classification['reason']}")
        print(f"**Destination:** {classification['route']}")


if __name__ == "__main__":
    main()
