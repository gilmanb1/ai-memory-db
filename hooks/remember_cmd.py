#!/usr/bin/env python3
"""
remember_cmd.py — Entry point for the /remember custom slash command.

Reads the text from the MEMORY_TEXT environment variable, delegates to
user_prompt_submit.py (which owns all parsing and storage logic), and
prints the plain-text confirmation rather than the JSON hook envelope.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    text = os.environ.get("MEMORY_TEXT", "").strip()
    if not text:
        print(
            "Usage: /remember <text>\n"
            "Prefixes: global:, decision:, 'global decision:'"
        )
        return

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

    if result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            print(data.get("additionalContext", result.stdout))
        except json.JSONDecodeError:
            print(result.stdout)

    # Surface the [memory] status line from stderr
    for line in result.stderr.splitlines():
        if line.startswith("[memory]"):
            print(line, file=sys.stderr)
            break

    if result.returncode != 0 and not result.stdout.strip():
        print(f"Failed (exit {result.returncode}): {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
