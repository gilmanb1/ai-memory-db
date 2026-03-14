#!/usr/bin/env python3
"""
forget_cmd.py — Entry point for the /forget custom slash command.

Reads the query from the MEMORY_TEXT environment variable, searches all
memory tables, displays matches, and prints them for the user to select.
When called with MEMORY_FORGET_ID and MEMORY_FORGET_TABLE, performs the
soft delete directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import the memory package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory import db


def main() -> None:
    # Mode 1: Direct soft-delete by ID + table
    forget_arg = os.environ.get("MEMORY_FORGET_ID", "").strip()
    if forget_arg and " " in forget_arg:
        parts = forget_arg.split(None, 1)
        forget_id = parts[0]
        forget_table = parts[1]
    elif forget_arg:
        forget_id = forget_arg
        forget_table = os.environ.get("MEMORY_FORGET_TABLE", "").strip()
    else:
        forget_id = ""
        forget_table = ""
    if forget_id and forget_table:
        conn = db.get_connection()
        try:
            # Get the text before deleting for confirmation
            results = db.search_all_by_id(conn, forget_id)
            if not results:
                print("No item found with that ID.")
                return
            item = results[0]
            if db.soft_delete(conn, forget_id, forget_table):
                print(f"Forgot: {item['text']}")
            else:
                print("Failed to forget item.")
        finally:
            conn.close()
        return

    # Mode 2: Search by text query
    query = os.environ.get("MEMORY_TEXT", "").strip()
    if not query:
        print(
            "Usage: /forget <search text>\n"
            "Searches all memory tables and lets you select which to forget."
        )
        return

    conn = db.get_connection()
    try:
        # Try exact ID match first
        id_results = db.search_all_by_id(conn, query)
        if id_results:
            results = id_results
        else:
            results = db.search_all_by_text(conn, query)

        if not results:
            print(f"No memories found matching: {query}")
            return

        # Print numbered list for user selection
        print(f"Found {len(results)} matching memories:\n")
        for i, item in enumerate(results, 1):
            text_preview = item["text"][:80] + ("..." if len(item["text"]) > 80 else "")
            print(f"  {i}. [{item['table']}] {text_preview}")
            print(f"     ID: {item['id']}")
            print()

        print("To forget a memory, run:")
        print(f'  /forget-confirm <number>')
        print()
        # Output structured data for the slash command to use
        print("---FORGET_DATA---")
        import json
        print(json.dumps(results))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
