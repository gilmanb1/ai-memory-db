#!/usr/bin/env python3
"""
seed_demo_db.py — Create a demo DuckDB with the realistic test corpus.

Generates a self-contained knowledge.duckdb that the dashboard can serve
without any external dependencies (no Ollama, no API key needed).
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import memory.config as _cfg

DEMO_DB = Path(__file__).parent / "knowledge.duckdb"


def main():
    # Point config at the demo DB
    _cfg.DB_PATH = DEMO_DB

    from memory import db
    from test_memory import _mock_embed, _noop_decay

    # Remove old demo DB if exists
    if DEMO_DB.exists():
        DEMO_DB.unlink()

    # Build the corpus
    import test_corpus
    test_corpus.set_helpers(_mock_embed, _noop_decay)

    conn = db.get_connection(db_path=str(DEMO_DB))
    meta = test_corpus.build_corpus(conn, db)
    conn.close()

    # Print summary
    conn = db.get_connection(read_only=True, db_path=str(DEMO_DB))
    stats = db.get_stats(conn)
    conn.close()

    print(f"Demo database created: {DEMO_DB}")
    print(f"Size: {DEMO_DB.stat().st_size / 1024:.0f} KB")
    print(f"Facts: {stats['facts']['total']}")
    print(f"Entities: {stats['entities']['total']}")
    print(f"Relationships: {stats['relationships']['total']}")
    print(f"Decisions: {stats['decisions']['total']}")
    print(f"Guardrails: {stats.get('guardrails', {}).get('total', 0)}")
    print(f"Procedures: {stats.get('procedures', {}).get('total', 0)}")
    print(f"Error Solutions: {stats.get('error_solutions', {}).get('total', 0)}")
    print(f"Sessions: {stats['sessions']['total']}")


if __name__ == "__main__":
    main()
