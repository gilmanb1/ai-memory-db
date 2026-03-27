"""
cli.py — CLI for inspecting the Claude Code memory database.

Usage:
    python -m memory stats
    python -m memory facts [--class short|medium|long] [--limit N]
    python -m memory search <query>
    python -m memory entities
    python -m memory decisions
    python -m memory relationships
    python -m memory sessions
    python -m memory observations [--limit N]
    python -m memory consolidate [--scope SCOPE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DB_PATH


NO_LIMIT = 2**31 - 1  # effectively unlimited


def _effective_limit(limit: int) -> int:
    """Convert 0 (meaning 'no limit') to a very large number for SQL LIMIT."""
    return NO_LIMIT if limit <= 0 else limit


def _require_db():
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}")
        print("The memory system has not run yet. Start a Claude Code session first.")
        sys.exit(1)


def cmd_stats(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)
    stats = db.get_stats(conn)
    conn.close()

    print(f"Database: {DB_PATH}")
    print(f"Size: {DB_PATH.stat().st_size / 1024:.1f} KB")
    print()
    print("Facts:")
    f = stats["facts"]
    print(f"  Active: {f['total']}  (long: {f['long']}, medium: {f['medium']}, short: {f['short']})")
    print(f"  Inactive (forgotten): {f['inactive']}")
    print()
    print(f"Ideas:          {stats['ideas']['total']} active, {stats['ideas']['inactive']} inactive")
    print(f"Entities:       {stats['entities']['total']}")
    print(f"Relationships:  {stats['relationships']['total']}")
    print(f"Decisions:      {stats['decisions']['total']}")
    print(f"Open questions: {stats['questions']['total']}  ({stats['questions']['resolved']} resolved)")
    print(f"Sessions:       {stats['sessions']['total']}")
    obs = stats.get("observations", {})
    if obs.get("total", 0) or obs.get("inactive", 0):
        print()
        print("Observations:")
        print(f"  Active: {obs.get('total', 0)}  (long: {obs.get('long', 0)}, medium: {obs.get('medium', 0)})")
        print(f"  Inactive (superseded): {obs.get('inactive', 0)}")


def cmd_facts(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)

    limit = _effective_limit(args.limit)
    if args.temporal_class:
        facts = db.get_facts_by_temporal(conn, args.temporal_class, limit)
    else:
        facts = []
        for tc in ["long", "medium", "short"]:
            facts.extend(db.get_facts_by_temporal(conn, tc, limit))
        facts.sort(key=lambda f: f.get("decay_score", 0), reverse=True)
        facts = facts[:limit]

    conn.close()

    if not facts:
        print("No facts found.")
        return

    for f in facts:
        tc = f.get("temporal_class", "?")
        score = f.get("decay_score", 0)
        cat = f.get("category", "")
        print(f"  [{tc:6s} {score:.2f}] [{cat}] {f['text']}")


def cmd_search(args):
    import os

    _require_db()
    from .embeddings import embed, is_ollama_available
    if not is_ollama_available():
        print("Ollama not available. Cannot perform semantic search.")
        print("Run: ollama pull nomic-embed-text && ollama serve")
        sys.exit(1)

    query_emb = embed(args.query)
    if query_emb is None:
        print("Failed to embed query.")
        sys.exit(1)

    from . import db
    from .scope import resolve_scope, scope_display_name

    scope = args.scope if args.scope is not None else resolve_scope(os.getcwd())
    limit = _effective_limit(args.limit)
    threshold = 0.3
    type_filter = args.type

    conn = db.get_connection(read_only=True)

    # Define all searchable types
    search_types = {}
    if not type_filter or type_filter == "facts":
        search_types["Facts"] = lambda: db.search_facts(conn, query_emb, limit=limit, threshold=threshold, scope=scope)
    if not type_filter or type_filter == "decisions":
        search_types["Decisions"] = lambda: db._vector_search(
            conn, "decisions", query_emb,
            "id, text, temporal_class, decay_score, scope",
            db._scope_filter(scope), limit, threshold,
        )
    if not type_filter or type_filter == "observations":
        search_types["Observations"] = lambda: db.search_observations(conn, query_emb, limit=limit, threshold=threshold, scope=scope)
    if not type_filter or type_filter == "guardrails":
        search_types["Guardrails"] = lambda: db.search_guardrails(conn, query_emb, limit=limit, threshold=threshold, scope=scope)
    if not type_filter or type_filter == "procedures":
        search_types["Procedures"] = lambda: db.search_procedures(conn, query_emb, limit=limit, threshold=threshold, scope=scope)
    if not type_filter or type_filter == "error_solutions":
        search_types["Error Solutions"] = lambda: db.search_error_solutions(conn, query_emb, limit=limit, threshold=threshold, scope=scope)

    # Run searches, collecting results by type
    all_results = {}
    for type_name, search_fn in search_types.items():
        try:
            results = search_fn()
            if results:
                all_results[type_name] = results
        except Exception:
            pass  # table may not exist yet

    conn.close()

    total_count = sum(len(v) for v in all_results.values())
    if total_count == 0:
        print(f"No results found for \"{args.query}\".")
        return

    print(f"Search: \"{args.query}\"\n")

    def _display_text(r, type_name):
        """Extract the primary display text from a result dict."""
        if type_name == "Guardrails":
            return r.get("warning", r.get("text", ""))
        if type_name == "Procedures":
            return r.get("task_description", r.get("text", ""))
        if type_name == "Error Solutions":
            return r.get("error_pattern", r.get("text", ""))
        return r.get("text", "")

    for type_name, results in all_results.items():
        count = len(results)
        label = "match" if count == 1 else "matches"
        print(f"### {type_name} ({count} {label})")
        for r in results:
            score = r.get("score", 0)
            tc = r.get("temporal_class", "")
            item_scope = r.get("scope", "")
            scope_label = scope_display_name(item_scope) if item_scope else ""
            tc_part = f" [{tc}]" if tc else ""
            print(f"  [{score:.2f}]{tc_part} {_display_text(r, type_name)} ({scope_label})")
        print()

    type_count = len(all_results)
    scope_label = scope_display_name(scope) if scope else "all"
    print(f"---\n{total_count} results across {type_count} {'type' if type_count == 1 else 'types'} | scope: {scope_label} + global")


def cmd_entities(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)
    entities = db.get_top_entities(conn, _effective_limit(args.limit))
    conn.close()

    if not entities:
        print("No entities found.")
        return

    for e in entities:
        print(f"  {e}")


def cmd_decisions(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)
    decisions = db.get_decisions(conn, _effective_limit(args.limit))
    conn.close()

    if not decisions:
        print("No decisions found.")
        return

    for d in decisions:
        tc = d.get("temporal_class", "?")
        print(f"  [{tc}] {d['text']}")


def cmd_relationships(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)
    rels = db.get_all_relationships(conn)
    conn.close()

    if not rels:
        print("No relationships found.")
        return

    for r in rels:
        strength = r.get("strength", 1.0)
        count = r.get("session_count", 1)
        print(f"  {r['from']} --[{r['rel_type']}]--> {r['to']}")
        print(f"    {r['description']}  (strength: {strength:.1f}, sessions: {count})")


def cmd_sessions(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)
    rows = conn.execute("""
        SELECT id, trigger, cwd, message_count, summary, created_at
        FROM sessions
        ORDER BY created_at DESC
        LIMIT ?
    """, [_effective_limit(args.limit)]).fetchall()
    conn.close()

    if not rows:
        print("No sessions found.")
        return

    for sid, trigger, cwd, msg_count, summary, created in rows:
        print(f"  {str(created)[:19]}  [{trigger}]  {msg_count} msgs  {sid[:12]}...")
        if summary:
            print(f"    {summary}")
        print()


def cmd_promote(args):
    _require_db()
    from . import db
    from .config import GLOBAL_SCOPE

    table = args.table
    valid_tables = ["facts", "ideas", "decisions", "entities", "relationships"]
    if table not in valid_tables:
        print(f"Invalid table. Must be one of: {', '.join(valid_tables)}")
        sys.exit(1)

    conn = db.get_connection()
    success = db.promote_to_global(conn, args.item_id, table)
    conn.close()

    if success:
        print(f"Promoted {table} item {args.item_id[:12]}... to global scope.")
    else:
        print(f"Item {args.item_id} not found in {table}.")
        sys.exit(1)


def cmd_scopes(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)
    rows = conn.execute("""
        SELECT scope, COUNT(*) as cnt FROM (
            SELECT scope FROM facts WHERE is_active = TRUE
            UNION ALL
            SELECT scope FROM ideas WHERE is_active = TRUE
            UNION ALL
            SELECT scope FROM decisions WHERE is_active = TRUE
        ) GROUP BY scope ORDER BY cnt DESC
    """).fetchall()
    conn.close()

    if not rows:
        print("No scoped items found.")
        return

    from .scope import scope_display_name
    for scope, count in rows:
        print(f"  {scope_display_name(scope):30s}  {count:>4} items  ({scope})")


def cmd_observations(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)

    limit = _effective_limit(args.limit)
    observations = []
    for tc in ["long", "medium"]:
        observations.extend(db.get_observations_by_temporal(conn, tc, limit))
    observations.sort(key=lambda o: o.get("proof_count", 0), reverse=True)
    observations = observations[:limit]

    conn.close()

    if not observations:
        print("No observations found.")
        return

    for o in observations:
        oid = o["id"][:8]
        tc = o.get("temporal_class", "?")
        proof = o.get("proof_count", 0)
        print(f"  [{oid}] [{tc}] {o['text']} (proof: {proof} facts)")


def cmd_session_learned(args):
    _require_db()
    from . import db
    conn = db.get_connection(read_only=True)

    # Resolve session id
    if args.session_id:
        # Allow prefix match
        rows = conn.execute(
            "SELECT id, trigger, cwd, message_count, summary, created_at "
            "FROM sessions WHERE id LIKE ? ORDER BY created_at DESC LIMIT 1",
            [args.session_id + "%"],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, trigger, cwd, message_count, summary, created_at "
            "FROM sessions ORDER BY created_at DESC LIMIT 1"
        ).fetchall()

    if not rows:
        print("No session found.")
        conn.close()
        sys.exit(1)

    sid, trigger, cwd, msg_count, summary, created = rows[0]

    # Derive scope display name from cwd
    scope_name = Path(cwd).name if cwd else "unknown"

    print(f"## Session: {sid}")
    print(f"  {str(created)[:19]}  [{trigger}]  {msg_count or '?'} msgs")
    print(f"  Scope: {scope_name}")
    if summary:
        print(f"  Summary: {summary}")
    print()

    limit = _effective_limit(args.limit)
    total = 0

    # Tables: (label, table_name, text_column)
    standard_tables = [
        ("Facts", "facts", "text"),
        ("Ideas", "ideas", "text"),
        ("Decisions", "decisions", "text"),
        ("Observations", "observations", "text"),
        ("Guardrails", "guardrails", "warning"),
        ("Procedures", "procedures", "task_description"),
        ("Error Solutions", "error_solutions", "error_pattern"),
    ]

    # Session narratives
    narratives = conn.execute(
        "SELECT pass_number, narrative, is_final FROM session_narratives "
        "WHERE session_id = ? ORDER BY pass_number",
        [sid],
    ).fetchall()
    if narratives:
        print("### Narrative")
        for pass_num, narrative, is_final in narratives:
            tag = " (final)" if is_final else ""
            print(f"  Pass {pass_num}{tag}: {narrative}")
        print()

    # Standard tables
    for label, table, text_col in standard_tables:
        try:
            items = conn.execute(
                f"SELECT id, {text_col}, temporal_class, created_at FROM {table} "
                f"WHERE source_session = ? AND is_active = TRUE "
                f"ORDER BY created_at LIMIT ?",
                [sid, limit],
            ).fetchall()
        except Exception:
            items = []

        count = len(items)
        total += count
        print(f"### {label} ({count} extracted)")
        for _, text, tc, _ in items:
            print(f"  [{tc}] {text}")
        print()

    # Entities — linked via fact_entity_links for facts from this session
    try:
        fact_ids = conn.execute(
            "SELECT id FROM facts WHERE source_session = ? AND is_active = TRUE",
            [sid],
        ).fetchall()
        entity_names = []
        if fact_ids:
            fid_list = [r[0] for r in fact_ids]
            placeholders = ", ".join("?" * len(fid_list))
            entity_names = [r[0] for r in conn.execute(
                f"SELECT DISTINCT entity_name FROM fact_entity_links WHERE fact_id IN ({placeholders})",
                fid_list,
            ).fetchall()]
    except Exception:
        entity_names = []

    total += len(entity_names)
    print(f"### Entities ({len(entity_names)} extracted)")
    if entity_names:
        print(f"  {', '.join(entity_names)}")
    print()

    # Relationships
    try:
        rels = conn.execute(
            "SELECT id, from_entity, to_entity, rel_type, description "
            "FROM relationships WHERE source_session = ?",
            [sid],
        ).fetchall()
    except Exception:
        rels = []

    total += len(rels)
    print(f"### Relationships ({len(rels)} extracted)")
    for _, from_e, to_e, rel_type, desc in rels:
        print(f"  {from_e} --[{rel_type}]--> {to_e}")
    print()

    conn.close()

    print("---")
    print(f"Total: {total} items extracted from session {sid}")


def cmd_consolidate(args):
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable is required.")
        sys.exit(1)

    _require_db()
    from . import db
    from .scope import resolve_scope
    from .consolidation import run_consolidation, run_semantic_forgetting

    scope = args.scope or resolve_scope(os.getcwd())
    conn = db.get_connection()

    print(f"Running consolidation for scope: {scope}")
    c_stats = run_consolidation(conn, api_key, scope)
    print(f"  Batches processed: {c_stats['batches']}")
    print(f"  Observations created: {c_stats['created']}")
    print(f"  Observations updated: {c_stats['updated']}")
    print(f"  Observations deleted: {c_stats['deleted']}")

    print()
    print("Running semantic forgetting...")
    f_stats = run_semantic_forgetting(conn, scope)
    print(f"  Pairs checked: {f_stats['pairs_checked']}")
    print(f"  Superseded: {f_stats['superseded']}")

    conn.close()
    print()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Claude Code Memory Inspector",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("stats", help="Show database statistics")

    p_facts = sub.add_parser("facts", help="List facts")
    p_facts.add_argument("--class", dest="temporal_class", choices=["short", "medium", "long"])
    p_facts.add_argument("--limit", type=int, default=0)

    p_search = sub.add_parser("search", help="Semantic search across all memory types")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--limit", type=int, default=0)
    p_search.add_argument("--scope", type=str, default=None, help="Project scope (default: resolve from cwd)")
    p_search.add_argument("--type", type=str, default=None,
                          choices=["facts", "decisions", "observations", "guardrails", "procedures", "error_solutions"],
                          help="Filter to a specific memory type")

    p_entities = sub.add_parser("entities", help="List known entities")
    p_entities.add_argument("--limit", type=int, default=0)

    p_decisions = sub.add_parser("decisions", help="List decisions")
    p_decisions.add_argument("--limit", type=int, default=0)

    sub.add_parser("relationships", help="List relationship graph")

    p_sessions = sub.add_parser("sessions", help="List stored sessions")
    p_sessions.add_argument("--limit", type=int, default=0)

    p_promote = sub.add_parser("promote", help="Promote an item to global scope")
    p_promote.add_argument("table", help="Table name: facts, ideas, decisions, entities, relationships")
    p_promote.add_argument("item_id", help="The item UUID to promote")

    sub.add_parser("scopes", help="List all project scopes and item counts")

    p_obs = sub.add_parser("observations", help="List synthesized observations")
    p_obs.add_argument("--limit", type=int, default=0)

    p_consolidate = sub.add_parser("consolidate", help="Manually trigger consolidation")
    p_consolidate.add_argument("--scope", type=str, default="", help="Project scope (default: resolve from cwd)")

    p_session_learned = sub.add_parser("session-learned", help="Show what was learned from a session")
    p_session_learned.add_argument("session_id", nargs="?", default="", help="Session ID (prefix ok). Default: most recent")
    p_session_learned.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "stats": cmd_stats,
        "facts": cmd_facts,
        "search": cmd_search,
        "entities": cmd_entities,
        "decisions": cmd_decisions,
        "relationships": cmd_relationships,
        "sessions": cmd_sessions,
        "promote": cmd_promote,
        "scopes": cmd_scopes,
        "observations": cmd_observations,
        "consolidate": cmd_consolidate,
        "session-learned": cmd_session_learned,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
