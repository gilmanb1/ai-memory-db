#!/usr/bin/env python3
"""
audit_cmd.py — Analyze the memory database for quality issues.

Checks for: stale facts, contradictions, low-recall items, orphaned entities,
review queue backlog, scope imbalances, and data freshness.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if (_project_root / "memory" / "__init__.py").exists():
    sys.path.insert(0, str(_project_root))
else:
    sys.path.insert(0, str(Path.home() / ".claude"))


def main() -> None:
    from memory import db
    from memory.config import DB_PATH, DEDUP_THRESHOLD, FORGET_THRESHOLD

    if not DB_PATH.exists():
        print("No database found. Start using Claude Code to build your knowledge base.")
        return

    conn = db.get_connection(read_only=True)
    try:
        now = datetime.now(timezone.utc)
        issues = []
        stats = {}

        # ── 1. Overall stats ─────────────────────────────────────────────
        s = db.get_stats(conn)
        total_facts = s["facts"]["total"]
        stats["total_facts"] = total_facts
        stats["total_entities"] = s["entities"]["total"]
        stats["total_decisions"] = s["decisions"]["total"]

        print("## Memory Audit Report\n")
        print(f"Database: {DB_PATH}")
        print(f"Size: {DB_PATH.stat().st_size / 1024:.1f} KB")
        print(f"Total active items: {total_facts} facts, {s['entities']['total']} entities, "
              f"{s['decisions']['total']} decisions, {s['relationships']['total']} relationships")
        print()

        # ── 2. Stale facts (low decay score, haven't been seen recently) ─
        try:
            stale = conn.execute("""
                SELECT COUNT(*) FROM facts
                WHERE is_active = TRUE AND decay_score < 0.3
                  AND last_seen_at < ?
            """, [now - timedelta(days=30)]).fetchone()[0]
            if stale > 0:
                issues.append(f"WARNING: {stale} facts have low decay scores (<0.3) and haven't been seen in 30+ days")
                # Show the worst ones
                worst = conn.execute("""
                    SELECT id, text, decay_score, temporal_class, last_seen_at FROM facts
                    WHERE is_active = TRUE AND decay_score < 0.3
                    ORDER BY decay_score ASC LIMIT 5
                """).fetchall()
                for fid, text, ds, tc, ls in worst:
                    issues.append(f"  [{tc} {ds:.2f}] {text[:80]}  (last seen: {str(ls)[:10]})")
        except Exception:
            pass

        # ── 3. Low-recall items (never recalled, may be irrelevant) ──────
        try:
            never_recalled = conn.execute("""
                SELECT COUNT(*) FROM facts
                WHERE is_active = TRUE AND times_recalled = 0
                  AND created_at < ?
            """, [now - timedelta(days=7)]).fetchone()[0]
            if never_recalled > 5:
                issues.append(f"INFO: {never_recalled} facts have never been recalled (created 7+ days ago)")
        except Exception:
            pass

        # ── 4. Orphaned entities (no relationships, no fact links) ───────
        try:
            orphaned = conn.execute("""
                SELECT e.name FROM entities e
                WHERE e.is_active = TRUE
                  AND e.name NOT IN (
                    SELECT from_entity FROM relationships WHERE is_active = TRUE
                    UNION SELECT to_entity FROM relationships WHERE is_active = TRUE
                  )
                  AND e.name NOT IN (
                    SELECT entity_name FROM fact_entity_links
                  )
            """).fetchall()
            if orphaned:
                names = [r[0] for r in orphaned[:10]]
                issues.append(f"INFO: {len(orphaned)} orphaned entities (no relationships or fact links): {', '.join(names)}")
        except Exception:
            pass

        # ── 5. Review queue backlog ──────────────────────────────────────
        try:
            pending = conn.execute(
                "SELECT COUNT(*) FROM review_queue WHERE status = 'pending'"
            ).fetchone()[0]
            if pending > 0:
                issues.append(f"ACTION: {pending} items in review queue need attention. Run /review")
        except Exception:
            pass

        # ── 6. Scope imbalance ───────────────────────────────────────────
        try:
            scopes = conn.execute("""
                SELECT scope, COUNT(*) as cnt FROM facts
                WHERE is_active = TRUE GROUP BY scope ORDER BY cnt DESC
            """).fetchall()
            if len(scopes) > 1:
                max_count = scopes[0][1]
                for scope, count in scopes[1:]:
                    if count < max_count * 0.1:
                        from memory.scope import scope_display_name
                        issues.append(f"INFO: Scope '{scope_display_name(scope)}' has only {count} facts ({count*100//max_count}% of largest scope)")
        except Exception:
            pass

        # ── 7. Data freshness ────────────────────────────────────────────
        try:
            latest = conn.execute(
                "SELECT MAX(created_at) FROM facts WHERE is_active = TRUE"
            ).fetchone()[0]
            if latest:
                if isinstance(latest, str):
                    latest = datetime.fromisoformat(latest)
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                age = now - latest
                if age > timedelta(days=7):
                    issues.append(f"WARNING: No new facts in {age.days} days. Last extraction: {str(latest)[:10]}")
        except Exception:
            pass

        # ── 8. Inactive/superseded items ─────────────────────────────────
        try:
            inactive = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE is_active = FALSE"
            ).fetchone()[0]
            superseded = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE superseded_by IS NOT NULL AND superseded_by != ''"
            ).fetchone()[0]
            if inactive > 0:
                issues.append(f"INFO: {inactive} inactive facts ({superseded} superseded). Consider running consolidation.")
        except Exception:
            pass

        # ── 9. Embedding coverage ────────────────────────────────────────
        try:
            no_emb = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE is_active = TRUE AND embedding IS NULL"
            ).fetchone()[0]
            if no_emb > 0:
                issues.append(f"WARNING: {no_emb} active facts have no embeddings. Semantic search won't find them.")
        except Exception:
            pass

        # ── 10. Potential contradictions (high similarity, different text)
        try:
            # Quick check: pairs with similarity 0.85-0.93
            from memory.config import COHERENCE_SIMILARITY_LOW, COHERENCE_SIMILARITY_HIGH
            rows = conn.execute("""
                SELECT id, text, embedding FROM facts
                WHERE is_active = TRUE AND embedding IS NOT NULL
                ORDER BY importance DESC LIMIT 50
            """).fetchall()
            contradiction_count = 0
            for i in range(len(rows)):
                for j in range(i + 1, min(len(rows), i + 10)):
                    if rows[i][2] and rows[j][2]:
                        dot = sum(a * b for a, b in zip(rows[i][2], rows[j][2]))
                        na = sum(a * a for a in rows[i][2]) ** 0.5
                        nb = sum(b * b for b in rows[j][2]) ** 0.5
                        if na > 0 and nb > 0:
                            sim = dot / (na * nb)
                            if COHERENCE_SIMILARITY_LOW <= sim <= COHERENCE_SIMILARITY_HIGH:
                                contradiction_count += 1
            if contradiction_count > 0:
                issues.append(f"WARNING: {contradiction_count} potential contradiction pairs detected (similarity 0.85-0.93)")
        except Exception:
            pass

        # ── Output ───────────────────────────────────────────────────────
        if not issues:
            print("### No Issues Found")
            print("The knowledge base looks healthy.")
        else:
            actions = [i for i in issues if i.startswith("ACTION")]
            warnings = [i for i in issues if i.startswith("WARNING")]
            infos = [i for i in issues if i.startswith("INFO")]

            if actions:
                print("### Action Required")
                for a in actions:
                    print(f"  {a}")
                print()
            if warnings:
                print("### Warnings")
                for w in warnings:
                    print(f"  {w}")
                print()
            if infos:
                print("### Information")
                for i in infos:
                    print(f"  {i}")
                print()

        print("---")
        print(f"Audit complete. {len(issues)} issue(s) found.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
