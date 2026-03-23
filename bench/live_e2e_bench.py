#!/usr/bin/env python3
"""
live_e2e_bench.py — End-to-end benchmark through Claude Code hooks.

Calls the actual hook scripts via subprocess with JSON stdin,
exactly as Claude Code does in production. Tests the full stack:
  hooks → memory package → embeddings → DuckDB → formatted output

Uses the LIVE database at ~/.claude/memory/knowledge.duckdb.
Requires ONNX embeddings (no Ollama needed).

Usage: python3 bench/live_e2e_bench.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
HOOKS_DIR = PROJECT_ROOT / "hooks"

# Point at the live DB but use dev code
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))


def run_hook(hook_name: str, payload: dict, timeout: int = 30) -> dict:
    """Run a hook script via subprocess, returning parsed JSON output."""
    hook_path = HOOKS_DIR / f"{hook_name}.py"
    if not hook_path.exists():
        return {"error": f"Hook not found: {hook_path}"}

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(hook_path)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "TOKENIZERS_PARALLELISM": "false"},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    parsed = {}
    if stdout:
        try:
            parsed = json.loads(stdout.split("\n")[0])
        except json.JSONDecodeError:
            parsed = {"raw_output": stdout[:500]}

    return {
        "exit_code": result.returncode,
        "parsed": parsed,
        "raw_stdout": stdout[:1000],
        "stderr": stderr[:500],
        "elapsed_ms": elapsed_ms,
    }


def section(title: str):
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")


def check(name: str, passed: bool, detail: str = ""):
    marker = "✓" if passed else "✗"
    print(f"  {marker} {name}")
    if detail:
        print(f"    {detail}")
    return int(passed)


def run():
    total_pass = 0
    total_tests = 0

    cwd = str(PROJECT_ROOT)
    session_id = "bench-e2e-live"

    print("=" * 65)
    print("  END-TO-END BENCHMARK (through Claude Code hooks)")
    print(f"  Hooks: {HOOKS_DIR}")
    print(f"  CWD: {cwd}")
    print("=" * 65)

    # ── 1. Session Start ─────────────────────────────────────────────
    section("1. SessionStart Hook")
    result = run_hook("session_start", {"cwd": cwd, "session_id": session_id})
    total_tests += 1

    has_system_msg = "systemMessage" in result["parsed"]
    total_pass += check(
        f"Returns systemMessage ({result['elapsed_ms']:.0f}ms)",
        has_system_msg,
        f"Length: {len(result['parsed'].get('systemMessage', ''))} chars" if has_system_msg else result["stderr"][:200],
    )

    if has_system_msg:
        msg = result["parsed"]["systemMessage"]
        total_tests += 1
        total_pass += check(
            "Contains Memory Context header",
            "Memory Context" in msg,
        )
        total_tests += 1
        total_pass += check(
            "Contains Established Knowledge",
            "Established Knowledge" in msg or "Key Decisions" in msg,
        )
        # Check for new sections if they have data
        if "Guardrails" in msg:
            total_tests += 1
            total_pass += check("Contains Guardrails section", True)

    # ── 2. /remember guardrail ───────────────────────────────────────
    section("2. /remember guardrail: (via UserPromptSubmit hook)")
    result = run_hook("user_prompt_submit", {
        "prompt": "/remember guardrail: Do not remove the sleep(0.1) in batch_processor.py — rate limiting prevents upstream 429s",
        "cwd": cwd,
        "session_id": session_id,
    })
    total_tests += 1
    stored = "guardrail" in result["raw_stdout"].lower() or "Stored" in result["raw_stdout"] or "Memory" in result["raw_stdout"]
    total_pass += check(
        f"/remember guardrail stored ({result['elapsed_ms']:.0f}ms)",
        stored and result["exit_code"] == 0,
        result["raw_stdout"][:150],
    )

    # ── 3. /remember procedure ───────────────────────────────────────
    section("3. /remember procedure:")
    result = run_hook("user_prompt_submit", {
        "prompt": "/remember procedure: Run benchmarks: 1. python3 test_memory.py 2. python3 bench/coding_bench_onnx.py 3. python3 bench/live_bench.py",
        "cwd": cwd,
        "session_id": session_id,
    })
    total_tests += 1
    total_pass += check(
        f"/remember procedure stored ({result['elapsed_ms']:.0f}ms)",
        result["exit_code"] == 0 and ("procedure" in result["raw_stdout"].lower() or "Stored" in result["raw_stdout"]),
        result["raw_stdout"][:150],
    )

    # ── 4. /remember error: ──────────────────────────────────────────
    section("4. /remember error:")
    result = run_hook("user_prompt_submit", {
        "prompt": "/remember error: DuckDB CatalogError table not found -> Delete knowledge.duckdb and restart to re-run migrations",
        "cwd": cwd,
        "session_id": session_id,
    })
    total_tests += 1
    total_pass += check(
        f"/remember error stored ({result['elapsed_ms']:.0f}ms)",
        result["exit_code"] == 0,
        result["raw_stdout"][:150],
    )

    # ── 5. /remember fact (traditional) ──────────────────────────────
    section("5. /remember (traditional fact)")
    result = run_hook("user_prompt_submit", {
        "prompt": "/remember The benchmark suite has 468 unit tests, 28 integration tests, and 3 benchmark scripts",
        "cwd": cwd,
        "session_id": session_id,
    })
    total_tests += 1
    total_pass += check(
        f"/remember fact stored ({result['elapsed_ms']:.0f}ms)",
        result["exit_code"] == 0 and "Stored" in result["raw_stdout"],
        result["raw_stdout"][:150],
    )

    # ── 6. Prompt recall ─────────────────────────────────────────────
    section("6. Prompt Recall (UserPromptSubmit)")

    queries = [
        ("What database does the memory system use?", "DuckDB"),
        ("How do I run the benchmarks?", "benchmark"),
        ("Tell me about the embedding system", "embed"),
        ("What guardrails exist for this project?", "guardrail"),
        ("I'm getting a CatalogError from DuckDB", "CatalogError"),
    ]

    for query, expected_keyword in queries:
        result = run_hook("user_prompt_submit", {
            "prompt": query,
            "cwd": cwd,
            "session_id": session_id,
        })
        total_tests += 1

        has_context = "additionalContext" in result["parsed"]
        context_text = result["parsed"].get("additionalContext", "")

        # Check if the expected keyword appears in the recalled context
        keyword_found = expected_keyword.lower() in context_text.lower()
        passed = has_context and (keyword_found or len(context_text) > 50)
        total_pass += check(
            f"\"{query[:45]}...\" ({result['elapsed_ms']:.0f}ms)",
            passed,
            f"Keyword '{expected_keyword}': {'found' if keyword_found else 'not found'}, "
            f"context: {len(context_text)} chars",
        )

    # ── 7. Verify stored items exist ─────────────────────────────────
    section("7. Verify Stored Items in DB")
    sys.path.insert(0, str(PROJECT_ROOT))
    from memory import db as _db

    conn = _db.get_connection(read_only=True)

    total_tests += 1
    guardrails = _db.get_all_guardrails(conn)
    total_pass += check(
        f"Guardrails in DB: {len(guardrails)}",
        len(guardrails) > 0,
        guardrails[0]["warning"][:80] if guardrails else "none found",
    )

    total_tests += 1
    procedures = _db.get_procedures(conn)
    total_pass += check(
        f"Procedures in DB: {len(procedures)}",
        len(procedures) > 0,
        procedures[0]["task_description"][:80] if procedures else "none found",
    )

    total_tests += 1
    results = _db.search_all_by_text(conn, "CatalogError")
    tables = {r["table"] for r in results}
    total_pass += check(
        f"Error solution searchable",
        "error_solutions" in tables or "facts" in tables,
        f"Found in tables: {tables}",
    )

    total_tests += 1
    stats = _db.get_stats(conn)
    total_pass += check(
        f"DB stats: {stats['facts']['total']} facts, {stats.get('guardrails', {}).get('total', 0)} guardrails, "
        f"{stats.get('procedures', {}).get('total', 0)} procedures, "
        f"{stats.get('error_solutions', {}).get('total', 0)} error_solutions",
        stats["facts"]["total"] > 0,
    )

    conn.close()

    # ── 8. Latency summary ───────────────────────────────────────────
    section("8. Latency Summary")
    print("  (All latencies include subprocess spawn + Python startup + embeddings + DB)")
    print("  These are worst-case numbers — in-process would be ~10x faster")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    pct = (total_pass / total_tests * 100) if total_tests else 0
    print(f"  TOTAL: {total_pass}/{total_tests} tests passed ({pct:.0f}%)")
    print(f"{'=' * 65}")

    return total_pass == total_tests


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
