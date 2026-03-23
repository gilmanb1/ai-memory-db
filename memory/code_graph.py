"""
code_graph.py — Parse Python source files, extract symbols and imports,
store a dependency graph in DuckDB for impact analysis and navigation.

Uses the stdlib `ast` module for parsing. All file paths stored relative
to the repo root.
"""
from __future__ import annotations

import ast
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

from .config import GLOBAL_SCOPE

# ── Schema DDL ────────────────────────────────────────────────────────────────

CODE_SYMBOLS_DDL = """\
CREATE TABLE IF NOT EXISTS code_symbols (
    id              VARCHAR PRIMARY KEY,
    file_path       VARCHAR NOT NULL,
    symbol_name     VARCHAR NOT NULL,
    symbol_type     VARCHAR NOT NULL,
    line_number     INTEGER,
    signature       TEXT,
    docstring       TEXT,
    scope           VARCHAR DEFAULT '__global__',
    parsed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS code_symbols_unique
    ON code_symbols(file_path, symbol_name, symbol_type);
"""

CODE_DEPENDENCIES_DDL = """\
CREATE TABLE IF NOT EXISTS code_dependencies (
    id              VARCHAR PRIMARY KEY,
    from_file       VARCHAR NOT NULL,
    to_file         VARCHAR NOT NULL,
    import_name     VARCHAR,
    import_type     VARCHAR DEFAULT 'import',
    scope           VARCHAR DEFAULT '__global__',
    parsed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS code_deps_unique
    ON code_dependencies(from_file, to_file, import_name);
"""

SKIP_DIRS = {
    "venv", ".venv", "node_modules", ".git", "__pycache__",
    ".tox", "build", "dist", ".eggs",
}


# ── Table bootstrap ──────────────────────────────────────────────────────────

def ensure_code_graph_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create code_symbols and code_dependencies tables if they don't exist."""
    for ddl in (CODE_SYMBOLS_DDL, CODE_DEPENDENCIES_DDL):
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    """Return naive local datetime (matches DuckDB CURRENT_TIMESTAMP behavior)."""
    return datetime.now()


def _format_arguments(args: ast.arguments) -> str:
    """Format an ast.arguments node into a readable signature string."""
    parts: list[str] = []

    # positional-only
    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")

    # regular args
    for a in args.args:
        parts.append(a.arg)

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only
    for a in args.kwonlyargs:
        parts.append(a.arg)

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return f"({', '.join(parts)})"


def _scope_filter(scope: Optional[str]) -> tuple[str, list]:
    """Scope SQL fragment matching the pattern in db.py."""
    if scope is None:
        return ("", [])
    return ("AND (scope = ? OR scope = ?)", [scope, GLOBAL_SCOPE])


# ── Pure parsing (no DB required) ────────────────────────────────────────────

def parse_python_file(file_path: str) -> dict:
    """
    Parse a single .py file and extract symbols and imports.

    Returns {"symbols": [...], "imports": [...]} or {} on failure.
    """
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError):
        return {}

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return {}

    symbols: list[dict] = []
    imports: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({
                "name": node.name,
                "type": "function",
                "line": node.lineno,
                "signature": _format_arguments(node.args),
                "docstring": ast.get_docstring(node),
            })

        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(
                ast.unparse(b) if hasattr(ast, "unparse") else "..."
                for b in node.bases
            )
            symbols.append({
                "name": node.name,
                "type": "class",
                "line": node.lineno,
                "signature": f"({bases})" if bases else "()",
                "docstring": ast.get_docstring(node),
            })
            # Extract methods
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({
                        "name": f"{node.name}.{item.name}",
                        "type": "method",
                        "line": item.lineno,
                        "signature": _format_arguments(item.args),
                        "docstring": ast.get_docstring(item),
                    })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "names": [alias.asname or alias.name],
                    "type": "import",
                })

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Prepend dots for relative imports
            if node.level and node.level > 0:
                module = "." * node.level + module
            names = [alias.name for alias in node.names]
            imports.append({
                "module": module,
                "names": names,
                "type": "from_import",
            })

    return {"symbols": symbols, "imports": imports}


def resolve_import_path(
    import_module: str,
    from_file: str,
    repo_root: str,
) -> Optional[str]:
    """
    Convert an import like 'memory.db' to a relative file path like 'memory/db.py'.

    Handles absolute imports, relative imports (leading dots), and package imports.
    Returns None if the resolved file doesn't exist on disk.
    """
    root = Path(repo_root)

    # Relative import — dots indicate parent traversal from the importing file
    if import_module.startswith("."):
        dots = 0
        for ch in import_module:
            if ch == ".":
                dots += 1
            else:
                break
        remainder = import_module[dots:]

        from_dir = Path(from_file)
        # Go up `dots` levels from the file's directory
        base = from_dir.parent
        for _ in range(dots - 1):
            base = base.parent

        if remainder:
            parts = remainder.split(".")
            candidate = base / "/".join(parts)
        else:
            # bare relative import like "." — refers to the package __init__
            candidate = base
    else:
        # Absolute import
        parts = import_module.split(".")
        candidate = root / "/".join(parts)

    # Try as a module file first, then as a package
    as_file = candidate.with_suffix(".py")
    as_pkg = candidate / "__init__.py"

    if as_file.exists():
        try:
            return str(as_file.relative_to(root))
        except ValueError:
            return str(as_file)

    if as_pkg.exists():
        try:
            return str(as_pkg.relative_to(root))
        except ValueError:
            return str(as_pkg)

    return None


# ── Repo-wide parsing and storage ────────────────────────────────────────────

def parse_repo(
    repo_root: str,
    conn: duckdb.DuckDBPyConnection,
    scope: str,
    max_files: int = 1000,
) -> dict:
    """
    Parse all .py files in a repo, storing symbols and dependencies in DuckDB.

    Skips files that haven't changed since last parse (mtime-based).
    Returns stats dict.
    """
    ensure_code_graph_tables(conn)

    root = Path(repo_root)
    stats = {
        "files_scanned": 0,
        "files_parsed": 0,
        "symbols_found": 0,
        "dependencies_found": 0,
        "skipped_unchanged": 0,
    }

    py_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            if fname.endswith(".py"):
                py_files.append(Path(dirpath) / fname)
                if len(py_files) >= max_files:
                    break
        if len(py_files) >= max_files:
            break

    now = _now()

    for full_path in py_files:
        stats["files_scanned"] += 1
        try:
            rel_path = str(full_path.relative_to(root))
        except ValueError:
            continue

        # Check if file needs re-parsing
        try:
            file_mtime_ts = os.path.getmtime(full_path)
        except OSError:
            continue

        row = conn.execute(
            "SELECT MAX(parsed_at) FROM code_symbols WHERE file_path = ?",
            [rel_path],
        ).fetchone()
        last_parsed = row[0] if row and row[0] else None

        if last_parsed is not None:
            # Compare as naive local timestamps (DuckDB stores naive local)
            file_mtime_local = datetime.fromtimestamp(file_mtime_ts)
            if isinstance(last_parsed, datetime) and last_parsed.tzinfo is not None:
                last_parsed = last_parsed.replace(tzinfo=None)
            if file_mtime_local <= last_parsed:
                stats["skipped_unchanged"] += 1
                continue

        # Parse the file
        result = parse_python_file(str(full_path))
        if not result:
            continue
        stats["files_parsed"] += 1

        # Clear old data for this file
        conn.execute(
            "DELETE FROM code_symbols WHERE file_path = ?", [rel_path]
        )
        conn.execute(
            "DELETE FROM code_dependencies WHERE from_file = ?", [rel_path]
        )

        # Store symbols
        for sym in result.get("symbols", []):
            stats["symbols_found"] += 1
            conn.execute(
                """INSERT INTO code_symbols
                   (id, file_path, symbol_name, symbol_type,
                    line_number, signature, docstring, scope, parsed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (file_path, symbol_name, symbol_type)
                   DO UPDATE SET line_number = excluded.line_number,
                                 signature  = excluded.signature,
                                 docstring  = excluded.docstring,
                                 scope      = excluded.scope,
                                 parsed_at  = excluded.parsed_at
                """,
                [
                    _uid(), rel_path, sym["name"], sym["type"],
                    sym["line"], sym["signature"], sym.get("docstring"),
                    scope, now,
                ],
            )

        # Store dependencies
        for imp in result.get("imports", []):
            resolved = resolve_import_path(imp["module"], rel_path, repo_root)
            if resolved is None:
                continue  # external / unresolvable import
            for name in imp["names"]:
                stats["dependencies_found"] += 1
                conn.execute(
                    """INSERT INTO code_dependencies
                       (id, from_file, to_file, import_name,
                        import_type, scope, parsed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (from_file, to_file, import_name)
                       DO UPDATE SET import_type = excluded.import_type,
                                     scope      = excluded.scope,
                                     parsed_at  = excluded.parsed_at
                    """,
                    [
                        _uid(), rel_path, resolved, name,
                        imp["type"], scope, now,
                    ],
                )

    return stats


# ── Query functions ──────────────────────────────────────────────────────────

def get_dependents(
    conn: duckdb.DuckDBPyConnection,
    file_path: str,
    scope: Optional[str] = None,
) -> list[dict]:
    """What files depend on (import from) this file?"""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(
        f"""SELECT from_file, import_name
            FROM code_dependencies
            WHERE to_file = ? {scope_sql}
            ORDER BY from_file""",
        [file_path] + scope_params,
    ).fetchall()
    return [{"from_file": r[0], "import_name": r[1]} for r in rows]


def get_dependencies(
    conn: duckdb.DuckDBPyConnection,
    file_path: str,
    scope: Optional[str] = None,
) -> list[dict]:
    """What does this file depend on?"""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(
        f"""SELECT to_file, import_name
            FROM code_dependencies
            WHERE from_file = ? {scope_sql}
            ORDER BY to_file""",
        [file_path] + scope_params,
    ).fetchall()
    return [{"to_file": r[0], "import_name": r[1]} for r in rows]


def get_file_symbols(
    conn: duckdb.DuckDBPyConnection,
    file_path: str,
) -> list[dict]:
    """All symbols defined in a file."""
    rows = conn.execute(
        """SELECT id, symbol_name, symbol_type, line_number,
                  signature, docstring, scope
           FROM code_symbols
           WHERE file_path = ?
           ORDER BY line_number""",
        [file_path],
    ).fetchall()
    return [
        {
            "id": r[0],
            "symbol_name": r[1],
            "symbol_type": r[2],
            "line_number": r[3],
            "signature": r[4],
            "docstring": r[5],
            "scope": r[6],
        }
        for r in rows
    ]


def search_symbol(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    scope: Optional[str] = None,
) -> list[dict]:
    """Find symbols by name (case-insensitive LIKE search)."""
    scope_sql, scope_params = _scope_filter(scope)
    rows = conn.execute(
        f"""SELECT id, file_path, symbol_name, symbol_type,
                   line_number, signature, docstring, scope
            FROM code_symbols
            WHERE LOWER(symbol_name) LIKE LOWER(?) {scope_sql}
            ORDER BY file_path, line_number""",
        [f"%{name}%"] + scope_params,
    ).fetchall()
    return [
        {
            "id": r[0],
            "file_path": r[1],
            "symbol_name": r[2],
            "symbol_type": r[3],
            "line_number": r[4],
            "signature": r[5],
            "docstring": r[6],
            "scope": r[7],
        }
        for r in rows
    ]


def get_impact_analysis(
    conn: duckdb.DuckDBPyConnection,
    file_path: str,
    scope: Optional[str] = None,
) -> dict:
    """
    'What breaks if I change this file?'

    Returns dependents, symbols defined in the file, and any guardrails
    linked to it.
    """
    dependents = get_dependents(conn, file_path, scope=scope)
    symbols = get_file_symbols(conn, file_path)

    # Try to pull guardrails if the function exists in db module
    guardrails: list[dict] = []
    try:
        from .db import get_guardrails_for_files
        guardrails = get_guardrails_for_files(conn, [file_path], scope=scope)
    except (ImportError, Exception):
        pass

    return {
        "file": file_path,
        "dependents": dependents,
        "symbols": symbols,
        "guardrails": guardrails,
    }
