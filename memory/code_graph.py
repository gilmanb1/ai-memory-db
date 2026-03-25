"""
code_graph.py — Parse source files, extract symbols and imports,
store a dependency graph in DuckDB for impact analysis and navigation.

Uses the stdlib `ast` module for Python parsing. Language-agnostic via
the LanguageParser protocol — register parsers for additional languages.
All file paths stored relative to the repo root.
"""
from __future__ import annotations

import ast
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import duckdb

from .config import GLOBAL_SCOPE, CODE_GRAPH_SKIP_DIRS, CODE_GRAPH_MAX_FILES


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    symbols: list[dict] = field(default_factory=list)
    imports: list[dict] = field(default_factory=list)


@runtime_checkable
class LanguageParser(Protocol):
    extensions: set[str]
    def parse_file(self, file_path: str) -> ParseResult | None: ...
    def resolve_import(self, import_module: str, from_file: str, repo_root: str) -> str | None: ...


# ── Parser registry ──────────────────────────────────────────────────────────

_PARSERS: dict[str, LanguageParser] = {}


def register_parser(parser: LanguageParser) -> None:
    for ext in parser.extensions:
        _PARSERS[ext] = parser


def get_parser(file_path: str) -> LanguageParser | None:
    ext = Path(file_path).suffix
    return _PARSERS.get(ext)


def get_registered_extensions() -> set[str]:
    return set(_PARSERS.keys())


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


def ensure_code_graph_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """No-op kept for backward compatibility — tables are created by migration 10."""
    pass


def _detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    lang_map = {
        ".py": "python", ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript",
        ".go": "go", ".rs": "rust",
    }
    return lang_map.get(ext, "unknown")


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


# ── Python parser class ─────────────────────────────────────────────────────

class PythonParser:
    extensions = {".py"}

    def parse_file(self, file_path: str) -> ParseResult | None:
        result = parse_python_file(file_path)
        if not result:
            return None
        return ParseResult(symbols=result["symbols"], imports=result["imports"])

    def resolve_import(self, import_module: str, from_file: str, repo_root: str) -> str | None:
        return resolve_import_path(import_module, from_file, repo_root)


# Register built-in parsers
register_parser(PythonParser())

# Try to register tree-sitter parsers
try:
    from .parsers import register_all_parsers
    register_all_parsers()
except ImportError:
    pass  # tree-sitter not installed


# ── Single-file parsing (for PostToolUse hook) ───────────────────────────────

def parse_single_file(
    file_path: str,
    repo_root: str,
    conn: duckdb.DuckDBPyConnection,
    scope: str,
) -> dict:
    """Parse a single file and update code graph. Returns stats."""
    root = Path(repo_root)
    full_path = Path(file_path)

    try:
        rel_path = str(full_path.relative_to(root))
    except ValueError:
        rel_path = str(full_path)

    parser = get_parser(file_path)
    if parser is None:
        return {"parsed": False, "reason": "no parser for extension"}

    result = parser.parse_file(str(full_path))
    if result is None:
        return {"parsed": False, "reason": "parse failed"}

    language = _detect_language(file_path)
    now = _now()

    # Clear old data
    conn.execute("DELETE FROM code_symbols WHERE file_path = ?", [rel_path])
    conn.execute("DELETE FROM code_dependencies WHERE from_file = ?", [rel_path])

    sym_count = 0
    dep_count = 0

    for sym in result.symbols:
        sym_count += 1
        conn.execute(
            """INSERT INTO code_symbols
               (id, file_path, symbol_name, symbol_type, language,
                line_number, signature, docstring, scope, parsed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (file_path, symbol_name, symbol_type)
               DO UPDATE SET line_number = excluded.line_number,
                             signature  = excluded.signature,
                             docstring  = excluded.docstring,
                             language   = excluded.language,
                             scope      = excluded.scope,
                             parsed_at  = excluded.parsed_at
            """,
            [_uid(), rel_path, sym["name"], sym["type"], language,
             sym.get("line"), sym.get("signature"), sym.get("docstring"),
             scope, now],
        )

    for imp in result.imports:
        resolved = parser.resolve_import(imp["module"], rel_path, repo_root)
        if resolved is None:
            continue
        for name in imp.get("names", [imp.get("module", "")]):
            dep_count += 1
            conn.execute(
                """INSERT INTO code_dependencies
                   (id, from_file, to_file, import_name,
                    import_type, language, scope, parsed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (from_file, to_file, import_name)
                   DO UPDATE SET import_type = excluded.import_type,
                                 language   = excluded.language,
                                 scope      = excluded.scope,
                                 parsed_at  = excluded.parsed_at
                """,
                [_uid(), rel_path, resolved, name,
                 imp.get("type", "import"), language, scope, now],
            )

    # Update file index
    try:
        mtime = os.path.getmtime(full_path)
    except OSError:
        mtime = 0.0
    conn.execute(
        """INSERT INTO code_file_index (file_path, language, mtime, symbol_count, scope, parsed_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT (file_path, scope)
           DO UPDATE SET language = excluded.language, mtime = excluded.mtime,
                         symbol_count = excluded.symbol_count, parsed_at = excluded.parsed_at
        """,
        [rel_path, language, mtime, sym_count, scope, now],
    )

    return {"parsed": True, "symbols": sym_count, "dependencies": dep_count}


# ── Repo-wide parsing and storage ────────────────────────────────────────────

def parse_repo(
    repo_root: str,
    conn: duckdb.DuckDBPyConnection,
    scope: str,
    max_files: int = 0,
) -> dict:
    """
    Parse all supported source files in a repo, storing symbols and
    dependencies in DuckDB.

    Skips files that haven't changed since last parse (mtime-based).
    Returns stats dict.
    """
    if max_files <= 0:
        max_files = CODE_GRAPH_MAX_FILES

    root = Path(repo_root)
    stats = {
        "files_scanned": 0, "files_parsed": 0,
        "symbols_found": 0, "dependencies_found": 0,
        "skipped_unchanged": 0,
    }

    registered_exts = get_registered_extensions()
    if not registered_exts:
        return stats

    # Collect files
    all_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in CODE_GRAPH_SKIP_DIRS]
        for fname in filenames:
            if Path(fname).suffix in registered_exts:
                all_files.append(Path(dirpath) / fname)
                if len(all_files) >= max_files:
                    break
        if len(all_files) >= max_files:
            break

    # Load file index for mtime comparison
    try:
        index_rows = conn.execute(
            "SELECT file_path, mtime FROM code_file_index WHERE scope = ? OR scope = ?",
            [scope, GLOBAL_SCOPE],
        ).fetchall()
        file_index = {r[0]: r[1] for r in index_rows}
    except Exception:
        file_index = {}

    now = _now()

    for full_path in all_files:
        stats["files_scanned"] += 1
        try:
            rel_path = str(full_path.relative_to(root))
        except ValueError:
            continue

        try:
            file_mtime = os.path.getmtime(full_path)
        except OSError:
            continue

        # Check mtime
        if rel_path in file_index and file_mtime <= file_index[rel_path]:
            stats["skipped_unchanged"] += 1
            continue

        result = parse_single_file(str(full_path), repo_root, conn, scope)
        if result.get("parsed"):
            stats["files_parsed"] += 1
            stats["symbols_found"] += result.get("symbols", 0)
            stats["dependencies_found"] += result.get("dependencies", 0)

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
