"""
TypeScript / JavaScript parser using tree-sitter.

Handles .ts, .tsx, .js, .jsx files.
"""
from __future__ import annotations

import os
from pathlib import Path

import tree_sitter_typescript
import tree_sitter_javascript
from tree_sitter import Language, Parser

from ..code_graph import ParseResult


class TypeScriptParser:
    extensions = {".ts", ".tsx", ".js", ".jsx"}

    def __init__(self):
        self._ts_lang = Language(tree_sitter_typescript.language_typescript())
        self._tsx_lang = Language(tree_sitter_typescript.language_tsx())
        self._js_lang = Language(tree_sitter_javascript.language())
        self._parsers: dict[str, Parser] = {}

    def _get_parser(self, file_path: str) -> Parser:
        ext = Path(file_path).suffix.lower()
        if ext not in self._parsers:
            if ext == ".tsx":
                lang = self._tsx_lang
            elif ext in (".js", ".jsx"):
                lang = self._js_lang
            else:
                lang = self._ts_lang
            self._parsers[ext] = Parser(lang)
        return self._parsers[ext]

    # ── public API ────────────────────────────────────────────────────────

    def parse_file(self, file_path: str) -> ParseResult | None:
        """Parse a TS/JS file, returning symbols and imports."""
        try:
            source = Path(file_path).read_bytes()
        except (OSError, IOError):
            return None

        parser = self._get_parser(file_path)
        tree = parser.parse(source)
        if tree is None or tree.root_node is None:
            return None

        symbols: list[dict] = []
        imports: list[dict] = []

        self._walk(tree.root_node, source, symbols, imports, enclosing_class=None)

        return ParseResult(symbols=symbols, imports=imports)

    def resolve_import(
        self, import_module: str, from_file: str, repo_root: str
    ) -> str | None:
        """Resolve relative imports (./ and ../) to file paths within the repo."""
        if not import_module.startswith("."):
            return None  # bare / package import — can't resolve to a file

        root = Path(repo_root)
        from_dir = Path(from_file).parent

        # Resolve the relative path
        target = (root / from_dir / import_module).resolve()

        # Try appending common extensions / index files
        candidates = [
            target.with_suffix(ext)
            for ext in (".ts", ".tsx", ".js", ".jsx")
        ] + [
            target / f"index{ext}"
            for ext in (".ts", ".tsx", ".js", ".jsx")
        ]

        for candidate in candidates:
            if candidate.exists():
                try:
                    return str(candidate.relative_to(root))
                except ValueError:
                    return str(candidate)

        return None

    # ── tree walking ──────────────────────────────────────────────────────

    def _node_text(self, node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _find_child(self, node, type_name: str):
        """Return the first child of the given type, or None."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _walk(
        self,
        node,
        source: bytes,
        symbols: list[dict],
        imports: list[dict],
        enclosing_class: str | None,
    ) -> None:
        ntype = node.type

        # ── export_statement — unwrap and recurse into inner declaration ──
        if ntype == "export_statement":
            for child in node.children:
                self._walk(child, source, symbols, imports, enclosing_class)
            return

        # ── function_declaration ──────────────────────────────────────────
        if ntype == "function_declaration":
            name_node = self._find_child(node, "identifier")
            params_node = self._find_child(node, "formal_parameters")
            if name_node:
                symbols.append({
                    "name": self._node_text(name_node, source),
                    "type": "function",
                    "line": node.start_point[0] + 1,
                    "signature": self._node_text(params_node, source) if params_node else "()",
                    "docstring": None,
                })

        # ── class_declaration ─────────────────────────────────────────────
        elif ntype == "class_declaration":
            name_node = self._find_child(node, "identifier") or self._find_child(node, "type_identifier")
            cls_name = self._node_text(name_node, source) if name_node else "<anonymous>"
            symbols.append({
                "name": cls_name,
                "type": "class",
                "line": node.start_point[0] + 1,
                "signature": "()",
                "docstring": None,
            })
            # Recurse into class body for methods
            body = self._find_child(node, "class_body")
            if body:
                for child in body.children:
                    self._walk(child, source, symbols, imports, enclosing_class=cls_name)
            return

        # ── method_definition (inside class body) ─────────────────────────
        elif ntype == "method_definition":
            name_node = self._find_child(node, "property_identifier")
            params_node = self._find_child(node, "formal_parameters")
            if name_node:
                method_name = self._node_text(name_node, source)
                qualified = f"{enclosing_class}.{method_name}" if enclosing_class else method_name
                symbols.append({
                    "name": qualified,
                    "type": "method",
                    "line": node.start_point[0] + 1,
                    "signature": self._node_text(params_node, source) if params_node else "()",
                    "docstring": None,
                })

        # ── arrow_function assigned to const/let/var (variable_declarator) ─
        elif ntype in ("lexical_declaration", "variable_declaration"):
            for declarator in node.children:
                if declarator.type == "variable_declarator":
                    name_node = self._find_child(declarator, "identifier")
                    arrow = self._find_child(declarator, "arrow_function")
                    if name_node and arrow:
                        params_node = self._find_child(arrow, "formal_parameters")
                        symbols.append({
                            "name": self._node_text(name_node, source),
                            "type": "function",
                            "line": declarator.start_point[0] + 1,
                            "signature": self._node_text(params_node, source) if params_node else "()",
                            "docstring": None,
                        })

        # ── interface_declaration ─────────────────────────────────────────
        elif ntype == "interface_declaration":
            name_node = self._find_child(node, "type_identifier")
            if name_node:
                symbols.append({
                    "name": self._node_text(name_node, source),
                    "type": "interface",
                    "line": node.start_point[0] + 1,
                    "signature": None,
                    "docstring": None,
                })

        # ── type_alias_declaration ────────────────────────────────────────
        elif ntype == "type_alias_declaration":
            name_node = self._find_child(node, "type_identifier")
            if name_node:
                symbols.append({
                    "name": self._node_text(name_node, source),
                    "type": "type_alias",
                    "line": node.start_point[0] + 1,
                    "signature": None,
                    "docstring": None,
                })

        # ── import_statement ──────────────────────────────────────────────
        elif ntype == "import_statement":
            source_node = self._find_child(node, "string") or self._find_child(node, "string_literal")
            if source_node is None:
                # In some grammars the source is inside a deeper child
                for child in node.children:
                    if child.type in ("string", "string_literal"):
                        source_node = child
                        break
            if source_node:
                raw = self._node_text(source_node, source)
                module = raw.strip("'\"")
                # Collect imported names
                names: list[str] = []
                clause = self._find_child(node, "import_clause")
                if clause:
                    for child in clause.children:
                        if child.type == "identifier":
                            names.append(self._node_text(child, source))
                        elif child.type == "named_imports":
                            for spec in child.children:
                                if spec.type == "import_specifier":
                                    id_node = self._find_child(spec, "identifier")
                                    if id_node:
                                        names.append(self._node_text(id_node, source))
                        elif child.type == "namespace_import":
                            id_node = self._find_child(child, "identifier")
                            if id_node:
                                names.append(f"* as {self._node_text(id_node, source)}")
                if not names:
                    names = [module]
                imports.append({
                    "module": module,
                    "names": names,
                    "type": "import",
                })

        # ── generic recursion for nodes we didn't fully handle ────────────
        for child in node.children:
            if ntype not in (
                "export_statement", "class_declaration",
            ):
                self._walk(child, source, symbols, imports, enclosing_class)
