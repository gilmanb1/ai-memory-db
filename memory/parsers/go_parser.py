"""
Go parser using tree-sitter.

Handles .go files.
"""
from __future__ import annotations

import os
from pathlib import Path

import tree_sitter_go
from tree_sitter import Language, Parser

from ..code_graph import ParseResult


class GoParser:
    extensions = {".go"}

    def __init__(self):
        self._lang = Language(tree_sitter_go.language())
        self._parser = Parser(self._lang)

    # ── public API ────────────────────────────────────────────────────────

    def parse_file(self, file_path: str) -> ParseResult | None:
        """Parse a Go file, returning symbols and imports."""
        try:
            source = Path(file_path).read_bytes()
        except (OSError, IOError):
            return None

        tree = self._parser.parse(source)
        if tree is None or tree.root_node is None:
            return None

        symbols: list[dict] = []
        imports: list[dict] = []

        self._walk(tree.root_node, source, symbols, imports)

        return ParseResult(symbols=symbols, imports=imports)

    def resolve_import(
        self, import_module: str, from_file: str, repo_root: str
    ) -> str | None:
        """Resolve a Go import path to a directory within the repo, if local."""
        root = Path(repo_root)

        # Check if the import path matches a subdirectory within the repo
        # Go imports are package paths like "github.com/user/repo/pkg"
        # For local packages, try matching the tail of the import against repo dirs
        parts = import_module.split("/")
        for i in range(len(parts)):
            candidate = root / "/".join(parts[i:])
            if candidate.is_dir():
                try:
                    return str(candidate.relative_to(root))
                except ValueError:
                    return str(candidate)

        return None

    # ── helpers ───────────────────────────────────────────────────────────

    def _node_text(self, node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _find_child(self, node, type_name: str):
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _find_children(self, node, type_name: str):
        return [c for c in node.children if c.type == type_name]

    # ── tree walking ──────────────────────────────────────────────────────

    def _walk(
        self,
        node,
        source: bytes,
        symbols: list[dict],
        imports: list[dict],
    ) -> None:
        ntype = node.type

        # ── function_declaration ──────────────────────────────────────────
        if ntype == "function_declaration":
            name_node = self._find_child(node, "identifier")
            params_node = self._find_child(node, "parameter_list")
            if name_node:
                symbols.append({
                    "name": self._node_text(name_node, source),
                    "type": "function",
                    "line": node.start_point[0] + 1,
                    "signature": self._node_text(params_node, source) if params_node else "()",
                    "docstring": None,
                })

        # ── method_declaration ────────────────────────────────────────────
        elif ntype == "method_declaration":
            name_node = self._find_child(node, "field_identifier")
            params_node = self._find_child(node, "parameter_list")
            # Extract receiver type
            receiver_type = ""
            receiver = self._find_child(node, "parameter_list")
            if receiver:
                # The first parameter_list is the receiver in method_declaration
                # but tree-sitter-go uses a different structure
                pass
            # Walk receiver to find the type
            for child in node.children:
                if child.type == "parameter_list":
                    # First parameter_list is receiver, second is params
                    for param_child in child.children:
                        if param_child.type == "parameter_declaration":
                            type_node = (
                                self._find_child(param_child, "type_identifier")
                                or self._find_child(param_child, "pointer_type")
                            )
                            if type_node:
                                raw = self._node_text(type_node, source)
                                receiver_type = raw.lstrip("*")
                    break  # only first parameter_list

            if name_node:
                method_name = self._node_text(name_node, source)
                qualified = f"{receiver_type}.{method_name}" if receiver_type else method_name
                # Get actual params (second parameter_list)
                param_lists = self._find_children(node, "parameter_list")
                actual_params = param_lists[1] if len(param_lists) > 1 else None
                symbols.append({
                    "name": qualified,
                    "type": "method",
                    "line": node.start_point[0] + 1,
                    "signature": self._node_text(actual_params, source) if actual_params else "()",
                    "docstring": None,
                })

        # ── type_declaration (struct / interface) ─────────────────────────
        elif ntype == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name_node = self._find_child(child, "type_identifier")
                    if not name_node:
                        continue
                    name = self._node_text(name_node, source)
                    # Determine whether struct or interface
                    struct_node = self._find_child(child, "struct_type")
                    iface_node = self._find_child(child, "interface_type")
                    if struct_node:
                        symbols.append({
                            "name": name,
                            "type": "struct",
                            "line": child.start_point[0] + 1,
                            "signature": None,
                            "docstring": None,
                        })
                    elif iface_node:
                        symbols.append({
                            "name": name,
                            "type": "interface",
                            "line": child.start_point[0] + 1,
                            "signature": None,
                            "docstring": None,
                        })

        # ── import_declaration ────────────────────────────────────────────
        elif ntype == "import_declaration":
            # Single import: import "path"
            str_node = self._find_child(node, "import_spec")
            if str_node:
                self._extract_import_spec(str_node, source, imports)

            # Grouped: import ( ... )
            spec_list = self._find_child(node, "import_spec_list")
            if spec_list:
                for child in spec_list.children:
                    if child.type == "import_spec":
                        self._extract_import_spec(child, source, imports)

        # Recurse into children
        for child in node.children:
            self._walk(child, source, symbols, imports)

    def _extract_import_spec(
        self, spec_node, source: bytes, imports: list[dict]
    ) -> None:
        """Extract a single import spec (with optional alias)."""
        interpreted = self._find_child(spec_node, "interpreted_string_literal")
        if not interpreted:
            return
        raw = self._node_text(interpreted, source)
        module = raw.strip('"')
        # Optional alias
        alias_node = self._find_child(spec_node, "package_identifier") or self._find_child(spec_node, "dot") or self._find_child(spec_node, "blank_identifier")
        alias = self._node_text(alias_node, source) if alias_node else module.split("/")[-1]
        imports.append({
            "module": module,
            "names": [alias],
            "type": "import",
        })
