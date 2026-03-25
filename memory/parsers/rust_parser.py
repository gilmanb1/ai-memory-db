"""
Rust parser using tree-sitter.

Handles .rs files.
"""
from __future__ import annotations

import os
from pathlib import Path

import tree_sitter_rust
from tree_sitter import Language, Parser

from ..code_graph import ParseResult


class RustParser:
    extensions = {".rs"}

    def __init__(self):
        self._lang = Language(tree_sitter_rust.language())
        self._parser = Parser(self._lang)

    # ── public API ────────────────────────────────────────────────────────

    def parse_file(self, file_path: str) -> ParseResult | None:
        """Parse a Rust file, returning symbols and imports."""
        try:
            source = Path(file_path).read_bytes()
        except (OSError, IOError):
            return None

        tree = self._parser.parse(source)
        if tree is None or tree.root_node is None:
            return None

        symbols: list[dict] = []
        imports: list[dict] = []

        self._walk(tree.root_node, source, symbols, imports, enclosing_type=None)

        return ParseResult(symbols=symbols, imports=imports)

    def resolve_import(
        self, import_module: str, from_file: str, repo_root: str
    ) -> str | None:
        """Resolve a Rust use path to a file within the repo.

        Maps crate-relative paths:
            crate::foo::bar  ->  src/foo/bar.rs  or  src/foo/bar/mod.rs
            self::foo        ->  relative to current file
            super::foo       ->  parent module
        """
        root = Path(repo_root)

        parts = import_module.split("::")
        if not parts:
            return None

        # Handle crate:: prefix
        if parts[0] == "crate":
            parts = parts[1:]
            base = root / "src"
        elif parts[0] == "self":
            parts = parts[1:]
            base = (root / from_file).parent
        elif parts[0] == "super":
            parts = parts[1:]
            base = (root / from_file).parent.parent
        else:
            # External crate — can't resolve to a local file
            return None

        if not parts:
            return None

        # Build candidate path from remaining parts
        rel = Path(*parts)
        candidates = [
            base / rel.with_suffix(".rs"),
            base / rel / "mod.rs",
        ]

        for candidate in candidates:
            if candidate.exists():
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
        enclosing_type: str | None,
    ) -> None:
        ntype = node.type

        # ── function_item ─────────────────────────────────────────────────
        if ntype == "function_item":
            name_node = self._find_child(node, "identifier")
            params_node = self._find_child(node, "parameters")
            if name_node:
                fn_name = self._node_text(name_node, source)
                if enclosing_type:
                    qualified = f"{enclosing_type}.{fn_name}"
                    sym_type = "method"
                else:
                    qualified = fn_name
                    sym_type = "function"
                symbols.append({
                    "name": qualified,
                    "type": sym_type,
                    "line": node.start_point[0] + 1,
                    "signature": self._node_text(params_node, source) if params_node else "()",
                    "docstring": None,
                })

        # ── impl_item ────────────────────────────────────────────────────
        elif ntype == "impl_item":
            # Get the type being implemented
            type_node = self._find_child(node, "type_identifier")
            impl_name = self._node_text(type_node, source) if type_node else None

            # If it's a trait impl (impl Trait for Type), get the Type
            if impl_name is None:
                # Try generic_type or scoped_type_identifier
                for child in node.children:
                    if child.type in ("type_identifier", "generic_type", "scoped_type_identifier"):
                        impl_name = self._node_text(child, source)
                        break

            # Walk the declaration_list for methods
            decl_list = self._find_child(node, "declaration_list")
            if decl_list:
                for child in decl_list.children:
                    self._walk(child, source, symbols, imports, enclosing_type=impl_name)
            return  # don't recurse normally — we handled the body

        # ── struct_item ───────────────────────────────────────────────────
        elif ntype == "struct_item":
            name_node = self._find_child(node, "type_identifier")
            if name_node:
                symbols.append({
                    "name": self._node_text(name_node, source),
                    "type": "struct",
                    "line": node.start_point[0] + 1,
                    "signature": None,
                    "docstring": None,
                })

        # ── trait_item ────────────────────────────────────────────────────
        elif ntype == "trait_item":
            name_node = self._find_child(node, "type_identifier")
            if name_node:
                symbols.append({
                    "name": self._node_text(name_node, source),
                    "type": "trait",
                    "line": node.start_point[0] + 1,
                    "signature": None,
                    "docstring": None,
                })

        # ── enum_item ─────────────────────────────────────────────────────
        elif ntype == "enum_item":
            name_node = self._find_child(node, "type_identifier")
            if name_node:
                symbols.append({
                    "name": self._node_text(name_node, source),
                    "type": "enum",
                    "line": node.start_point[0] + 1,
                    "signature": None,
                    "docstring": None,
                })

        # ── use_declaration ───────────────────────────────────────────────
        elif ntype == "use_declaration":
            self._extract_use(node, source, imports, prefix_parts=[])

        # Recurse into children (skip impl_item body — handled above)
        for child in node.children:
            if ntype != "impl_item":
                self._walk(child, source, symbols, imports, enclosing_type)

    def _extract_use(
        self, node, source: bytes, imports: list[dict], prefix_parts: list[str]
    ) -> None:
        """Extract use declarations, handling nested/grouped paths."""
        for child in node.children:
            if child.type == "use_as_clause":
                # use foo::bar as baz;
                path = self._find_child(child, "scoped_identifier") or self._find_child(child, "identifier")
                if path:
                    module = self._node_text(path, source)
                    imports.append({
                        "module": module,
                        "names": [module.split("::")[-1]],
                        "type": "use",
                    })
                return

            if child.type == "scoped_identifier":
                module = self._node_text(child, source)
                imports.append({
                    "module": module,
                    "names": [module.split("::")[-1]],
                    "type": "use",
                })
                return

            if child.type == "identifier":
                module = self._node_text(child, source)
                imports.append({
                    "module": module,
                    "names": [module],
                    "type": "use",
                })
                return

            if child.type == "use_wildcard":
                path_node = self._find_child(child, "scoped_identifier") or self._find_child(child, "identifier")
                module = self._node_text(path_node, source) if path_node else "*"
                imports.append({
                    "module": module + "::*",
                    "names": ["*"],
                    "type": "use",
                })
                return

            if child.type == "scoped_use_list":
                # e.g., std::{io, fs} or std::io::{Read, Write}
                path_node = self._find_child(child, "scoped_identifier") or self._find_child(child, "identifier")
                base = self._node_text(path_node, source) if path_node else ""

                use_list = self._find_child(child, "use_list")
                if use_list:
                    for item in use_list.children:
                        if item.type == "identifier":
                            name = self._node_text(item, source)
                            full_path = f"{base}::{name}" if base else name
                            imports.append({
                                "module": full_path,
                                "names": [name],
                                "type": "use",
                            })
                        elif item.type == "scoped_identifier":
                            name = self._node_text(item, source)
                            full_path = f"{base}::{name}" if base else name
                            imports.append({
                                "module": full_path,
                                "names": [name.split("::")[-1]],
                                "type": "use",
                            })
                        elif item.type == "use_as_clause":
                            path = self._find_child(item, "scoped_identifier") or self._find_child(item, "identifier")
                            if path:
                                name = self._node_text(path, source)
                                full_path = f"{base}::{name}" if base else name
                                imports.append({
                                    "module": full_path,
                                    "names": [name.split("::")[-1]],
                                    "type": "use",
                                })
                        elif item.type in ("scoped_use_list", "use_wildcard"):
                            # Nested group — recurse
                            self._extract_use(item, source, imports, prefix_parts=[])
                return

            if child.type == "use_list":
                # Top-level use list (rare but possible)
                for item in child.children:
                    if item.type == "identifier":
                        name = self._node_text(item, source)
                        imports.append({
                            "module": name,
                            "names": [name],
                            "type": "use",
                        })
