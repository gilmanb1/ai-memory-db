"""
parsers — Tree-sitter based parsers for non-Python languages.

Each parser implements the LanguageParser protocol from code_graph.py.
Parsers are registered conditionally — if tree-sitter is not installed,
only the built-in Python parser (stdlib ast) is available.
"""
from __future__ import annotations


def register_all_parsers() -> None:
    """Try to register all tree-sitter parsers. Fails silently if deps missing."""
    from ..code_graph import register_parser

    try:
        from .typescript_parser import TypeScriptParser
        register_parser(TypeScriptParser())
    except ImportError:
        pass

    try:
        from .go_parser import GoParser
        register_parser(GoParser())
    except ImportError:
        pass

    try:
        from .rust_parser import RustParser
        register_parser(RustParser())
    except ImportError:
        pass
