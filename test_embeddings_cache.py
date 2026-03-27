"""
test_embeddings_cache.py — Cached local embeddings for test corpuses.

On first run, embeds all corpus texts via ONNX (or Ollama) and saves to a
JSON cache file. Subsequent runs load from cache (~0ms per embed).

Usage:
    from test_embeddings_cache import cached_embed, is_available
    if is_available():
        # Use cached_embed instead of _mock_embed for real semantic similarity
        test_corpus.set_helpers(cached_embed, _noop_decay)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CACHE_FILE = Path(__file__).parent / ".test_embedding_cache.json"

_cache: dict[str, list[float]] = {}
_available: bool | None = None
_embed_fn = None


def _load_cache():
    global _cache
    if CACHE_FILE.exists():
        try:
            _cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            _cache = {}


def _save_cache():
    CACHE_FILE.write_text(json.dumps(_cache))


def _init():
    """Initialize the real embedding backend."""
    global _available, _embed_fn
    if _available is not None:
        return _available
    try:
        from memory.embeddings import embed as real_embed, _init_onnx
        if _init_onnx():
            _embed_fn = real_embed
            _available = True
            _load_cache()
            return True
        # Try Ollama
        from memory.embeddings import is_ollama_available
        if is_ollama_available():
            _embed_fn = real_embed
            _available = True
            _load_cache()
            return True
    except Exception as e:
        print(f"[test_embeddings_cache] Init failed: {e}", file=sys.stderr)
    _available = False
    return False


def is_available() -> bool:
    """Check if real embeddings are available."""
    return _init()


def cached_embed(text: str) -> list[float] | None:
    """Embed text using ONNX/Ollama with persistent cache."""
    if not _init():
        return None

    text = text.strip()
    if not text:
        return None

    if text in _cache:
        return _cache[text]

    vec = _embed_fn(text)
    if vec:
        _cache[text] = vec
        # Save cache periodically (every 50 new embeddings)
        if len(_cache) % 50 == 0:
            _save_cache()
    return vec


def flush_cache():
    """Force save the cache to disk."""
    _save_cache()


def cache_stats() -> dict:
    """Return cache statistics."""
    return {
        "cached_embeddings": len(_cache),
        "cache_file": str(CACHE_FILE),
        "cache_file_exists": CACHE_FILE.exists(),
        "cache_size_kb": round(CACHE_FILE.stat().st_size / 1024, 1) if CACHE_FILE.exists() else 0,
    }
