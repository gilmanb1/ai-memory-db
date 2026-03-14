"""
embeddings.py — Ollama local embedding client.

Falls back gracefully when Ollama is not running: returns None and logs a
warning (once per process). The rest of the system handles None embeddings
by skipping vector search and deduplication (text-based logic still runs).
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from typing import Optional

from .config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

# Cache: avoid re-embedding the same text in one process run
_cache: dict[str, list[float]] = {}

# Warn only once per process to avoid log spam
_warned_once = False


def embed(text: str) -> Optional[list[float]]:
    """
    Return a float embedding for `text` from the local Ollama server.
    Returns None if Ollama is unreachable or the call fails.
    """
    global _warned_once
    text = text.strip()
    if not text:
        return None

    if text in _cache:
        return _cache[text]

    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            result = json.loads(resp.read())
            vec = result.get("embedding")
            if vec:
                _cache[text] = vec
            return vec
    except urllib.error.URLError as exc:
        if not _warned_once:
            print(
                f"[memory] Ollama not available ({OLLAMA_MODEL}). "
                f"Embedding-based dedup and recall disabled.",
                file=sys.stderr,
            )
            _warned_once = True
        return None
    except Exception as exc:
        if not _warned_once:
            print(
                f"[memory] Embedding failed ({OLLAMA_MODEL}): {exc}",
                file=sys.stderr,
            )
            _warned_once = True
        return None


def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed a list of texts, returning None for any that fail."""
    return [embed(t) for t in texts]


def is_ollama_available() -> bool:
    """Quick health-check — True if Ollama is up and the model is loaded."""
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False
