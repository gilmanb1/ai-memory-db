"""
embeddings.py — Local embedding client with ONNX Runtime (fast) and Ollama (fallback).

Tries ONNX Runtime first (in-process, ~2-5ms per embed). Falls back to Ollama HTTP
if ONNX is unavailable. Falls back gracefully when neither is available: returns None
and logs a warning (once per process).
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from typing import Optional

import numpy as np

from .config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, EMBEDDING_DIM

# Cache: avoid re-embedding the same text in one process run
_cache: dict[str, list[float]] = {}

# Warn only once per process to avoid log spam
_warned_once = False

# ── ONNX Runtime backend (lazy-loaded) ──────────────────────────────────

_onnx_session = None
_onnx_tokenizer = None
_onnx_available: Optional[bool] = None  # None = not checked yet


def _init_onnx() -> bool:
    """Lazy-load ONNX Runtime session and tokenizer. Returns True if available."""
    global _onnx_session, _onnx_tokenizer, _onnx_available
    if _onnx_available is not None:
        return _onnx_available
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            "nomic-ai/nomic-embed-text-v1.5", "onnx/model.onnx",
        )
        _onnx_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        _onnx_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        _onnx_available = True
        return True
    except Exception as e:
        print(f"[memory] ONNX embeddings unavailable ({e}), falling back to Ollama", file=sys.stderr)
        _onnx_available = False
        return False


def _onnx_embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts via ONNX Runtime. Returns list of 768-dim vectors."""
    inputs = _onnx_tokenizer(
        texts, return_tensors="np", padding=True, truncation=True, max_length=512,
    )
    input_names = {i.name for i in _onnx_session.get_inputs()}
    input_dict = {k: v for k, v in inputs.items() if k in input_names}

    outputs = _onnx_session.run(None, input_dict)
    token_embs = outputs[0]  # (batch, seq_len, hidden_dim)

    # Mean pooling with attention mask
    mask = np.expand_dims(inputs["attention_mask"], -1).astype(np.float32)
    pooled = (token_embs * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)

    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / np.maximum(norms, 1e-9)

    return [vec.tolist() for vec in pooled]


# ── Ollama HTTP backend ─────────────────────────────────────────────────

def _ollama_embed(text: str) -> Optional[list[float]]:
    """Embed via Ollama HTTP API. Returns None on failure."""
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
            return result.get("embedding")
    except Exception:
        return None


# ── Public API ──────────────────────────────────────────────────────────

def embed(text: str) -> Optional[list[float]]:
    """
    Return a float embedding for `text`.
    Uses ONNX Runtime (in-process, ~2-5ms) if available, else Ollama HTTP (~50-100ms).
    Returns None if neither backend is available.
    """
    global _warned_once
    text = text.strip()
    if not text:
        return None

    if text in _cache:
        return _cache[text]

    # Try ONNX first
    if _init_onnx():
        try:
            # Nomic requires "search_query: " or "search_document: " prefix
            prefixed = f"search_document: {text}"
            vecs = _onnx_embed_batch([prefixed])
            if vecs and len(vecs[0]) == EMBEDDING_DIM:
                _cache[text] = vecs[0]
                return vecs[0]
        except Exception:
            pass

    # Fallback to Ollama
    vec = _ollama_embed(text)
    if vec:
        _cache[text] = vec
        return vec

    if not _warned_once:
        print(
            f"[memory] No embedding backend available (ONNX failed, Ollama not running). "
            f"Embedding-based dedup and recall disabled.",
            file=sys.stderr,
        )
        _warned_once = True
    return None


def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed a list of texts, using ONNX batch inference when available."""
    if not texts:
        return []

    # Check cache first, identify misses
    results: list[Optional[list[float]]] = [None] * len(texts)
    miss_indices: list[int] = []
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        if text in _cache:
            results[i] = _cache[text]
        else:
            miss_indices.append(i)

    if not miss_indices:
        return results

    # Batch embed misses via ONNX
    if _init_onnx():
        try:
            miss_texts = [f"search_document: {texts[i].strip()}" for i in miss_indices]
            vecs = _onnx_embed_batch(miss_texts)
            for j, idx in enumerate(miss_indices):
                if vecs[j] and len(vecs[j]) == EMBEDDING_DIM:
                    text = texts[idx].strip()
                    _cache[text] = vecs[j]
                    results[idx] = vecs[j]
            return results
        except Exception:
            pass

    # Fallback: embed individually via Ollama
    for idx in miss_indices:
        results[idx] = embed(texts[idx])
    return results


def embed_query(text: str) -> Optional[list[float]]:
    """
    Embed a search query (uses "search_query: " prefix for nomic model).
    Use this for recall queries; use embed() for documents/facts.
    """
    text = text.strip()
    if not text:
        return None

    cache_key = f"__query__{text}"
    if cache_key in _cache:
        return _cache[cache_key]

    if _init_onnx():
        try:
            prefixed = f"search_query: {text}"
            vecs = _onnx_embed_batch([prefixed])
            if vecs and len(vecs[0]) == EMBEDDING_DIM:
                _cache[cache_key] = vecs[0]
                return vecs[0]
        except Exception:
            pass

    # Ollama doesn't distinguish query vs document
    vec = _ollama_embed(text)
    if vec:
        _cache[cache_key] = vec
        return vec
    return None


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


def is_available() -> bool:
    """True if any embedding backend (ONNX or Ollama) is available."""
    return _init_onnx() or is_ollama_available()
