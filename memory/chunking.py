"""
chunking.py — Split conversation text into small, overlapping chunks for precise retrieval.

Each conversation session is split into ~400-char windows with ~100-char overlap.
This makes vector search precise: a query about "Target coupon" matches the specific
window mentioning Target, not a diluted embedding of the entire 4000-char conversation.
"""
from __future__ import annotations

from .config import CHUNK_WINDOW_SIZE, CHUNK_OVERLAP


def split_into_chunks(text: str, window: int = CHUNK_WINDOW_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping windows.

    Returns a list of text chunks, each approximately `window` characters,
    with `overlap` characters shared between consecutive chunks.
    Splits at sentence/line boundaries when possible to keep chunks coherent.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    text = text.strip()

    # Short text: return as single chunk
    if len(text) <= window:
        return [text]

    chunks = []
    step = max(window - overlap, 1)  # ensure forward progress even with bad params
    pos = 0

    while pos < len(text):
        end = min(pos + window, len(text))
        chunk = text[pos:end]

        # Try to break at a sentence/line boundary within the last 20% of the window
        if end < len(text):
            break_zone_start = max(pos + int(window * 0.8), pos + 1)
            best_break = -1
            for sep in ['\n\n', '\n', '. ', '? ', '! ', '; ', ', ']:
                idx = text.rfind(sep, break_zone_start, end)
                if idx > 0:
                    best_break = idx + len(sep)
                    break
            if best_break > pos:
                chunk = text[pos:best_break]
                end = best_break

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        pos += step
        # Avoid tiny trailing chunks
        if len(text) - pos < overlap:
            break

    return chunks
