"""
routing.py — Classifies /remember text and routes to the appropriate storage system(s).

Routes:
  - "both"   → auto-memory (markdown files) AND DuckDB
  - "duckdb" → DuckDB only

Auto-memory types (matching the native mechanism):
  - feedback:  behavioral preferences, corrections, directives
  - user:      user profile, role, expertise, background
  - reference: external systems, URLs, dashboards, trackers
  - project:   deadlines, freezes, initiatives, milestones
"""
from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Optional

# ── Budget ────────────────────────────────────────────────────────────────
AUTO_MEMORY_MAX_INDEX_LINES = 180  # leave headroom under the 200-line truncation

# ── Classification patterns ───────────────────────────────────────────────
# Order matters: first match wins.

_FEEDBACK_PATTERNS = [
    r"\balways\b",
    r"\bnever\b",
    r"\bdon'?t\b",
    r"\bdo not\b",
    r"\bstop\b",
    r"\bavoid\b",
    r"\bi prefer\b",
    r"\bi want you to\b",
    r"\bi like when\b",
    r"\binstead of\b",
    r"\brather than\b",
    r"\bplease don'?t\b",
    r"\bmake sure (to|you)\b",
    r"\bwhen i ask\b",
    r"\bbefore (you|building|creating|implementing|coding|starting)\b",
]

_USER_PATTERNS = [
    r"\bi am a\b",
    r"\bi'?m a\b",
    r"\bmy role\b",
    r"\bi work (as|at|on|in)\b",
    r"\bmy background\b",
    r"\bmy expertise\b",
    r"\bi have experience\b",
    r"\bi'?m familiar with\b",
    r"\bi'?m new to\b",
    r"\bi'?m (a |an )?(senior|junior|lead|staff|principal)\b",
    r"\bi specialize\b",
    r"\bmy name is\b",
]

_REFERENCE_PATTERNS = [
    r"https?://",
    r"\btracked in\b",
    r"\bdashboard (at|is)\b",
    r"\bdocumented in\b",
    r"\bwiki\b",
    r"\bconfluence\b",
    r"\blinear\b",
    r"\bjira\b",
    r"\bslack (channel|workspace)\b",
    r"\bgrafana\b",
    r"\bnotion\b",
    r"\bgithub (issue|project|repo)\b",
]

_PROJECT_PATTERNS = [
    r"\bdeadline\b",
    r"\bfreeze\b",
    r"\brelease (branch|cut|date)\b",
    r"\bsprint\b",
    r"\bmilestone\b",
    r"\binitiative\b",
    r"\bpriority\b",
    r"\bblocked by\b",
    r"\bfreezing\b",
    r"\bmerge freeze\b",
    r"\bafter (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bdue (date|by)\b",
]


def _matches_any(text: str, patterns: list[str]) -> Optional[str]:
    """Return the first matching pattern, or None."""
    text_lower = text.lower()
    for pattern in patterns:
        m = re.search(pattern, text_lower)
        if m:
            return pattern
    return None


def classify_memory(text: str) -> dict:
    """
    Classify a /remember text into a routing decision.

    Returns:
        {
            "route": "both" | "duckdb",
            "auto_type": "feedback" | "user" | "reference" | "project" | None,
            "reason": str,
        }
    """
    # Check each category in priority order
    match = _matches_any(text, _FEEDBACK_PATTERNS)
    if match:
        return {
            "route": "both",
            "auto_type": "feedback",
            "reason": f"Behavioral directive detected (matched: {match})",
        }

    match = _matches_any(text, _USER_PATTERNS)
    if match:
        return {
            "route": "both",
            "auto_type": "user",
            "reason": f"User profile info detected (matched: {match})",
        }

    match = _matches_any(text, _REFERENCE_PATTERNS)
    if match:
        return {
            "route": "both",
            "auto_type": "reference",
            "reason": f"External reference detected (matched: {match})",
        }

    match = _matches_any(text, _PROJECT_PATTERNS)
    if match:
        return {
            "route": "both",
            "auto_type": "project",
            "reason": f"Project context detected (matched: {match})",
        }

    # Default: DuckDB only
    return {
        "route": "duckdb",
        "auto_type": None,
        "reason": "No auto-memory trigger matched; storing in DuckDB only",
    }


# ── Auto-memory file operations ──────────────────────────────────────────

def _slugify(text: str, max_len: int = 40) -> str:
    """Generate a filesystem-safe slug from text."""
    # Take first few meaningful words
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    slug = "_".join(words[:5])
    if len(slug) > max_len:
        slug = slug[:max_len]
    # Add a short hash to reduce collisions
    h = hashlib.md5(text.lower().encode()).hexdigest()[:6]
    return f"{slug}_{h}"


def _one_line_description(text: str) -> str:
    """Create a concise one-line description for the MEMORY.md index."""
    # Truncate at 80 chars
    desc = text.strip()
    if len(desc) > 80:
        desc = desc[:77] + "..."
    return desc


def write_auto_memory(
    text: str,
    auto_type: str,
    memory_dir: Path,
) -> Path:
    """
    Write a memory to the auto-memory system (markdown files + MEMORY.md index).

    Returns the path to the created/updated memory file.
    """
    slug = _slugify(text)
    filename = f"{auto_type}_{slug}.md"
    filepath = memory_dir / filename
    description = _one_line_description(text)

    # Write the memory file with frontmatter
    content = (
        f"---\n"
        f"name: {description}\n"
        f"description: {description}\n"
        f"type: {auto_type}\n"
        f"---\n"
        f"\n"
        f"{text}\n"
    )
    filepath.write_text(content)

    # Update MEMORY.md index
    _update_index(memory_dir, filename, description)

    return filepath


def _update_index(memory_dir: Path, filename: str, description: str) -> None:
    """Add or update an entry in MEMORY.md, respecting the line budget."""
    index_path = memory_dir / "MEMORY.md"

    if index_path.exists():
        lines = index_path.read_text().strip().split("\n")
    else:
        lines = ["# Memory Index", ""]

    # Check if this filename already exists in the index — update it
    new_entry = f"- [{filename}]({filename}) — {description}"
    updated = False
    for i, line in enumerate(lines):
        if filename in line:
            lines[i] = new_entry
            updated = True
            break

    if not updated:
        # Check budget before adding
        entry_lines = [l for l in lines if l.startswith("- [")]
        if len(entry_lines) >= AUTO_MEMORY_MAX_INDEX_LINES:
            # Budget exceeded — don't add (be opinionated: refuse rather than evict)
            return
        lines.append(new_entry)

    index_path.write_text("\n".join(lines) + "\n")


def find_auto_memory_file(text: str, memory_dir: Path) -> Optional[str]:
    """
    Find the auto-memory filename for a given text, if it exists.
    Reconstructs the expected filename from the text using the same slugify logic.
    Returns the filename (not full path) or None.
    """
    for auto_type in ("feedback", "user", "reference", "project"):
        slug = _slugify(text)
        filename = f"{auto_type}_{slug}.md"
        if (memory_dir / filename).exists():
            return filename
    return None


def delete_auto_memory(filename: str, memory_dir: Path) -> None:
    """
    Delete an auto-memory markdown file and remove its MEMORY.md index entry.
    Safe to call with a nonexistent filename.
    """
    # Delete the file
    filepath = memory_dir / filename
    if filepath.exists():
        filepath.unlink()

    # Remove from MEMORY.md index
    index_path = memory_dir / "MEMORY.md"
    if not index_path.exists():
        return

    lines = index_path.read_text().strip().split("\n")
    lines = [l for l in lines if filename not in l]
    index_path.write_text("\n".join(lines) + "\n")
