"""
config.py — All tuneable constants for the Claude Code memory system.
Edit these to customise behaviour without touching logic files.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
MEMORY_DIR = Path.home() / ".claude" / "memory"
DB_PATH    = MEMORY_DIR / "knowledge.duckdb"

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434"
OLLAMA_MODEL      = "nomic-embed-text"   # 768-dim; change to match your pull
EMBEDDING_DIM     = 768                  # must match the model above
OLLAMA_TIMEOUT    = 15                   # seconds per embedding request

# ── Claude API ─────────────────────────────────────────────────────────────
CLAUDE_MODEL      = "claude-sonnet-4-6"
EXTRACT_MAX_TOKENS = 4096
MAX_TRANSCRIPT_CHARS = 120_000           # chars sent to Claude for extraction

# ── Similarity thresholds ──────────────────────────────────────────────────
DEDUP_THRESHOLD   = 0.92   # cosine ≥ this → same item, update don't insert
RECALL_THRESHOLD  = 0.60   # cosine ≥ this → semantically relevant for recall

# ── Temporal class promotion thresholds ───────────────────────────────────
# LLM sets the initial class; these rules promote/adjust over time.
SESSION_COUNT_MEDIUM = 3   # seen in ≥ 3 sessions → at least medium
SESSION_COUNT_LONG   = 7   # seen in ≥ 7 sessions → long
AGE_MEDIUM_DAYS      = 7   # age > 7 days (still reinforced) → at least medium
AGE_LONG_DAYS        = 30  # age > 30 days (still active) → long

# ── Decay rates (per day, exponential) ────────────────────────────────────
# short items evaporate fast; long items are nearly permanent
DECAY_RATES = {"short": 0.18, "medium": 0.04, "long": 0.007}

# A decay_score < this causes the item to be marked inactive (forgotten)
FORGET_THRESHOLD = 0.05

# ── Recall limits ──────────────────────────────────────────────────────────
SESSION_LONG_FACTS_LIMIT   = 20   # long-term facts injected at session start
SESSION_MEDIUM_FACTS_LIMIT = 8    # medium-term facts injected at session start
SESSION_DECISIONS_LIMIT    = 10
SESSION_ENTITIES_LIMIT     = 25
PROMPT_FACTS_LIMIT         = 8    # facts recalled per user prompt
PROMPT_IDEAS_LIMIT         = 4
PROMPT_RELS_LIMIT          = 6
PROMPT_QUESTIONS_LIMIT     = 3

# ── Token budgets ────────────────────────────────────────────────────────
# Rough caps to prevent context window bloat from injected memory.
SESSION_TOKEN_BUDGET  = 3000   # max estimated tokens for session context
PROMPT_TOKEN_BUDGET   = 1500   # max estimated tokens for per-prompt context
CHARS_PER_TOKEN       = 4      # rough estimate for budget enforcement

# ── Project scoping ──────────────────────────────────────────────────────
GLOBAL_SCOPE          = "__global__"
AUTO_PROMOTE_PROJECT_COUNT = 3  # seen in N+ distinct projects → auto-promote to global
