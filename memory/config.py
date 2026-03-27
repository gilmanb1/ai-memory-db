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
SESSION_LONG_FACTS_LIMIT   = 30   # long-term facts injected at session start
SESSION_MEDIUM_FACTS_LIMIT = 15   # medium-term facts injected at session start
SESSION_DECISIONS_LIMIT    = 15
SESSION_ENTITIES_LIMIT     = 30
SESSION_GUARDRAILS_LIMIT   = 20   # guardrails always surface for active files
SESSION_PROCEDURES_LIMIT   = 10   # procedures injected at session start
PROMPT_FACTS_LIMIT         = 20   # facts recalled per user prompt
PROMPT_IDEAS_LIMIT         = 6
PROMPT_RELS_LIMIT          = 8
PROMPT_QUESTIONS_LIMIT     = 5
PROMPT_GUARDRAILS_LIMIT    = 10   # guardrails per prompt (file-matched)
PROMPT_PROCEDURES_LIMIT    = 5    # procedures per prompt
PROMPT_ERROR_SOLUTIONS_LIMIT = 5  # error→solution pairs per prompt

# ── Token budgets ────────────────────────────────────────────────────────
# Rough caps to prevent context window bloat from injected memory.
SESSION_TOKEN_BUDGET  = 3000   # max estimated tokens for session context
PROMPT_TOKEN_BUDGET   = 4000   # max estimated tokens for per-prompt context
CHARS_PER_TOKEN       = 4      # rough estimate for budget enforcement

# ── Routing ──────────────────────────────────────────────────────────────
# Set MEMORY_DEBUG=0 in environment to suppress routing explanations in /remember
MEMORY_DEBUG = True  # overridden by env var in remember_cmd.py

# ── Project scoping ──────────────────────────────────────────────────────
GLOBAL_SCOPE          = "__global__"
AUTO_PROMOTE_PROJECT_COUNT = 3  # seen in N+ distinct projects → auto-promote to global

# ── Incremental extraction ──────────────────────────────────────────
EXTRACTION_THRESHOLDS       = [40, 70, 90]   # context % triggers for incremental passes
CROSS_PASS_DEDUP_THRESHOLD  = 0.85           # tighter dedup between passes and against recalled items
DELTA_MIN_USER_MESSAGES     = 3              # skip extraction if delta has fewer user messages
DELTA_MIN_USER_CHARS        = 500            # skip extraction if delta has less user text
NARRATIVE_SEARCH_LIMIT      = 3              # max narratives returned in prompt recall
NARRATIVE_TOKEN_BUDGET      = 400            # tokens reserved for narratives in prompt recall

# ── Retrieval strategies ──────────────────────────────────────────────
RETRIEVAL_STRATEGIES    = ["semantic", "bm25", "graph", "temporal", "path", "code"]
RRF_K                   = 60               # reciprocal rank fusion constant
RERANK_ENABLED          = False            # cross-encoder reranking (requires extra model)
RERANK_MODEL            = None             # future: e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Consolidation ────────────────────────────────────────────────────
CONSOLIDATION_ENABLED        = True
CONSOLIDATION_BATCH_SIZE     = 10          # facts per LLM consolidation batch
CONSOLIDATION_MAX_OBS_CONTEXT = 20         # max observations shown to LLM per batch
OBSERVATION_SIMILARITY_THRESHOLD = 0.90    # cosine threshold for semantic forgetting sweep

# ── Conversation chunks ─────────────────────────────────────────────
CHUNKS_ENABLED             = True
CHUNK_WINDOW_SIZE          = 800     # chars per chunk window (larger = fewer chunks, less noise)
CHUNK_OVERLAP              = 200     # overlap between consecutive chunks
PROMPT_CHUNKS_LIMIT        = 3       # max chunks returned per prompt recall
PROMPT_SIBLINGS_LIMIT      = 5       # max sibling facts from same source chunks
CHUNK_MAX_DISPLAY_CHARS    = 800     # truncate chunk text in formatted output

# ── Importance scoring ──────────────────────────────────────────────
IMPORTANCE_DEFAULT      = 5              # default importance (1-10) if LLM doesn't assign
IMPORTANCE_WEIGHT       = True           # use importance in retrieval scoring

# ── Fact categories (code-oriented) ──────────────────────────────────
FACT_CATEGORIES = [
    "architecture",       # system design, patterns, layers
    "implementation",     # specific code patterns, idioms, conventions
    "operational",        # build, test, deploy, debug procedures
    "dependency",         # library versions, API contracts, integrations
    "decision_rationale", # why something was done a specific way
    "constraint",         # performance requirements, compatibility needs
    "bug_pattern",        # known bugs, error patterns, workarounds
    "user_preference",    # coding style, tool preferences
    "project_context",    # deadlines, team structure, ownership
    # Legacy categories still accepted:
    "technical", "decision", "personal", "contextual", "numerical",
]

# ── Outcome scoring (Feature 3) ──────────────────────────────────────
RECALL_UTILITY_WEIGHT   = True           # use recall_utility in retrieval scoring

# ── Correction detection ──────────────────────────────────────────────
CORRECTION_DETECTION_ENABLED = True
CORRECTION_LLM_FOR_AMBIGUOUS = True   # call LLM for ambiguous corrections (future)

# ── Prefetching (Feature 4) ─────────────────────────────────────────
PREFETCH_ENABLED        = True
PREFETCH_MAX_AGE_S      = 120.0          # prefetch cache TTL in seconds

# ── Failure probability (Feature 5) ─────────────────────────────────
FAILURE_PROB_WEIGHT     = True           # use failure_probability in retrieval scoring

# ── Community summaries (Feature 1) ─────────────────────────────────
COMMUNITY_MIN_CLUSTER_SIZE   = 3         # min items to form a community
COMMUNITY_MIN_ENTITY_OVERLAP = 2         # min shared entities to link items
COMMUNITY_MAX_LEVEL          = 2         # max hierarchy depth
SESSION_COMMUNITY_LIMIT      = 5         # summaries at session start
PROMPT_COMMUNITY_LIMIT       = 3         # summaries per prompt

# ── Coherence validation (Feature 7) ────────────────────────────────
COHERENCE_ENABLED            = True
COHERENCE_SIMILARITY_LOW     = 0.88      # lower bound for contradiction detection
COHERENCE_SIMILARITY_HIGH    = 0.92      # upper bound (above = dedup, not contradiction)

# ── Reflect ──────────────────────────────────────────────────────────
REFLECT_MAX_ITERATIONS  = 6               # max agentic reflect iterations
REFLECT_MODEL           = "claude-sonnet-4-6"

# ── Session observations limits ──────────────────────────────────────
SESSION_OBSERVATIONS_LIMIT = 15           # observations injected at session start
PROMPT_OBSERVATIONS_LIMIT  = 8            # observations per prompt recall

# ── Latency ──────────────────────────────────────────────────────────
PROMPT_RECALL_TIMEOUT_MS = 250            # target latency for prompt recall

# ── Code graph ──────────────────────────────────────────────────────
CODE_GRAPH_ENABLED       = True
CODE_GRAPH_MAX_FILES     = 2000    # max files parsed per repo
CODE_GRAPH_SKIP_DIRS     = {
    "venv", ".venv", "node_modules", ".git", "__pycache__",
    ".tox", "build", "dist", ".eggs", "target", "vendor", "pkg",
    ".next", "out", ".turbo",
}
SESSION_CODE_CONTEXT_LIMIT = 20    # files shown in session context
PROMPT_CODE_CONTEXT_LIMIT  = 10    # files shown in prompt context
