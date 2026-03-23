"""
extract.py — LLM-powered knowledge extraction from conversation transcripts.

Calls claude-sonnet-4-6 with tool_use to extract structured knowledge: facts,
ideas, relationships, decisions, open questions, and named entities — each
tagged with a temporal_class. Using tool_use guarantees valid structured output.
"""
from __future__ import annotations

import json
import textwrap
from typing import Any

import anthropic

from .config import (
    CLAUDE_MODEL, EXTRACT_MAX_TOKENS, MAX_TRANSCRIPT_CHARS,
    DELTA_MIN_USER_MESSAGES, DELTA_MIN_USER_CHARS,
)
from typing import Optional

# ── Extraction tool schema ────────────────────────────────────────────────

EXTRACTION_TOOL = {
    "name": "store_knowledge",
    "description": "Store extracted knowledge from a conversation into the memory system.",
    "input_schema": {
        "type": "object",
        "required": [
            "session_summary", "facts", "ideas", "relationships",
            "key_decisions", "open_questions", "entities",
        ],
        "properties": {
            "session_summary": {
                "type": "string",
                "description": "2-4 sentence plain-English summary of the conversation.",
            },
            "facts": {
                "type": "array",
                "description": "5-25 concrete, specific facts. Preserve specific names, places, numbers, and dates. For coding conversations, capture file paths, function names, patterns, and rationale.",
                "items": {
                    "type": "object",
                    "required": ["text", "category", "confidence", "temporal_class"],
                    "properties": {
                        "text": {"type": "string", "description": "A concrete, specific fact. Include specific names, file paths, function names, amounts, and dates when mentioned."},
                        "category": {
                            "type": "string",
                            "enum": [
                                "architecture", "implementation", "operational",
                                "dependency", "decision_rationale", "constraint",
                                "bug_pattern", "user_preference", "project_context",
                                "technical", "decision", "personal", "contextual", "numerical",
                            ],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "temporal_class": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                        },
                        "importance": {
                            "type": "integer",
                            "description": "1-10 importance score. 1=trivial, 5=normal, 8+=critical knowledge that prevents errors if remembered.",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File paths associated with this fact (e.g., 'memory/db.py', 'src/auth.py'). Include when the fact is about specific files.",
                        },
                        "failure_probability": {
                            "type": "number",
                            "description": "0.0-1.0: How likely would an AI agent get this wrong without being told? 0.0=obvious from code, 0.5=moderate risk, 0.8+=almost certain to cause errors. Non-obvious constraints, counterintuitive patterns, and 'ugly but correct' code should score 0.7+.",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                },
            },
            "ideas": {
                "type": "array",
                "description": "2-10 ideas, hypotheses, proposals, or insights surfaced.",
                "items": {
                    "type": "object",
                    "required": ["text", "type", "temporal_class"],
                    "properties": {
                        "text": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["hypothesis", "proposal", "insight", "concern", "analogy"],
                        },
                        "temporal_class": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                        },
                    },
                },
            },
            "relationships": {
                "type": "array",
                "description": "4-15 relationships between entities or concepts.",
                "items": {
                    "type": "object",
                    "required": ["from", "to", "type", "description"],
                    "properties": {
                        "from": {"type": "string", "description": "Entity or concept A (max 6 words)."},
                        "to": {"type": "string", "description": "Entity or concept B (max 6 words)."},
                        "type": {
                            "type": "string",
                            "enum": [
                                "depends_on", "causes", "part_of", "contrasts_with",
                                "enables", "uses", "defined_by", "leads_to",
                                "replaces", "similar_to",
                            ],
                        },
                        "description": {"type": "string", "description": "One sentence describing the relationship."},
                    },
                },
            },
            "key_decisions": {
                "type": "array",
                "description": "0-10 decisions made or committed to. Include the rationale (WHY), not just the decision.",
                "items": {
                    "type": "object",
                    "required": ["text", "temporal_class"],
                    "properties": {
                        "text": {"type": "string", "description": "The decision AND its rationale. e.g., 'Use urllib3 instead of requests because requests bundles certifi which conflicts with corporate CA'"},
                        "temporal_class": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
                        },
                        "importance": {
                            "type": "integer",
                            "description": "1-10 importance. Decisions with non-obvious rationale or that prevent common mistakes should be 8+.",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files this decision applies to.",
                        },
                    },
                },
            },
            "guardrails": {
                "type": "array",
                "description": "0-5 'don't touch this' warnings. Extract when code is described as intentionally unusual, when a library was chosen for non-obvious operational reasons, or when past changes caused problems.",
                "items": {
                    "type": "object",
                    "required": ["warning", "rationale"],
                    "properties": {
                        "warning": {"type": "string", "description": "What NOT to do. e.g., 'Do not refactor the polling loop to exponential backoff'"},
                        "rationale": {"type": "string", "description": "WHY this constraint exists. e.g., 'Downstream API detects backoff patterns and penalizes exponential clients'"},
                        "consequence": {"type": "string", "description": "What breaks if violated. e.g., 'Client gets blocked for 24h'"},
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files this guardrail applies to.",
                        },
                        "line_range": {"type": "string", "description": "Optional line range, e.g., 'L45-L78'"},
                    },
                },
            },
            "procedures": {
                "type": "array",
                "description": "0-5 'how to do X' procedures. Extract when step-by-step processes are described for common development tasks.",
                "items": {
                    "type": "object",
                    "required": ["task_description", "steps"],
                    "properties": {
                        "task_description": {"type": "string", "description": "What task this procedure covers. e.g., 'Add a new database migration'"},
                        "steps": {"type": "string", "description": "Step-by-step instructions. e.g., '1. Add entry to MIGRATIONS list 2. Increment version number 3. Run test_memory.py'"},
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files involved in this procedure.",
                        },
                    },
                },
            },
            "error_solutions": {
                "type": "array",
                "description": "0-5 error→solution pairs. Extract when bugs are discussed and solutions found.",
                "items": {
                    "type": "object",
                    "required": ["error_pattern", "solution"],
                    "properties": {
                        "error_pattern": {"type": "string", "description": "The error message or pattern. e.g., 'ImportError: No module named onnxruntime'"},
                        "error_context": {"type": "string", "description": "When this error occurs. e.g., 'On macOS ARM when running extraction'"},
                        "solution": {"type": "string", "description": "How to fix it. e.g., 'pip install onnxruntime-silicon'"},
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files where this error occurs.",
                        },
                    },
                },
            },
            "open_questions": {
                "type": "array",
                "description": "0-8 questions raised but not definitively resolved.",
                "items": {"type": "string"},
            },
            "entities": {
                "type": "array",
                "description": "4-20 named things: person, project, tool, technology, organisation, file, module, function.",
                "items": {"type": "string"},
            },
        },
    },
}

# ── Extraction system prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a knowledge-extraction engine for a persistent long-term memory system
    optimised for software engineering and coding conversations.
    Your output is stored in a vector database and must be precise and durable.

    Given a conversation transcript, use the store_knowledge tool to extract
    all notable facts, ideas, relationships, decisions, open questions, named
    entities, guardrails, procedures, and error→solution pairs.

    TEMPORAL CLASS RULES (critical — you set the initial class):

    short  — Ephemeral, conversation-specific details irrelevant to future sessions.
             Examples: current error message, which file we were editing right now,
             a transient debugging step, "we just ran X".

    medium — Working context relevant for the next few weeks / sessions.
             Examples: architectural choice under active discussion, current project phase,
             a preference expressed this session, an active bug being investigated,
             a decision with a short-lived scope.

    long   — Durable, session-agnostic knowledge that should persist indefinitely.
             Examples: user's name/role/company, project name and purpose,
             core tech stack (language, DB, framework), an established constraint,
             a completed architectural decision that won't be revisited,
             a user preference stated as a lasting rule.

    IMPORTANCE SCORING (1-10, assign to facts and decisions):
    1-3  — Trivial or easily re-derivable from code.
    4-6  — Useful working context, nice to recall.
    7-8  — Important knowledge that significantly helps the agent.
    9-10 — Critical. Forgetting this would cause errors, broken code, or wasted effort.
           Examples: "don't refactor X because Y", library chosen for production constraint,
           a past incident that informed a design decision.

    FAILURE PROBABILITY (0.0-1.0, assign to facts):
    0.0-0.2 — Obvious from reading the code. Agent would get this right on its own.
    0.3-0.5 — Moderate risk. Useful context but not catastrophic if missed.
    0.6-0.8 — High risk. Agent would likely make a wrong assumption without this.
    0.9-1.0 — Near-certain failure. Counterintuitive patterns, "ugly but correct" code,
              non-obvious library constraints, past incidents that shaped decisions.

    CODE-SPECIFIC EXTRACTION RULES:
    - ALWAYS capture file paths associated with facts (e.g., "in memory/db.py")
    - Extract specific function/class names as entities
    - When code is described as intentionally unusual or "ugly but correct",
      extract as a GUARDRAIL with warning + rationale + consequence
    - When step-by-step processes are described, extract as PROCEDURES
    - When errors are discussed and solved, extract as ERROR_SOLUTIONS
    - Capture architectural rationale ("we chose X because Y") as high-importance decisions
    - Extract test patterns and strategies
    - Prefer the specific code-oriented categories (architecture, implementation,
      operational, dependency, decision_rationale, constraint, bug_pattern)
      over generic ones (technical, contextual)

    Be aggressive about classifying as LONG what is truly durable.
    Be aggressive about classifying as SHORT what is one-session trivia.
""")


# ── Transcript helpers ────────────────────────────────────────────────────

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", "").strip())
            elif btype == "tool_use":
                parts.append(f"[tool_use: {block.get('name', 'tool')}]")
            elif btype == "tool_result":
                parts.append(f"[tool_result: {_content_to_text(block.get('content', ''))}]")
        return "\n".join(p for p in parts if p)
    return ""


def parse_transcript(path: str) -> list[dict]:
    """Parse a Claude Code JSONL transcript into {role, text, timestamp} dicts."""
    messages: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg = entry.get("message", {})
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                text = _content_to_text(msg.get("content", ""))
                if text:
                    messages.append({
                        "role": role,
                        "text": text,
                        "timestamp": entry.get("timestamp", ""),
                    })
    except OSError:
        pass
    return messages


def build_conversation_text(messages: list[dict], max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    lines: list[str] = []
    for msg in messages:
        label = "USER" if msg["role"] == "user" else "ASSISTANT"
        text = msg["text"]
        if len(text) > 6_000:
            text = text[:6_000] + "\n... [message truncated]"
        lines.append(f"--- {label} ---\n{text}\n")
    full = "\n".join(lines)
    if len(full) > max_chars:
        half = max_chars // 2
        omitted = len(full) - max_chars
        full = (
            full[:half]
            + f"\n\n... [{omitted:,} chars omitted — middle of conversation] ...\n\n"
            + full[-half:]
        )
    return full


# ── Claude API call ───────────────────────────────────────────────────────

def extract_knowledge(conversation_text: str, api_key: str, model: Optional[str] = None) -> dict:
    """
    Call Claude to extract structured knowledge from the conversation.
    Uses tool_use for guaranteed structured output.
    Returns the parsed knowledge dict.
    """
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or CLAUDE_MODEL,
        max_tokens=EXTRACT_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Extract all knowledge from this conversation transcript.\n\n"
                + conversation_text
            ),
        }],
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "store_knowledge"},
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "store_knowledge":
            return block.input

    raise ValueError("Claude did not return a store_knowledge tool_use block")


# ══════════════════════════════════════════════════════════════════════════
# Incremental extraction — delta parsing, tool schema, prompts
# ══════════════════════════════════════════════════════════════════════════

# ── Delta parsing ────────────────────────────────────────────────────────

def parse_transcript_delta(
    path: str,
    byte_offset: int = 0,
) -> tuple[list[dict], int]:
    """
    Parse a JSONL transcript starting from byte_offset.

    Returns (messages, new_byte_offset) where new_byte_offset is the
    file position after the last line read — use it for the next delta.

    When byte_offset=0, equivalent to parse_transcript but also returns offset.
    """
    messages: list[dict] = []
    new_offset = byte_offset
    try:
        with open(path, "rb") as fh:
            fh.seek(byte_offset)
            while True:
                raw = fh.readline()
                if not raw:
                    break
                new_offset = fh.tell()
                raw_str = raw.decode("utf-8", errors="replace").strip()
                if not raw_str:
                    continue
                try:
                    entry = json.loads(raw_str)
                except json.JSONDecodeError:
                    continue
                msg = entry.get("message", {})
                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                text = _content_to_text(msg.get("content", ""))
                if text:
                    messages.append({
                        "role": role,
                        "text": text,
                        "timestamp": entry.get("timestamp", ""),
                    })
    except OSError:
        pass
    return messages, new_offset


def is_delta_substantial(
    messages: list[dict],
    min_user_msgs: int = DELTA_MIN_USER_MESSAGES,
    min_user_chars: int = DELTA_MIN_USER_CHARS,
) -> bool:
    """
    Check if a delta has enough substantive user content to warrant extraction.

    Returns False for tool-heavy deltas with little user input.
    This prevents wasted API calls on deltas that are mostly code/tool output.
    """
    user_messages = [m for m in messages if m["role"] == "user"]
    user_chars = sum(len(m["text"]) for m in user_messages)
    return len(user_messages) >= min_user_msgs or user_chars >= min_user_chars


# ── Incremental extraction tool schema ───────────────────────────────────

INCREMENTAL_EXTRACTION_TOOL = {
    "name": "store_incremental_knowledge",
    "description": "Store extracted knowledge from a conversation segment, with ability to supersede outdated items.",
    "input_schema": {
        "type": "object",
        "required": [
            "narrative_summary", "facts", "ideas", "relationships",
            "key_decisions", "open_questions", "entities", "supersedes",
        ],
        "properties": {
            "narrative_summary": {
                "type": "string",
                "description": (
                    "Cumulative 3-6 sentence narrative summary of the ENTIRE conversation so far "
                    "(not just this segment). Must be self-contained — a reader with no other "
                    "context should understand the conversation's purpose and key outcomes."
                ),
            },
            # Same structured fields as EXTRACTION_TOOL
            "facts": EXTRACTION_TOOL["input_schema"]["properties"]["facts"],
            "ideas": EXTRACTION_TOOL["input_schema"]["properties"]["ideas"],
            "relationships": EXTRACTION_TOOL["input_schema"]["properties"]["relationships"],
            "key_decisions": EXTRACTION_TOOL["input_schema"]["properties"]["key_decisions"],
            "guardrails": EXTRACTION_TOOL["input_schema"]["properties"]["guardrails"],
            "procedures": EXTRACTION_TOOL["input_schema"]["properties"]["procedures"],
            "error_solutions": EXTRACTION_TOOL["input_schema"]["properties"]["error_solutions"],
            "open_questions": EXTRACTION_TOOL["input_schema"]["properties"]["open_questions"],
            "entities": EXTRACTION_TOOL["input_schema"]["properties"]["entities"],
            # New: superseding
            "supersedes": {
                "type": "array",
                "description": (
                    "IDs of existing items that are NOW OUTDATED because the new conversation "
                    "segment explicitly contradicts, reverses, or replaces them. "
                    "Only include an ID if a new item in YOUR response fully replaces it."
                ),
                "items": {
                    "type": "object",
                    "required": ["old_id", "old_table", "reason"],
                    "properties": {
                        "old_id": {
                            "type": "string",
                            "description": "The UUID of the item being replaced.",
                        },
                        "old_table": {
                            "type": "string",
                            "enum": ["facts", "ideas", "decisions", "open_questions"],
                            "description": "Which table the old item lives in.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this item is outdated (1 sentence).",
                        },
                    },
                },
            },
        },
    },
}


# ── Incremental extraction prompts ───────────────────────────────────────

_INCREMENTAL_SYSTEM = SYSTEM_PROMPT + textwrap.dedent("""\

    INCREMENTAL EXTRACTION RULES:
    You are processing a SEGMENT of an ongoing conversation, not the full transcript.

    - Extract only NEW knowledge from the segment below.
    - Do NOT re-extract items listed under PRIOR PASS ITEMS — they are already stored.
    - If the conversation contradicts or updates any EXISTING DATABASE ITEMS, include
      their IDs in the supersedes array. When superseding, the replacement item in your
      facts/decisions/ideas must capture the FULL updated state, not just the delta.
    - Only mark items as superseded if the conversation EXPLICITLY contradicts or
      replaces them. Do not supersede items merely because they are related.
    - Your narrative_summary must cover everything in the conversation so far,
      not just this segment. It must be self-contained.
""")


def _build_incremental_user_message(
    delta_text: str,
    prior_narrative: Optional[str] = None,
    existing_items: Optional[list[dict]] = None,
    prior_items: Optional[list[dict]] = None,
) -> str:
    """Build the user message for incremental extraction."""
    parts: list[str] = []

    # Existing DB items (for superseding)
    if existing_items:
        parts.append("EXISTING DATABASE ITEMS (reference by ID if any are now outdated):")
        for item in existing_items:
            table_prefix = item["table"][:3]  # "fac", "ide", "dec", etc.
            parts.append(f"[{table_prefix}-{item['id']}] {item['text']}")
        parts.append("")

    # Prior pass items (don't re-extract)
    if prior_items:
        parts.append("PRIOR PASS ITEMS (already stored this session — do NOT re-extract):")
        for item in prior_items:
            table_prefix = item["table"][:3]
            parts.append(f"[{table_prefix}-{item['id']}] {item['text']}")
        parts.append("")

    # Prior narrative
    if prior_narrative:
        parts.append("PRIOR NARRATIVE (from earlier in this session — update, don't restart):")
        parts.append(prior_narrative)
        parts.append("")

    parts.append("--- NEW CONVERSATION SEGMENT ---")
    parts.append(delta_text)

    return "\n".join(parts)


def extract_knowledge_incremental(
    delta_text: str,
    api_key: str,
    prior_narrative: Optional[str] = None,
    existing_items: Optional[list[dict]] = None,
    prior_items: Optional[list[dict]] = None,
) -> dict:
    """
    Call Claude to extract knowledge from a conversation delta (segment).

    Unlike extract_knowledge which processes the full transcript,
    this function:
      - Receives only the new conversation since the last extraction pass
      - Receives context about what was already extracted (prior_items)
      - Receives relevant existing DB items (for cross-session superseding)
      - Receives the prior narrative to update (for cumulative summaries)

    Returns the parsed knowledge dict (same structure as extract_knowledge
    but with additional 'narrative_summary' and 'supersedes' fields).
    """
    user_message = _build_incremental_user_message(
        delta_text=delta_text,
        prior_narrative=prior_narrative,
        existing_items=existing_items,
        prior_items=prior_items,
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=EXTRACT_MAX_TOKENS,
        system=_INCREMENTAL_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
        tools=[INCREMENTAL_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "store_incremental_knowledge"},
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "store_incremental_knowledge":
            return block.input

    raise ValueError("Claude did not return a store_incremental_knowledge tool_use block")
