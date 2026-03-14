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

from .config import CLAUDE_MODEL, EXTRACT_MAX_TOKENS, MAX_TRANSCRIPT_CHARS

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
                "description": "5-25 concrete, specific facts stated or established.",
                "items": {
                    "type": "object",
                    "required": ["text", "category", "confidence", "temporal_class"],
                    "properties": {
                        "text": {"type": "string", "description": "A concrete, specific fact."},
                        "category": {
                            "type": "string",
                            "enum": ["technical", "decision", "personal", "contextual", "numerical"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "temporal_class": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
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
                "description": "0-10 decisions made or committed to.",
                "items": {
                    "type": "object",
                    "required": ["text", "temporal_class"],
                    "properties": {
                        "text": {"type": "string"},
                        "temporal_class": {
                            "type": "string",
                            "enum": ["short", "medium", "long"],
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
                "description": "4-20 named things: person, project, tool, technology, organisation.",
                "items": {"type": "string"},
            },
        },
    },
}

# ── Extraction system prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a knowledge-extraction engine for a persistent long-term memory system.
    Your output is stored in a vector database and must be precise and durable.

    Given a conversation transcript, use the store_knowledge tool to extract
    all notable facts, ideas, relationships, decisions, open questions, and
    named entities.

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

def extract_knowledge(conversation_text: str, api_key: str) -> dict:
    """
    Call Claude to extract structured knowledge from the conversation.
    Uses tool_use for guaranteed structured output.
    Returns the parsed knowledge dict.
    """
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
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
