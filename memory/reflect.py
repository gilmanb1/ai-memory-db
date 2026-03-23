"""
reflect.py — Agentic reflect loop for synthesized answers from memory.

Searches observations (consolidated knowledge) then raw facts,
iterating with tool_choice to let Claude decide when to search more
or conclude. Forced sequence: observations first, then facts, then auto.
"""
from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from . import db
from .config import (
    REFLECT_MAX_ITERATIONS,
    REFLECT_MODEL,
    RECALL_THRESHOLD,
    GLOBAL_SCOPE,
)
from .embeddings import embed


# ── Data types ────────────────────────────────────────────────────────────

@dataclass
class ReflectResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    iterations_used: int = 0
    tool_trace: list[str] = field(default_factory=list)
    error: Optional[str] = None


# ── Tool definitions ──────────────────────────────────────────────────────

REFLECT_TOOLS = [
    {
        "name": "search_observations",
        "description": (
            "Search consolidated observations (auto-synthesized knowledge). "
            "These represent higher-level insights derived from multiple facts."
        ),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "recall_facts",
        "description": (
            "Search raw facts stored in memory. Use this for ground truth "
            "verification or when observations don't cover the topic."
        ),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 15},
            },
        },
    },
    {
        "name": "done",
        "description": (
            "Signal that you have enough evidence to answer. Requires that you "
            "have searched at least once before calling this."
        ),
        "input_schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": {"type": "string", "description": "Your synthesized answer"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of the facts/observations that support your answer",
                },
            },
        },
    },
]


REFLECT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a reflection agent with access to a persistent memory system.
    Your goal is to answer the user's question by searching through stored
    knowledge — first consolidated observations, then raw facts.

    WORKFLOW:
    1. Search observations first for high-level synthesized knowledge.
    2. Search facts for supporting evidence or details not in observations.
    3. Iterate if needed: refine your search queries based on what you find.
    4. Call done() with your answer and the IDs of supporting sources.

    RULES:
    - You MUST search before answering. Never call done() without evidence.
    - Cite specific source IDs in your answer when possible.
    - If memory contains nothing relevant, say so honestly.
    - Be concise but thorough. Synthesize across multiple sources.
""")


# ── Tool execution ────────────────────────────────────────────────────────

def _execute_tool(
    conn,
    tool_name: str,
    tool_input: dict,
    scope: Optional[str],
) -> tuple[str, list[dict]]:
    """Execute a reflect tool and return (result_text, source_items)."""
    if tool_name == "search_observations":
        query = tool_input.get("query", "")
        max_results = tool_input.get("max_results", 10)
        query_embedding = embed(query)
        if not query_embedding:
            return "Embedding service unavailable. Cannot search observations.", []
        results = db.search_observations(conn, query_embedding, limit=max_results, scope=scope)
        if not results:
            return "No matching observations found.", []
        lines = []
        for r in results:
            proof = r.get("proof_count", 0)
            lines.append(f"[{r['id']}] {r['text']} (proof_count: {proof}, score: {r.get('score', 0):.2f})")
        return "\n".join(lines), results

    elif tool_name == "recall_facts":
        query = tool_input.get("query", "")
        max_results = tool_input.get("max_results", 15)
        query_embedding = embed(query)
        if not query_embedding:
            return "Embedding service unavailable. Cannot search facts.", []
        results = db.search_facts(conn, query_embedding, limit=max_results, scope=scope)
        if not results:
            return "No matching facts found.", []
        lines = []
        for r in results:
            tc = r.get("temporal_class", "")
            lines.append(f"[{r['id']}] [{tc}] {r['text']} (score: {r.get('score', 0):.2f})")
        return "\n".join(lines), results

    return f"Unknown tool: {tool_name}", []


# ── Main reflect loop ────────────────────────────────────────────────────

def run_reflect(
    question: str,
    api_key: str,
    scope: Optional[str] = None,
    max_iterations: int = REFLECT_MAX_ITERATIONS,
    db_path: Optional[str] = None,
) -> ReflectResult:
    """
    Run an agentic reflect loop to answer a question from memory.

    Forced sequence: iteration 1 = search_observations, iteration 2 = recall_facts,
    then auto until done or max iterations.
    """
    conn = db.get_connection(read_only=True, db_path=db_path)
    try:
        return _reflect_loop(conn, question, api_key, scope, max_iterations)
    finally:
        conn.close()


def _reflect_loop(
    conn,
    question: str,
    api_key: str,
    scope: Optional[str],
    max_iterations: int,
) -> ReflectResult:
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": question}]
    tool_trace = []
    all_sources = []
    has_searched = False

    forced_tools = ["search_observations", "recall_facts"]

    for iteration in range(max_iterations):
        # Determine tool_choice for this iteration
        if iteration < len(forced_tools):
            tool_choice = {"type": "tool", "name": forced_tools[iteration]}
        elif iteration == max_iterations - 1:
            tool_choice = {"type": "tool", "name": "done"}
        else:
            tool_choice = {"type": "auto"}

        try:
            response = client.messages.create(
                model=REFLECT_MODEL,
                max_tokens=4096,
                system=REFLECT_SYSTEM_PROMPT,
                messages=messages,
                tools=REFLECT_TOOLS,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            return ReflectResult(
                answer="",
                iterations_used=iteration + 1,
                tool_trace=tool_trace,
                error=f"API call failed: {exc}",
            )

        # Check for text-only response (LLM decided to answer without tools)
        has_tool_use = any(b.type == "tool_use" for b in response.content)
        if not has_tool_use:
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            answer = "\n".join(text_blocks) if text_blocks else "No answer generated."
            return ReflectResult(
                answer=answer,
                sources=all_sources,
                iterations_used=iteration + 1,
                tool_trace=tool_trace,
            )

        # Process tool_use blocks
        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            tool_trace.append(f"iter={iteration} tool={tool_name} input={json.dumps(tool_input)[:200]}")

            if tool_name == "done":
                # Evidence guardrail: reject done() without prior search
                if not has_searched:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "ERROR: You must search observations or facts before calling done(). Please search first.",
                        "is_error": True,
                    })
                    continue

                answer = tool_input.get("answer", "")
                source_ids = tool_input.get("source_ids", [])
                return ReflectResult(
                    answer=answer,
                    sources=[s for s in all_sources if s.get("id") in set(source_ids)] if source_ids else all_sources,
                    iterations_used=iteration + 1,
                    tool_trace=tool_trace,
                )

            # Execute search tool
            result_text, sources = _execute_tool(conn, tool_name, tool_input, scope)
            all_sources.extend(sources)
            has_searched = True

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        # Add assistant message and tool results to conversation
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Exhausted iterations
    return ReflectResult(
        answer="Unable to formulate answer within iteration budget.",
        sources=all_sources,
        iterations_used=max_iterations,
        tool_trace=tool_trace,
        error="max_iterations reached",
    )
