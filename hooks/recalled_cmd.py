#!/usr/bin/env python3
"""Show what the memory system recalled for the last prompt."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

RECALL_LOG = Path.home() / ".claude" / "memory" / "last_recall.json"


def main():
    if not RECALL_LOG.exists():
        print("No recall log found. The memory system has not recalled anything yet this session.")
        return

    try:
        data = json.loads(RECALL_LOG.read_text())
    except Exception as e:
        print(f"Error reading recall log: {e}")
        return

    lines = []
    lines.append("## Last Recalled Context")
    lines.append("")

    if data.get("prompt"):
        lines.append(f"**Prompt:** {data['prompt']}")
        lines.append("")

    if data.get("session_id"):
        lines.append(f"**Session:** `{data['session_id'][:12]}`")
        lines.append("")

    # Facts
    facts = data.get("facts", [])
    if facts:
        lines.append(f"### Facts ({len(facts)})")
        for f in facts:
            scope = f.get("scope", "")
            tc = f.get("temporal_class", "")
            score = f.get("score", 0)
            badge = f" [{scope}]" if scope else ""
            badge += f" [{tc}]" if tc else ""
            badge += f" (score: {score})" if score else ""
            lines.append(f"- `{f.get('id', '?')}` {f.get('text', '')}{badge}")
        lines.append("")

    # Guardrails
    guardrails = data.get("guardrails", [])
    if guardrails:
        lines.append(f"### Guardrails ({len(guardrails)})")
        for g in guardrails:
            scope = f" [{g.get('scope', '')}]" if g.get("scope") else ""
            lines.append(f"- `{g.get('id', '?')}` {g.get('text', '')}{scope}")
        lines.append("")

    # Procedures
    procedures = data.get("procedures", [])
    if procedures:
        lines.append(f"### Procedures ({len(procedures)})")
        for p in procedures:
            lines.append(f"- `{p.get('id', '?')}` {p.get('text', '')}")
        lines.append("")

    # Error Solutions
    error_solutions = data.get("error_solutions", [])
    if error_solutions:
        lines.append(f"### Error Solutions ({len(error_solutions)})")
        for e in error_solutions:
            lines.append(f"- `{e.get('id', '?')}` {e.get('text', '')}")
        lines.append("")

    # Observations
    observations = data.get("observations", [])
    if observations:
        lines.append(f"### Observations ({len(observations)})")
        for o in observations:
            lines.append(f"- `{o.get('id', '?')}` {o.get('text', '')}")
        lines.append("")

    # Relationships
    relationships = data.get("relationships", [])
    if relationships:
        lines.append(f"### Relationships ({len(relationships)})")
        for r in relationships:
            lines.append(f"- {r}")
        lines.append("")

    # Entities
    entities = data.get("entities_hit", [])
    if entities:
        lines.append(f"### Entities Hit ({len(entities)})")
        lines.append(f"- {', '.join(entities)}")
        lines.append("")

    # Code Context
    code_context = data.get("code_context", [])
    if code_context:
        lines.append(f"### Code Context ({len(code_context)})")
        for c in code_context:
            lines.append(f"- `{c.get('file', '?')}` ({c.get('symbols', 0)} symbols)")
        lines.append("")

    # Check if anything was recalled at all
    has_items = any([facts, guardrails, procedures, error_solutions, observations, relationships, entities, code_context])
    if not has_items:
        lines.append("_No items were recalled for this prompt._")
        lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
