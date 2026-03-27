#!/usr/bin/env bash
# install.sh — Install the Claude Code memory system into ~/.claude/
# Usage: bash install.sh [--user | --project]
#   --user     Install hooks globally for all Claude Code sessions (default)
#   --project  Install hooks locally for the current project only

set -euo pipefail

MODE="user"
if [[ "${1:-}" == "--project" ]]; then
  MODE="project"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Determine target settings and hooks directory ─────────────────────────
if [[ "$MODE" == "user" ]]; then
  SETTINGS_FILE="$HOME/.claude/settings.json"
  HOOKS_DIR="$HOME/.claude/hooks"
  MEMORY_DIR="$HOME/.claude/memory"
  HOOK_PREFIX="$HOME/.claude/hooks"
else
  SETTINGS_FILE=".claude/settings.json"
  HOOKS_DIR=".claude/hooks"
  MEMORY_DIR="$HOME/.claude/memory"   # DB always lives in home
  HOOK_PREFIX='$CLAUDE_PROJECT_DIR/.claude/hooks'
fi

echo "=== Claude Code Memory System — Install (${MODE} mode) ==="
echo ""

# ── Check prerequisites ───────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
echo "ok uv found: $(uv --version)"

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 is not installed."
  exit 1
fi
echo "ok python3 found: $(python3 --version)"

# ── Copy memory package ───────────────────────────────────────────────────
echo ""
echo "-> Installing memory package to ~/.claude/memory/ ..."
mkdir -p "$HOME/.claude/memory"
cp -r "$SCRIPT_DIR/memory/"* "$HOME/.claude/memory/"
echo "ok Memory package installed"

# ── Copy hook scripts ─────────────────────────────────────────────────────
echo ""
echo "-> Installing hooks to ${HOOKS_DIR}/ ..."
mkdir -p "$HOOKS_DIR"
for hook in pre_compact session_start session_end user_prompt_submit status_line _extract_worker remember_cmd forget_cmd reflect_cmd memory_mcp_server post_tool_use knowledge_cmd recalled_cmd audit_cmd health_cmd; do
  cp "$SCRIPT_DIR/hooks/${hook}.py" "${HOOKS_DIR}/${hook}.py"
  chmod +x "${HOOKS_DIR}/${hook}.py"
  echo "  ok ${hook}.py"
done

# ── Install custom slash commands ─────────────────────────────────────────
echo ""
echo "-> Installing slash commands to ~/.claude/commands/ ..."
mkdir -p "$HOME/.claude/commands"
for cmd in remember forget forget-confirm memories facts search-memory decisions entities relationships sessions scopes reflect knowledge recalled session-learned review export-memory import-memory restore-memory snapshots audit-memory memory-health; do
  cp "$SCRIPT_DIR/commands/${cmd}.md" "$HOME/.claude/commands/${cmd}.md"
  echo "  ok ${cmd}.md"
done

# ── Create locks directory ────────────────────────────────────────────────
mkdir -p "$HOME/.claude/memory/locks"

# ── Merge hook config + status line into settings.json ────────────────────
echo ""
echo "-> Configuring hooks + status line in ${SETTINGS_FILE} ..."

mkdir -p "$(dirname "$SETTINGS_FILE")"

# Read existing settings or start empty
if [[ -f "$SETTINGS_FILE" ]]; then
  existing=$(cat "$SETTINGS_FILE")
else
  existing="{}"
fi

python3 - "$SETTINGS_FILE" "$HOOK_PREFIX" "$existing" << 'PYEOF'
import sys, json
from pathlib import Path

settings_path = sys.argv[1]
hook_prefix   = sys.argv[2]
existing_json = sys.argv[3]

try:
    settings = json.loads(existing_json)
except json.JSONDecodeError:
    settings = {}

new_hooks = {
    "PreCompact": [{
        "matcher": "",
        "hooks": [{"type": "command", "command": f"{hook_prefix}/pre_compact.py"}]
    }],
    "SessionStart": [{
        "matcher": "",
        "hooks": [{"type": "command", "command": f"{hook_prefix}/session_start.py"}]
    }],
    "SessionEnd": [{
        "matcher": "",
        "hooks": [{"type": "command", "command": f"{hook_prefix}/session_end.py"}]
    }],
    "UserPromptSubmit": [{
        "matcher": "",
        "hooks": [{"type": "command", "command": f"{hook_prefix}/user_prompt_submit.py"}]
    }],
    "PostToolUse": [{
        "matcher": "",
        "hooks": [{"type": "command", "command": f"{hook_prefix}/post_tool_use.py"}]
    }],
}

# Merge — preserve any existing hooks for other events
existing_hooks = settings.get("hooks", {})
for event, config in new_hooks.items():
    existing_hooks[event] = config
settings["hooks"] = existing_hooks

# Configure status line
settings["statusLine"] = {
    "type": "command",
    "command": f"{hook_prefix}/status_line.py",
}

Path(settings_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings_path).write_text(json.dumps(settings, indent=2))
print(f"  ok {settings_path} updated")
PYEOF

# ── Verify DB directory ───────────────────────────────────────────────────
mkdir -p "$HOME/.claude/memory"

# ── Run tests ─────────────────────────────────────────────────────────────
echo ""
echo "-> Running test suite ..."
if python3 "$SCRIPT_DIR/test_memory.py" 2>&1 | tail -3; then
  echo "ok All tests passed"
else
  echo "!!  Some tests failed — check output above"
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Claude Code Memory System installed (${MODE} mode)"
echo ""
echo "  Hooks (5 automatic + PostToolUse):"
echo "    SessionStart     -> inject long-term facts, guardrails, decisions"
echo "    UserPromptSubmit -> 6-way semantic recall per prompt"
echo "    PreCompact       -> extract knowledge before compaction"
echo "    SessionEnd       -> final extraction + auto-snapshot"
echo "    StatusLine       -> monitor context %, trigger at 40/70/90%"
echo "    PostToolUse      -> code graph update + guardrail enforcement"
echo ""
echo "  Slash commands (22):"
echo "    Storage:    /remember, /forget, /forget-confirm"
echo "    Retrieval:  /knowledge, /search-memory, /reflect, /recalled, /session-learned"
echo "    Management: /review, /audit-memory, /memory-health"
echo "    Inspection: /memories, /facts, /decisions, /entities, /relationships,"
echo "                /sessions, /scopes, /observations"
echo "    Backup:     /snapshots, /export-memory, /import-memory, /restore-memory"
echo ""
echo "  Database: ~/.claude/memory/knowledge.duckdb"
echo "  Snapshots: ~/.claude/memory/snapshots/"
echo "  Settings: ${SETTINGS_FILE}"
echo ""
echo "  Requirements:"
echo "    - Ollama running with nomic-embed-text pulled:"
echo "        ollama pull nomic-embed-text"
echo "    - ANTHROPIC_API_KEY set in your environment"
echo ""
echo "  Restart Claude Code to activate."
echo "========================================================"
