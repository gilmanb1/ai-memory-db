#!/usr/bin/env bash
# install-remote.sh — One-command install of ai-memory-db from GitHub.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/gilmanb1/ai-memory-db/master/install-remote.sh | bash
#
# Or with a specific version:
#   curl -fsSL https://raw.githubusercontent.com/gilmanb1/ai-memory-db/v1.0.0/install-remote.sh | bash

set -euo pipefail

REPO="https://github.com/gilmanb1/ai-memory-db.git"
VERSION="${AI_MEMORY_DB_VERSION:-master}"
INSTALL_DIR=$(mktemp -d)

echo "=== ai-memory-db installer ==="
echo ""

# ── Check prerequisites ────────────────────────────────────────────────
for cmd in git python3 uv; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not installed."
    case "$cmd" in
      uv) echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
      *) echo "  Please install $cmd and try again." ;;
    esac
    exit 1
  fi
done

# ── Clone and install ──────────────────────────────────────────────────
echo "-> Downloading ai-memory-db ($VERSION)..."
git clone --depth 1 --branch "$VERSION" "$REPO" "$INSTALL_DIR" 2>/dev/null || \
  git clone --depth 1 "$REPO" "$INSTALL_DIR"

echo "-> Running installer..."
cd "$INSTALL_DIR"
bash install.sh "$@"

# ── Cleanup ────────────────────────────────────────────────────────────
echo ""
echo "-> Cleaning up temporary files..."
rm -rf "$INSTALL_DIR"

echo ""
echo "Done. Restart Claude Code to activate."
