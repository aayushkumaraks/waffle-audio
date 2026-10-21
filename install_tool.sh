#!/usr/bin/env bash
# ==============================================================================
# Waffle Audio — Tool & MCP Server Installer Launcher
# ==============================================================================
set -e

# Ensure PATH includes standard local bin directories
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run python install_tool.py "$@"
