#!/usr/bin/env bash
# run_claude_from_plan.sh — Shell wrapper for the Claude dispatcher.
#
# Usage:
#   bash scripts/run_claude_from_plan.sh              # Execute plan
#   bash scripts/run_claude_from_plan.sh --dry-run    # Dry run
#   bash scripts/run_claude_from_plan.sh --force-model opus  # Force model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate venv if it exists
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

exec python3 "$SCRIPT_DIR/dispatch_claude.py" "$@"
