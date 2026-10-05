#!/usr/bin/env bash
# run_phase7.sh — shell entry point of the Phase 7 durable runner (P7.5).
#
#   bash scripts/run_phase7.sh launch calib     # v6 calibration -> profile -> held-out (P7.2)
#   bash scripts/run_phase7.sh launch blends    # P7.3 blends (waits for P7.2 and its gate)
#   bash scripts/run_phase7.sh launch pinn      # P7.4 30 training runs -> report
#   bash scripts/run_phase7.sh status           # outputs/phase7/runs/STATUS.md
#
# `launch` freezes HEAD into a source snapshot outside the working tree and
# starts a detached supervisor under caffeinate; it survives the terminal or
# the dispatching session ending. Re-running `launch` for a queue resumes it:
# completed jobs are skipped (hash-verified), interrupted attempts are kept as
# evidence and the exact registered command is re-run; failed jobs are not
# retried. See scripts/phase7_supervisor.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
case "${1:-status}" in
  launch) exec "$PY" "$SCRIPT_DIR/phase7_supervisor.py" launch --queue "$2" ${3:+--commit "$3"} ;;
  status) exec "$PY" "$SCRIPT_DIR/phase7_supervisor.py" status ;;
  *) echo "usage: $0 {launch QUEUE [COMMIT]|status}" >&2; exit 2 ;;
esac
