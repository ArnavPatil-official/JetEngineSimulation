#!/usr/bin/env bash
# Compatibility wrapper for the registered benchmark queue
# (docs/phase8_registration.md section 4; recovery registration
# docs/phase8_queue_recovery_registration.json).
# Usage unchanged: run_benchmark_queue.sh OUT_DIR "ARM VARIANT WORKLOAD WORKERS [native]" ...
# OUT_DIR and the specs must match a registered queue exactly. The queue now runs
# in scripts/phase8/ac_workflow.py, which takes the owner lease, skips validated
# terminal results, rechecks AC power immediately before each run and writes
# write-once records. It never matches process names or reads log text.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$1"; shift
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/phase8/ac_workflow.py" queue \
    --registration docs/phase8_queue_recovery_registration.json --out-dir "$OUT" -- "$@"
