#!/usr/bin/env bash
# Sequential registered benchmark queue (docs/phase8_registration.md section 4).
# Usage: run_benchmark_queue.sh OUT_DIR "ARM VARIANT WORKLOAD WORKERS" ...
# Each run is write-once; a failed or refused run is logged and the queue moves
# on. Every run starts only on AC power (timings on battery are not comparable).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$1"; shift
LOG="$OUT/queue_logs"
mkdir -p "$LOG"
for spec in "$@"; do
  read -r arm variant workload workers <<<"$spec"
  name="arm${arm}${variant}_${workload}_${workers}w"
  until pmset -g ps | head -1 | grep -q "AC Power"; do
    echo "$(date -u +%FT%TZ) waiting for AC power before $name" >> "$LOG/queue.log"
    sleep 300
  done
  echo "$(date -u +%FT%TZ) start $name" >> "$LOG/queue.log"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/phase8/benchmark.py" --arm "$arm" --variant "$variant" \
      --workload "$workload" --workers "$workers" --out-dir "$OUT" > "$LOG/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) end $name exit=$?" >> "$LOG/queue.log"
done
echo "$(date -u +%FT%TZ) queue done" >> "$LOG/queue.log"
