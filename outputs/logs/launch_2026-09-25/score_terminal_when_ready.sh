#!/usr/bin/env bash
# Supervise the already-launched P5.1 jobs; the reporter enforces all six runs.
set -u
while true; do
    complete=1
    for seed in 42 43 44; do
        [ -f "outputs/logs/train_sajben_v5_a3_s${seed}.done" ] || complete=0
    done
    [ "$complete" -eq 1 ] && break
    running=0
    for job_pid in "$@"; do
        kill -0 "$job_pid" 2>/dev/null && running=1
    done
    [ "$running" -eq 0 ] && break
    sleep 30
done
date -u
 git rev-parse HEAD
.venv/bin/python -u scripts/validation/sajben_report_p43.py
report_exit=$?
printf '{"exit_code":%s}\n' "$report_exit" > outputs/logs/launch_2026-09-25/terminal_report.done
exit "$report_exit"
