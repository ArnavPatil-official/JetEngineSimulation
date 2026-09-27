#!/usr/bin/env python3
"""
Verify the protected-file baseline (outputs/logs/static_thrust_accounting_protected_sha256.json).

The baseline is never regenerated. A protected file that has been archived
(Phase 6 P6.6) is verified by content at its new path, through the explicit
old -> new mapping in outputs/archive/pre_phase6/MAPPING.json. Exit status 1
if any protected file is missing or its SHA-256 differs.

Usage: .venv/bin/python scripts/validation/verify_protected_hashes.py
       .venv/bin/python scripts/validation/verify_protected_hashes.py --phase7
         (also the Phase 7 extension outputs/phase7/protected_sha256_phase7.json:
          v5/Phase 6 evidence, P7.0/P7.1 files, every model checkpoint, mechanisms,
          Sajben data; frozen before any Phase 7 computation)
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE = ROOT / "outputs" / "logs" / "static_thrust_accounting_protected_sha256.json"
MAPPING = ROOT / "outputs" / "archive" / "pre_phase6" / "MAPPING.json"
PHASE7_BASELINE = ROOT / "outputs" / "phase7" / "protected_sha256_phase7.json"


def verify(baseline_path: Path = BASELINE) -> list[str]:
    baseline = json.loads(baseline_path.read_text())
    mapping = json.loads(MAPPING.read_text()) if MAPPING.exists() else {}
    bad = []
    for old, sha in baseline.items():
        path = ROOT / mapping.get(old, old)
        ok = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha
        print(f"{'OK ' if ok else 'BAD'} {old}" + (f" -> {mapping[old]}" if old in mapping else ""))
        if not ok:
            bad.append(old)
    moved = sum(1 for k in baseline if k in mapping)
    print(f"protected files: {len(baseline)} ({moved} verified at archived paths); "
          f"mismatched/missing: {len(bad)} {bad}")
    return bad


if __name__ == "__main__":
    bad = verify()
    if "--phase7" in sys.argv[1:]:
        print(f"--- Phase 7 extension: {PHASE7_BASELINE.relative_to(ROOT)}")
        bad += verify(PHASE7_BASELINE)
    sys.exit(1 if bad else 0)
