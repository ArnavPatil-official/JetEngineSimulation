#!/usr/bin/env python3
"""
P8.0: write the Phase 8 protected-file list once (docs/phase8_registration.md section 5).

Contents: the Phase 7 list, every tracked outputs/phase7/ artifact and the
Phase 7 status record, and the Python v6 source path (the A0 reference, not
edited in Phase 8). Refuses to overwrite: the list is never regenerated.
Verify with scripts/validation/verify_protected_hashes.py --phase8.

Usage: .venv/bin/python scripts/phase8/freeze_protected_phase8.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PHASE7_LIST = ROOT / "outputs" / "phase7" / "protected_sha256_phase7.json"
OUT = ROOT / "outputs" / "phase8" / "protected_sha256_phase8.json"
V6_SOURCES = ["integrated_engine.py", "scripts/optimization/lto_v5.py", "scripts/optimization/lto_v6.py"]


def tracked(*pathspecs: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *pathspecs], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return [p for p in out if not p.endswith(".DS_Store")]


def main() -> int:
    if OUT.exists():
        print(f"{OUT.relative_to(ROOT)} exists; refusing to overwrite")
        return 1
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "outputs/phase7", "simulation",
                            *V6_SOURCES], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        print("protected paths have uncommitted changes:\n" + dirty)
        return 1
    paths = set(json.loads(PHASE7_LIST.read_text()))
    paths |= set(tracked("outputs/phase7", "outputs/phase7_execution_status.md"))
    paths |= set(tracked("simulation/*.py")) | set(V6_SOURCES)
    table = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(table)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
