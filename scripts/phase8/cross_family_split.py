#!/usr/bin/env python3
"""D3 cross-family split (docs/phase8_family_split.md).

Reads identifier/design columns only (icao_families reader), applies the
registered eligibility rule, and draws the stratified seeded held-out
families. Writes outputs/phase8/cross_family_split.json once, including its
own content SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
import icao_families as F  # noqa: E402

OUT = ROOT / "outputs" / "phase8" / "cross_family_split.json"
SEED = 20261001
N_HELDOUT = 8
FIXED_CALIBRATION = "Trent 1000"
STRATA = (("S1", 0.0, 100.0), ("S2", 100.0, 200.0), ("S3", 200.0, float("inf")))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def family_of(r) -> str:
    f = F.family(r["Manufacturer"], r["Engine Identification"])
    return "Trent 7000" if f.startswith("Trent7000") else f


def eligible(r) -> bool:
    status = (r.get("Current Engine Status") or "").strip()
    return (r.get("Eng Type") == "TF" and status == "" and r.get("Data Superseded") != "Yes"
            and all((num(r.get(c)) or 0.0) > 0.0
                    for c in ("Pressure Ratio", "B/P Ratio", "Rated Thrust (kN)")))


def main() -> int:
    if OUT.exists():
        sys.exit(f"{OUT} exists; write-once")
    sheet, _sheets, _header, keep, rows = F.read_sheet_columns(F.XLSX, "Gaseous Emissions and Smoke")
    fams: dict[str, list[dict]] = {}
    for r in rows:
        if eligible(r):
            fams.setdefault(family_of(r), []).append(r)
    info = {}
    for f, rs in sorted(fams.items()):
        thrust = statistics.median(num(r["Rated Thrust (kN)"]) for r in rs)
        stratum = next(s for s, lo, hi in STRATA if lo <= thrust < hi)
        info[f] = {"stratum": stratum, "median_rated_thrust_kN": thrust, "n_eligible_records": len(rs),
                   "geared": bool(re.match(r"^PW1\d{3}G", rs[0]["Engine Identification"])),
                   "uids": sorted(r["UID No"] for r in rs)}
    counts = {s: sum(v["stratum"] == s for v in info.values()) for s, _, _ in STRATA}
    total = sum(counts.values())
    quota = {s: N_HELDOUT * counts[s] / total for s in counts}
    alloc = {s: int(np.floor(q)) for s, q in quota.items()}
    order = sorted(counts, key=lambda s: (-(quota[s] - alloc[s]), ("S3", "S1", "S2").index(s)))
    for s in order[: N_HELDOUT - sum(alloc.values())]:
        alloc[s] += 1
    rng = np.random.default_rng(SEED)
    held = []
    for s, _, _ in STRATA:
        pool = sorted(f for f, v in info.items() if v["stratum"] == s and f != FIXED_CALIBRATION)
        if alloc[s] > len(pool):
            sys.exit(f"stratum {s} cannot supply {alloc[s]} held-out families")
        held += [str(x) for x in rng.choice(pool, size=alloc[s], replace=False)]
    calibration = sorted(f for f in info if f not in held)
    if len(calibration) < 8 or FIXED_CALIBRATION not in calibration:
        sys.exit("calibration pool below 8 families or Trent 1000 missing; no split written")
    doc = {"registration": "docs/phase8_family_split.md", "seed": SEED, "databank": F.EDITION,
           "xlsx_sha256": hashlib.sha256(F.XLSX.read_bytes()).hexdigest(),
           "columns_read": sorted(keep.values()), "n_eligible_families": total,
           "strata_counts": counts, "heldout_allocation": alloc,
           "heldout_families": sorted(held), "calibration_families": calibration,
           "families": info}
    payload = json.dumps(doc, indent=1, sort_keys=True)
    doc["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    with OUT.open("x") as f:
        f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{total} eligible families; held-out {sorted(held)}; calibration {len(calibration)}; "
          f"sha256 {doc['content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
