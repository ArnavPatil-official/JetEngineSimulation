#!/usr/bin/env python3
"""
Phase 6 P6.1 Step 3 — seeded, rated-thrust-stratified grouped split of the ICAO
Trent 1000 engine models into a calibration group and a held-out group.

Uses cycle INPUTS only (model name, Unique ID, Pressure Ratio, Bypass Ratio,
Rated Thrust); no fuel-flow, emissions or other target column is read.

Grouping unit: the engine model (never the record). Leakage guard: models that
share an identical certification input tuple (OPR, BPR, rated thrust) in any
record are merged into one group (e.g. Trent 1000-C and -D), so no held-out
engine has an input-identical twin in the calibration group. This yields 18
groups from the 28 model names.

Stratification: groups are sorted by rated thrust (then mean OPR, then name),
taken in consecutive pairs, and one member of each pair is assigned to
calibration by a seeded draw (numpy default_rng(seed)); the group containing
the calibration reference engine (Trent 1000-AE3) is forced into calibration
(its pair's draw is still consumed, so the other assignments do not depend on
the override). With 18 groups this gives 9 calibration / 9 held-out groups.

Output: outputs/phase6/split_p61.json (refuses to overwrite).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ICAO_CSV = ROOT / "data" / "icao_engine_data.csv"
INPUT_COLS = ["Engine ID", "Unique ID", "Pressure Ratio", "Bypass Ratio", "Rated Thrust (kN)"]
REFERENCE_MODEL = "Trent 1000-AE3"


def model_name(engine_id: str) -> str:
    return engine_id.split("BYPASS RATIO")[0].strip()


def input_groups(df: pd.DataFrame) -> list[list[str]]:
    parent = {m: m for m in df["model"].unique()}

    def root(m):
        while parent[m] != m:
            m = parent[m]
        return m

    for _, g in df.groupby(["Pressure Ratio", "Bypass Ratio", "Rated Thrust (kN)"]):
        ms = sorted(g["model"].unique())
        for m in ms[1:]:
            parent[root(m)] = root(ms[0])
    groups: dict[str, list[str]] = {}
    for m in parent:
        groups.setdefault(root(m), []).append(m)
    return [sorted(v) for v in groups.values()]


def make_split(seed: int = 42) -> dict:
    df = pd.read_csv(ICAO_CSV, usecols=INPUT_COLS)
    df["model"] = df["Engine ID"].map(model_name)
    groups = input_groups(df)

    def key(g):
        sub = df[df["model"].isin(g)]
        return (float(sub["Rated Thrust (kN)"].max()), float(sub["Pressure Ratio"].mean()), g[0])

    groups.sort(key=key)
    if len(groups) % 2:
        raise ValueError("odd number of groups; pairing rule needs an even count")
    rng = np.random.default_rng(seed)
    calib, held = [], []
    pairs = []
    for i in range(0, len(groups), 2):
        pair = groups[i:i + 2]
        pick = int(rng.integers(2))
        forced = [j for j, g in enumerate(pair) if REFERENCE_MODEL in g]
        if forced:
            pick = forced[0]
        calib.append(pair[pick])
        held.append(pair[1 - pick])
        pairs.append({"pair": pair, "calibration": pair[pick], "forced_reference": bool(forced)})

    def records(gs):
        ms = [m for g in gs for m in g]
        return sorted(df.loc[df["model"].isin(ms), "Unique ID"].unique().tolist())

    return {
        "description": "P6.1 grouped split (inputs only; see scripts/validation/phase6_split.py)",
        "seed": seed,
        "n_model_names": int(df["model"].nunique()),
        "n_groups": len(groups),
        "calibration_groups": calib,
        "heldout_groups": held,
        "calibration_models": sorted(m for g in calib for m in g),
        "heldout_models": sorted(m for g in held for m in g),
        "calibration_records": records(calib),
        "heldout_records": records(held),
        "pairs": pairs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "phase6" / "split_p61.json")
    args = ap.parse_args()
    split = make_split(args.seed)
    if args.out.exists():
        old = json.loads(args.out.read_text())
        if old != split:
            raise SystemExit(f"{args.out} exists and differs; refusing to overwrite")
        print(f"{args.out} exists and is identical (reproduced)")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(split, indent=2) + "\n")
    for label in ("calibration", "heldout"):
        print(f"{label}: {len(split[label + '_groups'])} groups, "
              f"{len(split[label + '_models'])} models, {len(split[label + '_records'])} records")
        for g in split[label + "_groups"]:
            print("   ", ", ".join(g))


if __name__ == "__main__":
    main()
