#!/usr/bin/env python3
"""A4c public engine inputs: family representatives and architecture policy
(docs/phase8_p84c_registration.md section 6; JSON ``input_convergence``).

Input-only. The ICAO databank is read through the whitelisted
``icao_families.read_sheet_columns`` reader (identifier/design columns; no
target column is decoded), with the xlsx and the frozen cross-family split
checked against their registered hashes. The two original Trent 1000-E
records come from ``data/icao_engine_data.csv`` read with input columns only.

Scope: the lowest eligible UID of each of the 20 families plus both Trent
1000-E records (UIDs deduplicated); not all 89 eligible records.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
import icao_families as F  # noqa: E402

REGISTRATION = ROOT / "docs" / "phase8_p84c_registration.json"
SPLIT = ROOT / "outputs" / "phase8" / "cross_family_split.json"
ICAO_CSV = ROOT / "data" / "icao_engine_data.csv"
XLSX = F.XLSX
XLSX_KEY = "data/empirical/raw/icao_edb_2026-03.xlsx (untracked)"
TRENT_1000_E = ("11RR052", "12RR058")
# input columns only; fuel flow and emission columns are never parsed
ICAO_CSV_INPUT_COLUMNS = ["Unique ID", "Engine ID", "Mode", "Pressure Ratio", "Bypass Ratio",
                          "Rated Thrust (kN)"]
DATABANK_INPUT_COLUMNS = ("UID No", "Engine Identification", "Manufacturer", "Pressure Ratio",
                          "B/P Ratio", "Rated Thrust (kN)")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_registration(path: Path = REGISTRATION) -> dict:
    return json.loads(Path(path).read_text())


def architecture_table(reg: dict) -> dict:
    """family -> {"shafts", "geared", "class"} from the registered policy."""
    pol = reg["input_convergence"]["architecture_policy"]
    out = {}
    for cls, shafts, geared in (("three_shaft", 3, False), ("two_shaft_direct", 2, False),
                                ("geared_two_shaft", 2, True)):
        for fam in pol[cls]["families"]:
            if fam in out:
                raise ValueError(f"family {fam} has two architecture classes")
            out[fam] = {"shafts": shafts, "geared": geared, "class": cls}
    return out


def architecture(family: str, reg: dict) -> dict:
    table = architecture_table(reg)
    if family not in table:
        raise KeyError(f"no registered architecture for family {family!r}")
    return table[family]


def verify_split(raw: bytes, reg: dict) -> dict:
    """The frozen cross-family split, with its file and content hashes checked."""
    want = reg["input_convergence"]["split_file"]
    if sha256_bytes(raw) != want["sha256"]:
        raise RuntimeError("cross_family_split.json differs from the registered file")
    doc = json.loads(raw)
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    content = sha256_bytes(json.dumps(body, indent=1, sort_keys=True).encode())
    if content != doc["content_sha256"] or content != want["content_sha256"]:
        raise RuntimeError("cross_family_split.json content hash does not verify")
    return doc


def select_representatives(split: dict, reg: dict) -> list[dict]:
    """Lowest eligible UID of every eligible family (deterministic, sorted by family).
    Every family must have a registered architecture; coverage must be complete."""
    fams = split["families"]
    if len(fams) != split["n_eligible_families"]:
        raise RuntimeError("family table and eligible-family count disagree")
    table = architecture_table(reg)
    if set(fams) != set(table):
        raise RuntimeError(f"architecture policy does not cover exactly the eligible families: "
                           f"missing {sorted(set(fams) - set(table))}, extra {sorted(set(table) - set(fams))}")
    reps = []
    for fam in sorted(fams):
        uids = fams[fam]["uids"]
        if not uids:
            raise RuntimeError(f"family {fam} has no eligible UID")
        reps.append({"family": fam, "uid": min(uids), **table[fam],
                     "geared_flag_split": bool(fams[fam]["geared"])})
        if table[fam]["geared"] != bool(fams[fam]["geared"]):
            raise RuntimeError(f"geared policy disagrees with the split flag for {fam}")
    return reps


def _num(x) -> float:
    v = float(x)
    if not (v > 0.0) or v != v or v == float("inf"):
        raise ValueError(f"design input {x!r} is not finite and positive")
    return v


def read_databank_inputs(uids: list[str], xlsx: Path, reg: dict) -> dict:
    """Design inputs of the given UIDs from the whitelisted databank columns only."""
    want = reg["inputs_sha256"][XLSX_KEY]
    if not Path(xlsx).exists():
        raise FileNotFoundError(f"ICAO databank {xlsx} not present (untracked raw data); no PASS without it")
    if sha256_bytes(Path(xlsx).read_bytes()) != want:
        raise RuntimeError("ICAO databank xlsx differs from the registered edition")
    _sheet, _sheets, _header, keep, rows = F.read_sheet_columns(Path(xlsx), "Gaseous Emissions and Smoke")
    if not set(DATABANK_INPUT_COLUMNS) <= set(keep.values()):
        raise RuntimeError("databank reader does not expose the required input columns")
    out = {}
    for r in rows:
        uid = (r.get("UID No") or "").strip()
        if uid in uids:
            if uid in out:
                raise RuntimeError(f"duplicate databank UID {uid}")
            out[uid] = {"identification": (r.get("Engine Identification") or "").strip(),
                        "manufacturer": (r.get("Manufacturer") or "").strip(),
                        "opr": _num(r["Pressure Ratio"]), "bpr": _num(r["B/P Ratio"]),
                        "rated_kN": _num(r["Rated Thrust (kN)"])}
    missing = sorted(set(uids) - set(out))
    if missing:
        raise RuntimeError(f"UIDs not found in the databank: {missing}")
    return out


def read_trent_1000e_inputs(csv: Path, reg: dict) -> dict:
    """OPR, BPR and rated thrust of the two original Trent 1000-E records
    (input columns of data/icao_engine_data.csv only)."""
    import pandas as pd

    want = reg["inputs_sha256"]["data/icao_engine_data.csv"]
    if sha256_bytes(Path(csv).read_bytes()) != want:
        raise RuntimeError("data/icao_engine_data.csv differs from the registered file")
    df = pd.read_csv(csv, usecols=ICAO_CSV_INPUT_COLUMNS, float_precision="round_trip")
    out = {}
    for uid in TRENT_1000_E:
        g = df[df["Unique ID"] == uid]
        if g.empty or not g["Engine ID"].str.startswith("Trent 1000-E ").all():
            raise RuntimeError(f"{uid} is not a Trent 1000-E record")
        vals = g[["Pressure Ratio", "Bypass Ratio", "Rated Thrust (kN)"]].drop_duplicates()
        if len(vals) != 1:
            raise RuntimeError(f"{uid} has inconsistent design inputs across modes")
        v = vals.iloc[0]
        out[uid] = {"identification": "Trent 1000-E", "manufacturer": "Rolls-Royce",
                    "opr": _num(v["Pressure Ratio"]), "bpr": _num(v["Bypass Ratio"]),
                    "rated_kN": _num(v["Rated Thrust (kN)"])}
    return out


def build_cases(reps: list[dict], databank: dict, trent_e: dict, reg: dict) -> list[dict]:
    """Representative cases plus both Trent 1000-E records, deduplicated by UID."""
    cases, seen = [], set()
    for r in reps:
        cases.append({"case": f"{r['family']}:{r['uid']}", "uid": r["uid"], "family": r["family"],
                      "source": "ICAO databank (whitelisted columns)", "shafts": r["shafts"],
                      "geared": r["geared"], "architecture_class": r["class"], **databank[r["uid"]]})
        seen.add(r["uid"])
    arch = architecture("Trent 1000", reg)
    for uid in TRENT_1000_E:
        if uid in seen:
            continue
        cases.append({"case": f"Trent 1000-E:{uid}", "uid": uid, "family": "Trent 1000",
                      "source": "data/icao_engine_data.csv (input columns)", "shafts": arch["shafts"],
                      "geared": arch["geared"], "architecture_class": arch["class"], **trent_e[uid]})
        seen.add(uid)
    families = {c["family"] for c in cases}
    if len(families) != reg_n_families(reg) or not all(u in seen for u in TRENT_1000_E):
        raise RuntimeError("cases do not cover every eligible family and both Trent 1000-E records")
    return cases


def reg_n_families(reg: dict) -> int:
    return sum(len(v["families"]) for k, v in reg["input_convergence"]["architecture_policy"].items()
               if isinstance(v, dict) and "families" in v)


def load_cases(xlsx: Path = XLSX, reg: dict | None = None) -> list[dict]:
    reg = reg or load_registration()
    split = verify_split(SPLIT.read_bytes(), reg)
    reps = select_representatives(split, reg)
    databank = read_databank_inputs([r["uid"] for r in reps], xlsx, reg)
    return build_cases(reps, databank, read_trent_1000e_inputs(ICAO_CSV, reg), reg)


if __name__ == "__main__":
    for c in load_cases():
        print(f"{c['case']:<28} {c['shafts']}-shaft{' geared' if c['geared'] else '':<7} "
              f"OPR {c['opr']:.2f} BPR {c['bpr']:.2f} F {c['rated_kN']:.1f} kN")
