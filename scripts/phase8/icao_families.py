#!/usr/bin/env python3
"""
P8.6 / D3: list candidate engine families in the full ICAO Engine Emissions
Databank WITHOUT reading any target column (fuel flow, emission indices,
smoke, nvPM, characteristic values). Only identifier and design columns are
parsed (manufacturer, engine identification, combustor, UID, rated thrust,
pressure ratio, bypass ratio, engine type, and the "out of production"/
"out of service"/eliminated markers when present), so no held-out target is
seen before the cross-family split is registered.

The .xlsx is read with the standard library (zip + XML); no extra dependency.
The databank file is not in git: its edition and sha256 are recorded.

Usage: .venv/bin/python scripts/phase8/icao_families.py
Outputs (write-once): outputs/phase8/icao_edb_families.csv, outputs/phase8/icao_edb_families.json
"""

import collections
import csv
import hashlib
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
XLSX = ROOT / "data" / "empirical" / "raw" / "icao_edb_2026-03.xlsx"
EDITION = "ICAO Aircraft Engine Emissions Databank, EASA download 131424, 'Emissions Databank (03/2026)', accessed 2026-09-29"
OUT_CSV = ROOT / "outputs" / "phase8" / "icao_edb_families.csv"
OUT_JSON = ROOT / "outputs" / "phase8" / "icao_edb_families.json"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

# columns that may be read; matched case-insensitively on the header text
ALLOWED = [r"^manufacturer$", r"^engine identification$", r"^combustor description$", r"^uid", r"^eng type$",
           r"rated thrust", r"^pressure ratio$", r"^b/p ratio$|bypass ratio", r"out of production",
           r"out of service", r"^eliminated", r"superseded", r"^ac$|engine status"]


def col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet_columns(path: Path, sheet_name: str | None = None):
    """Header and the allowed columns only; returns (sheet, headers, rows as dicts)."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    sheets = [(s.get("name"), "xl/" + target[s.get(REL)].lstrip("/").removeprefix("xl/"))
              for s in wb.find("m:sheets", NS)]
    name, part = next((s for s in sheets if sheet_name is None or s[0] == sheet_name), sheets[0])

    def value(c):
        v = c.find("m:v", NS)
        if c.get("t") == "s" and v is not None:
            return shared[int(v.text)]
        if c.get("t") == "inlineStr":
            return "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
        return v.text if v is not None else None

    rows = ET.fromstring(z.read(part)).find("m:sheetData", NS).findall("m:row", NS)
    header = {col_index(c.get("r")): (value(c) or "").strip() for c in rows[0].findall("m:c", NS)}
    keep = {i: h for i, h in header.items() if any(re.search(p, h.lower()) for p in ALLOWED)}
    out = []
    for r in rows[1:]:
        d = {}
        for c in r.findall("m:c", NS):
            i = col_index(c.get("r"))
            if i in keep:                      # target columns are never decoded
                d[keep[i]] = value(c)
        if d:
            out.append(d)
    return name, [s[0] for s in sheets], header, keep, out


def family(manufacturer: str, ident: str) -> str:
    """Engine family = the identification up to the first variant separator
    (e.g. 'Trent 1000-AE3' -> 'Trent 1000', 'CFM56-7B27' -> 'CFM56-7B',
    'LEAP-1A26' -> 'LEAP-1A', 'PW1127G-JM' -> 'PW1100G'). Rules are listed in
    the JSON output; ambiguous cases keep the full identification."""
    s = ident.strip()
    rules = [(r"^(Trent \d+)", None), (r"^(Trent XWB)", None), (r"^(CFM56-\d[A-Z]?)", None),
             (r"^(LEAP-1[ABC])", None), (r"^(GEnx-1B|GEnx-2B)", None), (r"^(GE90)", None), (r"^(GE9X)", None),
             (r"^(CF6-\d+)", None), (r"^(CF34-\d+)", None), (r"^(PW1[1-9])\d\dG", "{0}00G"),
             (r"^(PW4\d)", "{0}00"), (r"^(PW2\d)", "{0}00"), (r"^(PW6\d)", "{0}00"),
             (r"^(V25\d\d)", "V2500"), (r"^(BR7\d\d)", None), (r"^(RB211-\S+?)-", None),
             (r"^(JT8D)", None), (r"^(JT9D)", None), (r"^(PS-90)", None), (r"^(D-\d+)", None),
             (r"^(AE ?3007)", None), (r"^(Tay)", None), (r"^(SaM146)", None), (r"^(Passport)", None)]
    for pat, fmt in rules:
        m = re.match(pat, s)
        if m:
            base = m.group(1)
            if fmt == "V2500":
                return "V2500"
            return fmt.format(base) if fmt else base
    return s


def main() -> int:
    for p in (OUT_CSV, OUT_JSON):
        if p.exists():
            print(f"{p.relative_to(ROOT)} exists; refusing to overwrite")
            return 1
    sha = hashlib.sha256(XLSX.read_bytes()).hexdigest()
    sheet, all_sheets, header, keep, rows = read_sheet_columns(XLSX, "Gaseous Emissions and Smoke")
    hcol = {h.lower(): h for h in keep.values()}
    man = next(h for k, h in hcol.items() if k == "manufacturer")
    ident = next(h for k, h in hcol.items() if k == "engine identification")
    fams = collections.defaultdict(lambda: {"manufacturer": set(), "engines": set(), "rows": 0})
    for r in rows:
        if not r.get(ident):
            continue
        f = family(r.get(man) or "", r[ident])
        fams[f]["manufacturer"].add((r.get(man) or "").strip())
        fams[f]["engines"].add(r[ident].strip())
        fams[f]["rows"] += 1
    table = sorted(({"family": f, "manufacturer": "; ".join(sorted(v["manufacturer"])),
                     "n_engine_ids": len(v["engines"]), "n_databank_rows": v["rows"]}
                    for f, v in fams.items()), key=lambda d: (-d["n_engine_ids"], d["family"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    OUT_JSON.write_text(json.dumps({
        "stage": "P8.6 / D3 candidate families (identifier columns only; no target column decoded)",
        "databank": EDITION, "xlsx_sha256": sha, "sheet_read": sheet, "sheets_in_file": all_sheets,
        "columns_read": sorted(keep.values()),
        "columns_not_read": sorted(h for i, h in header.items() if i not in keep and h),
        "n_rows_with_identification": sum(1 for r in rows if r.get(ident)),
        "n_families": len(table),
        "family_rule": "scripts/phase8/icao_families.py:family (regex prefixes; unmatched = full identification)",
    }, indent=2) + "\n")
    for d in table[:40]:
        print(f"{d['family']:<22} {d['manufacturer'][:28]:<28} ids {d['n_engine_ids']:>3}  rows {d['n_databank_rows']:>3}")
    print(f"... {len(table)} families; sheet {sheet!r}; columns read: {sorted(keep.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
