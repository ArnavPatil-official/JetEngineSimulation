#!/usr/bin/env python3
"""
P8.6: CAT-JET empirical database (SQLite, long format).

Schema: data/empirical/schema.sql. Vocabulary, units, tiers and quality
classes: data/empirical/vocabulary.yaml. Every value is stored in SI with its
original value and unit. Quantities, units and classes not in the vocabulary
are rejected. A missing value is an absent row.

QA (plan P8.6): SI round trips, corrected flow/speed recomputed from raw
fields, duplicate detection (one rig in several papers counts once),
plausibility ranges and turbine/nozzle sanity, and a cold-air energy balance
where inlet and exit temperatures and specific work are all given.

Usage:
    .venv/bin/python scripts/phase8/empirical_db.py build  [--db PATH]
    .venv/bin/python scripts/phase8/empirical_db.py init   [--db PATH]
    .venv/bin/python scripts/phase8/empirical_db.py qa     [--db PATH]
    .venv/bin/python scripts/phase8/empirical_db.py stats  [--db PATH]
    .venv/bin/python scripts/phase8/empirical_db.py export [--db PATH] --out DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
EMP = ROOT / "data" / "empirical"
SCHEMA = EMP / "schema.sql"
VOCAB = EMP / "vocabulary.yaml"
DEFAULT_DB = EMP / "catjet_empirical.sqlite"
PDF_DIR = EMP / "pdf"                       # not in git; entries carry each PDF's sha256
ENTRIES = Path(__file__).resolve().parent / "empirical_entries"
TABLES = ["source", "experiment", "operating_point", "observation", "digitisation", "split"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_vocabulary(path: Path = VOCAB) -> dict:
    """PyYAML (YAML 1.1) reads 1.0e7 (no exponent sign) as a string, so every
    numeric field is converted here; the vocabulary text is not edited."""
    v = yaml.safe_load(Path(path).read_text())
    for q in v["quantities"].values():
        q["range"] = [float(x) for x in q["range"]]
    for table in v["units"].values():
        for u, c in table.items():
            table[u] = [float(x) for x in c] if isinstance(c, list) else float(c)
    v["reference_state"] = {k: float(x) for k, x in v["reference_state"].items()}
    return v


# ---------------------------------------------------------------- units
def _factor_offset(vocab: dict, quantity: str, unit: str) -> tuple[float, float]:
    q = vocab["quantities"].get(quantity)
    if q is None:
        raise ValueError(f"quantity {quantity!r} is not in the vocabulary")
    table = vocab["units"][q["dim"]]
    if unit not in table:
        raise ValueError(f"unit {unit!r} not allowed for {quantity!r} ({q['dim']}: {sorted(table)})")
    conv = table[unit]
    return (float(conv[0]), float(conv[1])) if isinstance(conv, list) else (float(conv), 0.0)


def to_si(vocab: dict, quantity: str, value: float, unit: str) -> float:
    f, o = _factor_offset(vocab, quantity, unit)
    return value * f + o


def from_si(vocab: dict, quantity: str, value_si: float, unit: str) -> float:
    f, o = _factor_offset(vocab, quantity, unit)
    return (value_si - o) / f


def sigma_to_si(vocab: dict, quantity: str, sigma: float, unit: str) -> float:
    """An uncertainty scales by the factor only (no offset)."""
    f, _ = _factor_offset(vocab, quantity, unit)
    return abs(sigma * f)


# ---------------------------------------------------------------- database
def connect(db: Path = DEFAULT_DB, create: bool = False) -> sqlite3.Connection:
    db = Path(db)
    if not create and not db.exists():
        raise FileNotFoundError(f"{db} does not exist (run `init`)")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if create:
        conn.executescript(SCHEMA.read_text())
        meta = {"schema_sha256": sha256_file(SCHEMA), "vocabulary_sha256": sha256_file(VOCAB),
                "vocabulary_version": str(load_vocabulary()["version"])}
        for k, v in meta.items():
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (k, v))
        conn.commit()
    check_meta(conn)
    return conn


def check_meta(conn: sqlite3.Connection) -> None:
    """The database must have been built with the committed schema and vocabulary."""
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    if meta.get("schema_sha256") != sha256_file(SCHEMA):
        raise RuntimeError("schema.sql changed since this database was created")
    if meta.get("vocabulary_sha256") != sha256_file(VOCAB):
        raise RuntimeError("vocabulary.yaml changed since this database was created (amend + migrate)")


def add_source(conn, source_id, citation, type_, access_date, identifier=None, pdf_path=None,
               pdf_sha256=None, notes=None) -> None:
    if pdf_path is not None:
        digest = sha256_file(pdf_path)
        if pdf_sha256 is not None and pdf_sha256 != digest:
            raise ValueError(f"{pdf_path} sha256 differs from the given value")
        pdf_sha256 = digest
    conn.execute("INSERT INTO source VALUES (?,?,?,?,?,?,?)",
                 (source_id, citation, identifier, type_, access_date, pdf_sha256, notes))


def add_experiment(conn, experiment_id, source_id, independence_key, test_article, component,
                   working_fluid, status, fidelity_tier, quality_class, facility=None,
                   geometry=None, notes=None) -> None:
    vocab = load_vocabulary()
    if quality_class not in vocab["quality_classes"]:
        raise ValueError(f"quality class {quality_class!r} not in the vocabulary")
    if int(fidelity_tier) not in vocab["fidelity_tiers"]:
        raise ValueError(f"fidelity tier {fidelity_tier!r} not in the vocabulary")
    conn.execute("INSERT INTO experiment VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (experiment_id, source_id, independence_key, facility, test_article, component,
                  json.dumps(geometry) if geometry is not None else None, working_fluid, status,
                  int(fidelity_tier), quality_class, notes))


def add_operating_point(conn, op_id, experiment_id, run_label, p_amb_Pa=None, T_amb_K=None) -> None:
    conn.execute("INSERT INTO operating_point VALUES (?,?,?,?,?)",
                 (op_id, experiment_id, run_label, p_amb_Pa, T_amb_K))


def add_observation(conn, op_id, quantity, value, unit, role, location, sigma=None,
                    sigma_kind=None, derived_from=None) -> int:
    vocab = load_vocabulary()
    if not math.isfinite(value):
        raise ValueError("non-finite value: store nothing instead")
    if (sigma is None) != (sigma_kind is None):
        raise ValueError("sigma and sigma_kind go together")
    if sigma_kind is not None and sigma_kind not in vocab["sigma_kinds"]:
        raise ValueError(f"sigma kind {sigma_kind!r} not in the vocabulary")
    value_si = to_si(vocab, quantity, value, unit)
    sigma_si = sigma_to_si(vocab, quantity, sigma, unit) if sigma is not None else None
    cur = conn.execute(
        "INSERT INTO observation(op_id, quantity, value_si, unit_si, value_original, unit_original,"
        " sigma_si, sigma_kind, role, derived_from, location) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (op_id, quantity, value_si, vocab["quantities"][quantity]["unit"], value, unit, sigma_si,
         sigma_kind, role, json.dumps(derived_from) if derived_from is not None else None, location))
    return int(cur.lastrowid)


def add_digitisation(conn, obs_id, figure, tool, axis_calibration, repeats) -> None:
    """``repeats``: the three repeat digitisations in the observation's original unit."""
    if len(repeats) != 3:
        raise ValueError("three repeat digitisations are required")
    row = conn.execute("SELECT quantity, unit_original FROM observation WHERE obs_id=?",
                       (obs_id,)).fetchone()
    vocab = load_vocabulary()
    si = [to_si(vocab, row["quantity"], r, row["unit_original"]) for r in repeats]
    conn.execute("INSERT INTO digitisation VALUES (?,?,?,?,?,?)",
                 (obs_id, figure, tool, json.dumps(axis_calibration), json.dumps(repeats),
                  max(si) - min(si)))


# ---------------------------------------------------------------- QA
@dataclass
class Issue:
    level: str        # "error" (must be fixed) or "check" (inspect by hand)
    kind: str
    where: str
    detail: str


def _op_values(conn) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in conn.execute("SELECT op_id, quantity, value_si FROM observation"):
        out.setdefault(r["op_id"], {})[r["quantity"]] = r["value_si"]
    return out


def qa_round_trip(conn) -> list[Issue]:
    vocab = load_vocabulary()
    issues = []
    for r in conn.execute("SELECT obs_id, quantity, value_si, value_original, unit_original FROM observation"):
        back = from_si(vocab, r["quantity"], r["value_si"], r["unit_original"])
        if not math.isclose(back, r["value_original"], rel_tol=1e-12, abs_tol=1e-12):
            issues.append(Issue("error", "si_round_trip", f"obs {r['obs_id']}",
                                f"{r['value_original']} -> {back}"))
    return issues


def qa_plausibility(conn) -> list[Issue]:
    vocab = load_vocabulary()
    issues = []
    for r in conn.execute("SELECT obs_id, op_id, quantity, value_si FROM observation"):
        q = vocab["quantities"].get(r["quantity"])
        if q is None:
            issues.append(Issue("error", "vocabulary", f"obs {r['obs_id']}", f"unknown {r['quantity']}"))
            continue
        lo, hi = q["range"]
        if not lo <= r["value_si"] <= hi:
            issues.append(Issue("error", "range", f"obs {r['obs_id']} ({r['op_id']})",
                                f"{r['quantity']} = {r['value_si']} outside [{lo}, {hi}]"))
    comp = {r["op_id"]: r["component"] for r in conn.execute(
        "SELECT o.op_id, e.component FROM operating_point o JOIN experiment e USING(experiment_id)")}
    for op, v in _op_values(conn).items():
        if comp.get(op) == "turbine":
            if "Tt_in" in v and "Tt_out" in v and v["Tt_out"] >= v["Tt_in"]:
                issues.append(Issue("error", "turbine_temperature", op, "Tt_out >= Tt_in"))
            if "Pt_in" in v and "Pt_out" in v and v["Pt_out"] >= v["Pt_in"]:
                issues.append(Issue("error", "turbine_pressure", op, "Pt_out >= Pt_in"))
            if "Pt_in" in v and "Pt_out" in v and "PR" in v and \
                    not math.isclose(v["PR"], v["Pt_in"] / v["Pt_out"], rel_tol=5e-3):
                issues.append(Issue("check", "pressure_ratio", op,
                                    f"PR {v['PR']} vs Pt_in/Pt_out {v['Pt_in'] / v['Pt_out']:.6g}"))
        if comp.get(op) == "nozzle" and "Pt_in" in v and "p_back" in v and "NPR" in v and \
                not math.isclose(v["NPR"], v["Pt_in"] / v["p_back"], rel_tol=5e-3):
            issues.append(Issue("check", "nozzle_pressure_ratio", op,
                                f"NPR {v['NPR']} vs Pt_in/p_back {v['Pt_in'] / v['p_back']:.6g}"))
    return issues


def qa_corrected(conn, rtol: float = 5e-3) -> list[Issue]:
    """Stored corrected flow/speed vs recomputed from raw fields (standard day)."""
    ref = load_vocabulary()["reference_state"]
    issues = []
    for op, v in _op_values(conn).items():
        if {"mdot_corr", "mdot", "Tt_in", "Pt_in"} <= v.keys():
            calc = v["mdot"] * math.sqrt(v["Tt_in"] / ref["T_ref_K"]) / (v["Pt_in"] / ref["p_ref_Pa"])
            if not math.isclose(calc, v["mdot_corr"], rel_tol=rtol):
                issues.append(Issue("check", "corrected_flow", op,
                                    f"stored {v['mdot_corr']:.6g}, recomputed {calc:.6g}"))
        if {"N_corr", "N", "Tt_in"} <= v.keys():
            calc = v["N"] / math.sqrt(v["Tt_in"] / ref["T_ref_K"])
            if not math.isclose(calc, v["N_corr"], rel_tol=rtol):
                issues.append(Issue("check", "corrected_speed", op,
                                    f"stored {v['N_corr']:.6g}, recomputed {calc:.6g}"))
    return issues


def qa_energy_balance(conn, rtol: float = 0.02) -> list[Issue]:
    """Cold-air turbines: specific work vs h(Tt_in) - h(Tt_out) of dry air (Cantera air.yaml)."""
    import cantera as ct
    air = ct.Solution("air.yaml")
    fluid = {r["op_id"]: (r["working_fluid"], r["component"]) for r in conn.execute(
        "SELECT o.op_id, e.working_fluid, e.component FROM operating_point o JOIN experiment e "
        "USING(experiment_id)")}
    issues = []
    for op, v in _op_values(conn).items():
        wf, comp = fluid.get(op, (None, None))
        if wf != "cold_air" or comp != "turbine" or not {"work_specific", "Tt_in", "Tt_out"} <= v.keys():
            continue
        p = v.get("Pt_in", 101325.0)
        air.TPX = v["Tt_in"], p, "O2:0.21, N2:0.78, AR:0.01"
        h_in = air.enthalpy_mass
        air.TPX = v["Tt_out"], p, "O2:0.21, N2:0.78, AR:0.01"
        dh = h_in - air.enthalpy_mass
        if not math.isclose(dh, v["work_specific"], rel_tol=rtol):
            issues.append(Issue("check", "energy_balance", op,
                                f"work {v['work_specific']:.6g} J/kg vs dh {dh:.6g} J/kg"))
    return issues


def qa_duplicates(conn) -> list[Issue]:
    issues = []
    # the same rig in several sources must share an independence_key
    rows = conn.execute("SELECT experiment_id, source_id, independence_key, lower(trim(test_article)) AS art, "
                        "lower(trim(coalesce(facility,''))) AS fac FROM experiment").fetchall()
    by_article: dict[tuple, set] = {}
    for r in rows:
        by_article.setdefault((r["art"], r["fac"]), set()).add(r["independence_key"])
    for (art, fac), keys in by_article.items():
        if len(keys) > 1:
            issues.append(Issue("check", "same_article_different_key", f"{art} @ {fac}",
                                f"independence keys {sorted(keys)}: same rig counted twice?"))
    # identical value vectors at two operating points of experiments sharing a key
    sig: dict[tuple, list] = {}
    for op, v in _op_values(conn).items():
        key = conn.execute("SELECT e.independence_key FROM operating_point o JOIN experiment e "
                           "USING(experiment_id) WHERE o.op_id=?", (op,)).fetchone()[0]
        sig.setdefault((key, tuple(sorted((q, round(x, 9)) for q, x in v.items()))), []).append(op)
    for (key, _), ops in sig.items():
        if len(ops) > 1:
            issues.append(Issue("check", "duplicate_operating_point", key, f"identical values at {ops}"))
    return issues


def run_qa(conn) -> list[Issue]:
    return (qa_round_trip(conn) + qa_plausibility(conn) + qa_corrected(conn)
            + qa_energy_balance(conn) + qa_duplicates(conn))


def independent_sources(conn, component: str, role: str | None = None) -> int:
    """Number of distinct independence keys for a component (optionally for one split role)."""
    q = "SELECT COUNT(DISTINCT e.independence_key) FROM experiment e"
    args: list = [component]
    if role is not None:
        q += " JOIN split s USING(experiment_id) WHERE e.component=? AND s.role=?"
        args.append(role)
    else:
        q += " WHERE e.component=?"
    return int(conn.execute(q, args).fetchone()[0])


# ---------------------------------------------------------------- build
def build(db: Path = DEFAULT_DB, pdf_dir: Path = PDF_DIR) -> list[str]:
    """Rebuild the database from the entry modules (sorted by name) into a new
    file, then replace ``db``. The .sqlite is derived: never edit it by hand."""
    import importlib
    db = Path(db)
    tmp = db.with_suffix(".building.sqlite")
    tmp.unlink(missing_ok=True)
    conn = connect(tmp, create=True)
    names = sorted(p.stem for p in ENTRIES.glob("*.py") if p.stem != "__init__")
    sys.path.insert(0, str(ENTRIES.parent))
    try:
        for name in names:
            importlib.import_module(f"empirical_entries.{name}").enter(conn, sys.modules[__name__], pdf_dir)
        conn.commit()
    except Exception:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    conn.close()
    tmp.replace(db)
    return names


# ---------------------------------------------------------------- export
def export_csv(conn, out_dir: Path) -> dict[str, str]:
    """One CSV per table (sorted by primary key) plus sha256 of each; used to hash-freeze splits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for t in TABLES:
        cur = conn.execute(f"SELECT * FROM {t} ORDER BY 1")
        cols = [d[0] for d in cur.description]
        path = out_dir / f"{t}.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(cols)
            for row in cur:
                w.writerow([repr(x) if isinstance(x, float) else x for x in row])
        digests[path.name] = sha256_file(path)
    (out_dir / "SHA256SUMS.json").write_text(json.dumps(digests, indent=2) + "\n")
    return digests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "init", "qa", "stats", "export"])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        print(f"built {a.db} from entries: {', '.join(build(a.db))}")
        return 0
    if a.cmd == "init":
        connect(a.db, create=True).close()
        print(f"initialised {a.db}")
        return 0
    conn = connect(a.db)
    if a.cmd == "qa":
        issues = run_qa(conn)
        for i in issues:
            print(f"{i.level.upper():5s} {i.kind}: {i.where}: {i.detail}")
        n_err = sum(i.level == "error" for i in issues)
        print(f"{len(issues)} issues, {n_err} errors")
        return 1 if n_err else 0
    if a.cmd == "stats":
        for t in TABLES:
            print(f"{t}: {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
        for comp in ("turbine", "nozzle", "combustor", "engine"):
            print(f"independent sources, {comp}: {independent_sources(conn, comp)}")
        return 0
    if a.out is None:
        ap.error("export needs --out")
    for name, d in export_csv(conn, a.out).items():
        print(f"{d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
