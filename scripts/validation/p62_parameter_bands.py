#!/usr/bin/env python3
"""
P6.2 — refit-conditioned range bands over the fixed-parameter ranges
(registered in outputs/phase6/p61_registration.json, "bands_P6_2", amendment A1).

Seeded (42) Monte Carlo, 64 draws. Each fixed parameter is drawn uniformly over
its registered range, in this fixed order:
    combustor_pressure_loss, eta_compressor (isentropic envelope),
    eta_turbine_polytropic, fpr_rated, fan e_poly -> eta_fan = fan_isentropic(e_poly, fpr_rated),
    eta_b[TAKE-OFF], eta_b[APPROACH], eta_b[IDLE];  xi fixed at 0; no beta.
For EACH draw the fitted parameters are re-fit on the calibration group
(least_squares from the v5 optimum, max_nfev 30). Then the design point
(Trent 1000-AE3 at matched take-off thrust) and the held-out group-weighted
fuel-flow MAPE are recomputed.

Reported as refit-conditioned RANGE bands (P5-P95, min-max): range propagation
under assumed uniform ranges, NOT statistical confidence intervals. A one-at-a-
time endpoint sweep (each parameter at each end of its range, others central,
refit) is also reported, to name the limiting assumption; it is not gating.

Outputs: outputs/p62_bands_v5.csv (one row per draw / OAT case), outputs/p62_bands_v5.json
Usage:   .venv/bin/python scripts/validation/p62_parameter_bands.py
"""

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lto_v5 as v5  # noqa: E402

OUT_CSV = ROOT / "outputs" / "p62_bands_v5.csv"
OUT_JSON = ROOT / "outputs" / "p62_bands_v5.json"
AE3_UID = "02P23RR126"
SCALAR_KEYS = ("combustor_pressure_loss", "eta_compressor", "eta_turbine_polytropic", "fpr_rated")


def draw_fixed(rng, reg) -> dict:
    fr = reg["fixed_ranges"]
    d = dict(reg["fixed_central"])
    for k in SCALAR_KEYS:
        d[k] = float(rng.uniform(*fr[k]["range"]))
    e_fan = float(rng.uniform(*fr["eta_fan"]["polytropic_range"]))
    d["eta_fan"] = v5.fan_isentropic(e_fan, d["fpr_rated"])
    d["eta_b"] = {m: float(rng.uniform(*fr["eta_b"]["range"][m])) for m in v5.MODES}
    d["_fan_e_poly"] = e_fan
    return d


def oat_cases(reg) -> list[tuple[str, dict]]:
    fr, c = reg["fixed_ranges"], reg["fixed_central"]
    cases = []
    for k in SCALAR_KEYS:
        for end, val in zip(("lo", "hi"), fr[k]["range"]):
            cases.append((f"{k}={end}", dict(c, **{k: val})))
    for end, e in zip(("lo", "hi"), fr["eta_fan"]["polytropic_range"]):
        cases.append((f"fan_e_poly={end}", dict(c, eta_fan=v5.fan_isentropic(e, c["fpr_rated"]))))
    for end, i in (("lo", 0), ("hi", 1)):
        cases.append((f"eta_b={end}", dict(c, eta_b={m: fr["eta_b"]["range"][m][i] for m in v5.MODES})))
    return cases


def evaluate(model, obj, x0, free, fixed, ae3, held) -> dict:
    model.fixed = {k: v for k, v in fixed.items() if not k.startswith("_")}
    res = v5.polish(obj, free, x0, 30)
    params = obj.params(free, res.x)
    dp = model.predict(params, ae3).set_index(ae3["Mode"])
    ph = model.predict(params, held)
    ape = v5._ape(ph["ff"].to_numpy(), held["Fuel Flow (kg/s)"].to_numpy())
    to = dp.loc["TAKE-OFF"]
    return {
        **{f"fit_{k}": v for k, v in params.items()},
        "cal_sse": float(res.fun @ res.fun), "refit_nfev": int(res.nfev),
        "heldout_mape_pct": v5._wmean(ape, held["w"]),
        "heldout_unreachable": int((ph["status"] == "unreachable").sum()),
        "dp_status": to["status"], "dp_ff_kg_s": to["ff"], "dp_tsfc_mg_Ns": to.get("tsfc_mg_Ns"),
        "dp_T4_K": to.get("T4"), "dp_T3_K": to.get("T3"), "dp_phi": to["phi"],
        "dp_m_core_kg_s": to.get("m_core"),
    }


def main() -> None:
    for p in (OUT_CSV, OUT_JSON):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    reg = v5.load_registration()
    spec = reg["bands_P6_2"]
    split = v5.load_split()
    fit = json.loads(v5.V5_FIT.read_text())
    free = fit["free"]
    x0 = [fit["params"][k] for k in free]
    cal = v5.calibration_rows(split)
    held = v5.attach_groups(v5.load_rows(split["heldout_records"], with_targets=True),
                            split["heldout_groups"])
    ae3 = v5.load_rows([AE3_UID], with_targets=False)
    model = v5.V5Model(reg["fixed_central"], nox_fit_exclude_models=split["heldout_models"])
    obj = v5.Objective(model, cal, fixed_fit={k: v for k, v in fit["params"].items() if k not in free})
    rng = np.random.default_rng(42)
    rows = []
    try:
        rows.append({"case": "central", **evaluate(model, obj, x0, free, reg["fixed_central"], ae3, held)})
        for i in range(64):
            d = draw_fixed(rng, reg)
            rows.append({"case": f"draw_{i:02d}", **{f"fixed_{k}": d[k] for k in SCALAR_KEYS},
                         "fixed_eta_fan": d["eta_fan"], "fixed_fan_e_poly": d["_fan_e_poly"],
                         **{f"fixed_eta_b_{m}": d["eta_b"][m] for m in v5.MODES},
                         **evaluate(model, obj, x0, free, d, ae3, held)})
            print(f"draw {i}: heldout {rows[-1]['heldout_mape_pct']:.3f} %  "
                  f"T4 {rows[-1]['dp_T4_K']}", flush=True)
        for name, d in oat_cases(reg):
            rows.append({"case": f"oat_{name}", **evaluate(model, obj, x0, free, d, ae3, held)})
    finally:
        model.close()
    df = pd.DataFrame(rows)
    draws = df[df["case"].str.startswith("draw_")]
    out = {"method": spec["method"], "reported_as": spec["reported_as"], "n_draws": len(draws),
           "central": df.iloc[0].to_dict(), "bands": {}, "oat": {}}
    for col in ("heldout_mape_pct", "dp_ff_kg_s", "dp_tsfc_mg_Ns", "dp_T4_K", "dp_phi",
                "dp_m_core_kg_s", *[f"fit_{k}" for k in free]):
        v = pd.to_numeric(draws[col], errors="coerce").dropna()
        out["bands"][col] = {"p5": float(v.quantile(0.05)), "p95": float(v.quantile(0.95)),
                             "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    oat = df[df["case"].str.startswith("oat_")]
    for col in ("heldout_mape_pct", "dp_tsfc_mg_Ns", "dp_T4_K", "fit_W_ref"):
        if col in oat:
            out["oat"][col] = dict(zip(oat["case"], pd.to_numeric(oat[col], errors="coerce")))
    out["design_point_takeoff"] = {k: out["bands"][k] for k in ("dp_ff_kg_s", "dp_tsfc_mg_Ns", "dp_T4_K")}
    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out["bands"], indent=2))


if __name__ == "__main__":
    main()
