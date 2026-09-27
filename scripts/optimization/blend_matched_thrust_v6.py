#!/usr/bin/env python3
"""
P7.3 — matched-thrust SAF blends at calibration v6 (registered in
``outputs/phase7/p73_registration.json`` before the first run).

Engine: Trent 1000-AE3 inputs (ICAO 02P23RR126), the SELECTED v6 fitted
parameters (``outputs/phase7/calibration_v6.json``) held fixed for every fuel
and every draw, single-zone equilibrium combustor, analytic turbine and nozzle.
Thrust is matched at each operating point: ICAO take-off (primary), approach,
idle, and an EXTRAPOLATED climb at 85 % rated thrust (no climb data in the
ICAO CSV; eta_b at climb = the take-off eta_b of the same draw).

Fuels (MASS fractions, ``fuels_v7.mass_blend``): Jet A (Dooley 2012), and Jet A
+ HEFA / FT / ATJ at 10, 20, 30, 50 %; neat HEFA, FT, ATJ as CONTEXT. Mass
fractions are not volumetric certification limits. The alternative Jet A
surrogate (Dooley 2010) is run at the central values only: it defines the
fuel-representation spread S.

Sensitivity: the 64 FIXED-parameter draws of P6.2 (``outputs/p62_bands_v5.csv``
rows draw_00..draw_63: pressure loss, eta_c, eta_t, FPR_rated, eta_fan, eta_b
per mode). The per-draw refitted v5 parameters in that file are NOT used. The
same draws are applied to every fuel, so differences are paired. These are
conditional fixed-calibration sensitivity bands, not refit-conditioned P6.2
bands and not statistical confidence intervals.

Claim rule (registered): see ``claim()``.

Outputs (write-once), outputs/phase7/:
  p73_blends_v6_central.csv   fuel x mode, central values (incl. Dooley 2010)
  p73_blends_v6_draws.csv     draw x fuel x mode
  p73_blends_v6_claims.csv    every registered pair x mode x quantity, pass/fail + reasons
  p73_blends_v6_nvpm.csv      Brem (2015) screening, with unavailable statuses
  p73_blends_v6_lifecycle_corsia.csv  CORSIA common-draw paired differences
  p73_blends_v6.json, p73_blends_v6.md  summary
Usage: .venv/bin/python scripts/optimization/blend_matched_thrust_v6.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import lto_v5 as v5  # noqa: E402
import lto_v6 as v6  # noqa: E402
from simulation import fuels_v7  # noqa: E402

REGISTRATION = ROOT / "outputs" / "phase7" / "p73_registration.json"
DEFAULT_OUT = ROOT / "outputs" / "phase7"
AE3_UID = "02P23RR126"

OPERATING_POINTS = {            # name -> (thrust fraction, eta_b source mode, in calibration domain)
    "TAKE-OFF": (1.00, "TAKE-OFF", True),
    "APPROACH": (0.30, "APPROACH", True),
    "IDLE": (0.07, "IDLE", True),
    "CLIMB85": (0.85, "TAKE-OFF", False),
}
PATHWAYS = ("HEFA", "FT", "ATJ")
FRACTIONS = (10, 20, 30, 50)
JETA = "JetA"
JETA_ALT = "JetA_dooley2010"
QUANTITIES = ("ff", "tsfc_mg_Ns", "T4", "phi", "nox_corr_g_s", "lifecycle_g_s")
N_DRAWS = 64
SIGN_AGREEMENT_MIN = 0.95


# --------------------------------------------------------------------------
# Fuels and pairs
# --------------------------------------------------------------------------
def fuel_parts() -> dict[str, dict[str, float]]:
    """Fuel name -> {registered surrogate: mass fraction}."""
    ja = fuels_v7.PRODUCTION["JetA"]
    out = {JETA: {ja: 1.0}}
    for p in PATHWAYS:
        for f in FRACTIONS:
            out[f"{p}-{f}"] = {ja: 1.0 - f / 100.0, fuels_v7.PRODUCTION[p]: f / 100.0}
    for p in PATHWAYS:
        out[f"{p}-100"] = {fuels_v7.PRODUCTION[p]: 1.0}
    out[JETA_ALT] = {"JetA_dooley2010": 1.0}
    return out


def saf_fraction(name: str) -> float:
    return 0.0 if name in (JETA, JETA_ALT) else int(name.split("-")[1]) / 100.0


def pairs() -> list[dict]:
    """Registered pair list: (a, b, family). Relative differences use b's central value."""
    out = []
    for p in PATHWAYS:
        for f in FRACTIONS:
            out.append({"a": f"{p}-{f}", "b": JETA, "family": "SAF_vs_JetA"})
    for f in FRACTIONS:
        for a, b in (("HEFA", "FT"), ("HEFA", "ATJ"), ("FT", "ATJ")):
            out.append({"a": f"{a}-{f}", "b": f"{b}-{f}", "family": "pathway_like_fraction"})
    for p in PATHWAYS:
        out.append({"a": f"{p}-100", "b": JETA, "family": "context_neat_vs_JetA"})
    for a, b in (("HEFA", "FT"), ("HEFA", "ATJ"), ("FT", "ATJ")):
        out.append({"a": f"{a}-100", "b": f"{b}-100", "family": "context_neat_pathway"})
    return out


# --------------------------------------------------------------------------
# Lifecycle (CORSIA) and nvPM (Brem 2015)
# --------------------------------------------------------------------------
def corsia() -> dict:
    c = yaml.safe_load((ROOT / "data" / "corsia_lca_values.yaml").read_text())
    tri = {p: c["pathways"][p]["triangular"] for p in PATHWAYS}
    return {"base": float(c["baseline_fossil_gCO2e_MJ"]), "tri": tri}


def corsia_central(c: dict) -> dict:
    return {"fossil": c["base"], **{p: float(c["tri"][p]["mode"]) for p in PATHWAYS}}


def corsia_common_draws(c: dict, m: int, seed: int) -> list[dict]:
    """Same generator and order as P6.3 (blend_matched_thrust_v5.corsia().common)."""
    rng = np.random.default_rng(seed)
    return [{"fossil": c["base"], **{p: float(rng.triangular(t["min"], t["mode"], t["max"]))
                                      for p, t in c["tri"].items()}} for _ in range(m)]


def component_class(surrogate: str) -> str:
    for k, v in fuels_v7.PRODUCTION.items():
        if v == surrogate:
            return "fossil" if k == "JetA" else k
    if surrogate.startswith("JetA"):
        return "fossil"
    raise KeyError(surrogate)


@lru_cache(maxsize=None)
def lhv_liquid(surrogate: str) -> float:
    """Liquid-basis LHV [MJ/kg]: gas-phase CRECK LHV minus the ONE registered
    heat of vaporisation (n-dodecane, 0.360 MJ/kg) applied to every species."""
    return fuels_v7.properties(fuels_v7.surrogate(surrogate))["lhv_liquid_basis_MJ_kg"]


def lifecycle_factor(parts: dict[str, float], lcef: dict) -> float:
    """g CO2e per kg of fuel: sum_i mass_i * LHV_i(liquid) * LCEF_i. Multiplied by
    the cycle fuel flow [kg/s] it gives g/s. The vaporisation correction enters
    here once (liquid basis); the cycle itself stays on the gas-phase basis."""
    tot = sum(parts.values())
    return sum((w / tot) * lhv_liquid(s) * lcef[component_class(s)] for s, w in parts.items())


def nvpm_brem(fuel: str, thrust_pct: float, cfg: dict) -> dict:
    """Brem (2015) percent change in nvPM number EI vs the Jet A reference,
    hydrogen linearly mass-blended between the registered references. Strict
    validity F > 30 % and 0 <= dH < 0.6; otherwise 'unavailable' (no number)."""
    h_ref, h_saf = cfg["h_ref_pct"], cfg["h_saf_pct"]
    f = saf_fraction(fuel)
    dh = f * (h_saf - h_ref)
    reasons = []
    if not thrust_pct > 30.0:
        reasons.append(f"thrust {thrust_pct:g} % not > 30 %")
    if not (0.0 <= dh < 0.6):
        reasons.append(f"dH {dh:.2f} not in [0, 0.6)")
    val = fuels_v7.nvpm_brem_dEIn_pct(dh, thrust_pct)
    if reasons:
        assert math.isnan(val)
        return {"dH_pct": dh, "status": "unavailable", "reason": "; ".join(reasons), "dEIn_pct": None}
    return {"dH_pct": dh, "status": "screening", "reason": "", "dEIn_pct": float(val)}


# --------------------------------------------------------------------------
# Claim rule
# --------------------------------------------------------------------------
def claim(delta: float, paired: np.ndarray, spread: float, in_domain: bool,
          all_converged: bool, extra_sign_frac: float | None = None) -> tuple[bool, list[str]]:
    """Registered rule. A difference is CLAIMED iff
      (i)   the quantity/operating point is in the calibration domain;
      (ii)  every paired draw converged for both fuels;
      (iii) the central difference is nonzero;
      (iv)  at least 95 % of the 64 paired draw differences have its sign
            (a zero paired difference counts as disagreement);
      (v)   |central difference| > S, the same-mode, same-quantity central
            Dooley 2012 vs Dooley 2010 spread at unchanged v6 calibration;
      (vi)  lifecycle only: >= 95 % sign agreement over the CORSIA common draws.
    Returns (claimed, reasons for rejection)."""
    why = []
    if not in_domain:
        why.append("out of domain (extrapolated operating point)")
    if not all_converged:
        why.append("unconverged/unreachable draw")
    if not (delta != 0.0 and np.isfinite(delta)):
        why.append("zero or non-finite central difference")
    frac = float(np.mean(np.sign(paired) == np.sign(delta))) if len(paired) else 0.0
    if len(paired) != N_DRAWS or frac < SIGN_AGREEMENT_MIN:
        why.append(f"sign agreement {frac:.3f} < 0.95 over {len(paired)} draws")
    if not abs(delta) > spread:
        why.append("abs(difference) <= Dooley2012-vs-Dooley2010 spread")
    if extra_sign_frac is not None and extra_sign_frac < SIGN_AGREEMENT_MIN:
        why.append(f"CORSIA sign agreement {extra_sign_frac:.3f} < 0.95")
    return (not why), why


# --------------------------------------------------------------------------
# Cycle runs
# --------------------------------------------------------------------------
def draw_fixed(row: pd.Series, central: dict) -> dict:
    """Fixed values of one P6.2 draw. Only fixed_* columns are read; the
    refitted fit_* columns of that file are deliberately ignored."""
    return dict(central, combustor_pressure_loss=float(row["fixed_combustor_pressure_loss"]),
                eta_compressor=float(row["fixed_eta_compressor"]),
                eta_turbine_polytropic=float(row["fixed_eta_turbine_polytropic"]),
                fpr_rated=float(row["fixed_fpr_rated"]), eta_fan=float(row["fixed_eta_fan"]),
                eta_b={m: float(row[f"fixed_eta_b_{m}"]) for m in v5.MODES})


def load_draws() -> list[tuple[str, dict]]:
    bands = pd.read_csv(ROOT / "outputs" / "p62_bands_v5.csv")
    d = bands[bands["case"].str.startswith("draw_")].reset_index(drop=True)
    if list(d["case"]) != [f"draw_{i:02d}" for i in range(N_DRAWS)]:
        raise RuntimeError("P6.2 draw table is not the registered 64 draws")
    return [(r["case"], r) for _, r in d.iterrows()]


def tasks_for(params: dict, fixed: dict, ae3: pd.Series, fuels: dict) -> tuple[list, list]:
    keys, tasks = [], []
    for name, parts in fuels.items():
        comp = fuels_v7.mass_blend(parts)
        for op, (x, eta_mode, _dom) in OPERATING_POINTS.items():
            st = v5.mode_state(params, fixed, ae3["Pressure Ratio"], ae3["Bypass Ratio"],
                               ae3["Rated Thrust (kN)"], x)
            tasks.append((st, fixed["eta_compressor"], fixed["eta_turbine_polytropic"],
                          fixed["eta_b"][eta_mode], x * ae3["Rated Thrust (kN)"], None, comp))
            keys.append((name, op))
    return keys, tasks


def run(model, params, fixed, ae3, fuels) -> pd.DataFrame:
    keys, tasks = tasks_for(params, fixed, ae3, fuels)
    res = list(model.pool.map(v5.solve_task, tasks, chunksize=4))
    df = pd.DataFrame(res)
    df.insert(0, "fuel", [k[0] for k in keys])
    df.insert(1, "op", [k[1] for k in keys])
    return df


def main(out_dir: Path = DEFAULT_OUT, n_workers: int = 6) -> dict:
    reg = json.loads(REGISTRATION.read_text())
    for rel, h in reg["inputs_sha256"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != h:
            raise RuntimeError(f"registered input {rel} changed")
    names = ("p73_blends_v6_central.csv", "p73_blends_v6_draws.csv", "p73_blends_v6_claims.csv",
             "p73_blends_v6_nvpm.csv", "p73_blends_v6_lifecycle_corsia.csv", "p73_blends_v6.json",
             "p73_blends_v6.md")
    paths = {n: v6._out(out_dir, n) for n in names}
    reg6 = v6.load_registration_v6()
    hold = json.loads((out_dir / v6.HOLDOUT_JSON).read_text())
    if not hold["blend_gate"]["open"]:
        raise SystemExit(f"P7.3 gate closed: {hold['blend_gate']['reason']}")
    fit = json.loads((out_dir / v6.FIT_JSON).read_text())
    params = fit["params"]
    split = v5.load_split()
    ae3 = v5.load_rows([AE3_UID], with_targets=False).iloc[0]
    fuels = fuel_parts()
    central_fx = reg6["fixed_central"]
    model = v6.make_model(reg6, split, n_workers)
    try:
        cen = run(model, params, central_fx, ae3, fuels)
        draw_frames = []
        study = {k: v for k, v in fuels.items() if k != JETA_ALT}
        for case, row in load_draws():
            d = run(model, params, draw_fixed(row, central_fx), ae3, study)
            d.insert(0, "draw", case)
            draw_frames.append(d)
            print(f"{case}: {int((d['status'] != 'converged').sum())} unconverged", flush=True)
        draws = pd.concat(draw_frames, ignore_index=True)
    finally:
        model.close()

    c = corsia()
    lc_c = corsia_central(c)
    for df in (cen, draws):
        df["saf_mass_fraction"] = [saf_fraction(f) for f in df["fuel"]]
        df["lifecycle_factor_gCO2e_per_kg"] = [lifecycle_factor(fuels[f], lc_c) for f in df["fuel"]]
        df["lifecycle_g_s"] = df["ff"] * df["lifecycle_factor_gCO2e_per_kg"]
        df["in_calibration_domain"] = [OPERATING_POINTS[o][2] for o in df["op"]]
    cen["lhv_liquid_MJ_kg"] = [sum(w * lhv_liquid(s) for s, w in fuels[f].items()) for f in cen["fuel"]]
    cen["lhv_gas_MJ_kg"] = [fuels_v7.properties(fuels_v7.mass_blend(fuels[f]))["lhv_gas_MJ_kg"]
                            for f in cen["fuel"]]

    cd = corsia_common_draws(c, reg["lifecycle"]["n_corsia_common"], reg["lifecycle"]["seed"])
    C = cen.set_index(["fuel", "op"])
    D = draws.set_index(["draw", "fuel", "op"])
    rows, lc_rows = [], []
    for pr in pairs():
        a, b, fam = pr["a"], pr["b"], pr["family"]
        for op, (_x, _e, dom) in OPERATING_POINTS.items():
            ca, cb = C.loc[(a, op)], C.loc[(b, op)]
            conv = bool((draws.loc[(draws["fuel"].isin([a, b])) & (draws["op"] == op), "status"]
                         == "converged").all()) and ca["status"] == cb["status"] == "converged"
            lc_frac = None
            for q in QUANTITIES:
                delta = float(ca[q] - cb[q])
                pa = D.xs((a, op), level=("fuel", "op"))[q]
                pb = D.xs((b, op), level=("fuel", "op"))[q]
                paired = (pa - pb).to_numpy(dtype=float)
                spread = abs(float(C.loc[(JETA, op), q] - C.loc[(JETA_ALT, op), q]))
                extra = None
                if q == "lifecycle_g_s":
                    la = np.array([lifecycle_factor(fuels[a], d) for d in cd]) * ca["ff"]
                    lb = np.array([lifecycle_factor(fuels[b], d) for d in cd]) * cb["ff"]
                    dd = la - lb
                    extra = lc_frac = float(np.mean(np.sign(dd) == np.sign(delta)))
                    lc_rows.append({"a": a, "b": b, "family": fam, "op": op, "delta_central": delta,
                                    "corsia_p5": float(np.percentile(dd, 5)),
                                    "corsia_p95": float(np.percentile(dd, 95)),
                                    "corsia_sign_agreement": lc_frac})
                ok, why = claim(delta, paired, spread, dom, conv, extra)
                rows.append({
                    "family": fam, "a": a, "b": b, "op": op, "quantity": q,
                    "central_a": float(ca[q]), "central_b": float(cb[q]), "delta_central": delta,
                    "delta_rel_pct_of_b": 100.0 * delta / float(cb[q]) if cb[q] else np.nan,
                    "spread_S_dooley2012_vs_2010": spread,
                    "paired_sign_agreement": float(np.mean(np.sign(paired) == np.sign(delta))),
                    "paired_n": int(len(paired)),
                    "paired_delta_p5": float(np.percentile(paired, 5)),
                    "paired_delta_p95": float(np.percentile(paired, 95)),
                    "paired_delta_min": float(paired.min()), "paired_delta_max": float(paired.max()),
                    "corsia_sign_agreement": extra, "in_domain": dom, "all_converged": conv,
                    "claimed": ok, "rejection_reasons": "; ".join(why),
                })
    claims = pd.DataFrame(rows)
    nv = []
    for f in fuels:
        if f == JETA_ALT:
            continue
        for op, (x, _e, _d) in OPERATING_POINTS.items():
            r = nvpm_brem(f, 100.0 * x, reg["nvpm"])
            nv.append({"fuel": f, "op": op, "thrust_pct": 100.0 * x, **r,
                       "label": ("screening at an extrapolated engine condition (85 % climb)" if op == "CLIMB85"
                                 else "screening (empirical relation), not validated engine nvPM")
                       if r["status"] == "screening" else "unavailable (outside the registered validity)"})
    nvpm = pd.DataFrame(nv)

    in_dom = claims[claims["in_domain"]]
    summary = {
        "registration": str(REGISTRATION.relative_to(ROOT)), "calibration": v6.FIT_JSON,
        "v6_params_fixed_for_all_fuels_and_draws": params, "n_draws": N_DRAWS,
        "n_unconverged_central": int((cen["status"] != "converged").sum()),
        "n_unconverged_draws": int((draws["status"] != "converged").sum()),
        "spread_S": {op: {q: abs(float(C.loc[(JETA, op), q] - C.loc[(JETA_ALT, op), q])) for q in QUANTITIES}
                     for op in OPERATING_POINTS},
        "claims": claims[claims["claimed"]][["family", "a", "b", "op", "quantity", "delta_rel_pct_of_b"]
                                            ].to_dict("records"),
        "n_comparisons": int(len(claims)), "n_claimed": int(claims["claimed"].sum()),
        "n_claimed_by_family": claims.groupby("family")["claimed"].sum().astype(int).to_dict(),
        "n_in_domain_comparisons": int(len(in_dom)),
        "labels": reg["labels"],
    }
    cen.to_csv(paths["p73_blends_v6_central.csv"], index=False)
    draws.to_csv(paths["p73_blends_v6_draws.csv"], index=False)
    claims.to_csv(paths["p73_blends_v6_claims.csv"], index=False)
    nvpm.to_csv(paths["p73_blends_v6_nvpm.csv"], index=False)
    pd.DataFrame(lc_rows).to_csv(paths["p73_blends_v6_lifecycle_corsia.csv"], index=False)
    v5._write_new(paths["p73_blends_v6.json"], v5._json(summary))
    v5._write_new(paths["p73_blends_v6.md"], report_md(summary, claims, cen, nvpm))
    print(json.dumps({k: summary[k] for k in ("n_unconverged_central", "n_unconverged_draws",
                                               "n_comparisons", "n_claimed", "n_claimed_by_family")}, indent=2))
    return summary


def report_md(summary: dict, claims: pd.DataFrame, cen: pd.DataFrame, nvpm: pd.DataFrame) -> str:
    L = ["# P7.3 matched-thrust blends at calibration v6", "",
         "Registered in `outputs/phase7/p73_registration.json` before this run. Engine: Trent 1000-AE3 "
         "inputs, v6 parameters held fixed for every fuel and draw. Bands are **conditional "
         "fixed-calibration sensitivity** over the 64 P6.2 fixed-parameter draws (paired across fuels); "
         "they are not refit-conditioned P6.2 bands and not confidence intervals. Mass-basis blends; "
         "not volumetric certification limits; no operational approval is implied. Neat SAF is context. "
         "CLIMB85 is an extrapolated operating point (no climb data): reported, never claimed.", "",
         f"Unconverged: central {summary['n_unconverged_central']}, draws {summary['n_unconverged_draws']}.",
         "", "## Fuel-representation spread S (|Dooley 2012 − Dooley 2010|, central, v6)", "",
         "| Point | " + " | ".join(QUANTITIES) + " |", "|---|" + "---|" * len(QUANTITIES)]
    for op, s in summary["spread_S"].items():
        L.append(f"| {op} | " + " | ".join(f"{s[q]:.4g}" for q in QUANTITIES) + " |")
    L += ["", f"## Claims: {summary['n_claimed']} of {summary['n_comparisons']} comparisons", "",
          "Rule: nonzero central difference, ≥ 95 % of 64 paired draws agree in sign, abs(Δ) > S "
          "(lifecycle also ≥ 95 % over 1000 CORSIA draws), in domain, all draws converged.", "",
          "| Family | a | b | Point | Quantity | Δ % of b | sign agr. | abs(Δ)/S | Claimed | Reasons |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in claims.iterrows():
        ratio = abs(r["delta_central"]) / r["spread_S_dooley2012_vs_2010"] if r["spread_S_dooley2012_vs_2010"] else np.inf
        L.append(f"| {r['family']} | {r['a']} | {r['b']} | {r['op']} | {r['quantity']} | "
                 f"{r['delta_rel_pct_of_b']:+.3f} | {r['paired_sign_agreement']:.2f} | {ratio:.2f} | "
                 f"{'**yes**' if r['claimed'] else 'no'} | {r['rejection_reasons']} |")
    L += ["", "## nvPM number (Brem et al. 2015; screening only)", "",
          "| Fuel | Point | ΔH (pts) | ΔEI_n % | Status |", "|---|---|---|---|---|"]
    for _, r in nvpm.iterrows():
        v = "—" if r["dEIn_pct"] is None or (isinstance(r["dEIn_pct"], float) and np.isnan(r["dEIn_pct"])) \
            else f"{r['dEIn_pct']:+.2f}"
        L.append(f"| {r['fuel']} | {r['op']} | {r['dH_pct']:.2f} | {v} | {r['label']}"
                 + (f" ({r['reason']})" if r["reason"] else "") + " |")
    L += ["", "## Method limits", ""] + [f"- {x}" for x in summary["labels"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    main(Path(a.out_dir), a.workers)
