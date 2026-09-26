#!/usr/bin/env python3
"""
P6.3 — SAF blend comparison at MATCHED THRUST (R2.5, R1.4).

Registered in outputs/phase6/p63_registration.json (docs/phase6_p63_registration.md)
before the first run. Replaces the free-phi / fixed-phi blend studies of
optimize_blend.py (blends compared at unequal thrust), which stay only as the
v4 reproduction path.

Engine: Trent 1000-AE3 inputs (OPR, BPR, rated thrust), v5 calibration
(outputs/calibration_v5_A2.json) and central fixed values; single-zone
equilibrium combustor, analytic turbine and nozzle. Each blend is run at the
ICAO thrust of every LTO mode (take-off primary); phi is SOLVED, so it is not a
design variable. Outputs per blend and mode: fuel flow, TSFC, T4, phi, NOx
(ICAO-derived correlation: a function of OPR and fuel flow only, with no fuel-
composition term) and lifecycle CO2e (CORSIA Doc 06 triangular draws).

Blend fractions are MASS fractions of the four component surrogates (lifecycle
CO2e is per kg), converted to the mixture's mole fractions for Cantera. Note:
simulation.fuels.make_saf_blend mixes its fractions on a mole basis, which is
why it is not used here (a nominal 50 % ATJ blend would be 42.5 % ATJ by mass).

Design (registered): 256-point scrambled Sobol sample (seed 42) over
(SAF total in [0, 0.5], three pathway weights in [0, 1]) plus four reference
blends (Jet-A1, HEFA-50, FT-50, ATJ-50). CORSIA: each design point carries its
own seeded draw (for the variance decomposition), and 1000 common-scenario
draws (seed 42) give lifecycle bands. P6.2 conditioning: the reference blends
are re-run under each of the 64 P6.2 draws (fixed values and refitted
parameters from outputs/p62_bands_v5.csv).

Outputs (write-once), outputs/results/:
  blend_matched_thrust_v5.csv          one row per design point x mode
  blend_matched_thrust_v5_refs_p62.csv reference blends x P6.2 draw x mode
  blend_matched_thrust_v5_rankings.csv pairwise differences vs the two spreads
  blend_matched_thrust_v5.json         summary
Usage: .venv/bin/python scripts/optimization/blend_matched_thrust_v5.py
"""

import json
import sys
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
from simulation.fuels import ATJ_SPK, FT_SPK, HEFA_SPK, JET_A1, SPECIES_ATOMS  # noqa: E402

REGISTRATION = ROOT / "outputs" / "phase6" / "p63_registration.json"
BANDS_CSV = ROOT / "outputs" / "p62_bands_v5.csv"
RESULTS = ROOT / "outputs" / "results"
OUT_CSV = RESULTS / "blend_matched_thrust_v5.csv"
OUT_REFS = RESULTS / "blend_matched_thrust_v5_refs_p62.csv"
OUT_RANK = RESULTS / "blend_matched_thrust_v5_rankings.csv"
OUT_JSON = RESULTS / "blend_matched_thrust_v5.json"
AE3_UID = "02P23RR126"
COMPONENTS = {"JetA": JET_A1, "HEFA": HEFA_SPK, "FT": FT_SPK, "ATJ": ATJ_SPK}
REFERENCES = {"Jet-A1": (1.0, 0.0, 0.0, 0.0), "HEFA-50": (0.5, 0.5, 0.0, 0.0),
              "FT-50": (0.5, 0.0, 0.5, 0.0), "ATJ-50": (0.5, 0.0, 0.0, 0.5)}
PAIRS = [("HEFA-50", "Jet-A1"), ("FT-50", "Jet-A1"), ("ATJ-50", "Jet-A1"),
         ("HEFA-50", "FT-50"), ("HEFA-50", "ATJ-50"), ("FT-50", "ATJ-50")]
QUANTITIES = ("ff", "tsfc_mg_Ns", "T4", "nox_corr_g_s", "lifecycle_g_s")
M_C, M_H = 12.011, 1.008


def component_mw(fuel) -> float:
    x = fuel.normalized_species()
    return sum(v * (SPECIES_ATOMS[s][0] * M_C + SPECIES_ATOMS[s][1] * M_H) for s, v in x.items())


def mass_blend_mole_fractions(p: tuple) -> dict:
    """Mass fractions (JetA, HEFA, FT, ATJ) -> mixture species mole fractions."""
    n = {}
    for (name, fuel), pi in zip(COMPONENTS.items(), p):
        if pi <= 0.0:
            continue
        mol = pi / component_mw(fuel)            # kmol of component surrogate per kg blend
        for sp, x in fuel.normalized_species().items():
            n[sp] = n.get(sp, 0.0) + mol * x
    tot = sum(n.values())
    return {sp: v / tot for sp, v in sorted(n.items())}


def sobol_design(n: int, seed: int) -> list[tuple]:
    from scipy.stats import qmc
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    out = []
    for s01, wh, wf, wa in u:
        s = 0.5 * s01
        w = wh + wf + wa
        out.append((1.0 - s, s * wh / w, s * wf / w, s * wa / w))
    return out


def lcef_per_kg(p: tuple, draws: dict) -> float:
    """Lifecycle gCO2e per kg of blend: sum_i p_i * LHV_i * L_CEF_i (Doc 06 eq. 1 weighting)."""
    return sum(pi * COMPONENTS[k].LHV_MJ_per_kg * draws[k] for k, pi in zip(COMPONENTS, p))


def corsia(seed_formula_seed: int):
    c = yaml.safe_load((ROOT / "data" / "corsia_lca_values.yaml").read_text())
    tri = {p: c["pathways"][p]["triangular"] for p in ("HEFA", "FT", "ATJ")}
    base = c["baseline_fossil_gCO2e_MJ"]

    def per_point(i: int) -> dict:          # same per-trial formula as optimize_blend.draw_lcef
        rng = np.random.default_rng(seed_formula_seed * 1_000_003 + i)
        d = {"JetA": base}
        for p, t in tri.items():
            d[p] = float(rng.triangular(t["min"], t["mode"], t["max"]))
        return d

    def common(m: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        return [{"JetA": base, **{p: float(rng.triangular(t["min"], t["mode"], t["max"]))
                                  for p, t in tri.items()}} for _ in range(m)]

    modes = {"JetA": base, **{p: t["mode"] for p, t in tri.items()}}
    return per_point, common, modes


def run_points(model, params, fixed, ae3, blends: dict) -> pd.DataFrame:
    """Thrust-matched cycle for every (blend, mode) with the given parameters."""
    keys, tasks = [], []
    for name, p in blends.items():
        comp = mass_blend_mole_fractions(p)
        for _, r in ae3.iterrows():
            x = v5.MODE_X[r["Mode"]]
            st = v5.mode_state(params, fixed, r["Pressure Ratio"], r["Bypass Ratio"],
                               r["Rated Thrust (kN)"], x)
            tasks.append((st, fixed["eta_compressor"], fixed["eta_turbine_polytropic"],
                          fixed["eta_b"][r["Mode"]], x * r["Rated Thrust (kN)"], None, comp))
            keys.append((name, r["Mode"], float(r["Target Thrust (kN)"])))
    res = list(model.pool.map(v5.solve_task, tasks, chunksize=4))
    df = pd.DataFrame(res)
    df.insert(0, "blend", [k[0] for k in keys])
    df.insert(1, "Mode", [k[1] for k in keys])
    df.insert(2, "target_kN", [k[2] for k in keys])
    return df


def main() -> None:
    for p in (OUT_CSV, OUT_REFS, OUT_RANK, OUT_JSON):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    reg63 = json.loads(REGISTRATION.read_text())
    reg = v5.load_registration()
    split = v5.load_split()
    fit = json.loads(v5.V5_FIT.read_text())
    ae3 = v5.load_rows([AE3_UID], with_targets=False)
    d = reg63["design"]
    per_point, common, lcef_mode = corsia(d["seed"])
    blends = dict(REFERENCES)
    blends.update({f"S{i:03d}": p for i, p in enumerate(sobol_design(d["n_sobol"], d["seed"]))})
    fixed = reg["fixed_central"]
    model = v5.V5Model(fixed, nox_fit_exclude_models=split["heldout_models"])
    try:
        df = run_points(model, fit["params"], fixed, ae3, blends)
        bands = pd.read_csv(BANDS_CSV)
        draws = bands[bands["case"].str.startswith("draw_")]
        ref_rows = []
        for _, b in draws.iterrows():
            fx = dict(fixed, combustor_pressure_loss=b["fixed_combustor_pressure_loss"],
                      eta_compressor=b["fixed_eta_compressor"],
                      eta_turbine_polytropic=b["fixed_eta_turbine_polytropic"],
                      fpr_rated=b["fixed_fpr_rated"], eta_fan=b["fixed_eta_fan"],
                      eta_b={m: b[f"fixed_eta_b_{m}"] for m in v5.MODES})
            pr = {k: b[f"fit_{k}"] for k in fit["free"]}
            r = run_points(model, pr, fx, ae3, REFERENCES)
            r.insert(0, "p62_case", b["case"])
            ref_rows.append(r)
        refs = pd.concat(ref_rows, ignore_index=True)
    finally:
        model.close()

    comp_cols = ["p_JetA", "p_HEFA", "p_FT", "p_ATJ"]
    frac = pd.DataFrame([blends[n] for n in df["blend"]], columns=comp_cols, index=df.index)
    df = pd.concat([df, frac], axis=1)
    idx = {n: i for i, n in enumerate(blends)}
    pt_draw = [per_point(idx[n]) for n in df["blend"]]
    for k in ("HEFA", "FT", "ATJ"):
        df[f"lcef_draw_{k}"] = [dd[k] for dd in pt_draw]
    df["lifecycle_g_s"] = [lcef_per_kg(blends[n], lcef_mode) * ff for n, ff in zip(df["blend"], df["ff"])]
    df["lifecycle_g_s_point_draw"] = [lcef_per_kg(blends[n], dd) * ff
                                      for n, dd, ff in zip(df["blend"], pt_draw, df["ff"])]
    cdraws = common(d["n_corsia_common"], d["seed"])
    lc = {}                                   # (blend, mode) -> lifecycle under each common draw
    for _, r in df[df["blend"].isin(REFERENCES)].iterrows():
        lc[(r["blend"], r["Mode"])] = np.array([lcef_per_kg(blends[r["blend"]], cd) * r["ff"]
                                                for cd in cdraws])
    for q in ("p5", "p95"):
        df[f"lifecycle_g_s_common_{q}"] = [
            float(np.percentile(lc[(b, m)], 5 if q == "p5" else 95)) if (b, m) in lc else np.nan
            for b, m in zip(df["blend"], df["Mode"])]
    refs["lifecycle_g_s"] = [lcef_per_kg(blends[b], lcef_mode) * ff for b, ff in zip(refs["blend"], refs["ff"])]

    # Rankings: |central difference| vs (a) Monte-Carlo spread and (b) P6.2 band (registered rule)
    rows = []
    cen = df[df["blend"].isin(REFERENCES)].set_index(["blend", "Mode"])
    for mode in v5.MODES:
        for q in QUANTITIES:
            ja = refs[(refs["blend"] == "Jet-A1") & (refs["Mode"] == mode)][q]
            band_b = float(np.percentile(ja, 95) - np.percentile(ja, 5))
            for a, b in PAIRS:
                delta = float(cen.loc[(a, mode), q] - cen.loc[(b, mode), q])
                if q == "lifecycle_g_s":
                    dd = lc[(a, mode)] - lc[(b, mode)]
                    spread_a = float(np.percentile(dd, 95) - np.percentile(dd, 5))
                else:
                    spread_a = 0.0             # deterministic cycle; Brent xtol 1e-12 in phi
                pa = refs[(refs["blend"] == a) & (refs["Mode"] == mode)].set_index("p62_case")[q]
                pb = refs[(refs["blend"] == b) & (refs["Mode"] == mode)].set_index("p62_case")[q]
                paired = (pa - pb).dropna()
                claim = bool(abs(delta) > spread_a and abs(delta) > band_b)
                rows.append({
                    "Mode": mode, "quantity": q, "blend_a": a, "blend_b": b,
                    "delta_central": delta,
                    "delta_rel_pct": 100.0 * delta / float(cen.loc[(b, mode), q]),
                    "spread_a_MC_p5_p95_width": spread_a,
                    "band_b_P62_JetA1_p5_p95_width": band_b,
                    "ranking_claimed": claim,
                    "paired_P62_sign_agreement_frac": float(np.mean(np.sign(paired) == np.sign(delta))),
                    "paired_P62_delta_p5": float(np.percentile(paired, 5)),
                    "paired_P62_delta_p95": float(np.percentile(paired, 95)),
                })
    rank = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    refs.to_csv(OUT_REFS, index=False)
    rank.to_csv(OUT_RANK, index=False)
    n_unreach = int((df["status"] == "unreachable").sum())
    summary = {
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "calibration": str(v5.V5_FIT.relative_to(ROOT)),
        "n_design_points": len(blends), "n_rows": len(df), "n_unreachable": n_unreach,
        "n_p62_draws": int(draws.shape[0]),
        "n_unreachable_p62": int((refs["status"] == "unreachable").sum()),
        "reference_central": {f"{b}|{m}": {q: float(cen.loc[(b, m), q]) for q in (*QUANTITIES, "phi")}
                              for b in REFERENCES for m in v5.MODES},
        "rankings_claimed": rank[rank["ranking_claimed"]][["Mode", "quantity", "blend_a", "blend_b",
                                                            "delta_rel_pct"]].to_dict("records"),
        "note": ("NOx is the ICAO-derived correlation (OPR, fuel flow); blend NOx differences "
                 "follow fuel flow only and are not a combustion-chemistry result."),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(rank.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("n_design_points", "n_unreachable",
                                               "n_unreachable_p62", "rankings_claimed")}, indent=2))


if __name__ == "__main__":
    main()
