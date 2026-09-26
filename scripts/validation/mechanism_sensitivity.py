#!/usr/bin/env python3
"""
P6.4 item 2 — mechanism dependence of the production (equilibrium) combustor.

T4, equilibrium product composition, phi and fuel flow at MATCHED take-off
thrust (Trent 1000-AE3 rated thrust, v5 calibration, analytic turbine and
nozzle), with each shipped mechanism profile and its native Jet-A surrogate:

    CRECK C1-C16   n-C12H26 (production "Jet-A1")
    HyChem A1      POSF10264 (C11H22 lumped species)
    HyChem A2      POSF10325 (C11H22 lumped species)

The combustor solves HP equilibrium, so the mechanism enters only through
species thermodynamics (no rate constants are used). Mechanism and surrogate
formula change together for CRECK vs HyChem; A1 vs A2 isolates the lumped
fuel species' thermo at an identical formula. Differences are compared with
the P6.2 refit-conditioned bands where those exist (outputs/p62_bands_v5.json).
A difference smaller than the bands is UNRESOLVED at that uncertainty level,
not zero.

Outputs: outputs/mechanism_sensitivity_v5.csv, outputs/mechanism_sensitivity_v5.json
Usage:   .venv/bin/python scripts/validation/mechanism_sensitivity.py
"""

import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

import lto_v5 as v5  # noqa: E402
from integrated_engine import (EmissionsEstimator, IntegratedTurbofanEngine,  # noqa: E402
                               LocalFuelBlend)
from simulation.combustor.combustor import Combustor  # noqa: E402

PROFILES = [
    ("CRECK C1-C16", "data/creck_c1c16_full.yaml", {"NC12H26": 1.0}),
    ("HyChem A1", "data/A1highT.yaml", {"POSF10264": 1.0}),
    ("HyChem A2", "data/A2NOx.yaml", {"POSF10325": 1.0}),
]
AE3_UID = "02P23RR126"
SPECIES = ("CO2", "H2O", "O2", "N2", "CO", "H2", "OH", "NO")
OUT_CSV = ROOT / "outputs" / "mechanism_sensitivity_v5.csv"
OUT_JSON = ROOT / "outputs" / "mechanism_sensitivity_v5.json"
BANDS = ROOT / "outputs" / "p62_bands_v5.json"


def main() -> None:
    for p in (OUT_CSV, OUT_JSON):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    reg = v5.load_registration()
    split = v5.load_split()
    fit = json.loads(v5.V5_FIT.read_text())
    ae3 = v5.load_rows([AE3_UID], with_targets=False)
    row = ae3[ae3["Mode"] == "TAKE-OFF"].iloc[0]
    state = v5.mode_state(fit["params"], reg["fixed_central"], row["Pressure Ratio"],
                          row["Bypass Ratio"], row["Rated Thrust (kN)"], 1.0)
    target = float(row["Target Thrust (kN)"])
    with contextlib.redirect_stdout(io.StringIO()):
        eng = IntegratedTurbofanEngine()
        eng.emissions = EmissionsEstimator(nox_fit_exclude_models=set(split["heldout_models"]))
    eng.design_point.update(state)
    eng.compressor.eta_c = reg["fixed_central"]["eta_compressor"]
    eng.turbine_design["eta_polytropic"] = reg["fixed_central"]["eta_turbine_polytropic"]
    rows = []
    for label, mech, comp in PROFILES:
        eng.mechanism_file = mech          # fuel-air ratio stoichiometry
        eng.combustor_creck = Combustor(mech)   # equilibrium combustor (production slot)
        with contextlib.redirect_stdout(io.StringIO()):
            r = eng.run_at_thrust(target, LocalFuelBlend(label, comp),
                                  combustor_efficiency=reg["fixed_central"]["eta_b"]["TAKE-OFF"])
        eq = Combustor(mech)._fresh_solutions()[1]
        eq.TPY = r["combustor"]["T_out"], r["combustor"]["p_out"], r["combustor"]["Y_out"]
        x = dict(zip(eq.species_names, eq.X))
        rows.append({
            "profile": label, "mechanism": mech, "fuel_species": ";".join(comp),
            "target_thrust_kN": target, "thrust_kN": r["performance"]["thrust_kN"],
            "phi": r["thrust_match"]["phi"], "fuel_flow_kg_s": r["performance"]["fuel_mass_flow"],
            "tsfc_mg_Ns": r["performance"]["tsfc_mg_per_Ns"], "T4_K": r["combustor"]["T_out"],
            "cp4": r["combustor"]["cp_out"], "gamma4": r["combustor"]["gamma_out"],
            **{f"X_{s}": float(x.get(s, 0.0)) for s in SPECIES},
        })
    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    for c in ("fuel_flow_kg_s", "tsfc_mg_Ns", "T4_K"):
        df[f"{c}_rel_diff_vs_CRECK_pct"] = 100.0 * (df[c] - ref[c]) / ref[c]
    bands = json.loads(BANDS.read_text()) if BANDS.exists() else None
    summary = {
        "condition": f"AE3 take-off, matched thrust {target:.1f} kN, v5 central fixed values, "
                     "single-zone equilibrium combustor",
        "max_abs_rel_diff_pct": {c: float(df[f"{c}_rel_diff_vs_CRECK_pct"].abs().max())
                                 for c in ("fuel_flow_kg_s", "tsfc_mg_Ns", "T4_K")},
        "A1_vs_A2_fuel_flow_rel_diff_pct": float(
            100 * (df.iloc[2]["fuel_flow_kg_s"] - df.iloc[1]["fuel_flow_kg_s"])
            / df.iloc[1]["fuel_flow_kg_s"]),
        "p62_bands": bands.get("design_point_takeoff") if bands else "not yet produced",
        "reading": "differences below the P6.2 bands are unresolved at that uncertainty level, not zero",
    }
    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
