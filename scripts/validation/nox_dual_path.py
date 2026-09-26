"""
Three-path NOx comparison (Phase 2.4, V5).

Paths compared per LTO mode (conditions from the frozen v3 calibration):

1. ICAO-derived correlation (fuel-flow/OPR proxy, NOT chemistry):
   EmissionsEstimator.estimate_nox — the log-log fit to the Trent 1000 family
   whose in-sample R^2 = 0.9969 was the misreported Highlights number.
2. Extended Zeldovich post-processing of CRECK equilibrium products
   (simulation/nox_chemistry.py) — thermal NO only, single zone, equilibrium
   O/O2/OH, residence time tau = V_combustor * rho / m_dot.
3. HyChem A-2 + NOx kinetic anchor (data/A2NOx.yaml, POSF10325): equilibrium
   products at the same T3/p3/phi with all N-oxides zeroed, then integrated in
   a constant-pressure reactor for the same tau with the full N-chemistry
   (thermal + N2O + NO2 routes). A fresh-reactant reactor cannot autoignite at
   LTO T3 in milliseconds, so this "kinetics on N-free burned products" design
   is the like-for-like kinetic anchor for path 2.

Reference column: ICAO certification EI-NOx for the calibration engine
(Trent 1000-AE3, UID 02P23RR126): Idle 5.53, Approach 13.66, Take-off 48.25
g/kg (data/icao_engine_data.csv).

The deliverable is the SPREAD between paths: blend-level NOx differences
smaller than this spread cannot support blend rankings in the manuscript.

Outputs: outputs/nox_dual_path.csv, outputs/plots/nox_path_comparison.png

--v5 (Phase 6, P6.6): per-mode states from the thrust-matched v5 cycle
(AE3 at ICAO thrust, outputs/calibration_v5_A2.json, A1 central fixed values,
single-zone combustor, NOx correlation refit without the held-out models).
Outputs: outputs/nox_dual_path_v5.csv, outputs/plots/nox_path_comparison_v5.png
(write-once). The three NOx paths are unchanged.
"""

import sys
import json
import contextlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt

from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY
from simulation.nox_chemistry import (
    zeldovich_ei_nox, combustor_residence_time, M_NO2,
)

CALIBRATION_JSON = PROJECT_ROOT / "outputs" / "calibration_trent1000_ae3_v3.json"
OUT_CSV = PROJECT_ROOT / "outputs" / "nox_dual_path.csv"
OUT_PLOT = PROJECT_ROOT / "outputs" / "plots" / "nox_path_comparison.png"

# Combustor volume assumption for residence time (documented in provenance):
# A_combustor_exit (0.207 m^2) x assumed length 0.5 m
V_COMBUSTOR_M3 = 0.207 * 0.5

# ICAO certification EI-NOx for Trent 1000-AE3 (UID 02P23RR126), g/kg
ICAO_EI_NOX = {"Idle": 5.53, "Approach": 13.66, "Takeoff": 48.25}

MODE_PHI_KEY = {"Idle": "phi_idle", "Approach": "phi_app", "Takeoff": "phi_to"}

# Fixed-order categorical palette (dataviz reference, slots 1-3 + neutral)
PATH_COLORS = {
    "ICAO correlation": "#2a78d6",
    "Zeldovich (CRECK eq.)": "#1baf7a",
    "HyChem A-2 kinetics": "#eda100",
}


def a2_kinetic_ei_nox(T3, p3, phi, tau):
    """Full-N-chemistry anchor: A2NOx equilibrium products, N-oxides zeroed,
    constant-pressure reactor integration for tau seconds."""
    gas = ct.Solution(str(PROJECT_ROOT / "data" / "A2NOx.yaml"))
    gas.TP = T3, p3
    gas.set_equivalence_ratio(phi, "POSF10325:1.0", "O2:0.21, N2:0.79")
    fuel_mass_frac = gas.Y[gas.species_index("POSF10325")]
    gas.equilibrate("HP")

    # Zero the N-oxides so their formation is resolved kinetically
    Y = gas.Y.copy()
    for sp in ("NO", "NO2", "N2O", "N"):
        if sp in gas.species_names:
            Y[gas.species_index(sp)] = 0.0
    gas.TPY = gas.T, gas.P, Y  # renormalizes mass fractions

    reactor = ct.ConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])
    net.advance(tau)

    g = reactor.thermo
    x_no = g.X[g.species_index("NO")]
    x_no2 = g.X[g.species_index("NO2")]
    mean_mw = g.mean_molecular_weight
    # EI as NO2-equivalent g per kg fuel (ICAO convention)
    g_no2_per_kg_mix = (x_no + x_no2) * M_NO2 / mean_mw * 1000.0
    return {
        "EI_NOx_g_per_kg_fuel": g_no2_per_kg_mix / fuel_mass_frac,
        "T_flame_K": g.T,
        "X_NO_ppm": x_no * 1e6,
        "X_NO2_ppm": x_no2 * 1e6,
    }


def v3_states():
    """Frozen v3 per-mode cycle states at fixed phi (the Phase 2.4 evidence path)."""
    with open(CALIBRATION_JSON) as fh:
        calib = json.load(fh)
    best = calib["best_params"]
    engine = IntegratedTurbofanEngine()
    for mode, state in calib["per_mode_states"].items():
        phi = best[MODE_PHI_KEY[mode]]
        engine.design_point["pi_c"] = state["pi_c"]
        engine.design_point["mass_flow_core"] = state["mass_flow_core_kg_s"]
        engine.design_point["combustor_pressure_loss"] = best["pressure_loss"]
        engine.design_point["fpr"] = 1.0 + 0.45 * state["power_fraction"] ** best["k_pi"]
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            res = engine.run_full_cycle(
                fuel_blend=FUEL_LIBRARY["Jet-A1"], phi=phi,
                combustor_efficiency=best["eta_combustor"],
            )
        yield mode, phi, state["pi_c"], best["pressure_loss"], engine, res


def v5_states():
    """Thrust-matched v5 per-mode cycle states (phi solved at the ICAO thrust)."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "optimization"))
    import lto_v5 as v5
    from integrated_engine import EmissionsEstimator
    reg, split = v5.load_registration(), v5.load_split()
    fit = json.loads(v5.V5_FIT.read_text())
    fixed = reg["fixed_central"]
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        engine = IntegratedTurbofanEngine()
        engine.emissions = EmissionsEstimator(nox_fit_exclude_models=set(split["heldout_models"]))
    engine.compressor.eta_c = fixed["eta_compressor"]
    engine.turbine_design["eta_polytropic"] = fixed["eta_turbine_polytropic"]
    names = {"IDLE": "Idle", "APPROACH": "Approach", "TAKE-OFF": "Takeoff"}
    for _, r in v5.load_rows(["02P23RR126"], with_targets=False).iterrows():
        x = v5.MODE_X[r["Mode"]]
        st = v5.mode_state(fit["params"], fixed, r["Pressure Ratio"], r["Bypass Ratio"],
                           r["Rated Thrust (kN)"], x)
        engine.design_point.update(st)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            res = engine.run_at_thrust(x * r["Rated Thrust (kN)"], FUEL_LIBRARY["Jet-A1"],
                                       combustor_efficiency=fixed["eta_b"][r["Mode"]])
        yield (names[r["Mode"]], res["thrust_match"]["phi"], st["pi_c"],
               fixed["combustor_pressure_loss"], engine, res)


def main():
    v5_mode = "--v5" in sys.argv[1:]
    out_csv = OUT_CSV.with_name("nox_dual_path_v5.csv") if v5_mode else OUT_CSV
    out_plot = OUT_PLOT.with_name("nox_path_comparison_v5.png") if v5_mode else OUT_PLOT
    if v5_mode:
        for pth in (out_csv, out_plot):
            if pth.exists():
                raise SystemExit(f"{pth} exists; refusing to overwrite")
    rows = []
    for mode, phi, pi_c, p_loss, engine, res in (v5_states() if v5_mode else v3_states()):
        T3 = res["compressor"]["T_out"]
        p3 = res["compressor"]["p_out"] * (1 - p_loss)
        m_dot_fuel = res["performance"]["fuel_mass_flow"]
        m_dot_total = res["performance"]["total_mass_flow"]
        opr = res["compressor"]["p_out"] / engine.design_point["P_ambient"]

        # Path 1: ICAO-derived correlation
        nox_corr_g_s = engine.emissions.estimate_nox(OPR=opr, m_dot_fuel=m_dot_fuel)
        ei_corr = nox_corr_g_s / m_dot_fuel

        # Path 2: Zeldovich on CRECK equilibrium products
        gas = ct.Solution("data/creck_c1c16_full.yaml")
        gas.TP = T3, p3
        gas.set_equivalence_ratio(phi, "NC12H26:1.0", "O2:0.21, N2:0.79")
        fuel_frac = gas.Y[gas.species_index("NC12H26")]
        gas.equilibrate("HP")
        tau = combustor_residence_time(V_COMBUSTOR_M3, m_dot_total, gas.density)
        zel = zeldovich_ei_nox(gas, tau, fuel_frac)

        # Path 3: HyChem A-2 kinetic anchor (same T3/p3/phi/tau)
        a2 = a2_kinetic_ei_nox(T3, p3, phi, tau)

        rows.append({
            "Mode": mode, "phi": phi, "pi_c": pi_c,
            "T3_K": T3, "p3_bar": p3 / 1e5,
            "T_flame_CRECK_K": zel["T_flame_K"],
            "T_flame_A2_K": a2["T_flame_K"],
            "tau_ms": tau * 1e3,
            "EI_ICAO_correlation": ei_corr,
            "EI_Zeldovich_CRECK": zel["EI_NOx_g_per_kg_fuel"],
            "EI_HyChem_A2": a2["EI_NOx_g_per_kg_fuel"],
            "EI_ICAO_certification": ICAO_EI_NOX[mode],
        })
        print(f"{mode:<9} T3={T3:6.1f} K  T_fl={zel['T_flame_K']:6.1f} K  "
              f"tau={tau*1e3:.1f} ms | corr={ei_corr:7.2f}  "
              f"Zeld={zel['EI_NOx_g_per_kg_fuel']:8.2f}  "
              f"A2={a2['EI_NOx_g_per_kg_fuel']:8.2f}  "
              f"ICAO cert={ICAO_EI_NOX[mode]:6.2f} g/kg")

    df = pd.DataFrame(rows)
    # Path spread per mode: max/min ratio across the three model paths
    paths = ["EI_ICAO_correlation", "EI_Zeldovich_CRECK", "EI_HyChem_A2"]
    df["path_spread_max_over_min"] = df[paths].max(axis=1) / df[paths].min(axis=1).clip(lower=1e-9)
    df.to_csv(out_csv, index=False)

    # Grouped-bar comparison figure (log scale: paths differ by orders)
    modes = df["Mode"].tolist()
    x = np.arange(len(modes))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, col) in enumerate([
            ("ICAO correlation", "EI_ICAO_correlation"),
            ("Zeldovich (CRECK eq.)", "EI_Zeldovich_CRECK"),
            ("HyChem A-2 kinetics", "EI_HyChem_A2")]):
        ax.bar(x + (i - 1) * width, df[col], width,
               label=label, color=PATH_COLORS[label], edgecolor="white")
    ax.scatter(x, df["EI_ICAO_certification"], marker="D", s=48,
               color="#1a1a19", zorder=3, label="ICAO certification (AE3)")
    ax.set_yscale("log")
    ax.set_xticks(x, modes)
    ax.set_ylabel("EI-NOx [g NO$_2$-eq / kg fuel]")
    ax.set_title("NOx model paths vs. ICAO certification\n" + (
        "(v5: single zone at overall phi; chemistry paths are low-side proxies)" if v5_mode else
        "(single-zone equilibrium-T chemistry paths are upper-bound proxies)"))
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)

    print(f"\nPath spread (max/min) per mode:")
    for _, r in df.iterrows():
        print(f"  {r['Mode']:<9} {r['path_spread_max_over_min']:8.1f}x")
    print(f"\nSaved: {out_csv}\n       {out_plot}")


if __name__ == "__main__":
    main()
