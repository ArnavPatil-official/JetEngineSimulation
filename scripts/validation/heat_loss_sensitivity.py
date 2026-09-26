"""
Combustor heat-loss sensitivity sweep (Phase 2.6, V9).

Sweeps the case/liner heat-loss fraction xi in {0, 2, 4, 6}% of heat release
at the calibrated take-off design point (frozen v3 calibration) for all four
fuels, and reports the effect on T4, thrust, TSFC, fuel flow, and on the
blend-to-blend deltas.

Decision rule (plan AB6): if xi = 4% moves outputs more than the blend
signal, the manuscript states that heat-loss treatment bounds the resolvable
blend effect size. This answers Reviewer 1 lines 303-310 quantitatively, and
eta_comb is renamed "lumped heat-delivery efficiency" in the provenance table.

Note: fuel flow is xi-independent by construction (FAR is set by phi), so the
sensitivity acts on temperatures and thrust only; LTO fuel-flow calibration
is unaffected by the heat-loss treatment.

Outputs: outputs/heat_loss_sensitivity.csv, outputs/plots/heat_loss_sensitivity.png

--v5 (Phase 6, P6.6): the same sweep at MATCHED take-off thrust (Trent
1000-AE3, 310.9 kN; v5 calibration outputs/calibration_v5_A2.json; A1 central
fixed values; phi solved by run_at_thrust). Here fuel flow is an output, so
heat loss acts on fuel flow and TSFC as well as on temperatures.
Outputs: outputs/heat_loss_sensitivity_v5.csv,
outputs/plots/heat_loss_sensitivity_v5.png (write-once).
"""

import sys
import json
import contextlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY, LocalFuelBlend
from simulation.fuels import HEFA_SPK, FT_SPK, ATJ_SPK

CALIBRATION_JSON = PROJECT_ROOT / "outputs" / "calibration_trent1000_ae3_v3.json"
OUT_CSV = PROJECT_ROOT / "outputs" / "heat_loss_sensitivity.csv"
OUT_PLOT = PROJECT_ROOT / "outputs" / "plots" / "heat_loss_sensitivity.png"

XI_VALUES = [0.0, 0.02, 0.04, 0.06]

# Fixed-order categorical palette (dataviz reference palette, slots 1-4)
FUEL_COLORS = {"Jet-A1": "#2a78d6", "HEFA-SPK": "#1baf7a",
               "FT-SPK": "#eda100", "ATJ-SPK": "#008300"}


def v5_runner():
    """Engine at the v5 AE3 take-off state; returns run(fuel, xi) at matched thrust."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "optimization"))
    import lto_v5 as v5
    from integrated_engine import EmissionsEstimator
    reg, split = v5.load_registration(), v5.load_split()
    fit = json.loads(v5.V5_FIT.read_text())
    fixed = reg["fixed_central"]
    r = v5.load_rows(["02P23RR126"], with_targets=False).set_index("Mode").loc["TAKE-OFF"]
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        engine = IntegratedTurbofanEngine()
        engine.emissions = EmissionsEstimator(nox_fit_exclude_models=set(split["heldout_models"]))
    engine.compressor.eta_c = fixed["eta_compressor"]
    engine.turbine_design["eta_polytropic"] = fixed["eta_turbine_polytropic"]
    engine.design_point.update(v5.mode_state(fit["params"], fixed, r["Pressure Ratio"],
                                             r["Bypass Ratio"], r["Rated Thrust (kN)"], 1.0))

    def run(fuel, xi):
        engine.design_point["combustor_heat_loss_fraction"] = xi
        return engine.run_at_thrust(float(r["Rated Thrust (kN)"]), fuel,
                                    combustor_efficiency=fixed["eta_b"]["TAKE-OFF"])
    return run


def main():
    v5_mode = "--v5" in sys.argv[1:]
    out_csv = OUT_CSV.with_name("heat_loss_sensitivity_v5.csv") if v5_mode else OUT_CSV
    out_plot = OUT_PLOT.with_name("heat_loss_sensitivity_v5.png") if v5_mode else OUT_PLOT
    if v5_mode:
        for pth in (out_csv, out_plot):
            if pth.exists():
                raise SystemExit(f"{pth} exists; refusing to overwrite")
        run = v5_runner()
    else:
        with open(CALIBRATION_JSON) as fh:
            calib = json.load(fh)
        best = calib["best_params"]
        engine = IntegratedTurbofanEngine()
        engine.design_point["combustor_pressure_loss"] = best["pressure_loss"]

        def run(fuel, xi):
            engine.design_point["combustor_heat_loss_fraction"] = xi
            return engine.run_full_cycle(fuel_blend=fuel, phi=best["phi_to"],
                                         combustor_efficiency=best["eta_combustor"])

    fuels = [
        FUEL_LIBRARY["Jet-A1"],
        LocalFuelBlend(HEFA_SPK.name, HEFA_SPK.species),
        LocalFuelBlend(FT_SPK.name, FT_SPK.species),
        LocalFuelBlend(ATJ_SPK.name, ATJ_SPK.species),
    ]

    rows = []
    for xi in XI_VALUES:
        for fuel in fuels:
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                res = run(fuel, xi)
            p = res["performance"]
            rows.append({
                "xi_pct": xi * 100,
                "Fuel": fuel.name,
                "T4_K": res["combustor"]["T_out"],
                "Thrust_kN": p["thrust_kN"],
                "TSFC_mg_per_Ns": p["tsfc_mg_per_Ns"],
                "Fuel_Flow_kg_s": p["fuel_mass_flow"],
            })
            print(f"xi={xi*100:3.0f}%  {fuel.name:<10} T4={rows[-1]['T4_K']:7.1f} K  "
                  f"thrust={rows[-1]['Thrust_kN']:6.1f} kN  "
                  f"TSFC={rows[-1]['TSFC_mg_per_Ns']:6.2f}  "
                  f"ff={rows[-1]['Fuel_Flow_kg_s']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Heat-loss effect vs blend signal
    base = df[df.xi_pct == 0]
    at4 = df[df.xi_pct == 4]
    blend_tsfc_spread = base["TSFC_mg_per_Ns"].max() - base["TSFC_mg_per_Ns"].min()
    jet_base = base[base.Fuel == "Jet-A1"].iloc[0]
    jet_4 = at4[at4.Fuel == "Jet-A1"].iloc[0]
    xi_tsfc_shift = abs(jet_4["TSFC_mg_per_Ns"] - jet_base["TSFC_mg_per_Ns"])
    xi_t4_shift = abs(jet_4["T4_K"] - jet_base["T4_K"])

    print("\n" + "=" * 70)
    print("HEAT-LOSS EFFECT vs BLEND SIGNAL (take-off design point)")
    print("=" * 70)
    print(f"Blend-to-blend TSFC spread at xi=0:      {blend_tsfc_spread:.3f} mg/(N·s)")
    print(f"xi=4% TSFC shift (Jet-A1):               {xi_tsfc_shift:.3f} mg/(N·s)")
    print(f"xi=4% T4 shift (Jet-A1):                 {xi_t4_shift:.1f} K")
    print(f"Heat loss dominates blend signal:        {xi_tsfc_shift > blend_tsfc_spread}")

    # Figure: T4 and TSFC vs xi, per fuel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for fuel in FUEL_COLORS:
        sub = df[df.Fuel == fuel]
        axes[0].plot(sub["xi_pct"], sub["T4_K"], marker="o", ms=5, lw=1.6,
                     color=FUEL_COLORS[fuel], label=fuel)
        axes[1].plot(sub["xi_pct"], sub["TSFC_mg_per_Ns"], marker="o", ms=5,
                     lw=1.6, color=FUEL_COLORS[fuel], label=fuel)
    axes[0].set_xlabel("Heat-loss fraction ξ [%]")
    axes[0].set_ylabel("T4 [K]")
    axes[0].set_title("Combustor exit temperature")
    axes[1].set_xlabel("Heat-loss fraction ξ [%]")
    axes[1].set_ylabel("TSFC [mg/(N·s)]")
    axes[1].set_title("TSFC (two-stream)")
    for ax in axes:
        ax.grid(True, lw=0.4, alpha=0.4)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Heat-loss sensitivity at " + ("matched take-off thrust (v5)" if v5_mode
                 else "calibrated take-off point"), y=1.02)
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_csv}\n       {out_plot}")


if __name__ == "__main__":
    main()
