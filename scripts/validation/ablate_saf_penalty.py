"""
Ablation of the unsourced SAF combustor-efficiency penalties (Phase 1.3).

Combustor.estimate_efficiency() historically applied name-triggered efficiency
penalties on top of the equivalence-ratio parabola:

    eta = 0.995 - 0.04 * (phi - 1.0)^2          (kept)
    if 'Bio-SPK' in blend name: eta -= 0.015     (removed - unsourced)
    elif 'HEFA' in blend name:  eta -= 0.010     (removed - unsourced)
    eta clipped to [0.90, 0.999]

The penalties had no literature source and directly manufactured blend
sensitivity in reported results. This script quantifies exactly what they did:
it runs the full engine cycle for each fuel at a fixed design point with the
penalized ("ON") and parabola-only ("OFF") efficiency and reports the deltas.
Both efficiencies are computed explicitly here, so the script reproduces the
historical behavior even after the penalty block is deleted from combustor.py.

Output: outputs/ablation_saf_penalty.csv
"""

import sys
import contextlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY, LocalFuelBlend
from simulation.fuels import HEFA_SPK, FT_SPK, ATJ_SPK

PHI = 0.5  # fixed design-point equivalence ratio (project default)
OUT_CSV = PROJECT_ROOT / "outputs" / "ablation_saf_penalty.csv"

# Historical penalty rule (removed from Combustor.estimate_efficiency)
PENALTIES = {"Bio-SPK": 0.015, "HEFA": 0.010}


def eta_parabola(phi: float) -> float:
    """The phi-parabola retained in Combustor.estimate_efficiency()."""
    eta = 0.995 - 0.04 * (phi - 1.0) ** 2
    return float(max(min(eta, 0.999), 0.90))


def eta_with_penalty(phi: float, name: str) -> float:
    """Historical behavior: parabola plus name-triggered SAF penalty."""
    eta = 0.995 - 0.04 * (phi - 1.0) ** 2
    if "Bio-SPK" in name:
        eta -= PENALTIES["Bio-SPK"]
    elif "HEFA" in name:
        eta -= PENALTIES["HEFA"]
    return float(max(min(eta, 0.999), 0.90))


def main() -> None:
    # Wrap simulation.fuels surrogates as LocalFuelBlend (the engine reads
    # .composition); names are preserved so the penalty rule sees the same text.
    fuels = [
        FUEL_LIBRARY["Jet-A1"],   # no penalty either way (reference)
        FUEL_LIBRARY["Bio-SPK"],  # -1.5% penalty when ON
        FUEL_LIBRARY["HEFA-50"],  # -1.0% penalty when ON
        LocalFuelBlend(HEFA_SPK.name, HEFA_SPK.species),  # -1.0% ('HEFA' in name)
        LocalFuelBlend(FT_SPK.name, FT_SPK.species),      # no penalty either way
        LocalFuelBlend(ATJ_SPK.name, ATJ_SPK.species),    # no penalty either way
    ]

    engine = IntegratedTurbofanEngine()

    rows = []
    for fuel in fuels:
        for case, eta in [("penalty_on", eta_with_penalty(PHI, fuel.name)),
                          ("penalty_off", eta_parabola(PHI))]:
            with open(os.devnull, "w") as devnull, \
                    contextlib.redirect_stdout(devnull):
                res = engine.run_full_cycle(
                    fuel_blend=fuel, phi=PHI, combustor_efficiency=eta
                )
            perf = res["performance"]
            rows.append({
                "Fuel": fuel.name,
                "Case": case,
                "Eta_Comb": eta,
                "TSFC_mg_per_Ns": perf["tsfc_mg_per_Ns"],
                "Thrust_kN": perf["thrust_kN"],
                "T4_K": res["combustor"]["T_out"],
                "Fuel_Flow_kg_s": perf["fuel_mass_flow"],
            })
            print(f"{fuel.name:<10} {case:<12} eta={eta:.4f} "
                  f"TSFC={rows[-1]['TSFC_mg_per_Ns']:.3f} "
                  f"thrust={rows[-1]['Thrust_kN']:.2f} kN "
                  f"T4={rows[-1]['T4_K']:.1f} K "
                  f"ff={rows[-1]['Fuel_Flow_kg_s']:.4f} kg/s")

    df = pd.DataFrame(rows)

    # Per-fuel deltas (ON - OFF): the entire effect of the removed penalties
    print("\n" + "=" * 78)
    print(f"PENALTY ABLATION AT phi={PHI} (delta = penalty_on - penalty_off)")
    print("=" * 78)
    deltas = []
    for fuel_name, grp in df.groupby("Fuel", sort=False):
        on = grp[grp["Case"] == "penalty_on"].iloc[0]
        off = grp[grp["Case"] == "penalty_off"].iloc[0]
        d = {
            "Fuel": fuel_name,
            "Delta_Eta": on["Eta_Comb"] - off["Eta_Comb"],
            "Delta_TSFC_mg_per_Ns": on["TSFC_mg_per_Ns"] - off["TSFC_mg_per_Ns"],
            "Delta_TSFC_pct": (on["TSFC_mg_per_Ns"] / off["TSFC_mg_per_Ns"] - 1) * 100,
            "Delta_Thrust_kN": on["Thrust_kN"] - off["Thrust_kN"],
            "Delta_Thrust_pct": (on["Thrust_kN"] / off["Thrust_kN"] - 1) * 100,
            "Delta_T4_K": on["T4_K"] - off["T4_K"],
            "Delta_Fuel_Flow_kg_s": on["Fuel_Flow_kg_s"] - off["Fuel_Flow_kg_s"],
        }
        deltas.append(d)
        print(f"{fuel_name:<10} dEta={d['Delta_Eta']:+.4f} "
              f"dTSFC={d['Delta_TSFC_pct']:+.3f}% "
              f"dThrust={d['Delta_Thrust_pct']:+.3f}% "
              f"dT4={d['Delta_T4_K']:+.2f} K")

    # TSFC ranking with and without penalties (does the fuel ordering change?)
    on_rank = (df[df["Case"] == "penalty_on"]
               .sort_values("TSFC_mg_per_Ns")["Fuel"].tolist())
    off_rank = (df[df["Case"] == "penalty_off"]
                .sort_values("TSFC_mg_per_Ns")["Fuel"].tolist())
    print(f"\nTSFC ranking, penalties ON : {on_rank}")
    print(f"TSFC ranking, penalties OFF: {off_rank}")
    print(f"Ranking flip caused by penalties: {on_rank != off_rank}")

    out = pd.concat([df, pd.DataFrame(deltas)], axis=0, ignore_index=True, sort=False)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
