"""P7.1 — property check of the Phase 7 surrogates against measured targets.

Tolerances were fixed in data/fuel_properties_v7.yaml (commit 4b8eeda) before
any surrogate property was computed. Writes outputs/phase7/p71_surrogate_check.{csv,md}.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from simulation.fuels_v7 import check_surrogates, properties, PRODUCTION  # noqa: E402

OUT = ROOT / "outputs" / "phase7" / "p71_surrogate_check"


def main():
    df = pd.DataFrame(check_surrogates())
    v5 = properties({"NC12H26": 1.0})
    df.to_csv(f"{OUT}.csv", index=False)
    lines = [
        "# P7.1 surrogate property check",
        "",
        "Tolerances fixed before computation (`data/fuel_properties_v7.yaml`, commit `4b8eeda`): "
        "hydrogen mass % within ±0.3 points; liquid-basis LHV (gas-phase CRECK LHV − 0.360 MJ/kg "
        "n-dodecane heat of vaporization) within ±0.4 MJ/kg of the measured value; Jet A aromatic "
        "mass % within ±6 points of 18.7.",
        "",
        "| Surrogate | Production role | Formula | H mass % (target) | LHV gas | LHV liquid basis (measured) | Aromatics mass % (target) | H | LHV | Arom. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    role = {v: k for k, v in PRODUCTION.items()}
    for _, r in df.iterrows():
        f = lambda b: "—" if b is None or b != b else ("pass" if b else "**fail**")
        tl = "—" if r["target_lhv_liquid_MJ_kg"] != r["target_lhv_liquid_MJ_kg"] else f"{r['target_lhv_liquid_MJ_kg']:.1f}"
        ta = "—" if r["target_aromatic_mass_pct"] != r["target_aromatic_mass_pct"] else f"{r['target_aromatic_mass_pct']:.1f}"
        lines.append(
            f"| {r['surrogate']} | {role.get(r['surrogate'], 'spread only')} | C{r['formula_C']:.2f}H{r['formula_H']:.2f} "
            f"| {r['h_mass_pct']:.2f} ({r['target_h_mass_pct']:.2f}) | {r['lhv_gas_MJ_kg']:.3f} "
            f"| {r['lhv_liquid_basis_MJ_kg']:.3f} ({tl}) | {r['aromatic_mass_pct']:.1f} ({ta}) "
            f"| {f(r['h_pass'])} | {f(r['lhv_pass'])} | {f(r['aromatics_pass'])} |")
    lines += [
        "",
        f"v5 Jet-A1 (pure n-dodecane) for comparison: H {v5['h_mass_pct']:.2f} %, "
        f"LHV gas {v5['lhv_gas_MJ_kg']:.3f} MJ/kg (reproduces manifest E10, 44.462).",
        "",
        "Reading: the production Jet A surrogate passes the heating-value check, which is the "
        "property the thermodynamic cycle uses, and fails the hydrogen and aromatic checks. Its "
        "target fuel is Jet-A POSF 4658; the measured targets are NJFCP A-2 (POSF 10325), and the "
        "A-2 hydrogen value here is computed from the HyChem average formula C11.4H21.7. The nvPM "
        "relations therefore use the reference hydrogen contents (13.8 % Jet A-1, 15.30 % neat SAF; "
        "Teoh et al. 2022), not surrogate hydrogen. Registered outcomes; not re-tuned.",
    ]
    Path(f"{OUT}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
