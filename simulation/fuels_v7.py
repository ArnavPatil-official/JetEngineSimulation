"""
Phase 7 fuel representation: property-matched, published surrogates.

Through v5 the Jet A fuel was pure n-dodecane. Its heating value (44.46 MJ/kg,
CRECK thermo) is SAF-like, so the model could not separate SAF from fossil fuel
at matched thrust (P6.3: <= 0.16 % fuel-flow difference). Real Jet A contains
~19 % aromatics by mass and ~32 % cycloparaffins (NJFCP A-2, POSF 10325), which
lowers its hydrogen content and heating value.

This module reads ``data/fuel_properties_v7.yaml`` (targets, surrogates,
sources, pre-set tolerances) and provides:

* surrogate compositions as mole fractions of CRECK species;
* properties computed from the composition and the production CRECK thermo:
  mean formula, molecular weight, hydrogen mass fraction, aromatic mass
  fraction, gas-phase lower heating value (298.15 K, H2O vapour);
* mass-basis blending (ASTM blend limits are volumetric; densities are not
  modelled, as in P6.3);
* the Brem et al. (2015) nvPM relation as given in Teoh et al. (2022) SI
  Eq. S1, evaluated only inside its stated validity range (NaN outside).

The v5 surrogates in ``simulation/fuels.py`` are untouched, so every v5
artifact still reproduces.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping

import yaml

_ROOT = Path(__file__).resolve().parent.parent
PROPS_YAML = _ROOT / "data" / "fuel_properties_v7.yaml"
CRECK_YAML = _ROOT / "data" / "creck_c1c16_full.yaml"

M_C, M_H = 12.011, 1.008
T_REF = 298.15

# CRECK species with a benzene ring, used for the aromatic mass fraction.
AROMATIC_SPECIES = {"C7H8", "XYLENE", "NPBENZ", "TMBENZ", "C6H5C4H9", "C6H5C2H5",
                    "TETRALIN", "C10H7CH3", "C10H8"}

# Production names for the Phase 7 fuels.
PRODUCTION = {
    "JetA": "JetA_dooley2012",
    "HEFA": "HEFA_liang2025",
    "FT": "FT_dooley2012",
    "ATJ": "ATJ_C1matched",
}


@lru_cache(maxsize=1)
def load_properties() -> dict:
    with open(PROPS_YAML) as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def _creck():
    import warnings
    import cantera as ct
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ct.Solution(str(CRECK_YAML))


def surrogate(name: str) -> Dict[str, float]:
    """Normalised mole fractions of a registered surrogate."""
    x = dict(load_properties()["surrogates"][name]["mole_fractions"])
    tot = sum(x.values())
    return {k: v / tot for k, v in x.items()}


def _atoms(sp: str):
    c = _creck().species(sp).composition
    return c.get("C", 0.0), c.get("H", 0.0)


def _h298(sp: str) -> float:
    """Standard molar enthalpy at 298.15 K [J/kmol] from CRECK thermo."""
    return _creck().species(sp).thermo.h(T_REF)


def properties(x: Mapping[str, float]) -> dict:
    """Properties of a composition given as mole fractions of CRECK species."""
    tot = sum(v for v in x.values() if v > 0)
    x = {k: v / tot for k, v in x.items() if v > 0}
    n_c = sum(v * _atoms(k)[0] for k, v in x.items())
    n_h = sum(v * _atoms(k)[1] for k, v in x.items())
    mw = n_c * M_C + n_h * M_H                                   # g/mol of mixture
    m_arom = sum(v * (_atoms(k)[0] * M_C + _atoms(k)[1] * M_H)
                 for k, v in x.items() if k in AROMATIC_SPECIES)
    # C_nC H_nH + (nC + nH/4) O2 -> nC CO2 + nH/2 H2O(g), per mole of mixture
    h_reac = sum(v * _h298(k) for k, v in x.items()) + (n_c + n_h / 4.0) * _h298("O2")
    h_prod = n_c * _h298("CO2") + (n_h / 2.0) * _h298("H2O")
    lhv_gas = (h_reac - h_prod) / mw / 1e6                       # (J/kmol)/(kg/kmol) -> MJ/kg
    return {
        "formula_C": n_c, "formula_H": n_h, "h_over_c": n_h / n_c,
        "mw_g_mol": mw,
        "h_mass_pct": 100.0 * n_h * M_H / mw,
        "aromatic_mass_pct": 100.0 * m_arom / mw,
        "lhv_gas_MJ_kg": lhv_gas,
        "lhv_liquid_basis_MJ_kg": lhv_gas - load_properties()["heat_of_vaporization_MJ_kg"]["value"],
    }


def mass_blend(parts: Mapping[str, float]) -> Dict[str, float]:
    """Mole fractions of a mass-basis blend of registered surrogates.

    ``parts`` maps surrogate name -> mass fraction (normalised here).
    """
    tot = sum(parts.values())
    moles: Dict[str, float] = {}
    for name, w in parts.items():
        if w <= 0:
            continue
        x = surrogate(name)
        mw = properties(x)["mw_g_mol"]
        n = (w / tot) / mw                       # mol of surrogate per g of blend
        for sp, xi in x.items():
            moles[sp] = moles.get(sp, 0.0) + n * xi
    s = sum(moles.values())
    return {k: v / s for k, v in moles.items()}


def check_surrogates() -> list[dict]:
    """P7.1: compare every registered surrogate with its target, using the
    tolerances fixed in the YAML before any value was computed."""
    cfg = load_properties()
    tol = cfg["tolerances"]
    rows = []
    for name, spec in cfg["surrogates"].items():
        p = properties(surrogate(name))
        tgt = cfg["targets"][spec["validate_against"]]
        row = {"surrogate": name, "target": spec["validate_against"], **p}
        if "average_formula" in tgt:
            f = tgt["average_formula"]
            h_t = 100.0 * f["H"] * M_H / (f["C"] * M_C + f["H"] * M_H)
        else:
            h_t = tgt["h_mass_pct"]
        row["target_h_mass_pct"] = h_t
        row["h_pass"] = abs(p["h_mass_pct"] - h_t) <= tol["h_mass_pct_abs"]
        if "lhv_liquid_MJ_kg" in tgt:
            row["target_lhv_liquid_MJ_kg"] = tgt["lhv_liquid_MJ_kg"]
            row["lhv_pass"] = abs(p["lhv_liquid_basis_MJ_kg"] - tgt["lhv_liquid_MJ_kg"]) <= tol["lhv_liquid_basis_abs_MJ_kg"]
        else:
            row["target_lhv_liquid_MJ_kg"] = float("nan")
            row["lhv_pass"] = None                   # no measured LHV target
        if name.startswith("JetA"):
            a_t = tgt["class_mass_pct"]["aromatics"]
            row["target_aromatic_mass_pct"] = a_t
            row["aromatics_pass"] = abs(p["aromatic_mass_pct"] - a_t) <= tol["aromatics_mass_pct_abs"]
        else:
            row["target_aromatic_mass_pct"] = float("nan")
            row["aromatics_pass"] = None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# nvPM relations (Teoh et al. 2022 SI). Outside validity -> NaN, never
# extrapolated.
# --------------------------------------------------------------------------

def nvpm_brem_dEIn_pct(dH_pct: float, F_pct: float) -> float:
    """Eq. S1 (Brem et al. 2015): percent change in nvPM number EI."""
    c = load_properties()["nvpm"]["brem2015_via_teoh2022_S1"]
    v = c["validity"]
    if not (F_pct > v["F_pct_gt"] and 0.0 <= dH_pct < v["dH_pct_lt"]):
        return float("nan")
    return (c["alpha0"] + c["alpha1"] * F_pct) * dH_pct


# The ICAO CAEP/11 fuel-composition correction (Teoh et al. 2022 SI Eq. S2) is
# deliberately NOT implemented: see amendment P7-A1 in data/fuel_properties_v7.yaml.
