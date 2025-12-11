"""
fuels.py

Fuel surrogate definitions and blending utilities for Jet-A1 and SAF components
(HEFA-SPK, FT-SPK, ATJ-SPK), compatible with Cantera composition strings.

Usage example
-------------
from fuels import JET_A1, HEFA_SPK, FT_SPK, ATJ_SPK, make_saf_blend

blend = make_saf_blend(p_j=0.6, p_h=0.2, p_f=0.1, p_a=0.1)
fuel_string = blend.as_composition_string()
gas.set_equivalence_ratio(phi=0.35,
                          fuel=fuel_string,
                          oxidizer="O2:1.0, N2:3.76")
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, List, Tuple


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FuelSurrogate:
    """
    Represents a fuel surrogate as a mapping of species -> unnormalized mole fractions.

    The internal 'species' dictionary does NOT need to be normalized; methods below
    handle normalization when generating composition strings.
    """
    name: str
    species: Dict[str, float]

    def normalized_species(self) -> Dict[str, float]:
        """Return a new dict of species -> mole fraction normalized to sum to 1."""
        total = sum(self.species.values())
        if total <= 0.0:
            raise ValueError(f"FuelSurrogate '{self.name}' has non-positive total fraction.")
        return {sp: val / total for sp, val in self.species.items() if val > 0.0}

    def as_composition_string(self) -> str:
        """
        Return a Cantera-compatible composition string, e.g.:

        'NC12H26:0.7, IC8H18:0.3'
        """
        norm = self.normalized_species()
        # Cantera accepts non-normalized too, but normalized is clearer.
        parts = [f"{sp}:{mf:.6g}" for sp, mf in norm.items()]
        return ", ".join(parts)

    def with_scaled_fraction(self, factor: float) -> Dict[str, float]:
        """
        Return a dict of species -> (factor * mole fraction) where mole fractions
        are normalized inside this surrogate.

        This is used when combining multiple FuelSurrogate objects into a blend.
        """
        norm = self.normalized_species()
        return {sp: factor * mf for sp, mf in norm.items()}


# ---------------------------------------------------------------------------
# Species names (adjust if your mechanisms use different labels)
# ---------------------------------------------------------------------------

# NOTE:
# - Update these strings if your YAML/CTI files use different species names.
#   For example, some mechanisms may use 'nC12H26' instead of 'NC12H26'.

SP_N_DODECANE = "NC12H26"   # n-dodecane, Jet-A1/HEFA/FT surrogate
SP_ISO_OCTANE = "IC8H18"    # iso-octane, ATJ/branched surrogate
SP_N_DECANE   = "NC10H22"   # n-decane (adjust if your CRECK file uses 'C10H22' etc.)


# ---------------------------------------------------------------------------
# Base fuel surrogates
# ---------------------------------------------------------------------------

# Jet-A1: represented as pure n-dodecane (standard single-component surrogate)
JET_A1 = FuelSurrogate(
    name="Jet-A1",
    species={
        SP_N_DODECANE: 1.0,
    },
)

# HEFA-SPK: paraffinic, mid/long-chain; simple 2-component representation
# Here: 85% n-dodecane + 15% iso-octane (iso-paraffinic fraction)
HEFA_SPK = FuelSurrogate(
    name="HEFA-SPK",
    species={
        SP_N_DODECANE: 0.85,
        SP_ISO_OCTANE: 0.15,
    },
)

# FT-SPK: heavily paraffinic; represented by a mix of n-decane, n-dodecane, and a bit
# of iso-octane for iso-paraffinic content.
FT_SPK = FuelSurrogate(
    name="FT-SPK",
    species={
        SP_N_DODECANE: 0.50,
        SP_N_DECANE:   0.35,
        SP_ISO_OCTANE: 0.15,
    },
)

# ATJ-SPK: branched paraffinic, short-to-mid chain; dominated by iso-octane
# plus some n-dodecane.
ATJ_SPK = FuelSurrogate(
    name="ATJ-SPK",
    species={
        SP_ISO_OCTANE: 0.80,
        SP_N_DODECANE: 0.20,
    },
)


# ---------------------------------------------------------------------------
# Blending utilities
# ---------------------------------------------------------------------------

def blend_surrogates(fuels: List[Tuple[FuelSurrogate, float]],
                     name: str = "Blend") -> FuelSurrogate:
    """
    Blend multiple FuelSurrogate objects by high-level fractions.

    Parameters
    ----------
    fuels : List[Tuple[FuelSurrogate, float]]
        List of (FuelSurrogate, fraction) pairs.
        These fractions do NOT need to sum to 1; they will be normalized internally.
    name : str
        Optional name for the resulting blended surrogate.

    Returns
    -------
    FuelSurrogate
        A new FuelSurrogate whose species dictionary is the
        mole-fraction-weighted combination of all inputs.
    """
    total_p = sum(p for _, p in fuels)
    if total_p <= 0.0:
        raise ValueError("Total blend fraction must be positive.")

    # Normalize top-level fractions
    normalized_top = [(fuel, p / total_p) for fuel, p in fuels if p > 0.0]

    combined: Dict[str, float] = {}
    for fuel, p_norm in normalized_top:
        scaled = fuel.with_scaled_fraction(p_norm)
        for sp, val in scaled.items():
            combined[sp] = combined.get(sp, 0.0) + val

    return FuelSurrogate(name=name, species=combined)


def make_saf_blend(
    p_j: float,
    p_h: float,
    p_f: float,
    p_a: float,
    enforce_astm: bool = True,
) -> FuelSurrogate:
    """
    Create a SAF blend C_F = {p1 J, p2 H, p3 F, p4 A} using Jet-A1,
    HEFA-SPK, FT-SPK, and ATJ-SPK surrogates.

    ASTM-like constraints (optional):
    - Jet-A1 fraction >= 0.5
    - HEFA + FT + ATJ <= 0.5

    Parameters
    ----------
    p_j, p_h, p_f, p_a : float
        Overall fractions for Jet-A1, HEFA-SPK, FT-SPK, and ATJ-SPK.
        These do NOT need to sum to 1; they will be normalized.
    enforce_astm : bool
        If True, raises a ValueError if the constraints are violated.

    Returns
    -------
    FuelSurrogate
        Blended surrogate representing the overall fuel.
    """
    if enforce_astm:
        total = p_j + p_h + p_f + p_a
        if total <= 0.0:
            raise ValueError("Total SAF blend fraction must be positive.")
        j_frac = p_j / total
        saf_frac = (p_h + p_f + p_a) / total
        if j_frac < 0.5 - 1e-8:
            raise ValueError(
                f"ASTM constraint violated: Jet-A1 fraction {j_frac:.3f} < 0.5"
            )
        if saf_frac > 0.5 + 1e-8:
            raise ValueError(
                f"ASTM constraint violated: SAF fraction {saf_frac:.3f} > 0.5"
            )

    fuels = [
        (JET_A1, p_j),
        (HEFA_SPK, p_h),
        (FT_SPK, p_f),
        (ATJ_SPK, p_a),
    ]
    return blend_surrogates(fuels, name="SAF-Blend")


# ---------------------------------------------------------------------------
# Simple manual test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: 60% Jet-A1, 20% HEFA, 10% FT, 10% ATJ
    blend = make_saf_blend(p_j=0.6, p_h=0.2, p_f=0.1, p_a=0.1)
    print(f"Blend name: {blend.name}")
    print("Species (unnormalized):", blend.species)
    print("Normalized species:", blend.normalized_species())
    print("Cantera composition string:", blend.as_composition_string())
