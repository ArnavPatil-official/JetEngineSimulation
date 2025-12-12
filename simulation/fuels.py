"""
Fuel Surrogate Definitions and Blending Utilities.

This module defines surrogate fuel representations for conventional jet fuel (Jet-A1)
and Sustainable Aviation Fuels (SAF) using n-alkane mixtures compatible with chemical
kinetics mechanisms like CRECK C1-C16.

Fuel Surrogates:
- Jet-A1: Pure n-dodecane (NC12H26)
- HEFA-SPK: Hydroprocessed esters and fatty acids (85% n-dodecane, 15% iso-octane)
- FT-SPK: Fischer-Tropsch synthetic paraffinic kerosene (n-decane + n-dodecane + iso-octane)
- ATJ-SPK: Alcohol-to-jet (80% iso-octane, 20% n-dodecane)

These surrogates approximate the combustion chemistry of real fuels while maintaining
computational tractability for detailed chemical kinetics simulations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping, List, Tuple


# ---------------------------------------------------------------------------
# Fuel Surrogate Data Structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FuelSurrogate:
    """
    Represents a fuel as a mixture of surrogate chemical species.

    Each fuel is represented by a weighted combination of simple hydrocarbons
    (e.g., n-dodecane, iso-octane) that approximate the combustion behavior
    of the real complex fuel mixture.

    Attributes:
        name: Fuel identifier (e.g., "Jet-A1", "HEFA-SPK")
        species: Dictionary mapping species names to mole fractions (unnormalized)
        LHV_MJ_per_kg: Lower Heating Value [MJ/kg] - for energy calculations
        carbon_fraction: Mass fraction of carbon (w_C = m_C / m_fuel) - for CO₂ emissions
    """
    name: str
    species: Dict[str, float]
    LHV_MJ_per_kg: float = 43.2  # Default: typical jet fuel LHV
    carbon_fraction: float = 0.857  # Default: C12H26 has 12*12/(12*12 + 26*1) = 0.857

    def normalized_species(self) -> Dict[str, float]:
        """
        Normalize species mole fractions to sum to 1.0.

        Returns:
            Dictionary of species -> normalized mole fractions
        """
        total = sum(self.species.values())
        if total <= 0.0:
            raise ValueError(f"FuelSurrogate '{self.name}' has non-positive total fraction.")
        return {sp: val / total for sp, val in self.species.items() if val > 0.0}

    def as_composition_string(self) -> str:
        """
        Convert fuel composition to Cantera-compatible format string.

        Returns:
            Composition string like "NC12H26:0.7, IC8H18:0.3" for use with
            Cantera's set_equivalence_ratio() method
        """
        norm = self.normalized_species()
        parts = [f"{sp}:{mf:.6g}" for sp, mf in norm.items()]
        return ", ".join(parts)

    def with_scaled_fraction(self, factor: float) -> Dict[str, float]:
        """
        Scale all species mole fractions by a constant factor.

        This is used when blending multiple fuel surrogates together to create
        a composite fuel (e.g., 60% Jet-A1 + 40% HEFA).

        Args:
            factor: Scaling factor to apply to normalized mole fractions

        Returns:
            Dictionary of species -> scaled mole fractions
        """
        norm = self.normalized_species()
        return {sp: factor * mf for sp, mf in norm.items()}


# ---------------------------------------------------------------------------
# Chemical Species Identifiers
# ---------------------------------------------------------------------------
# Species names must match those in the Cantera mechanism YAML file

SP_N_DODECANE = "NC12H26"   # n-dodecane: C12H26 linear alkane (kerosene-range)
SP_ISO_OCTANE = "IC8H18"    # iso-octane: C8H18 branched alkane (gasoline-range)
SP_N_DECANE   = "NC10H22"   # n-decane: C10H22 linear alkane (short kerosene)


# ---------------------------------------------------------------------------
# Pre-defined Fuel Surrogates
# ---------------------------------------------------------------------------

# Conventional Jet-A1: Pure n-dodecane surrogate
# This is the most common single-component representation for conventional jet fuel
# C12H26: M_C = 12*12 = 144 g/mol, M_H = 26*1 = 26 g/mol, M_total = 170 g/mol
# Carbon fraction: w_C = 144/170 = 0.847
JET_A1 = FuelSurrogate(
    name="Jet-A1",
    species={
        SP_N_DODECANE: 1.0,
    },
    LHV_MJ_per_kg=44.1,  # n-dodecane LHV
    carbon_fraction=0.847  # C12H26 carbon mass fraction
)

# HEFA-SPK: Hydroprocessed Esters and Fatty Acids - Synthetic Paraffinic Kerosene
# Bio-derived SAF with predominantly straight-chain alkanes and some branching
# Blend: 85% C12H26 (w_C=0.847) + 15% C8H18 (w_C=0.842)
# Weighted w_C ≈ 0.85*0.847 + 0.15*0.842 = 0.846
HEFA_SPK = FuelSurrogate(
    name="HEFA-SPK",
    species={
        SP_N_DODECANE: 0.85,  # Dominant straight-chain component
        SP_ISO_OCTANE: 0.15,  # Represents iso-paraffinic content
    },
    LHV_MJ_per_kg=44.0,  # Slightly lower than pure dodecane due to iso-octane
    carbon_fraction=0.846  # Weighted average
)

# FT-SPK: Fischer-Tropsch Synthetic Paraffinic Kerosene
# Gas-to-liquid or coal-to-liquid SAF with high paraffinic content
# Blend: 50% C12H26 + 35% C10H22 (w_C=0.845) + 15% C8H18
# Weighted w_C ≈ 0.50*0.847 + 0.35*0.845 + 0.15*0.842 = 0.846
FT_SPK = FuelSurrogate(
    name="FT-SPK",
    species={
        SP_N_DODECANE: 0.50,  # Long-chain paraffinic component
        SP_N_DECANE:   0.35,  # Medium-chain paraffinic component
        SP_ISO_OCTANE: 0.15,  # Iso-paraffinic fraction
    },
    LHV_MJ_per_kg=43.9,
    carbon_fraction=0.846
)

# ATJ-SPK: Alcohol-to-Jet Synthetic Paraffinic Kerosene
# Produced from biomass-derived alcohols; more branched structure
# Blend: 80% C8H18 (w_C=0.842) + 20% C12H26 (w_C=0.847)
# Weighted w_C ≈ 0.80*0.842 + 0.20*0.847 = 0.843
ATJ_SPK = FuelSurrogate(
    name="ATJ-SPK",
    species={
        SP_ISO_OCTANE: 0.80,  # Dominant branched component
        SP_N_DODECANE: 0.20,  # Minor straight-chain component
    },
    LHV_MJ_per_kg=43.5,  # Lower due to higher iso-octane content
    carbon_fraction=0.843
)


# ---------------------------------------------------------------------------
# Fuel Blending Utilities
# ---------------------------------------------------------------------------

def blend_surrogates(fuels: List[Tuple[FuelSurrogate, float]],
                     name: str = "Blend") -> FuelSurrogate:
    """
    Create a composite fuel blend from multiple fuel surrogates.

    This function combines multiple fuels (e.g., Jet-A1 + HEFA + FT) by blending
    their species compositions. The resulting surrogate represents the weighted
    average chemistry of the blend.

    Args:
        fuels: List of (FuelSurrogate, mass_fraction) pairs
               Fractions will be automatically normalized if they don't sum to 1.0
        name: Name for the resulting blended fuel

    Returns:
        New FuelSurrogate representing the blended fuel composition

    Example:
        blend = blend_surrogates([
            (JET_A1, 0.6),
            (HEFA_SPK, 0.4)
        ], name="Jet-A1/HEFA-40")
    """
    total_p = sum(p for _, p in fuels)
    if total_p <= 0.0:
        raise ValueError("Total blend fraction must be positive.")

    # Normalize blend fractions to sum to 1.0
    normalized_top = [(fuel, p / total_p) for fuel, p in fuels if p > 0.0]

    # Combine species from all fuels, weighted by their blend fractions
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
    Create a Sustainable Aviation Fuel (SAF) blend with ASTM certification constraints.

    ASTM D7566 requires that approved SAF blends contain at least 50% conventional
    Jet-A1 to ensure compatibility with existing aircraft fuel systems and maintain
    safety margins for combustion characteristics.

    Args:
        p_j: Mass fraction of Jet-A1 (conventional fuel)
        p_h: Mass fraction of HEFA-SPK
        p_f: Mass fraction of FT-SPK
        p_a: Mass fraction of ATJ-SPK
        enforce_astm: If True, validates ASTM D7566 blending constraints:
                      - Jet-A1 fraction >= 0.5
                      - Total SAF fraction <= 0.5

    Returns:
        FuelSurrogate representing the compliant SAF blend

    Raises:
        ValueError: If enforce_astm=True and blend violates ASTM constraints

    Example:
        # 60% Jet-A1, 30% HEFA, 10% FT (ASTM-compliant)
        blend = make_saf_blend(p_j=0.6, p_h=0.3, p_f=0.1, p_a=0.0)
    """
    if enforce_astm:
        total = p_j + p_h + p_f + p_a
        if total <= 0.0:
            raise ValueError("Total SAF blend fraction must be positive.")
        j_frac = p_j / total
        saf_frac = (p_h + p_f + p_a) / total
        if j_frac < 0.5 - 1e-8:
            raise ValueError(
                f"ASTM D7566 constraint violated: Jet-A1 fraction {j_frac:.3f} < 0.5"
            )
        if saf_frac > 0.5 + 1e-8:
            raise ValueError(
                f"ASTM D7566 constraint violated: SAF fraction {saf_frac:.3f} > 0.5"
            )

    fuels = [
        (JET_A1, p_j),
        (HEFA_SPK, p_h),
        (FT_SPK, p_f),
        (ATJ_SPK, p_a),
    ]
    return blend_surrogates(fuels, name="SAF-Blend")


if __name__ == "__main__":
    # Example usage: Create a 60% Jet-A1, 30% HEFA, 10% FT blend
    blend = make_saf_blend(p_j=0.6, p_h=0.3, p_f=0.1, p_a=0.0)
    print(f"Blend name: {blend.name}")
    print("Species composition:", blend.normalized_species())
    print("Cantera composition string:", blend.as_composition_string())
