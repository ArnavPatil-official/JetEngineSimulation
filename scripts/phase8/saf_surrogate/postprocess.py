"""Exact registered carbon/lifecycle/Brem formulas; not learned physics evidence."""
from __future__ import annotations

from .inputs import FUELS


def derived_outputs(query, ff, properties):
    carbon, lifecycle = 0.0, 0.0
    lca = properties["lifecycle"]
    for fuel in FUELS:
        weight = float(query[f"f_{fuel}"])
        component = properties["fuels"][properties["surrogates"][fuel]]
        factor = lca["baseline_fossil_gCO2e_MJ"] if fuel == "JetA" else lca["pathways"][fuel]["triangular"]["mode"]
        carbon += weight*component["carbon_mass_fraction"]
        lifecycle += weight*component["lhv_liquid_MJ_kg"]*factor
    ei = 44.01/12.011*carbon
    dH = 1.5*sum(float(query[f"f_{fuel}"]) for fuel in ("HEFA", "FT", "ATJ"))
    thrust = 100*float(query["thrust_fraction"])
    valid = thrust > 30 and 0 <= dH < .6
    reasons = []
    if not thrust > 30:
        reasons.append("thrust_pct_not_above_30")
    if not 0 <= dH < .6:
        reasons.append("hydrogen_change_outside_reference_envelope")
    return {"EI_CO2_kg_kg": ei, "CO2_g_s": 1000*ei*float(ff),
            "lifecycle_g_s": lifecycle*float(ff),
            "nvpm_dEI_number_pct": (-114.21+1.06*thrust)*dH if valid else None,
            "nvpm_status": "screening" if valid else "unavailable",
            "nvpm_reason": ";".join(reasons)}


def lifecycle_scenarios(properties):
    import numpy as np
    generator = np.random.default_rng(42)
    return [{"JetA": properties["lifecycle"]["baseline_fossil_gCO2e_MJ"],
             **{name: float(generator.triangular(**{
                 "left": properties["lifecycle"]["pathways"][name]["triangular"]["min"],
                 "mode": properties["lifecycle"]["pathways"][name]["triangular"]["mode"],
                 "right": properties["lifecycle"]["pathways"][name]["triangular"]["max"]}))
                for name in ("HEFA", "FT", "ATJ")}}
            for _ in range(1000)]
