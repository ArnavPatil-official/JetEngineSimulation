"""Registered exact-oracle wrapper; loaded only while the shared lease is owned."""
from __future__ import annotations

import importlib.util
import math
import sys
from functools import lru_cache


def load_original(root):
    path = root / "scripts/phase8/pinn_diagnostics/nozzle_verification.py"
    spec = importlib.util.spec_from_file_location("_nozzle_ode_original_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def admissibility(original, reg):
    """Analytic conservative interval bounds plus the registered dense gamma check."""
    g = reg["gas_and_geometry"]
    low = reg["regimes"]["smooth_subcritical"]["NPR"][1]
    high = reg["regimes"]["smooth_choked"]["NPR"][0]
    # exp(z)-1 <= z*exp(z) gives a uniform bound for M_exit at NPR<=low.
    mach_squared_upper = 2 * math.log(low) / g["gamma"][0] * low ** ((g["gamma"][1] - 1) / g["gamma"][1])
    # C(g)=[2/(g+1)]^[(g+1)/(2(g-1))] decreases with gamma:
    # its derivative sign is log(1+t)-t<0, t=(g-1)/2.
    gmax = g["gamma"][1]
    area_ratio_lower = (2 / (gmax + 1)) ** ((gmax + 1) / (2 * (gmax - 1))) / math.sqrt(mach_squared_upper)
    # For M>1, d log(A/A*)/d gamma<0 since log(1+z)>z/(1+z).
    # Hence A/A*(2,g)>=A/A*(2,gmax)>1.5 implies M_exit^+<2.
    # At M=2, NPR increases with gamma: the numerator derivative is positive.
    supersonic_area_lower = float(original.area_mach_ratio(2.0, gmax))
    npr_design_upper = (1 + 2 * (gmax - 1)) ** (gmax / (gmax - 1))
    if area_ratio_lower <= 1.5 or supersonic_area_lower <= 1.5 or npr_design_upper > high:
        raise ValueError("uniform smooth-regime admissibility bound failed")
    first_choking, design = [], []
    for index in range(1001):
        gamma = g["gamma"][0] + (gmax - g["gamma"][0]) * index / 1000
        for branch, values in (("subsonic", first_choking), ("supersonic", design)):
            mach = original.mach_from_area_ratio(1.5, gamma, branch, reg["oracle"]["root_xtol"])
            values.append((1 + 0.5 * (gamma - 1) * mach * mach) ** (gamma / (gamma - 1)))
    if not low < min(first_choking) or high < max(design):
        raise ValueError("dense smooth-regime admissibility check failed")
    return {"status": "PASS", "gamma_points": 1001, "subcritical_area_ratio_lower_bound": area_ratio_lower,
            "supersonic_area_at_M2_lower_bound": supersonic_area_lower, "design_NPR_upper_bound": npr_design_upper,
            "first_choking_NPR_min": min(first_choking), "design_exit_NPR_max": max(design)}


class ExactReference:
    def __init__(self, original, reg):
        self.original, self.reg = original, reg

    @lru_cache(maxsize=1024)
    def _profile(self, regime, npr_key, gamma, gas_R, xs):
        import numpy as np
        geo, cfg = self.reg["gas_and_geometry"], self.reg["oracle"]
        # Always include inlet/throat/exit for the original invariant gates;
        # return only requested supervised/scoring locations afterward.
        gate_xs = sorted(set(xs) | {-1.0, 0.0, 1.0})
        nozzle = self.original.QuasiOneD(gamma, gas_R, geo["p0_Pa"], geo["T0_K"],
                                        geo["x_m"], len(gate_xs), cfg["root_xtol"])
        nozzle.x = np.asarray(gate_xs, dtype=float)
        if regime == "smooth_subcritical":
            mach = math.sqrt(2 / (gamma - 1) * (npr_key ** ((gamma - 1) / gamma) - 1))
            result = nozzle.smooth_subsonic(mach)
            keys = ("mass_flow_rel", "total_enthalpy_rel", "total_pressure_rel", "area_ratio_roundtrip_rel", "inlet_mach_abs")
            valid_branch = result["errors"]["max_mach"] < 1
        elif regime == "smooth_choked":
            result = nozzle.choked_isentropic()
            keys = ("mass_flow_vs_choked_rel", "total_enthalpy_rel", "total_pressure_rel", "area_ratio_roundtrip_rel", "throat_mach_abs")
            valid_branch = result["errors"]["branches_ok"]
        else:
            raise ValueError("unregistered nozzle regime")
        if not valid_branch or any(result["errors"][key] > 1e-10 for key in keys):
            raise ValueError("independent exact-reference gate failed")
        state = result["state"]
        rho_ref = geo["p0_Pa"] / (geo["R_ref_J_kg_K"] * geo["T0_K"])
        scales = (rho_ref, math.sqrt(geo["R_ref_J_kg_K"] * geo["T0_K"]), geo["T0_K"], geo["p0_Pa"])
        scaled = np.column_stack([state[key] / scale for key, scale in zip(("rho", "u", "T", "p"), scales)])
        requested = [gate_xs.index(x) for x in xs]
        return scaled[requested], {key: result["errors"][key] for key in keys}

    def profile(self, case, xs):
        bounds = self.reg["regimes"][case["regime"]]["NPR"]
        if not bounds[0] <= case["NPR"] <= bounds[1]:
            raise ValueError("NPR outside registered smooth interval")
        npr_key = case["NPR"] if case["regime"] == "smooth_subcritical" else 0.0
        return self._profile(case["regime"], npr_key, case["gamma"], case["R"], tuple(float(x) for x in xs))
