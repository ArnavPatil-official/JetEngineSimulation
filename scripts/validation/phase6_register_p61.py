#!/usr/bin/env python3
"""
Phase 6 P6.1 Step 3 — write the machine-readable pre-registration
(outputs/phase6/p61_registration.json) BEFORE any pilot or fit.

Reads the committed split and the CALIBRATION group's ICAO rows only (for the
data-derived per-mode eta_b). Held-out records are listed by ID; none of their
target columns is read here. Refuses to overwrite an existing registration.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
import lto_v5 as v5  # noqa: E402

OUT = ROOT / "outputs" / "phase6" / "p61_registration.json"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the registration is frozen")
    split = v5.load_split()
    cal = v5.attach_groups(v5.load_rows(split["calibration_records"], with_targets=True),
                           split["calibration_groups"])
    eta_b = v5.calibration_eta_b(cal)
    fixed = dict(v5.FIXED, eta_b={m: eta_b[m]["central"] for m in v5.MODES})
    ranges = json.loads(json.dumps(v5.FIXED_RANGES))
    ranges["eta_compressor"]["range"] = list(v5.compressor_isentropic_range())
    ranges["eta_b"]["range"] = {m: list(eta_b[m]["range"]) for m in v5.MODES}
    reg = {
        "registered": "2026-09-26",
        "status": "frozen before any pilot or fit",
        "split": "outputs/phase6/split_p61.json",
        "formulation": "scripts/optimization/lto_v5.py (module docstring)",
        "fuel": "Jet-A1 (CRECK, production mechanism profile)",
        "components": {"turbine": "analytic", "nozzle": "analytic"},
        "fitted": {k: list(b) for k, b in v5.FIT_BOUNDS.items()},
        "fixed_central": fixed,
        "fixed_ranges": ranges,
        "t4_guard_K": 3800.0 * 5.0 / 9.0,
        "weighting": "each split group equal; records equal within a group; modes equal "
                     "(recertifications and input-twin models share one group weight)",
        "fit": {
            "objective": "thrust_matched_v5: sum_i w_i e_i^2, e_i = relative fuel-flow error "
                         f"(unreachable row: e_i = {v5.FAILED_ROW_ERROR})",
            "reported": "group-weighted MAPE (same weights)",
            "optimizer": "Optuna TPESampler(seed=42) over FIT_BOUNDS in FIT_ORDER, then "
                         "scipy least_squares (trf, bounds, x_scale = box widths, diff_step 1e-4) "
                         "from the best trial",
            "full": {"n_trials": 150, "polish_max_nfev": 100},
            "pilot": {"n_trials": 40, "polish_max_nfev": 30},
        },
        "identifiability": {
            "diagnostic": "profile likelihood: for each fitted parameter theta_j on a uniform grid over "
                          "its box, re-fit the others (least_squares, warm-started from the neighbouring "
                          "grid point); D(theta) = n_eff * ln(SSE_profile(theta) / SSE_min), "
                          "n_eff = number of calibration (group, mode) pairs",
            "n_eff": int(cal["Group"].nunique() * len(v5.MODES)),
            "threshold_chi2_1_95": 3.841,
            "rule": "IDENTIFIED iff D >= 3.841 at both box edges (95 % profile interval interior "
                    "to the box) AND the interval width <= 50 % of the box width",
            "local_check_reported_not_gating": "condition number of the box-scaled J^T J at the optimum",
            "pilot": {"grid_points": 9, "inner_max_nfev": 20, "start": "pilot fit optimum"},
            "full": {"grid_points": 17, "inner_max_nfev": 40, "start": "v5 fit optimum"},
            "pilot_consequence": "a parameter not IDENTIFIED in the pilot is moved to fixed-with-cited-"
                                 "range or dropped BEFORE the full fit; the full profile is then run on "
                                 "the remaining fitted set",
        },
        "baselines": {
            "B0": "constant TSFC: per mode, group-weighted calibration mean of ICAO FF / F_target, "
                  "times the held-out F_target",
            "B1": "F-A rule: nearest-rated-thrust calibration group (ties: mean over tied groups) "
                  "per-mode mean ICAO fuel flow x (held-out rated thrust / that group's rated thrust)",
        },
        "heldout_metrics": {
            "primary": "group-weighted MAPE over all held-out rows (weights as in the fit)",
            "secondary": ["record-level MAPE", "per-mode group-weighted MAPE"],
            "unreachable_rows": "APE = 100 %, reported by row",
        },
        "acceptance": {
            "A1_identifiability": "every parameter in the final fitted set IDENTIFIED (full profile)",
            "A2_skill_margin_pp": 0.25,
            "A2_rule": "PASS iff primary MAPE_model <= min(MAPE_B0, MAPE_B1) - 0.25 pp; "
                       "ESCALATE (plan section 9) iff MAPE_model > MAPE_B0 + 0.25 pp; otherwise "
                       "'no demonstrated skill' (fails A2, reported)",
            "A3_opr_trend": "per mode, OLS slope of TSFC_i = FF_i / F_target,i on OPR_i across held-out "
                            "groups (one point per group: group mean OPR, group-weighted mean TSFC), "
                            "computed for ICAO data and for model predictions; PASS iff the signs "
                            "agree in all three modes. Partial slope controlling for rated thrust is "
                            "reported, not gating",
            "A4_informativeness": "tests/test_holdout_informativeness.py passes on the v5 CSV",
        },
        "nox": "refit the NOx correlation on calibration-group models only (nox_fit_exclude_models = "
               "held-out models) and re-run nox_holdout_validation.py on the held-out group",
        "bands_P6_2": {
            "method": "seeded (42) Monte Carlo, 64 draws, each fixed parameter uniform over its cited "
                      "range (eta_b per mode over its calibration min-max; xi fixed); for EACH draw the "
                      "fitted parameters are RE-FIT on the calibration group (least_squares from the v5 "
                      "optimum, max_nfev 30), then the design point and held-out MAPE are recomputed",
            "reported_as": "refit-conditioned range bands (P5-P95 and min-max). These are range "
                           "propagation under assumed uniform ranges, NOT statistical confidence "
                           "intervals",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=2) + "\n")
    print(json.dumps({"eta_b": eta_b, "eta_c_isentropic_range": ranges["eta_compressor"]["range"],
                      "n_eff": reg["identifiability"]["n_eff"]}, indent=2))


if __name__ == "__main__":
    main()
