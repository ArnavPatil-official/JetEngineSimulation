#!/usr/bin/env python3
"""
Phase 6 P6.1 Step 3 — write the machine-readable pre-registration
(outputs/phase6/p61_registration.json) BEFORE any pilot or fit.

Reads the committed split and the CALIBRATION group's ICAO rows only (for the
data-derived per-mode eta_b). Held-out records are listed by ID; none of their
target columns is read here. Refuses to overwrite an existing registration.

Amendment A1 (docs/plan_phase6_review.md, R6-A/R6-C): the rejected first
registration R0 is archived byte-identical under outputs/phase6/superseded/;
A1 drops beta (single-zone combustor), converts the fan efficiency exactly,
documents the compressor conversion and the eta_b proxy, and keeps R0's split,
objective, weighting, fitted set, baselines, margins and thresholds.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
import lto_v5 as v5  # noqa: E402

OUT = ROOT / "outputs" / "phase6" / "p61_registration.json"
R0 = ROOT / "outputs" / "phase6" / "superseded" / "p61_registration_R0_REJECTED.json"
R0_SHA256 = "89e7d11630f2aa0bd1cbfb1e2a8fccf8e57c98b69132388ffb2e7ddb35158dc8"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the registration is frozen")
    if hashlib.sha256(R0.read_bytes()).hexdigest() != R0_SHA256:
        raise SystemExit(f"{R0} is not the archived R0 registration")
    r0 = json.loads(R0.read_text())
    hv = v5.creck_heating_values()
    if (abs(hv["Q_CO_MJ_KG"] - v5.Q_CO_MJ_KG) > 5e-5
            or abs(hv["Q_FUEL_MJ_KG"] - v5.Q_FUEL_MJ_KG) > 5e-5):
        raise SystemExit(f"heating values do not reproduce: {hv}")
    split = v5.load_split()
    cal = v5.attach_groups(v5.load_rows(split["calibration_records"], with_targets=True),
                           split["calibration_groups"])
    eta_b = v5.calibration_eta_b(cal)
    fixed = dict(v5.FIXED, eta_b={m: eta_b[m]["central"] for m in v5.MODES})
    ranges = json.loads(json.dumps(v5.FIXED_RANGES))
    ranges["eta_compressor"]["range"] = list(v5.compressor_isentropic_range())
    e_lo, e_hi = v5.FIXED_RANGES["eta_compressor"]["polytropic_range"]
    ranges["eta_compressor"]["variable_cp_conversion_at_opr_43.2"] = [
        v5.compressor_isentropic_variable_cp(e_lo), v5.compressor_isentropic_variable_cp(e_hi)]
    ranges["eta_compressor"]["constant_gamma_at_calibration_opr_span"] = {
        str(o): [v5.compressor_isentropic(e_lo, o), v5.compressor_isentropic(e_hi, o)]
        for o in (float(cal["Pressure Ratio"].min()), float(cal["Pressure Ratio"].max()))}
    ranges["eta_fan"]["range"] = list(v5.fan_isentropic_envelope())
    ranges["eta_fan"]["at_central_fpr_1.45"] = [
        v5.fan_isentropic(e, v5.FIXED["fpr_rated"]) for e in v5.FIXED_RANGES["eta_fan"]["polytropic_range"]]
    ranges["eta_b"]["range"] = {m: list(eta_b[m]["range"]) for m in v5.MODES}
    reg = {
        "registered": "2026-09-26",
        "amendment": "A1",
        "status": "frozen before any pilot or fit (amends R0, which was rejected before any pilot)",
        "supersedes": {
            "path": str(R0.relative_to(ROOT)),
            "sha256": R0_SHA256,
            "registered_in_commit": "59d9665",
            "rejected_by": "docs/plan_phase6_review.md R6-A (commit c55e182)",
            "reason": "R0 fixed the burner-zone air fraction beta = 0.8 over an unsourced "
                      "ILLUSTRATIVE range; decision 2 requires a cited range, data "
                      "identification, or dropping the parameter",
        },
        "changes_from_R0": [
            "beta (combustor_air_fraction) DROPPED: single-zone combustor, all core air in one HP "
            "equilibrium at the overall phi (design_point combustor_air_fraction = 1.0, which "
            "disables the split). The split has no citable range for the quantity implemented "
            "(air frozen out of equilibrium and remixed at constant cp before the turbine); one "
            "1972 research-combustor air distribution (NASA TM X-2476, Fig. 2) is a single design "
            "and a different quantity; turbine cooling air (NASA/TM-2017-219501 Table 3) is injected "
            "into the turbine and is also a different quantity. Legacy beta stays in frozen v2-v4 "
            "and reproduction paths. Provisional input-only probe (AE3, W_ref 96, k 0.6): matched-"
            "thrust fuel flow beta 1.0 vs 0.8 = +0.41 % take-off, +0.59 % approach",
            "eta_fan range: exact polytropic-to-isentropic conversion (fan model's constant gamma "
            "1.4), transformed per Monte Carlo draw at the drawn FPR_rated; envelope reported",
            "eta_compressor: constant-gamma rated-OPR conversion retained; variable-cp conversion "
            "and calibration-OPR span reported (documentation, not substituted)",
            "eta_b: documented as an energy-efficiency proxy (assumptions in lto_v5."
            "eta_b_from_emissions); heating values reproduced by lto_v5.creck_heating_values",
            "NOx: worker engines refit the NOx correlation without held-out models "
            "(lto_v5._init_worker); no full-data-fit NOx value is reported as held-out evidence",
            "driver errors: only ThrustTargetUnreachable rows are penalised (e = 1.0); any other "
            "exception or non-finite prediction aborts the run (R6-B)",
        ],
        "model_structure": {
            "combustor": "single-zone HP equilibrium at overall phi (Cantera, CRECK), temperature "
                         "rise scaled by eta_b; no burner/dilution split",
            "turbine": "analytic", "nozzle": "analytic", "heat_loss_xi": 0.0,
        },
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
               "held-out models) and re-run nox_holdout_validation.py on the held-out group. The v5 "
               "cycle workers apply the same exclusion at initialisation (lto_v5._init_worker), so "
               "every v5 NOx(corr) value is from the calibration-only fit",
        "eta_b_proxy": {
            "formula": "eta_b = 1 - (EI_CO*Q_CO + EI_HC*Q_fuel)/(1000*Q_fuel), EI in g/kg",
            "Q_CO_MJ_kg": v5.Q_CO_MJ_KG, "Q_fuel_MJ_kg": v5.Q_FUEL_MJ_KG,
            "reproduced_from_CRECK": hv,
            "conventions": "298.15 K; lower heating values (H2O vapour); Q_fuel = n-C12H26 "
                           "(production Jet-A1 surrogate); HC counted as unburned fuel at Q_fuel "
                           "per unit mass (ICAO HC is methane-equivalent mass: approximation); CO "
                           "and HC are the only unreleased-energy carriers",
            "role": "energy-efficiency proxy fixed from calibration-group data and used as the "
                    "temperature-rise scaling T4 = T3 + eta_b (T_ad - T3); NOT an inverse-cycle "
                    "parameter identified from the fuel-flow objective",
            "data": "calibration-group records only",
        },
        "bands_P6_2": {
            "method": "seeded (42) Monte Carlo, 64 draws, each fixed parameter uniform over its cited "
                      "range (eta_b per mode over its calibration min-max; eta_fan transformed per draw "
                      "= fan_isentropic(e_poly ~ U(polytropic_range), drawn FPR_rated); eta_compressor "
                      "uniform over its constant-gamma isentropic range; xi fixed; no beta); for EACH "
                      "draw the "
                      "fitted parameters are RE-FIT on the calibration group (least_squares from the v5 "
                      "optimum, max_nfev 30), then the design point and held-out MAPE are recomputed",
            "reported_as": "refit-conditioned range bands (P5-P95 and min-max). These are range "
                           "propagation under assumed uniform ranges, NOT statistical confidence "
                           "intervals",
        },
    }
    # R0 decisions that A1 must preserve verbatim
    for key in ("split", "weighting", "fitted", "fit", "identifiability", "baselines",
                "heldout_metrics", "acceptance", "t4_guard_K", "fuel", "components"):
        if json.loads(json.dumps(reg[key])) != r0[key]:
            raise SystemExit(f"A1 changed the R0 decision '{key}'")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=2) + "\n")
    print(json.dumps({"eta_b": eta_b, "eta_c_isentropic_range": ranges["eta_compressor"]["range"],
                      "n_eff": reg["identifiability"]["n_eff"]}, indent=2))


if __name__ == "__main__":
    main()
