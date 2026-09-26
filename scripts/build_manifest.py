#!/usr/bin/env python3
"""
Phase 6 (P6.6, P6.7): one registry -> outputs/ARTIFACT_MANIFEST.md and docs/model_map.md.

Every manuscript-bound number is a row: what it is, the script that makes it,
the artifact it lives in, the model that computes it, the inputs it depends on,
and the data that constrains it (or "none — assumption, range cited"). Claim
numbers are READ FROM THE ARTIFACTS here, never typed, so the manifest cannot
drift from the pipeline. tests/test_manifest_integrity.py checks that both
documents equal this script's output, that every listed path exists, and that
no file under outputs/ is unreferenced (orphan sweep).

Usage: .venv/bin/python scripts/build_manifest.py [--check]
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "outputs" / "ARTIFACT_MANIFEST.md"
MODEL_MAP = ROOT / "docs" / "model_map.md"
ARCH = "outputs/archive/pre_phase6"


def J(path):
    return json.loads((ROOT / path).read_text())


def C(path):
    return pd.read_csv(ROOT / path)


@dataclass
class Row:
    id: str
    section: str
    content: object                      # str or callable -> str (reads artifacts)
    command: str
    artifacts: list
    seed: str
    model: str = ""
    inputs: str = ""
    constrained_by: str = ""
    extra_refs: list = field(default_factory=list)   # referenced, not primary

    def text(self) -> str:
        return self.content() if callable(self.content) else self.content


# --------------------------------------------------------------------------
# Claim extractors (read the artifacts)
# --------------------------------------------------------------------------
def v1():
    d = J("outputs/calibration_v5_A2.json")
    p = d["params"]
    other = [c for c in d["candidates"] if c["name"] != d["selected"]][0]
    return (f"v5 calibration (thrust-matched; amendment A2 selection `{d['selected']}`): "
            f"W_ref **{p['W_ref']:.2f} kg/s** (hand-set v4 core 79.9, ×{p['W_ref'] / 79.9:.3f}), "
            f"a {p['a_thrust']:.3f}, k_π {p['k_pi']:.3f}, k_ṁ {p['k_mdot']:.3f}; in-sample calibration "
            f"MAPE {d['calibration_weighted_mape_pct']:.2f} % ({d['n_rows']} rows, "
            f"{d['n_unreachable']} unreachable). Other candidate (registered fit): SSE "
            f"{other['sse']:.6f} vs {d['sse']:.6f}. **In-sample, not validation**")


def v2():
    d = J("outputs/identifiability_profile_v5.json")
    parts = [f"{k} [{v['interval_95'][0]:.4g}, {v['interval_95'][1]:.4g}]"
             for k, v in d["verdicts"].items()]
    return (f"Identifiability (profile likelihood, registered rule): IDENTIFIED "
            f"{', '.join(d['identified'])}; not identified: {d['not_identified'] or 'none'}. "
            f"95 % intervals: {'; '.join(parts)}")


def v3():
    s = C("outputs/holdout_icao_validation_summary_v5.csv").set_index("Scope")
    d = J("outputs/holdout_icao_validation_v5.json")
    g = "group-weighted MAPE (%)"
    modes = " / ".join(f"{m.lower()} {s.loc[m, 'Model ' + g]:.2f}·{s.loc[m, 'B0 ' + g]:.2f}·"
                       f"{s.loc[m, 'B1 ' + g]:.2f}" for m in ("TAKE-OFF", "APPROACH", "IDLE"))
    return (f"Held-out fuel flow ({int(s.loc['ALL', 'Rows'])} rows, 9 held-out groups, "
            f"{int(s.loc['ALL', 'Unreachable'])} unreachable), group-weighted MAPE model · B0 "
            f"(constant TSFC) · B1 (rated-thrust rescaling): **{s.loc['ALL', 'Model ' + g]:.2f} · "
            f"{s.loc['ALL', 'B0 ' + g]:.2f} · {s.loc['ALL', 'B1 ' + g]:.2f} %** ({modes}). "
            f"A2: {d['A2']['verdict']}; A3 (TSFC–OPR sign): {d['A3']['verdict']} "
            f"({', '.join(k for k, v in d['A3']['sign_agrees'].items() if not v)} disagrees); "
            f"A4 informativeness: test passes")


def v5():
    d = C("outputs/design_point_summary_v5.csv").set_index("mode").loc["TAKE-OFF"]
    return (f"Take-off design point (AE3, matched {d['target_thrust_kN']:.1f} kN): φ {d['phi_solved']:.4f}, "
            f"T3 {d['T3_K']:.1f} K, T4 {d['T4_K']:.1f} K, T5 {d['T5_K']:.1f} K, core "
            f"{d['mass_flow_core_kg_s']:.1f} / total {d['total_air_mass_flow_kg_s']:.0f} kg/s, thrust core "
            f"{d['thrust_core_kN']:.1f} + bypass {d['thrust_bypass_kN']:.1f} kN, TSFC "
            f"{d['TSFC_mg_per_Ns']:.2f} mg/(N·s), fuel flow {d['fuel_flow_kg_s']:.3f} vs ICAO "
            f"{d['icao_fuel_flow_kg_s']:.3f} kg/s ({d['fuel_flow_rel_err_pct']:+.1f} %, in-sample); "
            f"approach and idle rows in the same file")


def v6():
    s = C("outputs/nox_holdout_validation_summary_p61.csv").set_index("Scope")
    a = s.loc["ALL (held-out group)"]
    return (f"NOx correlation fitted on the calibration group only, scored on the held-out group "
            f"({int(a['n'])} rows): group-weighted MAPE **{a['Model_group_weighted_MAPE_pct']:.2f} %** vs "
            f"naive per-mode mean EI {a['Naive_group_weighted_MAPE_pct']:.2f} %. Inputs are ICAO OPR and "
            f"fuel flow; within-family; no composition term, so NOT evidence for blend NOx ranking")


def v6b():
    s = C("outputs/nox_holdout_validation_summary.csv").set_index("Scope")["MAPE_pct"]
    return (f"Supplementary: leave-one-engine-out NOx correlation over all 28 models, MAPE "
            f"**{s['ALL (held-out)']:.2f} %** (take-off {s['TAKE-OFF']:.2f} / approach {s['APPROACH']:.2f} / "
            f"idle {s['IDLE']:.2f}); in-sample reference {s['ALL (in-sample, reference)']:.2f} %. The withdrawn "
            f"R² = 0.9969 Highlight was the in-sample fit R² (training diagnostic)")


def v8():
    d = J("outputs/p62_bands_v5.json")
    b = d["bands"]

    def rng(k, f="{:.3f}"):
        return f"{f.format(b[k]['p5'])}–{f.format(b[k]['p95'])}"
    return (f"P6.2 refit-conditioned range bands ({d['n_draws']} draws, P5–P95; range propagation "
            f"under assumed uniform ranges, **not confidence intervals**): held-out MAPE "
            f"{rng('heldout_mape_pct', '{:.2f}')} %, take-off TSFC {rng('dp_tsfc_mg_Ns')} mg/(N·s), "
            f"T4 {rng('dp_T4_K', '{:.0f}')} K, fitted W_ref {rng('fit_W_ref', '{:.1f}')} kg/s. "
            f"One-at-a-time endpoints in the same file name the limiting assumption")


def b1():
    r = C("outputs/results/blend_matched_thrust_v5_rankings.csv")
    to = r[r["Mode"] == "TAKE-OFF"].set_index(["quantity", "blend_a", "blend_b"])
    ff = [f"{a} {to.loc[('ff', a, 'Jet-A1'), 'delta_rel_pct']:+.2f} %"
          for a in ("HEFA-50", "FT-50", "ATJ-50")]
    lc = [f"{a} {to.loc[('lifecycle_g_s', a, 'Jet-A1'), 'delta_rel_pct']:+.1f} %"
          for a in ("HEFA-50", "FT-50", "ATJ-50")]
    claimed = r[r["ranking_claimed"]]
    cl = sorted({f"{q}" for q in claimed["quantity"]})
    return (f"SAF blends at **matched thrust** (AE3, all LTO modes; φ solved). Take-off fuel flow vs "
            f"Jet-A1: {', '.join(ff)}; lifecycle CO₂e (CORSIA modes): {', '.join(lc)}. Rankings claimed "
            f"under the registered rule (difference > MC spread and > P6.2 band): "
            f"{len(claimed)} of {len(r)} comparisons, quantities: {', '.join(cl) or 'none'}")


def b2():
    v = C("outputs/results/variance_decomposition_v5.csv").set_index("Objective")
    lc = v.loc["lifecycle_g_s_point_draw"]
    ff = v.loc["ff"]
    draw = sum(lc[f"lcef_draw_{k}"] for k in ("HEFA", "FT", "ATJ"))
    return (f"Variance decomposition at matched take-off thrust (256 Sobol blends): fuel flow varies "
            f"{ff['rel_range_pct']:.2f} % across blends (R² {ff['R2']:.3f}); lifecycle CO₂e: blend "
            f"fractions {1 - draw:.1%} / CORSIA draw {draw:.1%} of explained variance (R² {lc['R2']:.3f})")


def b4():
    d = C("outputs/results/lca_rank_stability_v5.csv")
    m = d[d["ParetoOptimal_central"]]
    st = (m["Pareto_persistence"] >= 0.5).mean()
    return (f"CORSIA rank stability (1000 common draws): {len(m)} central Pareto members on (TSFC, "
            f"lifecycle, NOx); {st:.1%} stay Pareto-optimal in ≥ 50 % of draws → blend-selection "
            f"statements {'may carry scenario bands' if st >= 0.5 else 'are scenario-conditional'}")


def e3():
    d = C("outputs/nox_dual_path_v5.csv").set_index("Mode").loc["Takeoff"]
    return (f"NOx three-path comparison at v5 states: take-off EI correlation {d['EI_ICAO_correlation']:.1f}, "
            f"Zeldovich {d['EI_Zeldovich_CRECK']:.2f}, HyChem-A2 {d['EI_HyChem_A2']:.2f} g/kg vs certification "
            f"{d['EI_ICAO_certification']:.2f}; single zone at overall φ, chemistry paths are low-side "
            f"proxies. No blend NOx ranking")


def e4():
    d = C("outputs/heat_loss_sensitivity_v5.csv")
    j = d[d["Fuel"] == "Jet-A1"].set_index("xi_pct")["TSFC_mg_per_Ns"]
    b = d[d["xi_pct"] == 0]["TSFC_mg_per_Ns"]
    return (f"Heat-loss sensitivity at matched take-off thrust: ξ = 4 % raises Jet-A1 TSFC by "
            f"{j[4] - j[0]:.3f} mg/(N·s) ({100 * (j[4] / j[0] - 1):+.1f} %), "
            f"{(j[4] - j[0]) / (b.max() - b.min()):.0f}× the neat-fuel TSFC spread "
            f"({b.max() - b.min():.3f}) → heat-loss treatment bounds the resolvable blend effect")


def e9():
    d = J("outputs/mechanism_sensitivity_v5.json")
    c = C("outputs/mechanism_sensitivity_v5.csv").set_index("profile")
    bf = d["p62_bands"]["dp_ff_kg_s"]
    ff0 = c.loc["CRECK C1-C16", "fuel_flow_kg_s"]
    width = 100 * (bf["p95"] - bf["p5"]) / ff0
    a1, a2 = (c.loc[k, "fuel_flow_kg_s_rel_diff_vs_CRECK_pct"] for k in ("HyChem A1", "HyChem A2"))
    return (f"Mechanism/surrogate sensitivity (P6.4; AE3 take-off at matched thrust, v5 calibration "
            f"held fixed): fuel flow and TSFC vs CRECK n-dodecane **{a1:+.2f} % (HyChem A1), {a2:+.2f} % "
            f"(A2)** — larger than the P6.2 fuel-flow band width ({width:.2f} %), so resolved; mostly the "
            f"surrogate's heating value (44.46 vs 43.53 / 43.48 MJ/kg, "
            f"`outputs/logs/phase6_p64_surrogate_lhv.log`). T4 differs by ≤ "
            f"{d['max_abs_rel_diff_pct']['T4_K']:.2f} % and A1 vs A2 fuel flow by "
            f"{d['A1_vs_A2_fuel_flow_rel_diff_pct']:+.2f} % — within the band (unresolved, not zero). "
            f"Not re-calibrated per mechanism")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
SEC_V = "Model + validation (v5, thrust-matched)"
SEC_B = "SAF blends at matched thrust (P6.3)"
SEC_E = "Evidence and sensitivity"
SEC_P = "PINN record (P6.5 — closed; no further training)"

CYCLE = ("`integrated_engine.run_at_thrust` (compressor, single-zone HP-equilibrium combustor, "
         "analytic turbine and nozzle) via `scripts/optimization/lto_v5.py`")
ICAO_IN = "ICAO rated thrust, OPR, BPR per record; x·F_rated per mode"
FIXED_IN = "fixed values with cited ranges (V7)"

ROWS = [
    Row("V1", SEC_V, v1,
        "`calibrate_lto.py --tag v5 --pilot`; `--tag v5 --free W_ref a_thrust k_pi k_mdot`; `--tag v5 --a2-select`",
        ["outputs/calibration_v5_A2.json"], "42 (TPE)", CYCLE, f"{ICAO_IN}; {FIXED_IN}",
        "ICAO fuel flow of the calibration group (31 records, 9 groups)",
        ["outputs/calibration_v5_A2_evaluations.csv", "outputs/calibration_v5_A2_rows.csv",
         "outputs/calibration_v5.json", "outputs/calibration_v5_evaluations.csv",
         "outputs/calibration_v5_rows.csv", "outputs/phase6/p61_pilot_fit.json",
         "outputs/phase6/p61_pilot_fit_evaluations.csv", "outputs/phase6/p61_pilot_fit_rows.csv",
         "outputs/phase6/p61_amendment_A2.json"]),
    Row("V2", SEC_V, v2, "`identifiability_profile.py --v5 pilot`; `--v5 full`",
        ["outputs/identifiability_profile_v5.json", "outputs/identifiability_profile_v5.csv"], "—",
        "profile likelihood over the V1 objective", "V1 objective", "calibration-group fuel flow",
        ["outputs/identifiability_profile_v5_progress.log",
         "outputs/phase6/identifiability_profile_v5_pilot.json",
         "outputs/phase6/identifiability_profile_v5_pilot.csv",
         "outputs/phase6/identifiability_profile_v5_pilot_progress.log"]),
    Row("V3", SEC_V, v3, "`holdout_icao_validation.py --tag _v5`",
        ["outputs/holdout_icao_validation_summary_v5.csv", "outputs/holdout_icao_validation_v5.json",
         "outputs/holdout_icao_validation_v5.csv"], "via V1", CYCLE + "; baselines B0, B1 from calibration data",
        f"{ICAO_IN}; V1 parameters", "ICAO fuel flow of the held-out group (29 records, 9 groups), read once"),
    Row("V4", SEC_V,
        "Finding F-A: the v4 held-out fuel-flow validation (2.46 % on 177 rows; 2.50 % on 171) was a "
        "rated-thrust rescaling rule — predicted fuel flow ÷ thrust ratio constant to 1e-16 per mode; "
        "an AE3-ratio baseline scores 3.22 %. v4 M2 is withdrawn",
        "`tests/test_holdout_informativeness.py`",
        [f"{ARCH}/holdout_icao_validation_v4.csv", "outputs/parameter_provenance.md"], "—",
        "v4 fixed-φ cycle (superseded)", "rated-thrust ratio only", "—"),
    Row("V5", SEC_V, v5, "`design_point_summary.py --v5`", ["outputs/design_point_summary_v5.csv"],
        "deterministic", CYCLE, f"AE3 inputs; V1; {FIXED_IN}", "AE3 is a calibration engine (in-sample)"),
    Row("V6", SEC_V, v6, "`nox_holdout_validation.py --split outputs/phase6/split_p61.json`",
        ["outputs/nox_holdout_validation_summary_p61.csv", "outputs/nox_holdout_validation_p61.csv"],
        "deterministic", "EI_NOx = A·OPR^B·ṁ_f^C (ICAO-derived correlation; not chemistry)",
        "ICAO OPR, fuel flow", "ICAO EI-NOx of the calibration group"),
    Row("V6b", SEC_V, v6b,
        "`nox_holdout_validation.py`",
        ["outputs/nox_holdout_validation_summary.csv", "outputs/nox_holdout_validation.csv"],
        "deterministic", "same correlation", "ICAO OPR, fuel flow", "ICAO EI-NOx (leave-one-model-out)",
        ["outputs/plots/nox_holdout_pred_vs_icao.png", "outputs/plots/nox_holdout_error_boxplot.png"]),
    Row("V7", SEC_V,
        "Fixed parameters under the range rule: η_b per mode (data-derived CO/HC proxy), combustor "
        "pressure loss 0.045 [0.04, 0.05], η_c 0.86, turbine η_poly 0.90, η_fan 0.90, FPR 1.45 — each "
        "with a cited range (NASA/TM-2017-219501, NASA/CR-2005-213657, NASA/TM-2007-214690); β dropped "
        "(single-zone); ξ = 0",
        "`scripts/validation/phase6_register_p61.py` (registration A1)",
        ["outputs/phase6/p61_registration.json", "docs/phase6_p61_registration.md"], "—",
        "—", "—", "none — assumption, range cited (η_b: calibration-group CO/HC)",
        ["outputs/phase6/split_p61.json", "outputs/phase6/superseded/README.md",
         "outputs/phase6/superseded/p61_registration_R0_REJECTED.json",
         "outputs/phase6/superseded/phase6_p61_registration_R0_REJECTED.md"]),
    Row("V8", SEC_V, v8, "`scripts/validation/p62_parameter_bands.py`",
        ["outputs/p62_bands_v5.json", "outputs/p62_bands_v5.csv"], "42 (64 draws)",
        "V1 re-fit per draw + " + CYCLE, "V7 ranges", "calibration-group fuel flow (re-fit per draw)"),
    Row("B1", SEC_B, b1, "`scripts/optimization/blend_matched_thrust_v5.py`",
        ["outputs/results/blend_matched_thrust_v5_rankings.csv", "outputs/results/blend_matched_thrust_v5.csv",
         "outputs/results/blend_matched_thrust_v5.json", "outputs/results/blend_matched_thrust_v5_refs_p62.csv"],
        "42 (Sobol, CORSIA)", CYCLE + "; lifecycle = ṁ_f·Σ p_i·LHV_i·L_CEF_i",
        "blend mass fractions; AE3 inputs; V1; V7; CORSIA draws", "none for blend effects (no blend engine data); "
        "CORSIA Doc 06 ranges for lifecycle",
        ["outputs/phase6/p63_registration.json", "docs/phase6_p63_registration.md"]),
    Row("B2", SEC_B, b2, "`variance_decomposition.py --v5 --seed 42 --n-mc 1000`",
        ["outputs/results/variance_decomposition_v5.csv"], "42",
        "random-forest permutation importance over B1 design", "B1 outputs", "—",
        ["outputs/results/objective_correlation_v5.csv"]),
    Row("B3", SEC_B, b4, "same", ["outputs/results/lca_rank_stability_v5.csv"], "42",
        "Pareto set over B1 design under common CORSIA draws", "B1 outputs; CORSIA ranges", "CORSIA Doc 06"),
    Row("E1", SEC_E, "CORSIA L_CEF triangular ranges, every value cited to ICAO Doc 06 (Nov 2025) rows",
        "hand-built data file, values verified against the PDF", ["data/corsia_lca_values.yaml"], "—",
        "lifecycle accounting", "—", "ICAO Doc 06"),
    Row("E2", SEC_E, "EI-CO₂ = 3.664·w_C (3.100 kg/kg Jet-A1; Cantera cross-check 3.098)",
        "`simulation/fuels.py::carbon_fraction_of_composition`", ["outputs/parameter_provenance.md"], "—",
        "stoichiometry", "surrogate composition", "—"),
    Row("E3", SEC_E, e3, "`nox_dual_path.py --v5`",
        ["outputs/nox_dual_path_v5.csv", "outputs/plots/nox_path_comparison_v5.png"], "deterministic",
        "correlation / Zeldovich on CRECK equilibrium / HyChem-A2 kinetics (evidence path only)",
        "V5 states", "ICAO EI-NOx (AE3)"),
    Row("E4", SEC_E, e4, "`heat_loss_sensitivity.py --v5`",
        ["outputs/heat_loss_sensitivity_v5.csv", "outputs/plots/heat_loss_sensitivity_v5.png"],
        "deterministic", CYCLE, "ξ ∈ {0, 2, 4, 6} %", "none — structural argument for ξ = 0"),
    Row("E5", SEC_E, "PINN component ablation (±10 % thrust; basis of the P3.1 adjudication). v3-state "
        "decision evidence, not a production number",
        "`scripts/validation/ablate_pinn_components.py`",
        ["outputs/ablation_pinn_components.csv", "outputs/ablation_pinn_components_summary.csv"], "—",
        "v3 fixed-φ cycle", "—", "—"),
    Row("E6", SEC_E, "Sajben negative result (wall-Cp shape-L2 0.71–1.09 vs < 0.10)",
        "`scripts/validation/sajben_figure.py`",
        ["outputs/sajben_validation_errors.csv", "outputs/plots/sajben_wall_cp_validation.png"], "—",
        "LE-PINN (retired)", "Sajben geometry", "Hseih et al. 1987 experiment",
        ["outputs/sajben_validation_errors_rescored.csv"]),
    Row("E7", SEC_E, "SAF-penalty ablation (removed penalties flipped the TSFC ranking; Phase 1)",
        "`scripts/validation/ablate_saf_penalty.py`", ["outputs/ablation_saf_penalty.csv"], "—",
        "v1 cycle", "—", "—"),
    Row("E8", SEC_E, "Parameter provenance table (supplementary material)",
        "maintained by hand, updated each phase", ["outputs/parameter_provenance.md"], "—"),
    Row("E9", SEC_E, e9, "`scripts/validation/mechanism_sensitivity.py`",
        ["outputs/mechanism_sensitivity_v5.json", "outputs/mechanism_sensitivity_v5.csv"], "deterministic",
        CYCLE + " with CRECK / HyChem A1 / A2 thermo", "V1; V7", "—",
        ["outputs/logs/phase6_p64_surrogate_lhv.log"]),
    Row("E10", SEC_E, "Surrogate LHVs computed from CRECK thermo (gas-phase fuel, H₂O vapour, "
        "298.15 K): Jet-A1 44.462, HEFA 44.478, FT 44.520, ATJ 44.571 MJ/kg; used only for "
        "energy-weighted lifecycle CO₂e", "`simulation/fuels.py` (LHV_METHOD), "
        "`tests/test_phase6_registration.py`", ["simulation/fuels.py"], "—", "CRECK species enthalpies",
        "surrogate compositions", "—"),
    Row("E11", SEC_E, "Turbine p5 adjudication (analytic = work-consistent; PINN −41.5 %); "
        "v3-state decision evidence", "`scripts/validation/adjudicate_turbine_p5.py`",
        ["outputs/turbine_p5_adjudication.csv", "outputs/turbine_p5_adjudication_v5.csv"], "—",
        "analytic vs PINN turbine", "—", "—"),
]

PINN_ROWS = [
    ("P1", "Sajben P4.3 attempt 1 (ReLU, physics w 0.05 with warm-up): worse-wall Cp shape-L2 **0.258**, "
           "single seed, band not claimed", "`train_sajben.py` → `sajben_report_p43.py`",
     ["outputs/sajben_retrain_v5.md", "outputs/sajben_retrain_v5.csv"], "42"),
    ("P2", "Sajben P4.3 attempt 2: **0.245**, single seed, band not claimed", "same",
     ["outputs/sajben_retrain_v5.md"], "42"),
    ("P3", "Sajben P4.3 attempt 3 (tanh, μ from data), terminal: **0.159 ± 0.016** = **PARTIAL** vs gate "
           "0.10; matched data-only 0.115 ± 0.038; physics-on worse by 0.044 > seed spread",
     "same", ["outputs/sajben_retrain_v5.md"], "42, 43, 44"),
    ("P4", "Turbine surrogate attempt 1: held-out max |Δp5|/p5 **28.25 %**; gate (1 %) missed",
     "`train_turbine_surrogate.py`", ["outputs/turbine_surrogate_v5.md", "outputs/turbine_surrogate_fidelity_v5.csv",
                                     "outputs/turbine_surrogate_cycle_check_v5.csv", "outputs/turbine_envelope_v5.csv"], "42"),
    ("P5", "Turbine surrogate attempt 2: **6.39 %**; gate missed; turbine PINN retired", "same",
     ["outputs/turbine_surrogate_v5_a2.md", "outputs/turbine_surrogate_fidelity_v5_a2.csv",
      "outputs/turbine_surrogate_cycle_check_v5_a2.csv"], "42"),
    ("P6", "Physics-residual defect: under ReLU every second derivative is zero, so the Laplacian "
           "residual enforced Euler; no Reynolds-stress terms", "direct measurement",
     ["outputs/physics_residual_defect.md"], "—"),
    ("P7", "P4.1 Sajben data audit: quasi-1D broadcast training set; analytic ceiling **0.154**; SA-RANS "
           "0.089 / 0.084", "`sajben_data_audit.py`",
     ["outputs/sajben_data_audit.md", "outputs/sajben_data_audit_ceiling.csv",
      "outputs/plots/sajben_data_audit_wall_cp.png"], "—"),
    ("P8", "Reimplementation audit against Ma et al. (AST 168, 111002)", "hand-written",
     ["docs/le_pinn_vs_ma2025.md"], "—"),
]

RECORDS = [   # referenced, never cited as results
    ("Run records", ["outputs/phase5_execution_status.md", "outputs/phase6_execution_status.md", "outputs/logs/"]),
    ("Historical calibrations (inputs to the v3/v4 evidence and reproduction paths; superseded by V1)",
     ["outputs/calibration_trent1000_ae3.json", "outputs/calibration_trent1000_ae3_v2.json",
      "outputs/calibration_trent1000_ae3_v3.json", "outputs/calibration_trent1000_ae3_v4.json"]),
    ("Not manuscript sources", ["outputs/results/pinn_comparison_results.csv",
                                "outputs/plots/integrated_cycle_comparison.png",
                                "outputs/pinn_architecture_diagram.png", "outputs/pinn_architecture_diagram.py"]),
    ("Archive (dead numbers, never cite; mapping in MAPPING.json / README.md)", ["outputs/archive/"]),
]


def all_paths() -> list[str]:
    out = []
    for r in ROWS:
        out += r.artifacts + r.extra_refs
    for _, _, _, arts, _ in PINN_ROWS:
        out += arts
    for _, paths in RECORDS:
        out += paths
    return out


def cell(s: str) -> str:
    return s.replace("|", "\\|")


def build_manifest() -> str:
    L = ["# ARTIFACT MANIFEST — canonical sources for every manuscript-bound number", "",
         "Phase 6 (v5), generated by `scripts/build_manifest.py` from its registry — **do not edit by "
         "hand**. Claim numbers are read from the artifacts. A file under `outputs/` that is not listed "
         "here must not be cited (`tests/test_manifest_integrity.py` enforces it). v4 → v5 mapping: "
         "`docs/number_crosswalk_v5.md`. Model map: `docs/model_map.md`.", "",
         "Production configuration: v5 thrust-matched cycle — calibration `outputs/calibration_v5_A2.json` "
         "(registration A1 + amendment A2), single-zone HP-equilibrium combustor (no kinetics), analytic "
         "turbine and nozzle, ξ = 0, Jet-A1 = n-dodecane (CRECK thermo).", ""]
    for sec in (SEC_V, SEC_B, SEC_E):
        L += [f"## {sec}", "", "| # | Content | Script | Output file(s) | Seed |", "|---|---|---|---|---|"]
        for r in ROWS:
            if r.section == sec:
                files = ", ".join(f"`{a}`" for a in r.artifacts + r.extra_refs)
                L.append(f"| {r.id} | {cell(r.text())} | {cell(r.command)} | {files} | {r.seed} |")
        L.append("")
    L += [f"## {SEC_P}", "", "All rows are negative or partial results; none is a production number.", "",
          "| # | Content | Script | Output file(s) | Seed |", "|---|---|---|---|---|"]
    for pid, text, cmd, arts, seed in PINN_ROWS:
        L.append(f"| {pid} | {cell(text)} | {cell(cmd)} | {', '.join(f'`{a}`' for a in arts)} | {seed} |")
    L.append("")
    for title, paths in RECORDS:
        L += [f"## {title}", ""] + [f"- `{p}`" for p in paths] + [""]
    return "\n".join(L)


def build_model_map() -> str:
    L = ["# Model map — claim to evidence (P6.7; answers R1.12, R1.14)", "",
         "Generated by `scripts/build_manifest.py` from the same registry as "
         "`outputs/ARTIFACT_MANIFEST.md` — **do not edit by hand**.", "",
         "```mermaid", "flowchart LR",
         '  ICAO["ICAO databank<br/>rated thrust, OPR, BPR<br/>fuel flow, CO/HC, EI-NOx"]',
         '  NASA["Cited ranges<br/>NASA TM/CR cycle reports"]',
         '  CRECK["CRECK C1–C16 thermo<br/>(species enthalpies)"]',
         '  CORSIA["CORSIA Doc 06<br/>L_CEF ranges"]',
         '  SAJ["Sajben experiment"]',
         '  CYC["Thrust-matched cycle<br/>φ solved at ICAO thrust<br/>single-zone equilibrium combustor<br/>analytic turbine + nozzle"]',
         '  CAL["Calibration V1<br/>W_ref, a, k_π, k_ṁ"]',
         '  NOX["NOx correlation<br/>(ICAO-derived, not chemistry)"]',
         '  LCA["Lifecycle accounting"]',
         '  PINN["LE-PINN / turbine PINN<br/>(retired; negative results)"]',
         "  ICAO -->|calibration-group fuel flow| CAL",
         "  ICAO -->|inputs: thrust, OPR, BPR| CYC",
         "  NASA -->|fixed values + ranges V7| CYC",
         "  ICAO -->|CO/HC → η_b proxy| CYC",
         "  CRECK --> CYC",
         "  CAL --> CYC",
         "  ICAO -->|calibration-group EI-NOx| NOX",
         "  CYC -->|fuel flow, OPR| NOX",
         "  CYC -->|fuel flow| LCA",
         "  CORSIA --> LCA",
         "  SAJ -.->|test| PINN",
         "  CYC --> V3[\"V3 held-out fuel flow<br/>vs B0, B1\"]",
         "  ICAO -->|held-out group, read once| V3",
         "  CYC --> V5[\"V5 design point\"]",
         "  CYC --> B1[\"B1–B3 blends at matched thrust\"]",
         "  LCA --> B1",
         "  NOX --> B1",
         "  NASA -->|range propagation| V8[\"V8 parameter bands\"]",
         "  CAL --> V8",
         "```", "",
         "| Row | Reported quantity | Computed by | Depends on | Constrained by data |",
         "|---|---|---|---|---|"]
    for r in ROWS:
        short = r.text().split(":")[0][:90]
        L.append(f"| {r.id} | {cell(short)} | {cell(r.model or '—')} | {cell(r.inputs or '—')} | "
                 f"{cell(r.constrained_by or '—')} |")
    for pid, text, *_ in PINN_ROWS:
        L.append(f"| {pid} | {cell(text.split(':')[0][:90])} | PINN (retired) | — | see manifest |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed documents differ")
    args = ap.parse_args()
    missing = [p for p in all_paths() if not (ROOT / p).exists()]
    if missing:
        print("missing artifacts:", missing)
        return 1
    docs = {MANIFEST: build_manifest(), MODEL_MAP: build_model_map()}
    if args.check:
        stale = [str(p) for p, t in docs.items() if not p.exists() or p.read_text() != t]
        print("stale:", stale or "none")
        return 1 if stale else 0
    for p, t in docs.items():
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
