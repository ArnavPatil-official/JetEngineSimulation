# Phase 6 execution status

Plan: `docs/plan.md` (Phase 6, approved 2026-09-26). Branch `phase4`, start
`6a56dcc`. Test floor at start: 132 passed / 1 skipped
(`.venv/bin/python -m pytest tests/ -v`).

## Log

| Step | Commit | Result |
|---|---|---|
| Prep: Cantera Solution reuse | `ba333a8` | Reusing one Solution per role, reset to its as-constructed state per call: bit-identical to fresh Solutions (24 randomized combustor cases; 5 full cycles vs a clean HEAD worktree); ~5x faster. Without the reset, equilibrate('HP') differs at ~1e-7. Outside the plan's file list; needed to make P6.1 computationally feasible. |
| P6.1 Step 1 — F-A test | `f2dead8` | `tests/test_holdout_informativeness.py`: v4 diagnosis pinned (2 pass); v5 informativeness test fails until the v5 CSV exists (as intended). F-A added to `outputs/parameter_provenance.md`; manifest M2 flagged SUPERSEDED. 177 rows: model 2.46016229 % vs AE3-ratio baseline 3.21571920 %; 171 rows (excl. AE3 models): 2.50278929 % vs 3.24190221 %. |
| P6.1 Step 2 — run_at_thrust | `920420d`, `1f997f8` | Brent solve of φ for a static-thrust target; bracket from the lowest φ at which the cycle closes to min(φ = 1, T4 guard 2111 K). Explicit `ThrustTargetUnreachable`. 8 tests pass (round trip within 1e-6, monotone thrust and T4, warm start gives the same root, clean failures). |
| Feasibility probe (provisional) | — | AE3 inputs only (no fuel flow read); provisional p_loss 0.045, β 0.8, η_b 0.999. At the hand-set 79.9 kg/s core flow the take-off target (310.9 kN) is unreachable inside the T4 guard (max 274.6 kN); at 96 kg/s it is reached at φ 0.545, T4 1903 K. Part-power feasibility is narrow in (W_ref, k_pi, k_mdot). |
| P6.1 Step 3 — registration R0 | `59d9665` | **Rejected** by review `c55e182` (R6-A: unsourced illustrative β) before any pilot. Archived byte-identical: `outputs/phase6/superseded/p61_registration_R0_REJECTED.json`, `outputs/phase6/superseded/phase6_p61_registration_R0_REJECTED.md` (README gives hashes and reason). Split `outputs/phase6/split_p61.json` retained. |

## Review corrections (`docs/plan_phase6_review.md`, 2026-09-26)

The first dispatch (`outputs/logs/phase6_dispatch.log`) was interrupted by the
reviewer; it did not complete, although its wrapper printed DONE.

| Item | Commit | Result |
|---|---|---|
| R6-B solver failure path | `6c8144d` | Warm root above the T4 guard re-diagnosed through the cold bracket; cold and warm starts give the same `ThrustTargetUnreachable` and guarded bound (reviewer reproduction: target 291.72574888 kN; guard maximum 274.343 kN). New `CycleDoesNotClose` is the only "outside working range" signal; non-finite inputs and config errors raise. `lto_v5` no longer converts unexpected exceptions into penalties. 16 solver tests pass. |
| R6-D cache evidence | `6c8144d` | `tests/test_solution_reuse.py`: A/B/A order vs fresh Solutions, exact equality, all 5 shipped mechanism files + FAR + full cycle (7 pass). Removing the reset is detected for 4/5 mechanisms (A1highT agrees at these states anyway). `outputs/logs/solution_reuse_regression.log`. |
| R6-A β | (A1 commit) | No citable range for the implemented quantity → **dropped**: v5 is single-zone (all core air in one equilibrium). Registration amendment A1: `outputs/phase6/p61_registration.json`, `docs/phase6_p61_registration.md`. |
| R6-C definitions | (A1 commit) | Fan: exact conversion, per-draw transform, envelope 0.8880–0.9689. Compressor: constant-γ approximation documented (variable-cp 0.8356–0.8949). η_b proxy assumptions and reproducible heating values (`creck_heating_values`, tested). v5 workers refit NOx without held-out models. `tests/test_phase6_registration.py` (8 pass). |

Validation after R6-B/R6-D: pytest 157 passed / 1 skipped / **1 intentional
failure** (`test_v5_holdout_prediction_depends_on_the_cycle`: the v5 CSV does
not exist yet); emissions exit 0; 3 mechanisms validate; 40/40 protected
hashes match (`outputs/logs/phase6_review_*.log`).

## Continuation (second dispatch, then takeover)

The second dispatch (`outputs/logs/phase6_review_dispatch.log`, started 15:16)
committed through `f9cd774` and exited at 16:00 without a final report or commit
of its pilot results. Its detached pilot profile (started 15:42) ran to completion
at 16:11. An interactive session took over from there.

| Step | Commit | Result |
|---|---|---|
| Amendment A1 | `3f0b940` | β dropped (single-zone), exact fan conversion, η_b proxy documented; R0 archived. |
| P6.1 Step 4 drivers | `0d5a70e` | `lto_v5.fit` (TPE seed 42 + least_squares), `lto_v5.profile`; `calibrate_lto.py --tag v5 [--pilot] [--free ...]`, `identifiability_profile.py --v5 {pilot,full}`. |
| P6.5 | `9ae1c23` | `docs/le_pinn_vs_ma2025.md`; PINN manifest rows. |
| P6.4 item 1 | `1bffa26` | Production path described as equilibrium thermodynamics in docstrings. |
| P6.1 Step 5 code | `f9cd774` | `lto_v5.run_holdout` (model vs B0 vs B1, A2/A3), NOx holdout on calibration-group refit. Not yet run. |
| P6.1 Step 4 pilot | `472617d` | Pilot fit (111 evaluations): W_ref 102.40 kg/s, a 1.109, k_pi 1.348, k_mdot 0.414; in-sample calibration MAPE 1.80 % (93 rows, 0 unreachable); box-scaled JᵀJ condition 222. Pilot profile: **all four IDENTIFIED** → full-fit free set unchanged. Caveats: for W_ref, a, k_mdot no 9-point grid value lies inside the 95 % interval (reported bracket = ±1 grid step), and 9/36 inner re-fits stopped at the 20-evaluation cap (status 0); an unconverged inner fit can only overstate D. The full profile (17 points, inner ≤40) is the gating A1 test. |
| P6.2 surrogate LHV | `b55e55a` | LHVs computed from CRECK thermo (gas-phase fuel, H₂O vapour, 298.15 K) and tested; used only for energy-weighted lifecycle CO₂e. |

| P6.1 full fit (registered) | `e84a794` | `calibrate_lto.py --tag v5 --free W_ref a_thrust k_pi k_mdot` (232 evaluations, 16:12–16:29; `outputs/logs/phase6_p61_full_fit.log`). TPE best trial #83 (SSE 0.01369), polish stopped on xtol: W_ref 127.78, a 1.256, k_pi 0.249, k_mdot 1.048; **SSE 0.009845, calibration MAPE 6.94 %**, 0 unreachable, box-scaled JᵀJ condition 9.2e8. This is a **worse local optimum than the pilot's** (SSE 0.000460, MAPE 1.80 %, 21× lower SSE): the same second basin appears in the pilot profile at k_pi = 0.2 (SSE 0.00996, W_ref 127.4, k_mdot 1.07). No full-fit evaluation reached SSE < 0.001. |

Tests after the pilot/LHV commits: 166 passed / 1 skipped / 1 intentional failure
(`test_v5_holdout_prediction_depends_on_the_cycle`, v5 CSV not yet produced).

## Decision: amendment A2 (user, 2026-09-26)

The registration fixed the optimizer but had no rule for a full fit that lands
in a worse basin than an optimum already found on the same calibration data.
The user chose: v5 calibration = lower-calibration-SSE of two registered
polishes (≤100 evaluations) — from the TPE best trial (as recorded) and from the
pilot optimum. Registered in `e380ffb` before the selection run and before any
held-out read (`outputs/phase6/p61_amendment_A2.json`, registration doc §0a).

| Step | Commit | Result |
|---|---|---|
| A2 selection | `d9a47f3` | `calibrate_lto.py --tag v5 --a2-select` (`outputs/logs/phase6_p61_a2_select.log`). Selected **pilot-start polish**: W_ref 102.395 kg/s (v4 hand-set core 79.9; ×1.282), a 1.109, k_pi 1.348, k_mdot 0.414; calibration SSE 0.000460, MAPE 1.80 %, 0 unreachable, JᵀJ condition 222. The polish did not move (3 evaluations, xtol). Other candidate: registered fit, SSE 0.009845. Output `outputs/calibration_v5_A2.json`. |

| Full profile (A1) | `90e56e8` | `identifiability_profile.py --v5 full` from the A2 optimum (16:55–18:22). **All four IDENTIFIED** → **A1 PASS**. Edge D 51.9–198.3 (threshold 3.841); 95 % intervals W_ref [100, 110] and k_mdot [0.3625, 0.525] (grid brackets: no grid point inside), a [1.095, 1.144], k_pi [1.324, 1.353]; widths 2–13 % of box. No profile point below the fit SSE. 1/68 inner re-fits at the evaluation cap (k_pi 0.281, D 82.8). |
| Held-out test (A2–A4) | `a48e484` | `holdout_icao_validation.py --tag _v5`. 87 rows / 9 groups, 0 unreachable. Group-weighted MAPE: **model 1.82 %, B0 2.19 %, B1 1.08 %** (TO 1.71/2.05/0.58; APP 1.96/1.12/0.85; IDLE 1.81/3.40/1.79). **A2 FAIL — no demonstrated skill** (beats B0 by 0.37 pp, loses to B1 by 0.75 pp; not a §9 escalation, which is triggered only by losing to B0 by > 0.25 pp). **A3 FAIL**: TSFC–OPR slope sign agrees at take-off and idle, disagrees at approach (ICAO −1.1e-5, model +1.0e-5 per unit OPR; both near zero). **A4 PASS** (`tests/test_holdout_informativeness.py` 3/3). Fitted W_ref 102.40 kg/s vs hand-set 79.9 (×1.28): the old take-off gap as a fitted airflow. |
| NOx split validation | `a48e484` | `nox_holdout_validation.py --split outputs/phase6/split_p61.json`: EI_NOx = 8.8295·OPR^0.2348·ṁ^0.9558 fitted on 93 calibration rows / 15 models. Held-out group-weighted MAPE **3.77 %** vs naive per-mode mean EI **10.60 %** (APP 4.95 vs 5.74; IDLE 3.65 vs 7.80; TO 2.70 vs 18.25). Inputs are ICAO OPR and fuel flow (the correlation's own inputs); within-family only. |

**Reading of P6.1.** Thrust matching turns fuel flow into a genuine model output
(A4) with identified parameters (A1), and the model beats a constant-TSFC rule.
It does not beat rescaling the nearest calibration engine by rated thrust (B1),
which within one engine family is the stronger naive predictor. Per the plan,
this is reported as it came out; no tuning toward the held-out set.

## P6.3 / P6.6 / P6.8 progress (while P6.2 bands run)

| Item | Commit | Result |
|---|---|---|
| P6.3 registration + code | `4e8f00f` | `outputs/phase6/p63_registration.json`, `docs/phase6_p63_registration.md`; `scripts/optimization/blend_matched_thrust_v5.py`; `variance_decomposition.py --v5`. **Found and avoided:** `make_saf_blend` mixes its "mass fractions" on a mole basis (nominal ATJ-50 = 42.5 % ATJ by mass) while lifecycle CO₂e weights by mass; P6.3 converts mass fractions to mixture mole fractions itself (tested). `fuels.py` unchanged (v4 reproduction). Not yet run (needs P6.2 bands). |
| P6.6 design point | `ea18356` | `design_point_summary.py --v5`: AE3 take-off at 310.9 kN, φ 0.3488, T4 1718 K (v4 1895 K), core 102.4 / total 1034 kg/s, TSFC 7.70 mg/(N·s), fuel flow +2.9 % vs ICAO (in-sample); approach +0.6 %, idle −0.4 %. |
| P6.6 NOx three-path | `e02b22b` | `nox_dual_path.py --v5`: single zone at overall φ → Zeldovich 0.03 and HyChem-A2 0.09 g/kg at take-off vs certification 48.25 and correlation 49.26. The chemistry paths are low-side proxies in v5 (the v3 "upper-bound" wording referred to the β = 0.8 burner zone). |
| P6.6 heat loss | `219de8d` | `heat_loss_sensitivity.py --v5`: at matched thrust, ξ = 4 % raises Jet-A1 TSFC by 0.327 mg/(N·s) (+4.2 %), 13× the neat-fuel TSFC spread (0.025) → heat-loss treatment bounds the resolvable blend effect. |
| P6.6 hash verifier | `e9316d8` | `scripts/validation/verify_protected_hashes.py` (archived files verified via explicit mapping; baseline never regenerated): 40/40. |
| P6.8 small fixes | `371db20` | Turbine import-time banner moved under `__main__`. `tests/test_nozzle_pinn_fix.py`: both tests had been **silently failing** (missing checkpoint path, bool returns counted as passes); now assert and pass — through the analytic fallback, since the PINN fails its physics gates. `parse_sajben_cfd.py` emits zeros (committed dataset unchanged; NaN there). |
| P6.8 archive | `3cc017e` | `simulation/emissions.py`, `pareto_visual.py`, `visualize_results.py`, `dashboard.py`, `fetch_and_build_cfd_data.py`, vendored SU2 repo + zip → `archive/` (`archive/README.md`). pytest 174 passed / 1 skipped. |
| P6.8 checkpoint provenance | `d17f9b7` | `tests/test_checkpoint_provenance.py`: 11 checkpoints; dataset hashes verified against files; activation derived from source at the recorded SHA where not stored. |
| P6.8 requirements | `c6efc65` | Pinned to the v5 environment; CPU torch install documented. |

## Outstanding gates

- P6.2 bands running (`scripts/validation/p62_parameter_bands.py`).
- P6.4 item 2 (`scripts/validation/mechanism_sensitivity.py`, written, not run).
