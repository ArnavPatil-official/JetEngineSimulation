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

| P6.1 full fit (registered) | (this commit) | `calibrate_lto.py --tag v5 --free W_ref a_thrust k_pi k_mdot` (232 evaluations, 16:12–16:29; `outputs/logs/phase6_p61_full_fit.log`). TPE best trial #83 (SSE 0.01369), polish stopped on xtol: W_ref 127.78, a 1.256, k_pi 0.249, k_mdot 1.048; **SSE 0.009845, calibration MAPE 6.94 %**, 0 unreachable, box-scaled JᵀJ condition 9.2e8. This is a **worse local optimum than the pilot's** (SSE 0.000460, MAPE 1.80 %, 21× lower SSE): the same second basin appears in the pilot profile at k_pi = 0.2 (SSE 0.00996, W_ref 127.4, k_mdot 1.07). No full-fit evaluation reached SSE < 0.001. |

Tests after the pilot/LHV commits: 166 passed / 1 skipped / 1 intentional failure
(`test_v5_holdout_prediction_depends_on_the_cycle`, v5 CSV not yet produced).

## STOPPED — decision needed (registered optimizer did not find the known better optimum)

The registration fixes the optimizer (TPE seed 42, 150 trials, then least_squares
from the best trial) but has no rule for a full fit that lands in a worse basin
than an optimum already found on the same calibration data. Running the full
profile, held-out test, P6.2 bands or P6.4 sensitivity from `calibration_v5.json`
would report a known-suboptimal calibration; picking a procedure now is an
amendment made after seeing calibration (not held-out) outcomes. No held-out
target has been read. `outputs/calibration_v5.json` is write-once and kept.

## Outstanding gates

- P6.1 Step 5: resolve the optimum question above; then full profile (A1), held-out test (A2, A3, A4).
- P6.2 bands (`scripts/validation/p62_parameter_bands.py`, written, not run —
  needs `outputs/calibration_v5.json`).
- P6.4 item 2 (`scripts/validation/mechanism_sensitivity.py`, written, not run).
