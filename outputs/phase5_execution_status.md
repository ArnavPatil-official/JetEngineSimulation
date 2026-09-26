# Phase 5 execution status — 2026-09-26

Plans: `docs/plan.md` (Phase 5, original), `docs/plan_phase5_review.md` (review corrections) and
`docs/plan_phase5_nozzle_repair.md` (approved 2026-09-26: static-thrust accounting repair).
Branch `phase4`. **State: the approved accounting repair is done. The work is STILL STOPPED AT THE P5.2 STEP 2 GATE,
because the remaining 52.79 kN (17.0 %) take-off shortfall is unexplained and the required input
provenance is not established.** P5.1 finished independently overnight; the terminal report passed its
completion guard and P4.3 is closed with the registered outcome below.

## Completed and committed

| Item | Commit | Evidence |
|---|---|---|
| P5.0 housekeeping, P5.1 launch guard (before this review) | … `b5a4528` | git log |
| Review 1: reporter now scores the complete terminal set. Naming any attempt-3 run expands to all six; duplicates, stray terminal checkpoints, and checkpoint/marker identity mismatches are refused before any output is written | `61ee1aa` | `tests/test_sajben_report_guard.py` (12 tests) |
| Review 2: provenance records scoped. v1 φ row no longer claims the v2–v4 ridge; "identified" no longer means "optimum"; the unsupported casing-loss bound is removed; Phase-4 future-tense text is labelled superseded | `5ed6d30` | `outputs/parameter_provenance.md`, `scripts/validation/heat_loss_provenance.md` |
| Import-safe `calibrate_lto.py` reviewed. It reproduces the v4 objective bit-exactly and now refuses to overwrite an existing calibration record | `0de4b10` | `test_refactored_calibration_reproduces_v4_objective`, `test_calibration_refuses_to_overwrite_a_frozen_record` |
| P5.2 Step 1: identifiability profile on v4 reproduces F2 | `e148402` | `outputs/identifiability_profile_v4.{json,csv}`, `outputs/logs/identifiability_profile_v4.log` |
| P5.2 Step 2: take-off thrust decomposition | `f1bd920` | `outputs/takeoff_thrust_gap.{json,md}`, `outputs/logs/takeoff_thrust_gap.log` |
| Repair phase 1: static-thrust regression tests, which fail before the fix (6 failed, 1 passed) | `8e14f10` | `tests/test_static_thrust_accounting.py`, `outputs/logs/static_thrust_accounting_prefix_pytest.log` |
| Repair phases 2–3: analytic core nozzle F = ṁ·u_e (u_0 = 0); ideal-expansion boundary documented; `A_exit_effective` exposed as diagnostic metadata | `0be8227` | `integrated_engine.py` `run_nozzle` |
| Repair phase 4: decomposition repeated with frozen inputs; pre-repair evidence preserved | `1e42c8f` | `outputs/takeoff_thrust_gap_after_accounting.{json,md}`, `outputs/logs/takeoff_thrust_gap_after_accounting.log` |

### v4 identifiability (computed, closed form, cycle cross-checked)

Structural basis: FAR = φ·f_st exactly (f_st = 0.0670467098), so fuel flow = φ·f_st·β·ṁ_rated·x^k_mdot.
The profile uses exact inner minimisation. It was cross-checked against the real Cantera cycle at 39 points
(frozen point, F2 ridge, every box end, each profile's minimiser and edge re-optimisations). Max relative
fuel-flow difference was 3.4e-16, and **no cycle crashed**, so on every checked point the feasible set and the
closed form coincide. Runtime is 634 s, most of it the cycle checks; the profile alone takes about 15 s.

| Parameter | Verdict | Measurement |
|---|---|---|
| φ_to | **identified** | interval [0.5428, 0.5432] (0.4 % of box), edge rises 0.026 / 0.035. The v4 sampler's 0.5409 is *outside* this interval, so v4 is not the optimum |
| η_b, pressure_loss, k_pi | not identified | profiles exactly flat (edge rise 0) |
| k_mdot | not identified | flat valley [0.5753, 0.6251] = 7.1 % of box. It rises ≥ 0.28 at both edges, so only the width check exposes it |
| φ_idle, φ_app | not identified | flat valleys clipped by the 0.30 box edges |

F2 ridge: at k_mdot 0.58 / 0.60 / 0.6218 the objective is 1.619 % at every point (spread 9.7e-17), and idle
thrust is 28.56 / 27.58 / 26.53 kN (cycle). Note: at *both* 0.58 and 0.60, φ_app (0.2903, 0.2974) is outside its
[0.30, 0.40] box; F2's text mentions only 0.58. Within the boxes the zero-objective valley still spans
k_mdot 0.575–0.625. The profile minimum is ≈ 0 against v4's 1.619 %, so the v4 sampler was not at the optimum.

## Tests

**After the accounting repair (2026-09-26):** `.venv/bin/python -m pytest tests/ -v`: **132 passed, 1 skipped,
9 warnings**, which is the 125/1 floor plus 7 new static-thrust tests
(`outputs/logs/static_thrust_accounting_full_pytest.log`). `scripts/test_emissions.py` exits 0
(`outputs/logs/static_thrust_accounting_emissions.log`). All three Cantera mechanisms load and validate
(`outputs/logs/static_thrust_accounting_mechanisms.log`). Independent review verified that 40 protected and evidence files
(data YAMLs, `models/*.pt`, v1–v4 calibration and hold-out artifacts, `outputs/takeoff_thrust_gap.{json,md}`, and
`outputs/identifiability_profile_v4.*`) are byte-identical before and after. The pre-repair hashes are retained in
`outputs/logs/static_thrust_accounting_protected_sha256.json`. Independent emissions and mechanism checks
also exited 0; their logs have the corresponding `_review.log` suffix.

Before the repair: full suite `python -m pytest tests/ -v`: **125 passed, 1 skipped, 9 warnings** (baseline 104 / 1 / 9; +21 new:
12 report guard, 9 identifiability). This is the post-change P5.0 count. `scripts/test_emissions.py` was not run
because no cycle or emissions code changed. The full suite was repeated after P5.1
completed: **125 passed, 1 skipped, 9 warnings** in 93.83 s; retained output is
`outputs/logs/phase5_p51_final_pytest.log`.

Protected artifacts: all 33 files in the session hash manifest (data YAMLs, `models/*.pt`, v2–v4 calibration and
hold-out artifacts) match byte-for-byte.

## P5.1 complete — terminal attempt 3

| Seed | Training PID | Log | Completion (EDT) |
|---|---|---|---|
| 42 | 93999 | `outputs/logs/train_sajben_v5_a3_s42.log` | 02:33:53, exit 0, 5000 epochs |
| 43 | 94000 | `outputs/logs/train_sajben_v5_a3_s43.log` | 02:33:07, exit 0, 5000 epochs |
| 44 | 94001 | `outputs/logs/train_sajben_v5_a3_s44.log` | 02:33:54, exit 0, 5000 epochs |

Launch record: `outputs/logs/launch_2026-09-25/launch_record.txt`. All three `.done`
markers record exit 0 and match the checkpoint SHA-256 hashes. The existing three
data-only logs supply their captured `exit=0` trailers. All six checkpoints contain
finite weights and record the registered seed, CPU device, tanh activation, and 5000 epochs.

The queued reporter completed with exit 0; evidence is in
`outputs/logs/launch_2026-09-25/terminal_report.{log,done}`. It scored every historical
attempt and all six terminal runs. Canonical report: `outputs/sajben_retrain_v5.{md,csv}`.

- Physics-on worse-wall L2: 0.156 / 0.144 / 0.175; mean **0.158634**, sample sd **0.015618**.
  Every seed is partial, so the registered terminal outcome is **PARTIAL**. Production remains analytic.
- Matched data-only mean **0.114937**, sample sd **0.037796**; seeds straddle pass/partial.
- Difference **+0.043697** exceeds the registered spread **0.037796**: reading 3,
  **negative result about this residual formulation**. There is no attempt 4.

The physics-on checkpoint `git_sha` is `94dcf69`, sampled at save time. The launch record
identifies the code loaded at startup as `459b900`; subsequent training-module changes
were log-label text only. The original checkpoint bytes are preserved, with this
distinction recorded in the launch evidence rather than rewriting their metadata.

## Pre-repair GATE record: take-off thrust gap is structural (docs/plan.md P5.2 Step 2)

*Historical record from `f1bd920`, kept unchanged. The repair and the repeated diagnosis follow in the next section.*

| | kN |
|---|---|
| Core (analytic nozzle) | 55.3914545 |
| Bypass | 186.2185401 |
| Model total | 241.6099946 |
| ICAO rated (CSV) | 310.9 |
| Nozzle-inlet momentum the core nozzle subtracts (82.2182069 kg/s × 200.6901102 m/s) | 16.5003810 |
| Total with that term restored (diagnostic) | 258.1103756 |
| Still short | 52.7896244 (17.0 %) |

- **Missing/incorrect term.** `IntegratedTurbofanEngine.run_nozzle` computes F = ṁ(u_e − u_in), where u_in is
  the turbine-exit velocity. The engine-level equation subtracts only the freestream momentum, which is zero on a
  static stand ([NASA](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-force/)). u_in is itself set
  by continuity through `A_combustor_exit × 1.82`, which is an area choice.
- **Nozzle treatment.** The core is choked for a convergent nozzle (p05/p_amb 3.644 vs critical 1.820). The model
  always expands fully and never uses its 0.340 m² design exit area (full expansion needs 0.237 m²). A convergent
  treatment gives 70.60 kN gross vs 71.89 kN fully expanded, which is lower, so it does not explain the shortfall.
  The bypass stream is unchoked.
- **Inputs (diagnostic only, none adopted).** With the term restored, the gap closes only with total airflow ×1.2045
  (the core's 79.9 kg/s is hand-set and unsourced) or BPR 12.29 (against the ICAO-sourced 9.1). FPR cannot close it
  alone within [1.45, 2.0].

**Proposed repair (approved 2026-09-26 and implemented; see below):**
`docs/plan_phase5_nozzle_repair.md` defines the exact scope and acceptance tests.
It corrects the analytic core's static momentum accounting, preserves and explicitly
labels the ideal-expansion approximation, and reruns the decomposition with frozen
inputs. The PINN paths already receive `static_test_stand`; no change to those paths
is proposed without evidence of a separate defect. The remaining ~52.8 kN stays an
unexplained residual until the follow-up diagnosis establishes its causes. An unsourced
airflow or geometry must not be selected merely to close it. The original citation and
identifiability gates remain in force; all historical v1–v4 artifacts remain frozen.

## Static-thrust accounting repair (docs/plan_phase5_nozzle_repair.md) — done

At the frozen v4 take-off point, with all inputs unchanged
(`outputs/takeoff_thrust_gap_after_accounting.md`):

| | Pre-repair | After repair |
|---|---|---|
| Core kN | 55.3914545 | **71.8918355** |
| Bypass kN | 186.2185401 | 186.2185401 |
| Total kN | 241.6099946 | **258.1103756** |
| Gap to 310.9 kN | 69.2900054 | **52.7896244 (17.0 %)** |

The three kinds of result are kept separate:

1. **Corrected accounting (a model fix).** Core and bypass now both use the static engine-level balance
   F = ṁ_e·u_e + (p_e − p_amb)A_e, with u_0 = 0. The change is exactly the previously subtracted internal
   momentum; the residual is 1.8e-14 kN. Fuel flow (2.3182068896 kg/s), T4, turbine-exit state, core jet
   velocity and emissions are unchanged: the max relative difference is 0. TSFC changes from
   9.5948303 to 8.9814556 mg/(N·s), and specific thrust from 299.3965 to 319.8433 N·s/kg.
   The field named `thermal_efficiency` is a kinetic-efficiency proxy computed from jet velocity and
   fuel power; it remains **0.5545372519474216**, unchanged. Independent comparison:
   `outputs/logs/static_thrust_accounting_metrics_review.log`.
2. **Unexplained residual (not a model fix).** The remaining **52.7896244 kN** is not attributed to any
   audited term. Ideal full expansion bounds the core from above: the convergent counterfactual,
   from the same stagnation state with a throat area implied by continuity and not sourced, is 1.29 kN lower.
   The effective exit area (0.2374 m²) comes from continuity and is not a measured dimension. The
   configured 0.340 m² PINN geometry does not constrain the analytic flow.
3. **Illustrative input changes (none adopted).** On their own, total airflow ×1.2045 or BPR 12.29 would close
   the gap. FPR cannot close it within [1.45, 2.0]. Core airflow 79.9 kg/s is a hand-set design-point value;
   BPR 9.1 is the ICAO-sourced value. Neither closer has a source.

**Gate status.** The accounting defect is resolved. Under docs/plan.md P5.2 Step 2 and "no sourced value, no fixed
value", P5.2 registration does **not** resume. The residual needs either a sourced airflow/cycle input or a
separately justified model term. Both need the user's decision; the repair approval did not waive this.
The η_b / pressure_loss citation gate is also still open.

## Blocked until the gate is resolved

P5.2 Steps 3–4 (register the v5 objective, calibrate v5, v5 identifiability, hold-out thrust MAPE), P5.3, P5.4 (pins
"the versions that produced v5"), and P5.5. Also still open from Phase 5: η_b / pressure_loss need page-cited
values before v5 can fix them.

## Next required action

1. **User:** decide how to treat the remaining 52.79 kN take-off residual. The options are a sourced core/total airflow
   (or other cycle input), a separately justified model term, or an explicit decision on carrying the residual.
2. **User/executor:** page-cited η_b and pressure_loss values (still required before v5 can fix them).
3. P5.1 is closed; no training or report-supervisor jobs remain active.
