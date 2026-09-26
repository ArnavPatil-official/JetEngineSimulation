# Phase 5 execution status — 2026-09-25 23:15 EDT

Plans: `docs/plan.md` (Phase 5, original) and `docs/plan_phase5_review.md` (review corrections).
Branch `phase4`. **State: STOPPED AT THE P5.2 STEP 2 ESCALATION GATE: the take-off thrust gap is
structural.** The P5.1 training runs are independent of the gate and are still running.

## Completed and committed

| Item | Commit | Evidence |
|---|---|---|
| P5.0 housekeeping, P5.1 launch guard (before this review) | … `b5a4528` | git log |
| Review 1: reporter now scores the complete terminal set. Naming any attempt-3 run expands to all six; duplicates, stray terminal checkpoints, and checkpoint/marker identity mismatches are refused before any output is written | `61ee1aa` | `tests/test_sajben_report_guard.py` (12 tests) |
| Review 2: provenance records scoped. v1 φ row no longer claims the v2–v4 ridge; "identified" no longer means "optimum"; the unsupported casing-loss bound is removed; Phase-4 future-tense text is labelled superseded | `5ed6d30` | `outputs/parameter_provenance.md`, `scripts/validation/heat_loss_provenance.md` |
| Import-safe `calibrate_lto.py` reviewed. It reproduces the v4 objective bit-exactly and now refuses to overwrite an existing calibration record | `0de4b10` | `test_refactored_calibration_reproduces_v4_objective`, `test_calibration_refuses_to_overwrite_a_frozen_record` |
| P5.2 Step 1: identifiability profile on v4 reproduces F2 | `e148402` | `outputs/identifiability_profile_v4.{json,csv}`, `outputs/logs/identifiability_profile_v4.log` |
| P5.2 Step 2: take-off thrust decomposition | `f1bd920` | `outputs/takeoff_thrust_gap.{json,md}`, `outputs/logs/takeoff_thrust_gap.log` |

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

Full suite `python -m pytest tests/ -v`: **125 passed, 1 skipped, 9 warnings** (baseline 104 / 1 / 9; +21 new:
12 report guard, 9 identifiability). This is the post-change P5.0 count. `scripts/test_emissions.py` was not run
because no cycle or emissions code changed.

Protected artifacts: all 33 files in the session hash manifest (data YAMLs, `models/*.pt`, v2–v4 calibration and
hold-out artifacts) match byte-for-byte.

## Live jobs (not stopped, not relaunched)

| Seed | PID | Log | State at 23:15 |
|---|---|---|---|
| 42 | 93999 | `outputs/logs/train_sajben_v5_a3_s42.log` | running, epoch ~950 / 5000 |
| 43 | 94000 | `outputs/logs/train_sajben_v5_a3_s43.log` | running, epoch ~1000 / 5000 |
| 44 | 94001 | `outputs/logs/train_sajben_v5_a3_s44.log` | running, epoch ~950 / 5000 |

Launch record: `outputs/logs/launch_2026-09-25/launch_record.txt`. Each run writes `<log>.done` on exit.
**P5.1 is not complete.** Once all three `.done` markers show exit 0, run
`python scripts/validation/sajben_report_p43.py` (the guard refuses otherwise), then commit the checkpoints,
logs, markers and `outputs/sajben_retrain_v5.{md,csv}`. The growing logs are deliberately uncommitted.
On the current tree the guard refuses: the three physics-on checkpoints are missing, and the legacy data-only
runs pass on their log-trailer evidence.

## GATE: take-off thrust gap is structural (docs/plan.md P5.2 Step 2)

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

**Proposed repair scope (not implemented; needs the user's decision):**
1. Correct the core-nozzle thrust in `integrated_engine.py::run_nozzle` (and the PINN nozzle path's thrust
   bookkeeping, for ablations) to the engine-level static equation, F = ṁ_e·u_e + (p_e − p_amb)·A_e.
2. Decide the core nozzle treatment: ideal full expansion (current) or convergent/choked with pressure thrust.
   These differ by 1.3 kN at take-off.
3. Then re-run `takeoff_thrust_gap.py`. The remaining ~52.8 kN would be input-level, and the main unsourced input
   is the core mass flow (79.9 kg/s). It needs a sourced value, or the user must decide to make airflow a fitted
   parameter identified by the new thrust targets (to be decided under P5.2 Step 3).
4. After any change: full pytest, `scripts/test_emissions.py`, and update the nozzle tests that touch
   `thrust_momentum` (`tests/test_nozzle_pinn_fix.py`, `tests/test_le_pinn_benchmark.py`). Every v1–v4 thrust/TSFC
   number moves (+6.8 % at take-off from item 1 alone); v1–v4 artifacts stay frozen.

## Blocked until the gate is resolved

P5.2 Steps 3–4 (register the v5 objective, calibrate v5, v5 identifiability, hold-out thrust MAPE), P5.3, P5.4 (pins
"the versions that produced v5"), and P5.5. Also still open from Phase 5: η_b / pressure_loss need page-cited
values before v5 can fix them.

## Next required action

1. **User:** decide the thrust repair (items 1–3 above).
2. **Executor, independent of the gate:** when the three runs finish, run the reporter and commit P5.1 as described
   above.
