# Phase 6 P6.1 — pre-registration (frozen 2026-09-26, before any pilot or fit)

Machine-readable copy: `outputs/phase6/p61_registration.json` (written by
`scripts/validation/phase6_register_p61.py`, which refuses to overwrite).
Formulation code: `scripts/optimization/lto_v5.py`. Split:
`outputs/phase6/split_p61.json` (`scripts/validation/phase6_split.py`).

No held-out fuel-flow, emissions or other target value was read to make any
decision below. The only ICAO targets read so far are the calibration group's
CO/HC (for η_b, below). The earlier feasibility probe (commit history,
`outputs/phase6_execution_status.md`) used AE3 **inputs** only (thrust
targets, OPR, BPR) with provisional fixed values; it read no fuel flow.

## Supporting paths added in this phase

`scripts/optimization/lto_v5.py`, `scripts/validation/phase6_split.py`,
`scripts/validation/phase6_register_p61.py`, `outputs/phase6/`,
`tests/test_holdout_informativeness.py`, `tests/test_run_at_thrust.py`,
`simulation/combustor/combustor.py` (Solution reuse, bit-identical),
`integrated_engine.py` (`run_at_thrust`, `eta_fan` design-point key). Planned
edits: `scripts/optimization/calibrate_lto.py` (`--tag v5` path),
`scripts/validation/holdout_icao_validation.py` (`--tag _v5` path),
`scripts/validation/identifiability_profile.py` (v5 profiler),
`scripts/validation/nox_holdout_validation.py` (calibration-group refit).

## 1. Split

Unit: the engine model, never the record. **Leakage guard (executor
decision, flagged for review):** models sharing an identical certification
input tuple (OPR, BPR, rated thrust) are merged into one group (e.g. Trent
1000-C and -D are input-identical), so no held-out engine has an input twin in
calibration. 28 model names → 18 groups. Groups sorted by rated thrust (then
mean OPR), paired consecutively, one per pair to calibration by
`numpy.random.default_rng(42)`; the AE3 group forced into calibration.

| | groups | models | records |
|---|---|---|---|
| Calibration | 9 | 15 (E2, H2, A, AE3, G, C/D, C2/D2/L2, J3/K3/Q3, M3/N3) | 31 |
| Held-out | 9 | 13 (E, H, H3, A2, G2, G3, CE3/D3/L3/P3, J2/K2, R3) | 29 |

**Weighting:** each group equal, records equal within a group, modes equal —
recertifications and input twins share one group weight.

## 2. Formulation (thrust-matched, v5)

Per record and mode (x = Power % / 100 ∈ {1.00, 0.30, 0.07}): target thrust
x·F_rated (ICAO input); ṁ_rated = W_ref·(F_rated/310.9)^a; π_c =
1 + (OPR−1)x^k_pi; ṁ_core = ṁ_rated·x^k_mdot; FPR = 1 + (FPR_rated−1)x^k_pi;
BPR = ICAO value; φ solved by `run_at_thrust`; fuel flow is the model output.
Jet-A1, CRECK, analytic turbine and nozzle. T4 guard 2111 K (3800 °R: upper
end of the NASA N+3 T4 design space, NASA/TM-2017-219501 p. 4).

**Fitted** (shared across calibration): W_ref ∈ [60, 140] kg/s, a ∈ [0, 2],
k_pi ∈ [0.2, 1.5], k_mdot ∈ [0.2, 1.5].

## 3. Fixed parameters (P6.2 range rule, settled before fitting)

| Parameter | Central | Range | Basis |
|---|---|---|---|
| η_b (per mode) | TO 0.999894, APP 0.999834, IDLE 0.998152 | TO 0.999880–0.999920; APP 0.999771–0.999877; IDLE 0.997321–0.998682 | **Data-derived**: η_b = 1 − (EI_CO·Q_CO + EI_HC·Q_fuel)/(1000·Q_fuel) from the calibration group's ICAO CO/HC; Q_CO 10.1018, Q_fuel (n-C12H26 LHV) 44.4620 MJ/kg from CRECK thermo at 298.15 K; central = group-weighted mean, range = min–max over calibration records. Replaces v4's unidentified 0.9963. |
| Combustor pressure loss | 0.045 | 0.04–0.05 | NASA/TM-2017-219501 p. 4 (4 %); NASA/CR-2005-213657 Table 1 p. 10 (π_b 0.96); NASA/TM-2007-214690 p. 12 (5 %, NPSS example). Replaces v4's unidentified 0.0442. |
| η_c (isentropic, single-stage model) | 0.86 | 0.8225–0.8865 | Polytropic 0.89–0.93 (NASA/TM-2017-219501 p. 3: HPC nominal 0.91, N+3 ~0.89, LPC ~0.93; NASA/CR-2005-213657 Table 1: e_lpc 0.9036, e_hpc 0.9066) converted to isentropic at OPR 43.2, γ = 1.4 |
| η_poly turbine | 0.90 | 0.90–0.92 | NASA/TM-2017-219501 p. 3 (HPT 0.91; N+2 level 0.90), p. 4 (LPT 0.92); NASA/CR-2005-213657 Table 1 (0.9029, 0.9174). Central = N+2 level (Trent 1000 predates N+3). |
| η_fan (isentropic) | 0.90 | 0.89–0.965 | NASA/CR-2005-213657 Table 1 (e_fan 0.8961); NASA/TM-2017-219501 p. 3 (0.97, geared FPR 1.3, "may seem aggressive") |
| FPR rated | 1.45 | 1.3–1.7 | NASA/TM-2017-219501 Table 3 p. 12 (NASA CFM56 model 1.7; N+3 1.3) |
| β (burner air fraction) | 0.80 | 0.70–0.90 | **ILLUSTRATIVE** — no page-citable range found. Nearest public analogue: secondary flows 15–19 % (NASA/TM-2017-219501 Table 3). Propagated and named as an assumption. |
| ξ (heat loss) | 0 | fixed | structural argument, `scripts/validation/heat_loss_provenance.md` |

These are spans of cited public design values (NASA reference-cycle and
cycle-analysis reports), not Trent 1000 data. Sources (downloaded 2026-09-26
from ntrs.nasa.gov; SHA-256 of the PDFs read):

- NASA/TM-2017-219501, Jones, Haller & Tong, *An N+3 Technology Level Reference Propulsion System* — https://ntrs.nasa.gov/citations/20170005426 — `87407ac94e6724e0eb976d5e9d8b16c4bc66a4c48f1b56757399149a3c11eeb9`
- NASA/CR-2005-213657, Liew, Urip & Yang, *A Parametric Cycle Analysis of a Separate-Flow Turbofan With Interstage Turbine Burner* — https://ntrs.nasa.gov/citations/20050186906 — `d54b2786ea410bbfc363dcb4b01c1eff634f708af0086b3c6cd5a18ca98d706b`
- NASA/TM-2007-214690, Jones, *An Introduction to Thermodynamic Performance Analysis of Aircraft Gas Turbine Engine Cycles Using NPSS* — https://ntrs.nasa.gov/citations/20070018165 — `6c89213678f5aab5cf4b469a3519856e9a74989f9f4ab4675aaad3709770d913`

## 4. Fit

Objective `thrust_matched_v5`: Σ w_i e_i², e_i = relative fuel-flow error;
an unreachable row scores e_i = 1.0. Reported: group-weighted MAPE. Optimizer:
Optuna TPE (seed 42) over the box, then scipy `least_squares` (trf, bounds,
x_scale = box widths, diff_step 1e-4) from the best trial.
Full: 150 trials + ≤100 polish evaluations. **Pilot:** 40 trials + ≤30.

## 5. Identifiability (registered diagnostic and threshold)

Profile likelihood per fitted parameter on a uniform grid over its box, other
parameters re-fit (least_squares, warm-started from the neighbouring point).
D(θ) = n_eff·ln(SSE_profile(θ)/SSE_min), n_eff = 27 calibration (group, mode)
pairs. **IDENTIFIED iff D ≥ 3.841 (χ²₁, 95 %) at both box edges AND the
95 % profile interval is at most 50 % of the box width.** Condition number of
the box-scaled JᵀJ at the optimum is reported, not gating.
Pilot: 9 grid points, inner ≤20 evaluations, from the pilot optimum. Full: 17
points, inner ≤40. A parameter not identified in the pilot is moved to
fixed-with-cited-range or dropped **before** the full fit.

## 6. Baselines, metrics, acceptance thresholds

- **B0** constant TSFC: per mode, group-weighted calibration mean of ICAO
  FF / F_target, × held-out F_target.
- **B1** F-A rule: nearest-rated-thrust calibration group (ties: mean over
  tied groups), per-mode mean ICAO fuel flow × rated-thrust ratio.
- **Primary metric:** group-weighted held-out MAPE. Secondary: record-level
  MAPE, per-mode MAPE. Unreachable rows count APE = 100 % and are listed.
- **A1** every final fitted parameter IDENTIFIED (full profile).
- **A2** margin **0.25 percentage points**: PASS iff MAPE_model ≤
  min(MAPE_B0, MAPE_B1) − 0.25 pp. **Escalate (§9)** iff MAPE_model >
  MAPE_B0 + 0.25 pp. Otherwise "no demonstrated skill" (fails A2, reported).
- **A3** OPR trend, per mode: OLS slope of TSFC = FF/F_target on OPR across
  the 9 held-out groups (one point per group: group mean OPR, group-weighted
  mean TSFC), for ICAO data and for model predictions. PASS iff signs agree in
  all three modes. Partial slope controlling for rated thrust reported, not
  gating.
- **A4** `tests/test_holdout_informativeness.py` passes on the v5 CSV.

NOx: correlation refit on calibration-group models only; held-out validation
on the held-out group.

## 7. P6.2 bands (conditioning rule)

Seeded (42) Monte Carlo, 64 draws, each fixed parameter uniform over its
range (η_b per mode over its calibration min–max; ξ fixed). For **each draw
the fitted parameters are re-fit** on the calibration group (least_squares
from the v5 optimum, ≤30 evaluations), then the design point and held-out
MAPE are recomputed. Reported as **refit-conditioned range bands** (P5–P95,
min–max) — range propagation under assumed uniform ranges, **not** statistical
confidence intervals.
