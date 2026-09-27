# Phase 7 registrations

Plan: `docs/plan.md` (Phase 7, branch `phase7`). Each section below is
committed **before** the computation it governs. The machine-readable records
are authoritative: `outputs/phase7/p72_registration.json`,
`outputs/phase7/p73_registration.json`, `outputs/phase7/p74_registration.json`.
Phase number 7 and calibration version 6 are distinct: v6 artifacts are named
`*_v6`, Phase 7 study artifacts `p7x_*`.

Preserved and not re-run: P7.0 (`4b8eeda`, `data/fuel_properties_v7.yaml`) and
P7.1 (`a44c131`, `outputs/phase7/p71_surrogate_check.*`). The production Jet A
surrogate (Dooley et al. 2012) **passes** the heating-value check and **fails**
the hydrogen (14.12 vs 13.77 %) and aromatics (26.1 vs 18.7 %) checks. Its
target fuel is Jet-A POSF 4658; the comparison target is NJFCP A-2 POSF 10325.
A published composition for one fuel is not proof that it matches the other.
Amendment P7-A1 (the CAEP/11 nvPM form is excluded) stays in force.

Frozen before any Phase 7 computation: `outputs/phase7/protected_sha256_phase7.json`
(134 files: v5 and Phase 6 evidence, P7.0/P7.1 files, every model checkpoint,
mechanisms, Sajben data). Check it with
`scripts/validation/verify_protected_hashes.py --phase7`.

## P7.2 — calibration v6 (registered before any v6 computation)

**The only change is the fuel.** Jet A becomes the Dooley 2012 surrogate
(n-dodecane 0.404, iso-octane 0.295, n-propylbenzene 0.228,
1,3,5-trimethylbenzene 0.073 mole fraction), read from the frozen YAML. All of
the following are the Phase 6 A1 values, read from
`outputs/phase6/p61_registration.json` and hash-checked:

- the split;
- the four free parameters (W_ref, a_thrust, k_pi, k_mdot) and their box;
- the objective and group weighting;
- the fixed central values;
- the optimizer settings and full budget (TPE 150 trials seed 42, then
  least_squares with max_nfev 100);
- the identifiability rule;
- the baselines B0/B1, the metrics, the margin 0.25 pp and A1–A4.

**η_b proxy convention (explicit).** The per-mode η_b values stay the
registered Phase 6 numbers. Those were computed with Q_fuel = 44.462 MJ/kg
(n-dodecane). They are **not** recomputed with the Dooley 2012 heating value.
The proxy remains a fixed temperature-rise scaling, so the fuel is the only
change.

**Candidates.**
1. TPE (150 trials, seed 42), then a polish from the best trial.
2. A polish from the v5 A2 optimum, with the same max_nfev 100. This is a
   second registered start, **not** a new pilot.

The calibration is the candidate with the lower SSE. An exact tie goes to (1).
Both candidates are recorded.

**A1 profile.** 17 grid points, inner max_nfev 40, the existing rule, starting
from the v6 optimum. Added guard (registered here): a parameter is not called
IDENTIFIED if any grid point that decides its verdict has a returned inner
optimum with an unreachable (penalised) calibration row. The deciding points
are both box edges and the points bracketing each end of the 95 % interval. A
numerical failure converted to a penalty cannot identify a parameter.

Reported but not gating:
- inner fits stopped at the cap;
- grid resolution (whether any grid point lies inside the interval);
- J^T J condition.

**Held-out step.**
- Model, B0 and B1 MAPE, overall and per mode.
- A2 and A3 with unchanged thresholds.
- A4: the assertions of the v5 informativeness test, applied to the v6 table.
  Equivalence is tested on the v5 CSV.
- Calibration airflow, and v5-to-v6 changes.

No held-out value chooses anything.

**NOx.** Worker NOx correlations are refit without the held-out models (the
existing engine-model split). The correlation's inputs (ICAO OPR and fuel
flow) do not involve the fuel, so the P6.1 NOx split validation is unchanged.

**Dependency.** P7.3 runs only if A1 PASS **and** A2 is not ESCALATE (the
existing B0 escalation). Otherwise the result is recorded and P7.3 is
gate-closed. The held-out step always runs. P7.4 is independent.

**Commands.** `bash scripts/run_phase7.sh launch calib` runs these in a frozen
snapshot of the launch commit:
- `.venv/bin/python scripts/optimization/lto_v6.py calibrate`
- `.venv/bin/python scripts/optimization/lto_v6.py profile`
- `.venv/bin/python scripts/optimization/lto_v6.py holdout`

Each runs with 6 workers. Outputs go under `outputs/phase7/` (write-once).
Run records are under `outputs/phase7/runs/calib/`.
