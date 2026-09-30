# Phase 8 — physics-first revision

Date: 2026-09-29. Branch `phase8`, from the Phase 7 freeze `7524a7a`.
Completed Phase 7 plan: `docs/plan_phase7_completed.md`; closure in
`outputs/phase7_execution_status.md`. Full design document (the source of this
plan; read it for equations and rationale): Claude doc "CAT-JET Phase 8 —
Physics-First Revision Plan",
https://claude.ai/code/artifact/53b56f20-5553-4edb-96d3-8c36b12513a5.
Where the two differ, this file and `docs/phase8_registration.md` govern.

This plan authorizes repo work and Mac-local computation for the **STS slice
(D1)** only: P8.0, P8.1, P8.2, P8.3 and the first P8.6 database sources.
P8.4 onward are listed for order and gates; they need a plan revision before
execution. No manuscript edits, outreach, remote publishing, merge or tag.

**Revision P8-R2 (2026-09-29):** `docs/phase8_r2_plan.md` extends local work
authorization to P8.4–P8.15 in their registered gate order. Its prospective
run registrations and the existing Phase 8 gate rules remain required. This
paragraph supersedes only the original STS-slice scope limit above; the
original phase descriptions remain as the Phase 8 baseline.

## Objective

Turn the v6 thrust-matched cycle into a component-level model in this order:
C++ port with identical results (G0), then physics upgrades each verified
alone (G1), then staged calibration and source-level validation (G2), and
only then synthetic data, MLX residual models and robust SAF optimization (G3).

The target is honest: v6 held-out fuel flow (87 rows, 9 groups) is 1.830 %
group-weighted MAPE, B0 2.189 %, B1 1.076 %. The model loses to the B1
thrust-ratio rule. The old "~11 % MAPE" is the pre-Phase-1 in-sample figure and
must not be quoted as a baseline. v6 A1 failed: all four knobs (W_ref,
a_thrust, k_pi, k_mdot) are penalty-dependent. P8.4 exists to retire them.

## Decisions (answered by Arnav, 2026-09-29)

- **D1** STS slice before the deadline (check the 2027 Regeneron STS date):
  C++ parity port, benchmark, composition-carrying state, enthalpy turbine,
  choking nozzle, first database sources. Everything else is reported as work
  in progress.
- **D2** Write the off-design matching solver in C++; verify it code-to-code
  against pyCycle's high-bypass turbofan example.
- **D3** Add other ICAO engine families. Keep the registered v6 Trent split;
  add a cross-family held-out set (whole families held out).
- **D4** The reactor network becomes the claimed NOx/CO path only if it beats
  the V6 ICAO-derived correlation on held-out engines.
- **D5** Rig data from other machines calibrates non-dimensional loss and
  discharge relations only, never Trent-specific values.
- **D6 (changed)** Phase 7 froze without P7.4. A nozzle PINN returns only as
  model M3 under a Phase 8 registration, if time allows. The turbine PINN stays
  retired. `phase7-p74-wip` is unreviewed code, not a registration.

## Constraints

- Register and commit each procedure before its first computation (benchmark,
  gates, splits, calibration stages). A failed gate stops the steps below it
  and is recorded, never tuned away.
- Protected: everything in `outputs/phase8/protected_sha256_phase8.json`
  (Phase 7 list + v6 evidence + the Python v6 source path). The Python v6 path
  (`integrated_engine.py`, `simulation/`, `scripts/optimization/lto_v5.py`,
  `lto_v6.py`) is the A0 reference and is **not edited** in Phase 8. New
  physics lives in the C++ core; the optimized-Python benchmark arm lives in a
  new module under `scripts/phase8/`.
- Chemical mechanism YAML, `data/icao_engine_data.csv` and `models/*.pt` are
  read-only. Never reuse a v1–v6 artifact name; new outputs go under
  `outputs/phase8/`, write-once.
- Each physics upgrade reduces to the v6 answer in its simplifying limit and
  is added alone, so every change in results is attributable.
- New parameters need a dataset that constrains them and must pass the A1
  profile rule including the penalty guard; otherwise they return to a cited
  fixed value (range rule).
- Simulated/CFD rows never enter a validation set. Validation uses Tier 4
  experimental rows only. No held-out row is opened before its split is
  registered with a hash.
- Keep all Phase 7 negative results visible (A1/A2/A3 FAIL, P7.3 gate
  closed, P7.4 not run, Dooley 2012 H/aromatics failures).
- `.DS_Store` belongs to the user. Ask before any push to origin.
- Phase 7 detached jobs are finished; there is nothing to leave running. Check
  `ps` for other executors in this tree before editing.

## Repo Context

The v6 production path: `scripts/optimization/lto_v6.py` builds
`lto_v5.V5Model` with the Dooley 2012 Jet A surrogate; each row calls
`integrated_engine.IntegratedTurbofanEngine.run_at_thrust` (single-zone HP
equilibrium combustor, η_b scales the temperature rise, constant-cp analytic
turbine, fully expanded nozzle, φ solved by bracketed `brentq` with
xtol 1e-12, rtol 4ε; the warm-start guess only narrows the bracket).
Knobs: `lto_v5.mode_state` (W_ref, a_thrust, k_pi, k_mdot). Frozen v6
parameters: `outputs/phase7/calibration_v6.json`. Frozen v6 rows:
`calibration_v6_rows.csv` (93) and `holdout_icao_validation_v6.csv` (87).
AE3 = Trent 1000-AE3, record `02P23RR126`, a calibration engine.
Reproduction harness: `scripts/reproduce_check.py` (P6.8, rtol compare).

Environment: Apple M3 Pro, 11 cores, 18 GB. `.venv` Python with
cantera 3.2.0, torch 2.9.1 (CPU), optuna 4.6.0. **No cmake or conda is
installed**; Homebrew is.

## Relevant Files

Existing (read; edit only where stated):
- `integrated_engine.py`, `simulation/**`, `scripts/optimization/lto_v5.py`,
  `lto_v6.py` — A0 reference, read-only in Phase 8.
- `scripts/reproduce_check.py` — pattern for the G0 comparison (not edited).
- `scripts/validation/verify_protected_hashes.py` — add a `--phase8` flag.
- `scripts/build_manifest.py` — add a Phase 8 records entry so new
  `outputs/phase8/` files are registered (orphan test); regenerate, never
  hand-edit, `outputs/ARTIFACT_MANIFEST.md` and `docs/model_map.md`.
- `requirements.txt` — add Python dependencies as each step needs them
  (pybind11, pyarrow, emcee; mlx only at P8.10).
- `outputs/phase7/*_v6.*`, `outputs/phase6/split_p61.json`,
  `data/icao_engine_data.csv`, `data/A2NOx.yaml`, `data/fuel_properties_v7.yaml`.

New:
- `docs/phase8_registration.md` — P8.0 registration (baseline, ledger
  template, gates G0–G3, benchmark protocol).
- `docs/phase8_parameter_ledger.md` — the living parameter ledger.
- `outputs/phase8/protected_sha256_phase8.json`, `scripts/phase8/freeze_protected_phase8.py`.
- `scripts/phase8/profile_v6.py`, `outputs/phase8/profile_v6_*.{txt,json}`.
- `scripts/phase8/benchmark.py`, `scripts/phase8/v6_optimized.py` (arm 2).
- `scripts/phase8/g0_parity.py`, `outputs/phase8/g0_parity_*.{csv,json}`.
- `cpp/environment.yml`, `cpp/conda-lock-osx-arm64.txt`.
- `cpp/catjet_core/` (CMakeLists, `thermo`, `combustor`, `components`,
  `nozzle`, `engine`, `batch`, `bindings`, Catch2 tests).
- `simulation/catjet_backend.py` — Python wrapper keeping the `V5Model`
  call signature, backend selected by one flag. New file; `lto_v5.py` is not
  edited (a new driver passes the backend).
- `data/empirical/` (schema SQL, vocabulary, SQLite file, CSV exports),
  `scripts/phase8/empirical_db.py`.
- `tests/test_phase8_*.py`.
- `outputs/phase8_execution_status.md`.

## Implementation Phases

### P8.0 — registration (no computation except the profile)

1. Freeze `outputs/phase8/protected_sha256_phase8.json`: the Phase 7 list,
   all committed `outputs/phase7/` v6 and registration artifacts, and the
   Python v6 source path. Add `--phase8` to `verify_protected_hashes.py`.
2. Commit `docs/phase8_registration.md` (v6 baseline numbers, parameter-ledger
   template, gates G0–G3, benchmark protocol, workloads) **before** any
   benchmark or parity run.
3. cProfile the Python v6 `run_at_thrust` at the AE3 take-off point with the
   frozen v6 parameters and fuel. Commit the profile and a short summary of
   where the time goes. This is diagnostic, not a benchmark arm.

### P8.1 — C++ core at parity (G0) and benchmark

1. Toolchain (approved 2026-09-29): Miniforge in `~/miniforge3`,
   `auto_activate_base false`, env `catjet-cpp` from `cpp/environment.yml`
   pinning libcantera-devel 3.2.0 with cmake, eigen and catch2; commit the
   explicit lockfile `cpp/conda-lock-osx-arm64.txt`. Compiler: Command Line
   Tools with the MacOSX26.5 SDK (registration amendment P8-A1.3). pybind11
   and the Python interpreter come from `.venv`. Confirm the built module
   imports in `.venv`; if the ABI does not allow it, stop and report.
2. Hello-world: HP equilibrium of the v6 fuel/air at the AE3 combustor inlet
   in C++ matches Python Cantera to 1e-12.
3. Port compressor/fan, burner (v6 HP equilibrium + η_b scaling + dilution
   mixing as coded), analytic turbine, nozzle; unit-test each at 1e-12 against
   the Python functions.
4. Port the thrust-matched φ solve, reproducing SciPy `brentq` iterates
   exactly (port the algorithm; never loosen the tolerance).
5. G0 (registered): C++ reproduces every numeric column of the 93 calibration
   rows and 87 held-out rows and the AE3 design point at rtol 1e-9.
6. Benchmark arms 1–4 on W1–W4 as registered, arms 2–4 in variants a/b/c
   (P8-A1.2: b = fewer evaluations per solve, must pass G0; c = products-only
   equilibrium, own tolerance, outside G0); publish the report.

### P8.2 — composition-carrying state and enthalpy turbine (G1)

GasState (T, P, Y) at every station, properties from Cantera; enthalpy
mixing; liquid fuel vaporisation enthalpy (v7 value). HP/IP/LP turbines on
h(T,Y) with a polytropic path (~50 pressure steps), cooling-air re-entry
(NGV before the rotor, rotor after), cooling fractions by the range rule.
G1 checks: constant-cp limit reproduces v6 T5, p5 to 1e-10; energy closure
< 1e-10 of ṁh; polytropic integral converges as steps double. Scaled maps
come with P8.4.

### P8.3 — choking nozzles at fixed area (G1)

Separate core and bypass convergent nozzles, quasi-1D, frozen composition,
real-gas critical point (max ρu on the isentrope), Cd and Cv, pressure thrust
when choked. G1 checks: reduces to v6 fully expanded when unchoked with
Cd = Cv = 1; ṁ and F continuous at the critical pressure ratio.

### P8.6 — empirical database (starts in parallel with P8.1)

SQLite long format per the design document (tables source, experiment,
operating_point, observation, digitisation, split), controlled vocabulary and
quality classes registered first. QA tests: SI round-trips, corrected-flow
recomputation, duplicate detection, plausibility. First three sources: NASA
CR-168189 (E3 HPT rig), one Langley nozzle report (TP-2991, TP-3411 or NTRS
19870014999), one NASA cold-air turbine report (e.g. TN D-6967). Confirm each
report number and table on acquisition. Download the full ICAO databank and
list candidate families for D3 **without opening any held-out rows**; the
cross-family split is registered before any family's rows are scored.

### Later (need a plan revision before execution)

P8.4 off-design matching (retires the knobs), P8.5 reactor network,
P8.7–P8.8 staged calibration and locked-source validation (G2),
P8.9–P8.12 synthetic data, MLX residual models, active learning (G3),
P8.13–P8.15 UQ, robust SAF optimization, freeze.

## Commands to Run

- `.venv/bin/python -m pytest tests/ -v`
- `.venv/bin/python scripts/validation/verify_protected_hashes.py --phase7`
- `.venv/bin/python scripts/validation/verify_protected_hashes.py --phase8`
- `.venv/bin/python scripts/test_emissions.py` after any combustor/emissions change
- `.venv/bin/python scripts/phase8/profile_v6.py`
- From P8.1: `ctest` in `cpp/build`, `.venv/bin/python scripts/phase8/g0_parity.py`,
  `.venv/bin/python scripts/phase8/benchmark.py --arm N --workload WN`

## Tests

Floor at the freeze commit: record the count in
`outputs/phase8_execution_status.md`. Add tests only for new numerical
contracts: C++ vs Python component parity, G0 comparator, G1 limit and closure
checks, nozzle continuity at choking, database QA. No long benchmark or study
inside pytest.

## Acceptance Criteria (STS slice)

1. Registration and protected list committed before any benchmark/parity run.
2. G0 passes at rtol 1e-9 on all 180 rows and the AE3 point, or its failure is
   recorded and P8.2 does not start.
3. Benchmark report covers arms 1–4 and W1–W4 with medians of 5 after 1
   warm-up, solves/s, peak memory, named machine, and the cProfile breakdown;
   if arm 2 closes most of the gap, the report says so.
4. P8.2 and P8.3 each pass their G1 checks alone.
5. Database schema, vocabulary and QA tests committed, with ≥ 3 sources
   entered and the D3 family list recorded without held-out rows opened.
6. Tests and protected hashes pass; `outputs/phase8_execution_status.md`
   states truthfully what is done, running and not started.

## Rollback Notes

Phase 8 work is additive on `phase8`. Revert named commits; never reset.
`main` = v5.0, `phase7` = freeze `7524a7a` and `phase7-p74-wip` stay as they
are. No overwrite of protected files or old models.

## Escalation Guidance

Stop and report: G0 cannot be met without loosening tolerance; the Cantera C++
build cannot be imported from `.venv`; a planned source is unavailable or has
no usable tables; any step needs a dependency not yet approved; a gate fails.
