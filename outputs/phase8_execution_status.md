# Phase 8 execution status

Plan: `docs/plan.md` (Phase 8). Registration: `docs/phase8_registration.md`.
Branch `phase8`, from the Phase 7 freeze `7524a7a`.
Scope authorized now: the STS slice first, plus independent Tracks B, C and D
under the user's autonomous-execution directive (2026-09-29). P8.4–P8.15
still require P8-R2 registration and their scientific gates before execution.

**Phase 8 is NOT complete.** The table says what is done, running or not started.

Test floor at the freeze `7524a7a`: 231 passed, 1 skipped (the orphan-output failure seen at
`8c1d502` predates the freeze and was fixed in `7524a7a`).

| Step | Commit | State |
|---|---|---|
| P8.0 protected list (184 files) + registration | `53e58b0` | registered |
| P8.0 cProfile of Python v6 `run_at_thrust` | `2a863b3` | done (diagnostic); see below |
| P8-A1 amendment (G2: B2 + paired cluster bootstrap; benchmark variants b, c) | `4c6df84` | registered before any cross-family data or benchmark |
| P8.1 toolchain | `c8e63ad` | done: Miniforge 26.7.2 in `~/miniforge3` (auto_activate false, no shell rc change), env `catjet-cpp` (libcantera-devel 3.2.0, cmake 4.4.3, ninja, eigen 5.0.1, catch2 3.16), explicit lockfile; CLT clang 21 + MacOSX26.5 SDK |
| P8.1 step 2 HP-equilibrium parity (C++ vs Python, 1e-12) | `8a1878c` | **PASS** on 3 cases (worst T rel 1.6e-14; worst species at 0.16× tol). Attempt 1 failed only a stricter exact-P check that was not the plan rule (P rel 5e-16); kept as `p81_hello_equilibrium_attempt1_exactP.json` |
| P8.1 pybind11 import into `.venv` | `ae75c28` | works, with a required link recipe (below) |
| P8.1 A1 C++ port of v6 (brentq, compressor, fan, FAR, burner with per-call state reset, analytic turbine, nozzle, NOx, run_full_cycle, run_at_thrust) | `243cc1d` | unit parity **PASS** at 1e-12 (35 pytest cases incl. AE3 thrust solves with identical evaluation counts and identical unreachable reasons); brentq bit-exact vs SciPy 1.16.3 on 24 cases (Catch2) |
| P8.1 **G0** | script `fadc08d`, result (this commit) | **PASS** (`outputs/phase8/g0_parity.json`): 93 calibration rows max rel diff 2.2e-12; 87 held-out rows 1.4e-10 (the 'Model APE (%)' column; predicted fuel flow 9.4e-13); held-out summary 7.4e-12; AE3 design point (v5 artifact, 3 modes, incl. p5) 2.2e-12. Python backend also reproduces the frozen v6 artifacts today (max 1.9e-12). Wall times in that run are not benchmark data (4 helper jobs were loading the Mac) |
| P8.1 benchmark | — | not started |
| P8.2, P8.3 | — | not started |
| P8.6 schema + vocabulary v1 + QA code/tests | `02cb627` (v2 `b7daa51`) | registered (P8-R1) before any source |
| P8.6 source 1: NASA CR-168189 (E3 HPT cooled rig, Table 5.3.1-II) | `2a4a0f4` | entered: 23 test points, 191 observations, class A, QA 0 issues; transcription cross-checked against the table's SI and clearance-adjusted columns |
| P8.6 sources 2-3 | — | **blocked on digitisation**: TN D-6967 and Grey & Wilsted (NACA TR-933/TN-1757) have plotted data only; TP-2991 is a C-D vectoring nozzle (poor fit). See `data/empirical/acquisition_log.md` |
| P8.6 ICAO databank + D3 family list | `ac88f6c` | 888 records, 133 families by prefix rule; identifier/design/status columns only, no target decoded (`outputs/phase8/icao_edb_families.*`) |
| Track C1 WebPlotDigitizer importer + instructions | this commit | Three CSV/project repeats; axis calibration and source hashes stored for each observation; Hungarian point matching; sample-SD digitisation uncertainty; 22 focused tests pass. `data/empirical/DIGITISING.md` names the source figures, curves, axes, priority and repeat paths. Digitisation itself is a user-only task and has not started. |

## P8.0 profile (AE3 take-off, frozen v6, `outputs/phase8/profile_v6_ae3_takeoff*.{txt,json}`)

The solved fuel flow matches the frozen `calibration_v6_rows.csv` row to 1e-9.
- Steady-state solve: **0.334 s**, 25 cycle evaluations (36 `cycle` calls,
  14 Brent residuals). **97 %** of it is inside `combustor.run`, i.e. Cantera
  HP equilibrium plus property calls on the CRECK mechanism. One isolated HP
  `equilibrate` takes 14.6 ms, so each cycle is about one equilibrium solve.
  Python orchestration (compressor, turbine, nozzle, Brent, bookkeeping)
  is about 3 %.
- First solve in a new worker: 2.52 s, dominated by one-time Solution
  parsing (`_fresh_solutions` 1.46 s, `_calculate_fuel_air_ratio` 0.73 s).
- cProfile cannot see Cantera's compiled methods; their time is attributed
  to the calling Python function. The first run of the script labelled that
  time as "simulation/ modules"; those uncommitted outputs were replaced by a
  rerun with corrected labels and the equilibrate timing (same code path).

Consequence for the benchmark (to be tested, not assumed): a line-for-line
C++ port of v6 can remove at most the ~3 % Python share per solve, because
the same Cantera equilibrium dominates in both. Faster per-solve times need
fewer cycle evaluations per solve or cheaper equilibrium, and any such change
must still pass G0. Throughput gains come from parallelism (arm 4) and from
avoiding per-worker Solution parsing. This is the plan's "C++ speedup is small"
risk, now measured.

## P8.1 toolchain finding: two Cantera builds in one process

The `.venv` pip Cantera wheel (`_cantera.cpython-312-darwin.so`, Cantera built
in statically, with its own sundials 7.4, yaml-cpp, hdf5 dylibs) exports weak
definitions of Cantera's C++ symbols. A pybind11 module linked to the shared
conda `libcantera.3.2.0.dylib` had those symbols bound to the wheel's copies
(dyld: "_cantera... has weak-def symbol used by libcantera.3.2.0.dylib") and
the process aborted (SIGABRT) on the module's first `newSolution` whenever
`cantera` had been imported, in either import order. Alone, the module worked.

Fix (in `cpp/CMakeLists.txt`): Python modules link the static `libcantera.a`,
compile with hidden visibility and export only `_PyInit_<module>`. With it, the
probe (`cpp/bindings/probe.cpp`) and pip Cantera coexist in both import orders,
give bit-identical HP-equilibrium T at the AE3 take-off inlet
(1700.811664711411), and a C++ `CanteraError` reaches Python as `RuntimeError`.
The module still binds conda's own `libfmt` (the fmt that `libcantera.a` was
built against), which is correct. Every future `catjet_core` module must use
the same recipe; standalone executables may keep the shared library.
