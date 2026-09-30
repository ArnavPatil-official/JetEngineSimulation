# Phase 8 execution status

Plan: `docs/plan.md` (Phase 8). Registration: `docs/phase8_registration.md`.
Branch `phase8`, from the Phase 7 freeze `7524a7a`.
Scope authorized now: the STS slice (D1): P8.0, P8.1, P8.2, P8.3 and the
first P8.6 database sources.

**Phase 8 is NOT complete.** The table says what is done, running or not started.

Test floor at the freeze `7524a7a`: 231 passed, 1 skipped (the orphan-output failure seen at
`8c1d502` predates the freeze and was fixed in `7524a7a`).

| Step | Commit | State |
|---|---|---|
| P8.0 protected list (184 files) + registration | `53e58b0` | registered |
| P8.0 cProfile of Python v6 `run_at_thrust` | `2a863b3` | done (diagnostic); see below |
| P8-A1 amendment (G2: B2 + paired cluster bootstrap; benchmark variants b, c) | `4c6df84` | registered before any cross-family data or benchmark |
| P8.1 toolchain | `c8e63ad` | done: Miniforge 26.7.2 in `~/miniforge3` (auto_activate false, no shell rc change), env `catjet-cpp` (libcantera-devel 3.2.0, cmake 4.4.3, ninja, eigen 5.0.1, catch2 3.16), explicit lockfile; CLT clang 21 + MacOSX26.5 SDK |
| P8.1 step 2 HP-equilibrium parity (C++ vs Python, 1e-12) | (this commit) | **PASS** on 3 cases (worst T rel 1.6e-14; worst species at 0.16× tol). Attempt 1 failed only a stricter exact-P check that was not the plan rule (P rel 5e-16); kept as `p81_hello_equilibrium_attempt1_exactP.json` |
| P8.1 C++ port, G0, benchmark | — | not started |
| P8.2, P8.3 | — | not started |
| P8.6 database | — | not started |

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
