# Phase 8 execution status

## Active product direction — 2026-10-04

The product is a physics-informed neural SAF blend pre-screening tool with
measured computational advantage. This direction supersedes the prior
post-chain scientific sequence. No prose results draft.

**Working deadline: October 28, 2026 (America/New_York).** Prioritize the
working product. Preserve the local tag name `freeze-2026-10-18`.

The main AC chain is unchanged. At the 19:30 local check on 2026-10-04,
run 2 and its two registered reruns had finished. The validated run-2
completion retains **48 PASS, 1 INVALID_POWER and 7 FAIL** (including the
seven historical specs); both reruns are **PASS**. Benchmark-owner release
is recorded at 22:45:30 UTC. Owner PID 22543 still holds the main lease,
waiting for AC before A2 calibration; no A2 child has started. Remaining
original order is A2, separate validation build, Track 4 focused tests and
its single diagnostic, full pytest, then protected hashes. The original
scientific tree and built-module hashes still match the lease.

**Deferred, not failed:** A4c; P8.5 rerun 4; G2 cross-family; stages 1–3
calibration and digitising; P8.9–P8.15 beyond the new screening/nozzle/product
studies; Sajben low-label study. All registrations stay intact. A4c's held-out
allocation is released unused; its reservation directory does not exist and
no score was taken. Do not execute the older post-chain A4c/A5 instructions.

New post-chain order: commit fresh G0 rerun evidence; registered P7.3-A1
conditional v6 C++ screening; P8-S generation, MLX training and frozen one-shot
scoring; blend-conditioned nozzle ODE PINN; tested screening CLI/API; quantitative
freeze and figures with a local tag. Every new procedure is registered before
computation. Fresh `outputs/phase8/g0_rerun_20261003` is currently absent;
the user directed its run after chain completion on 2026-10-04. It is pending
until the validated idle gate succeeds and AC is confirmed. The exact preflight
and command are in the active plan. No new study result exists.

P7.3-A1 preserves the claim rule, 64 fixed-parameter draws, fuels, modes,
Brem domain and lifecycle basis, while explicitly permitting the historical
penalty-guard-only A1 FAIL. Label its outputs `conditional on v6 calibration`.
P8-S claims simulator fidelity and measured speed; real-world accuracy is
inherited from v6. Exact fuel/CO2/lifecycle relations are computed, not learned.
Locked Sobol and named-blend test sets are opened once after all models freeze.

The older entries below retain historical implementation and registration
context; their A4c/A5 deadlines and manual commands are superseded by this
explicit deferral. See `docs/plan.md` for the current execution scope.

### New study registrations and implementation isolation — 2026-10-04

- P7.3-A1: `71c29c4`, prospective original-proof clarification `2b845a7`,
  exact additive paths/raw checkpoint note `b8ad0d6`.
- P8-S: `0c3e586`; the operational fuel-flow bar is fixed at 10% of the
  smallest eligible central named blend effect. Other fidelity/ranking,
  physics-benefit and measured-speed gates remain distinct.
- Blend-conditioned nozzle ODE study: `3d60ef3`, prospective source/duplicate
  coverage clarification `60c29d4`.
- Shared post-chain source/G0 provenance: `e415531`; CLI/API and producer
  receipt protocol `7cf5c9f`; synchronous verification/envelope note `f54878d`.

Numerical consumers remain isolated in worktrees and have not been integrated
into the main scientific tree, trained or scored. After strict benchmark
queue/owner-release validation, isolated toy checks passed: **68 gate/API
cases and 39 P7.3 fake-backend cases**. A changed owner-birth fixture now
expects lease retention; toy lifecycle fixtures include the registered
pathway constants and label metadata. These checks import no simulator,
Cantera or MLX and open no scientific test labels. The source/binary proof
was unchanged afterward. This is not the pending scientific verification
receipt. The prospective fixture clarification is committed as `7480226`.
No new G0, P7.3-A1, P8-S or nozzle result exists, and no new score reservation
has been consumed. A record-gated post-chain driver is being prepared; it
is **not yet armed**. It must export strict original evidence before new
source integration and follow the current sequence without manual gaps.

The active chain has reached arm 4. `arm4b_W4_11w` is a terminal **FAIL**:
its warm-up is internally reproducible but fails the frozen reference check
on nine core-thrust cells. No median timing exists for that failed run. Its
raw record stays unchanged, it is not retried, and the chain continues under
its registered terminal-with-flags policy. Fresh P8-S speed measurements
remain after release and use the full v6 C++ teacher.

### Standalone validation and timing overlap — 2026-10-04

During the naming-cleanup verification, Codex launched standalone
`.venv/bin/python -m pytest tests/ -v` while benchmark ownership was active.
That was an execution mistake; no further standalone tests or heavy work are
launched during the benchmark. The run reported **500 passed, 3 skipped,
1 failed**, plus four passed subtests. `test_all20_eligible_families` failed
because this standalone process was not the registered live `full_pytest`
child of the active workflow. It does not replace the chain's required test.
Post-test Phase 7 and Phase 8 protected hash checks passed.

The estimated test interval was 17:19:21–17:24:20 local, derived from the
pytest cache write time and reported 299.16-second runtime, rather than an
authoritative launch log. Comparing that estimated interval with driver
timestamps indicates overlap with
`run2_ac/arm2a_W4_11w`, `arm2b_W1_1w`, `arm2b_W2_1w` and `arm2b_W3_1w`.
Preserve their raw results and existing records; these timings are not clean
speed evidence. The last row also independently has `INVALID_POWER`.
Any clean follow-up measurement requires a prospective registration after
benchmark ownership ends; the current queue and its two registered reruns
remain unchanged.


Plan: `docs/plan.md` (Phase 8). Registration: `docs/phase8_registration.md`.
Branch `phase8`, from the Phase 7 freeze `7524a7a`.
Scope authorized now: Slice 1 first, plus independent Tracks B, C and D
under the user's autonomous-execution directive (2026-09-29). P8.4–P8.15
are covered by `docs/phase8_r2_plan.md` in their scientific gate order;
each run-specific registration remains required before computation.
Freeze-package outputs: `outputs/freeze/`. Local freeze tag: `freeze-2026-10-18`.

**Phase 8 is NOT complete.** The table says what is done, running or not started.

Test floor at the freeze `7524a7a`: 231 passed, 1 skipped (the orphan-output failure seen at
`8c1d502` predates the freeze and was fixed in `7524a7a`).

| Step | Commit | State |
|---|---|---|
| P8.0 protected list (184 files) + registration | `53e58b0` | registered |
| P8.0 cProfile of Python v6 `run_at_thrust` | `2a863b3` | done (diagnostic); see below |
| P8-A1 amendment (G2: B2 + paired cluster bootstrap; benchmark variants b, c) | `4c6df84` | registered before any cross-family data or benchmark |
| P8-A2 ablation ladder | `421022e` | registered before P8.2 results; each step recalibrates on calibration groups and scores the Trent held-out group once |
| P8-R2 plan revision, P8.4–P8.15 | `1974420` | registered before P8.4+ numerical work. Adds design/off-design, combustor, four-stage calibration, family-level bootstrap, synthetic/MLX/G3 and robust-optimization order. The external Claude design artifact was inaccessible, so missing numerical/model details require prospective pre-run registrations. |
| P8.3 numerical G1 procedure | `2dfd6bb` | prospective registration for the choking-nozzle implementation; no G1 result yet |
| D2 pyCycle HBTF reference | `42a3003` | 26/26 published upstream reference values reproduced; 126-call envelope sweep had 6 nonconverged points, retained in the record |
| D3 MLX/PyTorch scaffold | `42a3003` | 33 toy parity tests passed; float64 CPU scoring path; no simulator or empirical-target training |
| P8.1 toolchain | `c8e63ad` | done: Miniforge 26.7.2 in `~/miniforge3` (auto_activate false, no shell rc change), env `catjet-cpp` (libcantera-devel 3.2.0, cmake 4.4.3, ninja, eigen 5.0.1, catch2 3.16), explicit lockfile; CLT clang 21 + MacOSX26.5 SDK |
| P8.1 step 2 HP-equilibrium parity (C++ vs Python, 1e-12) | `8a1878c` | **PASS** on 3 cases (worst T rel 1.6e-14; worst species at 0.16× tol). Attempt 1 failed only a stricter exact-P check that was not the plan rule (P rel 5e-16); kept as `p81_hello_equilibrium_attempt1_exactP.json` |
| P8.1 pybind11 import into `.venv` | `ae75c28` | works, with a required link recipe (below) |
| P8.1 A1 C++ port of v6 (brentq, compressor, fan, FAR, burner with per-call state reset, analytic turbine, nozzle, NOx, run_full_cycle, run_at_thrust) | `243cc1d` | unit parity **PASS** at 1e-12 (35 pytest cases incl. AE3 thrust solves with identical evaluation counts and identical unreachable reasons); brentq bit-exact vs SciPy 1.16.3 on 24 cases (Catch2) |
| P8.1 **G0** | script `fadc08d`, result `74f53c8` | **PASS** (`outputs/phase8/g0_parity.json`): 93 calibration rows max rel diff 2.2e-12; 87 held-out rows 1.4e-10 (the 'Model APE (%)' column; predicted fuel flow 9.4e-13); held-out summary 7.4e-12; AE3 design point (v5 artifact, 3 modes, incl. p5) 2.2e-12. Python backend also reproduces the frozen v6 artifacts today (max 1.9e-12). Wall times in that run are not benchmark data (4 helper jobs were loading the Mac) |
| P8.1 benchmark variants b, c (gates) | `72f4ce3`, `a83ac7f`, `c938bf2` | arm 2b G0 **PASS** (180 rows + AE3, worst 4.0e-13). Arm 2c approximation gate **PASS** (FF 2.2e-9 rel, T4 1.4e-7 K, MAPE change 3.5e-9 pp); the first c gate preceded an uncommitted post-gate edit of the shared probe bracket (0.20/None→0.25/0.10), which was **discarded**: bracket restored to the gated b code, c re-gated on it (`arm2c_gate_rev2.json`, identical numbers, source hash recorded). Native arms 3/4: variant b (V6Engine + probe), c (benchmark-only copy with 12-species products-only equilibrium) and the copy in full-equilibrium mode: **PASS** (`native_variants_gate.json`; full and b worst 1.8e-12, c same as arm 2c). Variant b saves little: median W3 solve 27 → 25 cycle evaluations. |
| P8.1 benchmark timings | legacy queue retired; recovery pending | Run 2 (`outputs/phase8/benchmark/run2_ac/`, 56 runs) started on AC 2026-10-01 ~14:50Z; arm 1a W1 0.305 s (25 evaluations), W2 14.55 s, W3 12.97 s (no timing gaps now that progress lines are timestamped). **Overlap note:** during arm1a_W4_1w the executor compiled and tested P8.4 (single niced job; load average ~4.5 logged per repeat; repeat 2 364.8 s vs 359.8 s for repeat 1). Run 1 (battery) is a superseded record. | A ~20 s single-core development run overlapped `arm1a_W2_11w` (repeats 1–2; median 2.457 s, repeat 5 3.10 s): flagged; a clean supplementary rerun is queued after run 2. |
| P8.2 implementation | `5d59209` | GasThermo state, enthalpy mixing, liquid-fuel enthalpy (360 kJ/kg), cited HP cooling (CR-168189: NGV 0.0641, rotor 0.0275), RK4 polytropic HP/IP/LP stages (P8.2-A1), HP inversion 1e-13 (P8.2-A2). |
| P8.2 **G1** | `df3a6c1` | **PASS** at AE3 TAKE-OFF, APPROACH, IDLE (`outputs/phase8/p82_g1.json`): constant-cp vs v6 T5/p5 ≤ 6e-16; A1 vs zero-cooling A2 interface ≤ 1.3e-16; worst closure 2.9e-11 (1e-10); 50→100 steps 2.9e-11 (1e-8). Burner energy closure is definitional (reported, not counted). A fresh 180-row `g0_parity` rerun is **not** in the record (v6 sources byte-identical to the G0 commit instead; see the registered decisions and AC work below). The P8.2 compressor's ideal-outlet `setState_SP` keeps Cantera's 1e-9 default (~1e-9 precision; noted for P8.4). |
| P8.3 implementation + **G1** | `88319ec`, `6bf7918`, `ef4e928` | Development Catch2 run failed 1/5 (mass flow 2.76e-10 off at 1e-12: `setState_SP` default 1e-9); recorded in `docs/phase8_p83_amendment_a1.md` before G1, isentrope inversion set to 1e-13. G1 **PASS** (`outputs/phase8/p83_g1.json`): constant-cp p*/p0 3.6e-15; mass-flow/force jumps at p*(1±1e-9) 1.2e-15 / 1.6e-9 (1e-8); unchoked Cd=Cv=1 vs v6 force 8.5e-14; real-gas core/bypass exit energy ≤ 5.6e-13, Y and elements unchanged. AE3 take-off core nozzle chokes (p0/pa 1.87, real-gas p*/p0 0.537). |
| P8-A2 ladder solver fix | `b415ddd` | Before any ladder result: P8.2's thrust solve replaced by `thrust_match.hpp`, a template copy of the v6 solve; pytest proves it identical to `V6Engine::run_at_thrust` (phi, evaluation count, fuel flow, reasons) at AE3 three modes, cold/warm, above/below range. |
| Ladder A1 calibration | `5e03692` | Frozen `lto_v6.run_calibration`, P8.2 level 1: selected polish_from_v5_A2, SSE 4.5887e-4 (v6 4.5708e-4), in-sample MAPE 1.786 % (v6 1.785 %), W_ref 103.474, a_thrust 1.10891, k_pi 1.34377, k_mdot 0.414514, 0 unreachable; TPE candidate SSE 8.2e-3. Took 74 min vs v6's 11 min for a similar evaluation count (not profiled yet). |
| Ladder **A1 held-out** (scored once) | `99a0578` | **1.835 %** vs v6 1.830 %, B0 2.189 %, B1 1.076 %. Take-off 1.724, approach 1.917, idle 1.863 (B1 0.579 / 0.855 / 1.795). A2 skill FAIL, A3 approach-sign FAIL (as v6). 0 unreachable. |
| Ladder A2 calibration | attempt 1 stopped | Stopped by the executor at 02:21Z (Mac on battery, ~35 min left; run needs >1 h). `calibration_p8_A2_attempt1_incomplete_provenance.json` kept; no parameters produced. Restart on AC: `ablation_ladder.py --step A2 --phase calibrate`. |
| Ladder A3 | merged into A4 (`5958df1`, P8-A3) | User decisions 2026-10-01: A3 is not separately scorable (no closed cycle before P8.4 matching); A4 has no knobs (P8-R2 governs). |
| P8.4b Trent three-shaft (ladder A4) | registration `c0fd761`; code `1cb32e4`-series | Cited inputs (T4 1800 K from Martinez *Aerospace engine data*, Trent XWB TET; FPR 1.45 v6; IPC/HPC split by stage count; pyCycle generic efficiencies/ducts; CR-168189 cooling; P8.3 Cd 0.96/Cv 0.95). Checks **FAIL** (`p84b_g1.json`, `1cb32e4`): IPC reaches its stall line at 0.5 rated, APPROACH/IDLE unreachable; AE3 take-off fuel flow 2.700 kg/s (ICAO 2.327); sensitivities 2.49–2.92. |
| P8.4b-A1 handling bleed | `33b8e15`, `8262d14`; checks `25d0201` | Checks **PASS** (`p84b_a1_g1.json`): AE3 TO/APP/IDLE converge, closure ≤ 7e-13; fuel flow 2.700/0.693/0.255 vs ICAO 2.327/0.643/0.244; bleed 0/15/24 %. |
| Ladder **A4** (scored once) | `2dd705f` | **21.24 %** held-out (v6 1.830, A1 1.835, B0 2.189, B1 1.076). 6 held-out rows unreachable (both Trent 1000-E records fail at the design point from the registered guess; scored e = 1.0). Converged rows over-predict: TO +15.4 %, APP +10.3 %, IDLE +8.3 %. Skill check ESCALATE; A3 sign FAIL. Knob-free cited inputs; P8.7 is where design-point variables become calibrated. |
| P8.5 registration + A1 + A2 notes | `bb5b61b`, `7143e6b`, `a822e5d` | Network registered before computation; A1: C++ port of Cantera's steady loop (threshold 10 rtol), enthalpy-based PSR, sequential zones; A2: pre-G1 probe found whole-network closure C/H ~4.6e-7, traced to **7 element-imbalanced HyChem fuel reactions in the protected `A2NOx.yaml`** (2–4e-7 C/H atoms per event); G1 rules unchanged. |
| P8.5 implementation | `a822e5d`, `2f2b57f` | Threaded network in `cpp/catjet_core/reactor_network.*`; dev builds in `cpp/build_next` (ignored) while ladder workers held `cpp/build`. |
| P8.5 **G1** | attempt 2 `722a0b6`, rev1 `04b384d`, rev2 `43adda9` | All **FAIL** under the unchanged rules. Root cause of the closure error found (P8.5-A3, `52d387c`): **Cantera cloning rounds the HyChem stoichiometry** (reaction 0 mass imbalance +6.1e-5 vs −4.0e-6 kg/kmol in the file), so the cloned network had a 15x larger, opposite-sign error. Without cloning the error equals the protected A2NOx's own imbalance (~2.7e-8; diagnostic now matches observation to 4–6 %). P8.5-A4 (`ac4ec11`): G1 script now uses the take-off design as registered (it had recomputed it per mode) and PSRs are solved at unit flow. Rev2: closure el 2–3e-8 / energy ≤ 4.6e-10; σ=0 T spread ≤ 2.5e-12 but strict trace-species rule fails; long-τ TO pass, APP 2.5 K, IDLE 0.041 K with dY 2.2e-5. Outputs now sensible: eta_b 0.996/0.987/0.928; CRECK spread within 0.4 %. The earlier "APP/IDLE barely burn" and the "attribution not confirmed" status were artefacts of cloning and the per-mode design; corrected here. |
| P8.4 registration | `f3be816` | Two-shaft HBTF mirroring pyCycle; thermo-matched (pyCycle JANAF → Cantera NASA-9) vs production (CRECK) modes; map scaling; 10×10 off-design Newton; G1 = 0.10 % vs pyCycle in thermo-matched mode, closure audit. Trent three-shaft application = P8.4b after G1. |
| P8.4 maps + thermo exports | `67eab80`, `a5f12db`, P8.4-A1 `e15d14c` | pyCycle 4.4.0 maps, JANAF (NASA-9 into Cantera; cp/R 3e-16, molar masses 2e-16 vs pyCycle; pyCycle's element_wts has C 12.0170, a transposition, mirrored for the fuel) and US 1976 table exported with hashes. P8.4-A1 (before any solve): matched mode uses shifting equilibrium at every station, fuel h = 0, CEA air elements. |
| P8.4 implementation | `0ba43f5` | C++ two-shaft HBTF: maps (scipy-verified 1e-12), Akima atmosphere (scipy-verified), element equations mirroring pyCycle, 4×4 design and 10×10 off-design damped Newton, closure audit; matched and production thermo. |
| P8.4 **G1** | `eaf61f5` | **PASS** (`outputs/phase8/p84_g1.json`): matched mode at DESIGN, OD_full_pwr, OD_part_pwr, 75 quantities each, worst rel diff **3.9e-5** (tolerance 0.10 %); no extrapolation; mass ≤ 2e-16, energy ≤ 9e-13, production element closure ≤ 1e-13. Production deltas (reported only): FAR +1.6–1.9 %, TSFC +1.4–1.7 %. Next: P8.4b (Trent 1000 three-shaft application, knob retirement) needs its own registration. |
| P8.6 schema + vocabulary v1 + QA code/tests | `02cb627` (v2 `b7daa51`) | registered (P8-R1) before any source |
| P8-R1b empirical vocabulary v3 | `55be484` | registered FAR, chemical combustion efficiency, total/static turbine PR and corrected torque before entering the newly inspected GE E3 core/LPT/HPT sources or digitised TN D-6967 points |
| P8.6 source 1: NASA CR-168189 (E3 HPT cooled rig, Table 5.3.1-II) | `2a4a0f4` | entered: 23 test points, 191 observations, class A, QA 0 issues; transcription cross-checked against the table's SI and clearance-adjusted columns |
| P8.6 further tabulated sources | this commit | GE E3 HPT CR-168289 (1 measured design point), GE E3 LPT CR-168290 (4 legible measured runs, partial map), GE E3 core CR-168069 (19 station rows, 17 with raw EI values) entered. Derived SQLite SHA-256 `3ad6cd2ddcfd0a4896aa8486569951b81b670c4b1bb3f641827d14ad4c277579`: 4 sources, 47 points, 374 observations, 0 QA issues. Nozzle source TN-1757 and TN D-6967 still need user digitisation; Langley C-D vectoring reports are a poor fit. |
| P8.6 ICAO databank + D3 family list | `ac88f6c` | 888 records, 133 families by prefix rule; identifier/design/status columns only, no target decoded (`outputs/phase8/icao_edb_families.*`) |
| Track C1 WebPlotDigitizer importer + instructions | `a849c9a` | Three CSV/project repeats; axis calibration and source hashes stored for each observation; Hungarian point matching; sample-SD digitisation uncertainty; 22 focused tests pass. `data/empirical/DIGITISING.md` names the source figures, curves, axes, priority and repeat paths. Digitisation itself is a user-only task and has not started. |
| Track C3 family split | `5a57bb7` registration, `c0475a8` draw | User decision 2026-10-01: admit geared turbofans. Registered rule (TF only, in production, not superseded, OPR/BPR/thrust > 0; databank designations with Trent 7000 grouped) gives **20 families**; stratified seeded draw (S1 < 100 kN, S2 100–200, S3 > 200; seed 20261001): held-out CF34-10, D-36, PW1500G, LEAP-1A, LEAP-1C, GE90, GEnx-1B, Trent XWB; calibration 12 incl. Trent 1000. sha256 63baef1b…c258. Sibling-leakage sensitivity pre-registered. No held-out target opened. |
| Track C4 B2 and family-cluster bootstrap | `9358ac4` | Registered per-mode weighted TSFC regression and paired family resampling implemented. Five tests pass, including the 93-row Trent calibration pool and synthetic eight-family CI reproduction; no cross-family targets opened. |

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

## Registered decisions and AC work (updated 2026-10-03)

The user's 2026-10-02 decisions, repeated on 2026-10-03, supersede the former
"Blocked / needs the user" list. No additional scientific decision is awaited.

- **P8.5-A5:** registration `18df7a4`; implementation `71c6aed`, reviewed
  corrections `d0dc7d4` and `b9d638f`; strict workflow/build evidence
  registered in `7d545ed` and implemented in `a79312d`. No edit to `data/A2NOx.yaml`.
  Independently audit every reaction/element in the uncloned mechanism; use
  exactly 10 times its registered dimensionless bound for closure and a
  balanced control at strict 1e-10. Mixed trace-species rule: relative above
  Y = 1e-8, absolute 1e-12 at or below it. Temperature convergence at
  tau/10tau/100tau must shrink monotonically and end below 0.1 K; a converged
  approach/idle kinetics limit is recorded as physics. Failed integration is
  still numerical failure. Twenty-six pure gate/audit tests passed; the actual
  independent audit and rerun 4 remain pending on AC, due 2026-10-16.
- **P8.3-A2:** registration `2f5bbe5`. Re-cited full-scale TP-2171 and
  modelling evidence; new Cv central 0.985, sensitivity range 0.95–1.00.
  Cd central 0.96, range 0.90–0.99. These are declared engineering envelopes,
  and gross-thrust coefficient is not equated with Cv or Cd. Cv is included
  in the error-budget sensitivity list. The prior changed after A4; A4
  remains scored once at `2dd705f` with its original inputs and result.
- **A4c:** prospective registration `0e32a29`, before the 2026-10-08 deadline
  and before any computation. Five shared parameters with declared cited
  centrals/ranges, the 93-row calibration group only, A1 profile plus the
  exact guard, and one registered fallback. Public OPR/BPR/rated thrust
  remain inputs; no per-engine fitted parameter. Score the Trent held-out
  set exactly once by 2026-10-15, or report in progress. Pre-computation C1
  clarification `9e4db00` registers the opt-in public-input flow bound,
  separate build and complete frozen-evidence checks. Implementation `ca60227`
  passed pure tests and independent review. Numerical work is in progress;
  no fit, profile or held-out target access has occurred.
- **Calibration-only work:** registered OAT Cv, Cd, T4, FPR, six efficiencies
  and cooling error budget, reporting illustrative shares of the known
  +15.4/+10.3/+8.3 % aggregates; input-only Trent 1000-E convergence repair
  and an all-20-family design-convergence assertion. Implementation `ca60227`
  uses public inputs, an opt-in solver domain and the separate validation
  core; old solver defaults remain unchanged. All-20 actual convergence is
  required in the queued full pytest. No error-budget or convergence result exists.
- **G0:** the requested file reads now succeed. Fresh output selection `634ecb1` has
  been implemented by Claude and its path rules statically checked; no G0
  regeneration has run. Use a new write-once directory, preserving old G0.

### Queue deadlock recovery

The old queue's process-name predicate matched the parked A2 chain command,
while A2 waited for that queue to exit. Verified PIDs 48047/48049 were stopped
under the user's authorization. The old queue owners 45935/45938 were then
paused, verified to have no numerical descendants and retired without
resuming their already parsed loop; caffeinate 45937 was also retired.
No benchmark repeat ran during this recovery. Seven existing run-2 result
directories and all old outputs are preserved; no rerun result files exist.

Replacement registration: `0c71b49`, pre-launch review correction `5ab1c30`,
separate-build prerequisite `7a6f9cb`,
`docs/phase8_queue_recovery_registration.json`. Completion must use registered
run records and validated completion files, never process-name matches.
A2 starts only after both benchmark completion files and the benchmark-owner
release record validate. Existing invalid-power and overlap flags remain;
terminal completion is not a clean benchmark claim.

**Reviewed replacement:** corrective commit `8635403` replaces the queue
predicate with registered result/completion evidence, atomic PID/birth-time
ownership and AC/source/dependency checks after waits and before execution.
A2 is chained to both benchmark completion files and owner-release evidence.
The user's explicit "Just do it with CODEX" instruction superseded the
executor-role restriction; Codex completed and independently reviewed the
remaining implementation in worktrees.

**Armed and WAITING_FOR_AC:** owner PID `22543`, verified birth time
`Sat Oct 3 22:23:31 2026`; session `20261004T022331Z-22543`, launch source
`8afb0e0`. Durable lease:
`outputs/phase8/operations/20261003_recovery/owner.lease.json`.
Its seven historical spec records are validated/skipped; the next spec is
`run2_ac/arm2a_W1_1w`. There are 49 remaining run-2 specs, then exactly two
registered rerun jobs. The lease records no child or pending child; only the
owner and its caffeinate helper are live. No benchmark repeat, scientific
run or score occurred during repair. AC is polled every 300 seconds and
rechecked immediately before execution. The old owners remain absent.

Registered AC order:

1. Benchmark run 2 (skip validated existing records; finish missing specs).
2. The two registered rerun jobs: arm1a_W2_11w and arm1a_W4_11w.
3. A2 calibration from the benchmark completion evidence; no held-out score.
4. Separate validation build in `cpp/build_next`, preserving `cpp/build`.
5. Track 4 focused pytest, then its single registered diagnostic attempt.
6. Full pytest with the validation core preloaded and its path asserted, then
   phase-7/phase-8 protected hashes.

Independent numerical tracks wait until the main sequence releases ownership
and the registered records validate. Parallel planning/implementation uses
worktrees (`phase8-p85-a5-20261003`, `phase8-p83-a2-a4c-20261003`,
`phase8-queue-recovery-20261003`). Non-blocking
defects go to `docs/FIXES.md`. Registration before scoring, one-shot held-out
scores, protected files and no push are never deferred. Freeze package:
`outputs/freeze/` on 2026-10-18; local tag `freeze-2026-10-18`, not created yet.

### Scientific commands after the main chain

A4c uses the validated `cpp/build_next` build; it does not rebuild the
benchmark core. Review and commit the exported build-provenance and actual
all-20 convergence records first. Run the calibration-only OAT error budget,
then primary fit and primary A1 profile, committing their complete artifacts
between phases. Only a registered primary A1 failure permits the single
fallback and its required profile. Validate the frozen evidence before the
one-shot Trent score; reserve before target access and preserve any failure.
No held-out score is an automatic action of the AC chain.

A5 separately runs the independent uncloned-mechanism audit, reviews and
commits its passing record, then runs rerun 4 using the validated separate
core. Do not rerun the generic build command or alter the mechanism. Both
tracks require a terminal main chain and released lease before numerical
work. Failed gates stay recorded; non-blocking defects remain in `docs/FIXES.md`.

## Superseded blocked list (2026-10-01 morning)

1. **AC power.** Benchmark run 2 (56 runs) is armed and waits for mains power
   (registered protocol); ladder A2 calibration also needs > 1 h. The Mac has
   been on battery since the session started.
2. **C3 family split** — only 15 eligible direct-drive families vs the
   required ≥ 8 held-out + ≥ 8 calibration (16). Options: admit geared
   turbofans (solver support = a gearbox speed ratio on the LP shaft, P8.4b),
   or relax the in-production rule. Scientific scope → user.
3. **Ladder A3 definition** — fixed-area nozzles with imposed v6 flows are
   not a closed cycle; either define A3 as "real-gas choked nozzle on the
   imposed flow with floating area" (no fixed area until A4) or merge A3
   into A4 (P8.4b matching). Scientific scope → user.
4. **P8.5 G1 FAIL** — closure (~4.6e-7), strict σ=0 spread and the
   long-residence limit at APPROACH/IDLE failed. The A2NOx HyChem fuel
   reactions are element-imbalanced, but the registered diagnostic shows
   that is *not* the main cause; root cause open (next: per-PSR element
   audit, solve_steady polish, extinction at low power). Any change to the
   closure rule or mechanism is a scientific decision → user.
5. **G0 re-run for G1 item 4** — reading `scripts/phase8/g0_parity.py` /
   `simulation/catjet_backend.py` was blocked by the permission classifier
   in this session, so no fresh 180-row g0_parity run is in the P8.2/P8.3
   G1 records (v6 sources byte-identical to the G0 commit instead).

## Track 4 — PINN diagnosis (diagnostic only; updated 2026-10-03 UTC)

Plan: `docs/plan.md` (Track 4 addendum + pre-run review corrections).
Registration: `docs/phase8_track4_registration.json`
(P8-TRACK4-20261002-attempt1, committed before any computation).
Implementation, tests, audit corrections and repair notes: `0ae5588`.
Notes: `docs/phase8_pinn_repair_notes.md`. Not an engine, turbine or PINN
validation; no empirical/held-out/Sajben/WIND data opened; G2/G3 not reopened.

**State: BLOCKED — no diagnostic has run.** The Mac is on battery, and the
registration requires mains power. At `0ae5588` the registered command
refused with exit 3 ("not on mains (AC) power"; no heavy Python job, sources
clean, HEAD readable) and created no attempt directory.

| Item | State |
|---|---|
| Track 4a turbine map (train once, score 65×65 once, gate < 0.001) | **not run (BLOCKED)** |
| MMS Ma Eqs. 22–26, tanh and SiLU | **not run (BLOCKED)** |
| Nozzle ladder rungs 1–4 | **not run (BLOCKED)** |
| Focused pytest `tests/test_phase8_pinn_diagnostics.py` | **pending, not passed** |
| Full pytest `tests/` | **pending, not passed** |
| Static checks at `0ae5588` | done: AST parse and pyflakes clean; `git diff --check` clean; `build_manifest.py --check` stale none; protected hashes 40/134/184, 0 mismatched; models, data, simulation, requirements, registration and plan byte-identical to HEAD; no retired label/sponsor word; envelope sha256 and registered box re-checked with stdlib parsing |

Historical retired turbine PINN (P4.4 attempt 2): 6.390 % held-out max
|Δp5|/p5. That result stays as recorded. The new analytic score will sit beside
it once it exists. Its root cause is not established. It is not the legacy 4.2 MPa scale
(see the notes, source correction 1).

Pending commands run after benchmark run 2, its two rerun jobs, A2 and the
separate validation build. The recovery registration owns the order:
1. `nice -n 15 .venv/bin/python -m pytest tests/test_phase8_pinn_diagnostics.py -v`
   (an implementation bug may be fixed and committed only before step 2).
2. `nice -n 15 .venv/bin/python -m scripts.phase8.pinn_diagnostics.run_diagnostics --registration docs/phase8_track4_registration.json`
   (exactly once; a FAIL ends attempt 1 and any rerun needs a new prospective attempt).
3. Inspect `report.json` aggregate status, `run_log.txt` and `hashes.json` (exit 0 alone is not proof), then commit
   `outputs/phase8/track4/20261002_attempt1/**` and this status.
4. Conditional freeze step: only if diagnostics completed before 2026-10-16
   (America/New_York), create `outputs/freeze/NUMBERS.md` with the completed
   entries marked diagnostic, add its record to `scripts/build_manifest.py`,
   run `.venv/bin/python scripts/build_manifest.py` then `--check`. No local tag
   now. Until then, no freeze file or manifest record exists (none is created
   as a placeholder).
5. The recovery registration’s full-pytest command preloads and verifies
   `cpp/build_next` before collection, then
   `.venv/bin/python scripts/validation/verify_protected_hashes.py --phase7 --phase8`.

Blockers and deferred work: Ma Eq. 25 thermal-unit ambiguity (flagged,
not repaired); loss-balancing cross-validation, M1 empirical residuals,
R7-1..R7-3 and the post-freeze Sajben low-label study remain deferred. The parked A2 chain was stopped
and legacy queue retired during the
authorized recovery above; the reviewed replacement owns their registered
sequence. Numerical results are still pending on AC.

## Recovery review checks (2026-10-03)

- Integrated fixtures: 121 tests and four subtests passed (57 queue/G0,
  26 A5, 38 A4c/input-convergence). The actual all-20 solve assertion was
  deselected on battery and remains required in full pytest.
- G0 output-path checks passed for unchanged defaults, fresh relative output
  selection and path-traversal rejection; no regeneration ran.
- Registration JSON parsing, Python AST, pyflakes, shell syntax, diff and
  G0 CLI help checks passed; build manifest reports no stale entry.
- Protected manifests 40/134/184 each have zero mismatches; an independent
  310-file retained/protected/historical snapshot also has zero mismatches.
- The required case-insensitive naming scan returned no matches.
- No Git process or index lock was present before integration/commits.
- Full pytest, Track 4, all-family convergence, error budget, A4c fitting/
  profiling/scoring, A5 audit/rerun 4 and G0 regeneration remain pending.

After the reviewed main AC sequence finishes, a fresh G0 command is:

```sh
cd /Users/arnavpatil/Documents/JetEngineSimulation && caffeinate -i nice -n 15 .venv/bin/python scripts/phase8/g0_parity.py --workers 6 --out-dir outputs/phase8/g0_rerun_20261003
```

This preserves historical G0 files and rules. The read-permission problem
did not recur: both requested source files were read successfully.
