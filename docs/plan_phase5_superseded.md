# Phase 5 — Close Phase 4, repair the calibration, make the repo reproducible — September 26, 2026

Supersedes the unfinished part of Phase 4. The Phase 4 plan is archived at
`docs/plan_phase4_superseded.md`; P4.0–P4.2, P4.4 and P4.5 are complete (commits
`592cc20` … `b3bfa43` on branch `phase4`). What carries over is folded in below.

**Scope: fixes to the repository only.** Out of scope for this plan: preprint text,
choice of preprint server or venue, the project website, and any outreach. Those
consume this phase's outputs; they are not part of it.

---

## Where Phase 4 left things

| Item | State |
|---|---|
| P4.0 commit inherited state, resolve LE-PINN stream | done |
| P4.1 Sajben data gate | done — GO on the archive's WIND RANS solution |
| P4.2 fine-tune regression | done — collapsed init diagnosed, guards added |
| P4.3 nozzle LE-PINN | **incomplete** — attempt 3 (terminal) registered; only its data-only half ran |
| P4.4 turbine surrogate | done — retired (held-out p5 6.39 % vs 1 % gate) |
| P4.5 heat loss | done — ξ = 0 by structural argument; identifiability finding recorded |
| P4.6 re-run on v5 | not started — now blocked on F2 below |
| P4.7 reproducibility refactor | not started |
| P4.8 preprint + site | out of scope here |

---

## Findings driving this phase

**F1 — P4.3 has no applied outcome.** Attempt 1 fails (0.258). Attempt 2's 0.245 sits
within 0.01 of the band edge, so under the reporting rule committed in `b3bfa43` it gets
no band call. Attempt 3's physics-on runs never produced output: `outputs/logs/
train_sajben_v5_a3_s{42,43,44}.log` are 0 bytes, all stamped 2026-09-19 19:42:33, while
the matched data-only runs completed 19:52–20:10. A 2-epoch smoke test on 2026-09-25 ran
the physics-on path cleanly (split guard 0 eval rows in train; physics loss live), so the
failure was a launch/environment event on the Mac, not a code defect. The smoke
checkpoint was moved to `outputs/_smoketest/`.

**F2 — The LTO calibration identifies one parameter from data (critical; supersedes the
"four identified" statement in `heat_loss_provenance.md` §5).**

§5 established that η_b, pressure_loss and k_pi are inert in the fuel-flow objective.
The remaining four are not independently identified either. Fuel flow per mode is
φ·f_st·β·ṁ_rated·x^k_mdot, so take-off (x = 1) pins φ_to, but idle and approach give two
equations in three unknowns (φ_idle, φ_app, k_mdot). Measured on 2026-09-26, moving along
that ridge with everything else at v4:

| k_mdot | φ_idle | φ_app | fuel-flow MAPE | idle thrust | approach thrust |
|---|---|---|---|---|---|
| 0.5800 | 0.2643 | 0.2903 | **1.619 %** | 28.56 kN | 84.44 kN |
| 0.6000 | 0.2787 | 0.2974 | **1.619 %** | 27.58 kN | 83.30 kN |
| 0.6218 (v4) | 0.2953 | 0.3053 | **1.619 %** | 26.53 kN | 82.04 kN |

Identical objective, 8 % spread in idle thrust. The only thing selecting v4's point is the
φ search box (φ_app = 0.2903 at k = 0.58 is just outside its [0.30, 0.40] range). So of
seven sampled parameters, the data determine **φ_to alone**; the other six are set by
priors and box edges. Every manuscript-bound T4, thrust and TSFC depends on them — k_pi
alone swings approach thrust 57 → 84 kN across its range.

**F3 — The data needed to fix F2 are already in the repo.** ICAO LTO modes are defined as
fractions of rated thrust, and `data/icao_engine_data.csv` carries both `Power (%)` and
`Rated Thrust (kN)`. The AE3 thrust targets are therefore 7 / 30 / 100 % of 310.9 kN.
The model currently gives 26.5 / 82.0 / 241.6 kN — the take-off figure is 22 % low, and
that gap is today handled in prose. Adding thrust to the objective both breaks the ridge
and turns the 22 % gap from a sentence into a measured calibration residual.

**F4 — Records that now say false things.** `outputs/parameter_provenance.md` still
reports k_mdot and the φ as calibrated and flags only pressure_loss as inert (its v4 row
calls η_b "converged"). `heat_loss_provenance.md` §5 says the objective identifies four
parameters. `le_pinn.py` lines 1910 and 1961 hard-code "alive ReLU units", so a tanh run
logs itself as ReLU.

**F5 — The repo is not reproducible from a clean clone.** Requirements are floors only;
there is no `REPRODUCE.md`; root-level working files (`AGENT_CONTEXT.md`,
`EXECUTION_PACKAGE.md`, `MANUSCRIPT_REPAIR_PLAN.md`, `presentation_package.md`,
`SAF_Optimization_Poster.pptx`, `Technical_Summary.docx`) sit untracked beside the code;
a vendored third-party SU2 GUI repo lives under `data/raw/`; two tests return `bool`
instead of asserting; `simulation/turbine/turbine.py` prints a banner at import.

---

## Objective

1. Give P4.3 its terminal outcome (F1).
2. Replace the fuel-flow-only calibration with one whose fitted parameters are identified
   by data, prove it with a test, and produce calibration v5 (F2, F3).
3. Re-run every production study on v5 and regenerate the number trail.
4. Correct every record that F2 and F4 falsify.
5. Make every manifest row reproducible from a clean clone (F5).

---

## Constraints

- Pre-register before running: the P5.2 objective, its weights, which parameters are fitted
  vs fixed, and the acceptance thresholds are committed before the first calibration run.
- No sourced value, no fixed value. Any parameter moved from "fitted" to "fixed" needs a
  page-level citation in `outputs/parameter_provenance.md` — the same rule P4.5 applied to ξ.
  If none is found, stop and escalate; do not pick a number.
- Never overwrite `models/*.pt` or any v1–v4 calibration / holdout artifact. v5 is added
  alongside.
- Seeds and seed-passing preserved; every new stochastic artifact records its seed.
- `python -m pytest tests/ -v` after every phase; `scripts/test_emissions.py` after any
  cycle change. Record the pass/skip count at P5.0 as the floor.
- Nothing is deleted in this phase — archive with `mv` into `outputs/archive/` or
  `archive/`, so any removal is reversible.
- All work on `phase4`; `main` stays at the reproducible v4 state until P5.5.

---

## Relevant Files

**Modify:** `scripts/optimization/calibrate_lto.py`, `scripts/validation/holdout_icao_validation.py`,
`scripts/validation/sajben_report_p43.py`, `simulation/nozzle/le_pinn.py` (log label only),
`outputs/parameter_provenance.md`, `scripts/validation/heat_loss_provenance.md` (§5),
`scripts/parse_sajben_cfd.py`, `requirements.txt`, `.gitignore`,
`simulation/turbine/turbine.py` (import-time print), `tests/test_nozzle_pinn_fix.py`.

**Create:** `tests/test_calibration_identifiability.py`, `scripts/validation/identifiability_profile.py`,
`scripts/validation/takeoff_thrust_gap.py`, `docs/number_crosswalk_v5.md`, `REPRODUCE.md`,
`tests/test_manifest_integrity.py`, `tests/test_checkpoint_provenance.py`.

**Regenerate:** `outputs/ARTIFACT_MANIFEST.md` (from scratch, not edited).

**Read only:** `models/*.pt`, `outputs/calibration_trent1000_ae3_v{1..4}.json`, all
`holdout_icao_validation*_v{1..4}` artifacts, `outputs/archive/**`.

---

## Repo Context

Execution uses the planner/reviewer and Claude Code dispatcher workflow in `AGENTS.md`.
The affected subsystems are the Sajben nozzle LE-PINN training/reporting pipeline,
LTO cycle calibration and ICAO validation, production study provenance, and clean-clone
reproduction. Chemical mechanisms remain read-only unless a later approved plan targets them.

## File-Level Edits

The per-file edits are specified in P5.0–P5.4 below. Additional explicitly targeted
paths are `scripts/validation/train_sajben.py` (launch/completion guard),
`outputs/sajben_retrain_v5.{md,csv}` (terminal report), `outputs/logs/` (launch evidence),
`docs/working/` and archive destinations (reversible housekeeping). Do not bundle
unrelated existing work into implementation commits.

### Execution ordering and evidence

- The user has directed execution and specifically requested that P5.1 start first.
  Do only the prerequisites needed to preserve empty launch logs and implement/check
  the launch guard, then launch the registered physics-on runs sleep-proof before
  continuing P5.0 and P5.2. Record the launcher PID and log locations.
- Preserve existing checkpoints byte-for-byte. Legacy data-only runs predate `.done`
  markers: do not manufacture a historical zero-exit marker. Inspect their completion
  evidence and distinguish it from observed process status. If the strict six-run guard
  cannot be satisfied from retained evidence, report that issue before publishing an
  outcome; do not silently weaken the registered gate or overwrite existing checkpoints.
- Stop dependent calibration work at any P5.2 escalation gate; independent P5.1 runs
  may continue. Never proceed to merge/tag with an unresolved gate or incomplete run.
- Complexity is high (scientific identifiability, multiple subsystems, long training,
  and reproducibility). Recommended executor model: Claude Opus.

---

## Implementation Phases

### P5.0 — Housekeeping and record corrections (no model changes)

1. Commit the attempt-3 data-only checkpoints and their three logs. Move the three empty
   physics-on logs to `outputs/logs/failed_launch_2026-09-19/` and commit them there —
   the empty files are evidence of what happened and belong in the record.
2. `le_pinn.py` lines 1910, 1961: label from the model's recorded activation, not the
   literal "ReLU".
3. `outputs/parameter_provenance.md`: mark η_b and k_pi **unidentified by the v4
   objective**, and k_mdot / φ_idle / φ_app **jointly unidentified (ridge)**, citing the F2
   table. Remove "converged" wherever it describes an unidentified parameter.
4. `heat_loss_provenance.md` §5: correct "identifies four" to "identifies φ_to", with a
   pointer to F2.
5. Decide each untracked root-level working file: commit under `docs/working/`, or add to
   `.gitignore`. Nothing stays in the ambiguous untracked state.
6. Run pytest; record the count as the Phase 5 floor.

**Acceptance:** `git status` clean except `.DS_Store`; no record in the repo states that
an unidentified parameter was calibrated.

---

### P5.1 — Finish P4.3 attempt 3 (F1)

The registration in `adb0184` stands unchanged; this phase only executes it.

1. Add a launch guard to `train_sajben.py`: write the log through `tee`, write a
   `.done` marker with the exit code on completion, and have `sajben_report_p43.py`
   refuse to report attempt 3 unless all six checkpoints (3 physics-on, 3 data-only)
   exist with exit code 0. A half-run terminal attempt must be impossible to report.
2. Relaunch the three physics-on seeds (42/43/44) exactly as registered. On macOS run
   under `caffeinate -i` so sleep cannot kill them; use `--device cpu` if MPS memory is
   suspected in the 2026-09-19 failure.
3. Score all six with `sajben_report_p43.py`. Apply the rule registered in `b3bfa43`:
   band on the seed mean, claimed only if all three seeds agree; physics-on vs matched
   data-only classified by the three pre-registered readings; ceiling-relative column
   supplementary.
4. Write the applied outcome into `outputs/sajben_retrain_v5.md`. That closes P4.3. No
   attempt 4 — the registration says attempt 3 is terminal.

**Acceptance:** all six runs complete; one applied P4.3 outcome recorded with every
attempt listed.

---

### P5.2 — Repair the calibration (F2, F3) — the core of this phase

**Step 1 — Build the test before the fix.**
`scripts/validation/identifiability_profile.py`: for each sampled parameter, profile the
objective — fix it at points across its range, re-optimise the others, record the best
objective at each point. A parameter is *identified* if its profile has an interior
minimum that rises by a pre-set margin toward both box edges. Also run the explicit
ridge check from F2. Wrap the verdict in `tests/test_calibration_identifiability.py`.
Run it on v4 first: it must report φ_to identified and the other six not. If it doesn't
reproduce F2, the test is wrong — fix the test before touching the calibration.

**Step 2 — Diagnose the take-off thrust gap before fitting to it.**
`scripts/validation/takeoff_thrust_gap.py`: at the take-off point, decompose the
241.6 vs 310.9 kN shortfall — core vs bypass contribution, sensitivity to total airflow,
BPR, FPR and nozzle treatment. Fitting thrust while a structural defect sits in the cycle
would just hide the defect in a parameter; this step decides whether the 22 % is a wrong
fixed input or a missing model term.
**Gate:** if the gap is structural (a missing term, not a mis-set input), stop and
escalate with the decomposition — calibrating over it would repeat F2 in a new place.

**Step 3 — Register v5's objective (commit before any run).** Record in
`calibrate_lto.py` and the commit message:
- Targets: fuel flow and thrust at idle / approach / take-off for AE3 (thrust =
  `Power (%)` × `Rated Thrust (kN)` from the CSV).
- Weighting between the two target sets, fixed in advance.
- Fitted parameters: only those the Step-1 profile, run on the new objective with a
  cheap pilot, shows as identified.
- Fixed parameters: η_b and pressure_loss at page-cited values (combustion and
  pressure-loss chapters of the gas-turbine texts already named in
  `heat_loss_provenance.md` §4 are the place to look). No citation → escalate.
- Acceptance thresholds for in-sample and held-out fit, stated as numbers.

**Step 4 — Calibrate and validate.**
`calibrate_lto.py --tag v5`, seed 42. Run the identifiability test on v5 — every fitted
parameter must pass. Extend `holdout_icao_validation.py` to report held-out **thrust**
MAPE alongside fuel flow across the 57 non-AE3 records.

**Report whatever comes out.** Fitting thrust will likely worsen fuel-flow MAPE from
v4's 2.50 %. That is expected: v4's figure was obtained with six free directions and no
competing target. A worse number from an identified model is the more honest number, and
the v4 → v5 table says so.

**Acceptance:** identifiability test passes on v5; v5 JSON lists fitted vs fixed with
citations for every fixed value; held-out fuel-flow and thrust MAPE both reported.

---

### P5.3 — Re-run production studies on v5 and regenerate the number trail

Every number moves, because the calibration changes — this is not the verification-only
pass Phase 4 anticipated.

1. Re-run at v5 with the adjudicated configuration (turbine analytic, nozzle analytic,
   ξ = 0), same seeds: `design_point_summary.py`, `optimize_blend.py` (free-φ and
   `--freeze-phi`, 1000 trials), `variance_decomposition.py`, `nox_holdout_validation.py`,
   `nox_dual_path.py`, `heat_loss_sensitivity.py`, `ablate_pinn_components.py`.
2. Regenerate `outputs/ARTIFACT_MANIFEST.md` from scratch. Add rows for the Phase 4
   negative results: Sajben attempts 1–3 with the data-only ablations, turbine surrogate
   attempts 1–2, the physics-residual defect, the identifiability profile.
3. `docs/number_crosswalk_v5.md`: v4 → v5 for every manuscript-bound number, same shape
   as `docs/number_crosswalk.md`.
4. Move v4 production outputs to `outputs/archive/pre_phase5/`.
5. Orphan sweep: every file under `outputs/` outside `archive/` is referenced by the
   manifest, or it moves to archive.

**Acceptance:** manifest regenerated; crosswalk complete; zero orphans.

---

### P5.4 — Reproducibility from a clean clone (F5)

1. Pin `requirements.txt` to the versions that produced v5 (`pip freeze`, trimmed to
   direct dependencies). Note the torch build constraint: CUDA-linked aarch64 wheels fail
   on CPU-only Linux; document the working install.
2. `REPRODUCE.md`: clone → environment → one command per manifest row → expected output
   file and tolerance. Verify on a fresh clone in a fresh virtualenv, not on the
   development tree.
3. Archive, don't delete, code that is not in the manifest pipeline: `simulation/emissions.py`,
   `scripts/visualization/pareto_visual.py`, `scripts/visualization/visualize_results.py`,
   `dashboard.py`, `fetch_and_build_cfd_data.py` → `archive/`. Keep an import-compat shim
   only where a live module still imports one.
4. `data/raw/cfd_datasets/github/nozzle_flow_cfd-main/` — not used by P4.1 route (a);
   move out of `data/` to `archive/third_party/` with its licence file.
5. `parse_sajben_cfd.py`: emit zeros, not NaN, in target columns 5–7, and fix the
   "zero-padded" docstring in `finetune_on_cfd_data`.
6. Remove the import-time print from `simulation/turbine/turbine.py`; convert the two
   `return bool` tests in `tests/test_nozzle_pinn_fix.py` to assertions.
7. New tests: `test_manifest_integrity.py` (every manifest path exists; no unreferenced
   output outside archive) and `test_checkpoint_provenance.py` (every `models/*_v5*.pt`
   records seed, device, dataset hash, activation, git SHA).

**Acceptance:** a fresh clone reproduces every manifest row within tolerance; pytest
green at or above the P5.0 floor.

---

### P5.5 — Freeze

1. Final pytest and `scripts/test_emissions.py`.
2. Merge `phase4` → `main`, tag `v5.0`.
3. The manuscript/preprint is written from `outputs/ARTIFACT_MANIFEST.md` and
   `docs/number_crosswalk_v5.md` only.

---

## Commands to Run

```bash
# P5.1 — on the Mac, one seed per line, sleep-proof
caffeinate -i python scripts/validation/train_sajben.py --attempt 3 --seed 42 --device cpu
caffeinate -i python scripts/validation/train_sajben.py --attempt 3 --seed 43 --device cpu
caffeinate -i python scripts/validation/train_sajben.py --attempt 3 --seed 44 --device cpu
python scripts/validation/sajben_report_p43.py

# P5.2
python scripts/validation/identifiability_profile.py --calibration outputs/calibration_trent1000_ae3_v4.json
python scripts/validation/takeoff_thrust_gap.py
python scripts/optimization/calibrate_lto.py --tag v5 --n-trials 100 --seed 42
python scripts/validation/identifiability_profile.py --calibration outputs/calibration_trent1000_ae3_v5.json
python scripts/validation/holdout_icao_validation.py \
    --calibration outputs/calibration_trent1000_ae3_v5.json --tag _v5

# P5.3
python scripts/validation/design_point_summary.py
python scripts/optimization/optimize_blend.py --n-trials 1000 --seed 42 \
    --calibration outputs/calibration_trent1000_ae3_v5.json
python scripts/optimization/optimize_blend.py --n-trials 1000 --seed 42 \
    --calibration outputs/calibration_trent1000_ae3_v5.json --freeze-phi \
    --output-csv outputs/results/optimization_results_phi_frozen.csv
python scripts/analysis/variance_decomposition.py --seed 42 --n-mc 1000
python scripts/validation/nox_holdout_validation.py --tag _v5

# every phase
python -m pytest tests/ -v
python scripts/test_emissions.py
```

---

## Tests

- Floor: the pass/skip count recorded at P5.0.
- **New** `test_calibration_identifiability.py` — reproduces F2 on v4; passes on v5.
- **New** `test_manifest_integrity.py`, `test_checkpoint_provenance.py` (P5.4).
- **New** report guard — `sajben_report_p43.py` refuses a terminal attempt with missing
  or non-zero-exit runs (P5.1).

---

## Acceptance Criteria

1. Repo records no longer describe any unidentified parameter as calibrated or converged.
2. P4.3 closed with one applied outcome; every attempt reported.
3. Identifiability test reproduces F2 on v4 and passes on v5.
4. Take-off thrust gap decomposed before any fit to thrust.
5. v5 objective, weights, fitted/fixed split and thresholds committed before the run;
   every fixed value cited.
6. Held-out fuel-flow and thrust MAPE reported for v5, alongside v4, whichever way they move.
7. All production studies on v5; manifest regenerated; crosswalk complete; zero orphans.
8. Fresh-clone reproduction verified; requirements pinned; nothing deleted, all removals
   archived.

---

## Rollback Notes

- `main` holds v4 until P5.5. `phase4` can be reset to `b3bfa43` to discard all of Phase 5.
- v1–v4 calibration artifacts and all `models/*.pt` untouched; v5 is additive.
- Every removal is an `mv` into `archive/` and reverses with the opposite `mv`.

---

## Escalation Guidance

**Stop and escalate — do not guess:**

- P5.2 Step 2 finds the take-off thrust gap is structural.
- No citable value for η_b or pressure_loss.
- The identifiability test fails to reproduce F2 on v4.
- On the new objective, some parameter the cycle needs is still unidentified — decide
  with the user whether to fix it at a sourced value or add a target (the ICAO NOx
  columns are the only other per-mode data in the CSV, and the NOx correlation was fit on
  the same records, so using them needs care).
- v5 held-out MAPE degrades beyond the pre-registered threshold.

**Order and effort:** P5.0 (hours) → P5.1 (runs overnight; independent, start first) and
P5.2 in parallel → P5.3 → P5.4 → P5.5. P5.2 is the judgement-heavy phase and the one
that changes what the project can claim; everything after it is mechanical.
