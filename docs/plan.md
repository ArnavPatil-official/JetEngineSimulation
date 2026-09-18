# Phase 4 — Earn the PINN, turn on heat loss, rebuild for preprint — September 17, 2026

Phase 3 plan archived at `docs/plan_phase3_completed.md` (commits `de45e44`…`3b94733`;
80 passed / 1 pre-existing skip). Phase 3.5 froze the numbers and produced
`outputs/ARTIFACT_MANIFEST.md` + `docs/number_crosswalk.md`. An interim code-repair
pass on 2026-09-17 (uncommitted) closed the last reviewer items that were still live
in the code — see "Inherited state" below.

**Target has changed.** This is no longer a journal resubmission. The destination is a
preprint plus a project website. That lowers the bar on venue formatting and raises it
on two things a preprint has no reviewer to catch: every claim must be checkable from
the repo, and every artifact on the site must trace to a manifest row.

**Two user decisions set this phase's scope:**

1. **Retrain the PINNs until they earn the title** — do not rescope the paper around the
   analytic model. Phase 3 demoted both surrogates; Phase 4 either fixes them on evidence
   or triggers the pre-registered fallback in P4.3/P4.4.
2. **Turn on combustor heat loss and recalibrate to v5** — answer Reviewer 1's combustor
   point in the model, not in a limitations paragraph. Every manuscript-bound number moves.

---

## Inherited state (2026-09-17 code-repair pass, uncommitted on `main`)

Closed, verified, tests green: the in-sample NOx fit now carries an explicit split
(`EmissionsEstimator(nox_fit_exclude_models=...)`) and its startup R² is labelled a
training diagnostic; `scripts/validation/nox_holdout_validation.py` added
(leave-one-engine-out MAPE **4.17%**, manifest row M7); the "PHYSICS VALIDATION
CHECKLIST" print relabelled as internal consistency by construction; the retired point
LCA factors removed from `scripts/test_emissions.py`; `PyYAML` added to
`requirements.txt`; `simulation/emissions.py` marked non-production;
`scripts/visualization/pareto_visual.py` marked superseded; compressor and turbine loss
taxonomy corrected in code.

**These changes are uncommitted.** P4.0 commits them before anything else.

---

## Audit findings driving this phase

**F1 — The Sajben "CFD" training set contains none of the physics the Sajben validation
tests (critical; gates the whole nozzle retrain).**

`scripts/parse_sajben_cfd.py` says so in its own docstring: the binary NASA flow
solutions (`sajben.cfl`, `sajben.cgd`) "require pyCGNS (not in requirements.txt) and are
**skipped**. Instead, a quasi-1D normal-shock solver is run on the real Sajben geometry."
Only the 81×51 grid and the inlet BCs are NASA's.

Inspection of `data/processed/master_shock_dataset.pt` (128,061 rows, 6 inputs / 9 targets)
confirms what that implies:

- the transverse velocity `v` is non-zero on 86% of rows, but it is produced by
  `_estimate_transverse_velocity` from the local geometry slope — inviscid streamline
  deflection, not a solved momentum field;
- `mu_eff` spans 1.17e-5 – 1.81e-5 Pa·s, which is Sutherland *molecular* viscosity. There is
  no eddy viscosity anywhere in the set, so there is no turbulence model and therefore no
  boundary layer;
- target columns 5–7 are **NaN on all 128,061 rows**. They are correctly masked out —
  `finetune_on_cfd_data` uses `targets[:, :5]` behind a finiteness filter — so this is not a
  training bug, but the function's docstring describes those columns as "zero-padded" when
  they are NaN, and `parse_sajben_cfd.py` should emit zeros rather than NaN. Fix in P4.7.

The validation target is the opposite: Hseih, Wardlaw, Collins & Coakley (1987),
`data/raw/data.Mach46.txt` — a viscous transonic diffuser with shock/boundary-layer
interaction, which is the entire reason that case is a benchmark.

Consequence: the current 0.71–0.72 wall-Cp shape-L2 is **not** an under-training result,
and more epochs on this dataset cannot reach the <0.10 gate. The training signal does not
contain the phenomenon. Any retraining plan that skips this is a plan to waste a week.

**F2 — Fine-tuning makes the model strictly worse, which is a defect independent of F1.**

`outputs/sajben_validation_errors.csv`:

| model | L2_Cp_upper | L2_Cp_lower |
|---|---|---|
| `le_pinn_sajben.pt` | 0.714 | 0.723 |
| `le_pinn_sajben_finetuned.pt` | 0.805 | **1.091** |

A lower-wall shape-L2 above 1.0 means the prediction is further from the experiment than
a flat line at the experimental mean. Fine-tuning from a checkpoint that already fits the
same dataset should not be able to do that. Something in `finetune_sajben.py` /
`finetune_on_cfd_data` — normalizer refit, `physics_loss_weight=0.05` dominating at
`lr=1e-5`, geometry flag, or an LR schedule — is broken. Diagnose before retraining, or
the retrain inherits it.

**F3 — The turbine PINN has no external target, and the only credible one makes it a
surrogate of the analytic model.**

`outputs/turbine_p5_adjudication.csv`: PINN p5 = 2.765 bar vs work-consistent analytic
4.728 bar (−41.5%). `compute_loss_components` in `simulation/turbine/turbine.py` penalises
an EOS residual, a monotonic-temperature term and a work-matching term — nothing pins the
pressure path, so p5 is free to drift. There is no turbine experimental or CFD dataset in
the repo.

Retraining it against the work-consistent analytic solution would make it a neural
surrogate **of an analytic model Phase 3 already adopted as production** — legitimate as a
speed/differentiability claim, dishonest as an accuracy claim. Whichever is chosen must be
stated in those words. See P4.4 for the pre-registered decision rule.

**F4 — Heat loss is a first-order term pinned at zero.**

`outputs/heat_loss_sensitivity.csv`: ξ = 0 → 4% moves T4 by **45 K** (2030.8 → 1985.6 K for
Jet-A1). Across all four fuels at fixed ξ, T4 spans **3.6 K**. The term being held at zero
is an order of magnitude larger than the signal the optimizer is being asked to resolve.

**F5 — v5 will invalidate every frozen number.**

`outputs/ARTIFACT_MANIFEST.md` rows M1–M7 and S1–S7, `docs/number_crosswalk.md`, and every
figure derive from calibration v4. Turning ξ on changes T4, hence turbine inlet state,
hence thrust, TSFC, and the NOx correlation's fuel-flow input. The crosswalk must be
**regenerated old→new**, not patched, or the project reacquires exactly the
"two sets of numbers" problem Reviewer 2 opened with.

---

## Objective

1. Establish whether the nozzle LE-PINN can be trained against data that contains the
   physics the Sajben benchmark measures; if it can, train it and clear the gate (F1, F2).
2. Give the turbine PINN an honest target, retrain, and state precisely what its agreement
   does and does not demonstrate (F3).
3. Source, wire and calibrate combustor liner heat loss; produce calibration v5 (F4).
4. Re-run every production study on v5 and regenerate the manifest and crosswalk (F5).
5. Refactor the repo so a stranger can reproduce every number from a clean clone.
6. Ship a preprint and a project website, both traceable to the manifest.

---

## Constraints

- **Pre-register every gate before running the study it judges.** P4.3 and P4.4 fix their
  pass/fail thresholds in this document; the outcome is not chosen after seeing the result.
  This is the single practice the reviewers' integrity findings demand.
- Never overwrite `models/*.pt` in place. New checkpoints take new names
  (`le_pinn_sajben_v5.pt`, `turbine_pinn_v5.pt`); existing files are read-only for this phase.
- Preserve seeds and seed-passing. Every checkpoint records `{seed, device, dataset hash,
  epochs, lr, loss weights, git SHA}` in its payload — no exceptions.
- `data/*.yaml` mechanism files stay read-only.
- v1–v4 calibration and holdout artifacts are preserved; v5 is added alongside.
- `python -m pytest tests/ -v` after every phase; `scripts/test_emissions.py` after any
  combustor change. No silent failures, no skipped reporting.
- Training scripts must run on CPU. Several default to `--device mps`; that is a Mac-only
  default and it blocks reproduction by anyone else.
- Do not edit `scripts/visualization/visualize_results.py` or `compare_pinn_le_pinn.py` —
  they carry the user's uncommitted LE-PINN work. P4.0 resolves them first.
- The manuscript/preprint text is written from `outputs/ARTIFACT_MANIFEST.md` only.

---

## Repo Context

Subsystems in scope: nozzle LE-PINN (`simulation/nozzle/`), turbine PINN
(`simulation/turbine/`), combustor (`simulation/combustor/`), cycle integration
(`integrated_engine.py`), LTO calibration and blend optimisation (`scripts/optimization/`),
validation suite (`scripts/validation/`), dataset generation (`scripts/parse_sajben_cfd.py`).

Out of scope: chemistry EI-NOx held-out validation (no data); cruise-point extension;
the dashboard (`dashboard.py`).

---

## Relevant Files

**Read first:** `scripts/parse_sajben_cfd.py` (docstring, lines 1–30), `data/raw/data.Mach46.txt`,
`outputs/ARTIFACT_MANIFEST.md`, `docs/number_crosswalk.md`, `outputs/parameter_provenance.md`.

**Modify:** `simulation/nozzle/le_pinn.py` (`finetune_on_cfd_data`), `simulation/turbine/turbine.py`
(loss components, training entry), `simulation/combustor/combustor.py` (ξ default path),
`integrated_engine.py` (`combustor_heat_loss_fraction` in `design_point`),
`scripts/optimization/calibrate_lto.py` (ξ handling, v5 tag),
`scripts/validation/train_sajben.py` + `finetune_sajben.py` (device, seeds, provenance),
`scripts/validation/sajben_validation.py` (split-aware reporting), `requirements.txt`.

**Create:** `scripts/validation/sajben_data_audit.py`, `scripts/validation/sajben_split.py`,
`scripts/validation/train_turbine_surrogate.py`, `scripts/validation/heat_loss_provenance.md`,
`docs/number_crosswalk_v5.md`, `REPRODUCE.md`, `site/` (static project site).

**Read only:** all `models/*.pt`, all v1–v4 calibration/holdout artifacts,
`outputs/archive/**`.

---

## Implementation Phases

### P4.0 — Commit the inherited state, resolve the LE-PINN stream, branch

1. Review and commit the 2026-09-17 code-repair changes (listed under "Inherited state").
   One commit, message referencing the reviewer items closed.
2. **User decision required, blocking:** `simulation/nozzle/le_pinn.py`,
   `scripts/validation/{train_sajben,finetune_sajben,sajben_validation,compare_pinn_le_pinn}.py`,
   `tests/test_le_pinn.py`, `tests/test_sajben_training_path.py` and
   `scripts/visualization/visualize_results.py` carry uncommitted user modifications
   (611 insertions / 438 deletions). Commit or discard before P4.2 touches `le_pinn.py` —
   do not merge blind.
3. Branch `phase4` from the resulting commit. All Phase 4 work lands there; `main` keeps a
   reproducible v4 state until P4.6 passes.

**Validation:** `git status` clean; `pytest tests/ -v` → 80 passed / 1 skipped.

---

### P4.1 — Sajben data-reality gate (F1) — GO/NO-GO, nothing downstream starts without it

`scripts/validation/sajben_data_audit.py`, a written finding, not a training run:

1. Quantify what `master_shock_dataset.pt` contains: fraction of rows with `v ≠ 0`; span of
   wall-normal gradients; whether any row encodes a boundary layer. Expected answer, to be
   confirmed or refuted: none of it.
2. Extract from `data/raw/data.Mach46.txt` the experimental quantity the gate is scored on
   (upper/lower wall Cp, four velocity stations) and state what a quasi-1D inviscid solution
   can reproduce of it, in principle, at best. Compute that ceiling: run the quasi-1D solver
   at the experimental conditions and score it with the same `_l2_relative` + `_normalise_01`
   used by `sajben_validation.py`. **If the analytic ceiling is itself above 0.10, no model
   trained on that data can pass, and this is proven rather than argued.**
3. Enumerate the three routes to training data that does contain the physics:
   - **(a) Real NASA flow solutions** — `sajben.cfl` / `sajben.cgd` via pyCGNS. Cheapest if
     the files hold converged RANS solutions. Verify contents before adopting.
   - **(b) Run 2D RANS** — an SU2 case for the Sajben geometry; note that
     `data/raw/cfd_datasets/github/nozzle_flow_cfd-main/` already vendors an SU2 harness.
     Most expensive, most defensible, and produces a reusable dataset.
   - **(c) Train on the experiment with a declared split** — e.g. fit on lower-wall Cp +
     two velocity stations, hold out upper-wall Cp + the other two. Cheapest, but the
     held-out set is then small and correlated; it must be described as such, and it cannot
     be called external validation.
4. **Gate:** pick one route, record the choice and its justification in the audit output.
   Routes (a) and (b) keep "external validation" language; route (c) does not, and the
   preprint wording changes accordingly.

**Acceptance:** `outputs/sajben_data_audit.md` states the analytic ceiling number, the route
chosen, and — if the ceiling exceeds 0.10 — says plainly that the existing 0.71 result was a
domain-mismatch artifact, not a training failure.

**Escalation:** if all three routes are unaffordable, stop and re-open the P4 scope decision
with the user. Do not proceed to P4.3 on data that cannot pass.

---

### P4.2 — Diagnose the fine-tune regression (F2)

1. Reproduce: score `le_pinn_sajben.pt` and `le_pinn_sajben_finetuned.pt` fresh; confirm the
   0.714/0.723 → 0.805/1.091 regression.
2. Bisect the fine-tune path. Candidates in order of suspicion: normalizer refit between
   train and eval; `physics_loss_weight=0.05` overwhelming a `lr=1e-5` data term;
   `geometry` flag not propagating to the physics residual; no LR schedule or early stop;
   `_safe_torch_load` silently dropping optimizer/normalizer state.
3. Fix the cause. Add a regression test that fails on any checkpoint scoring worse than its
   own initialisation on the held-out metric.

**Acceptance:** fine-tuning from a checkpoint never scores worse than that checkpoint;
new test in `tests/` covers it; cause documented in the commit message.

---

### P4.3 — Retrain the nozzle LE-PINN, with the gate fixed in advance (F1)

Runs only after P4.1 selects a route and P4.2 closes.

1. Build the training set for the chosen route. Record a content hash in the checkpoint.
2. Declare the train/test split **in the script, before training**, and make
   `sajben_validation.py` refuse to score a checkpoint whose recorded split overlaps the
   evaluation set. Leakage should be impossible by construction, not by discipline.
3. Train. CPU-runnable, seeded, provenance recorded.
4. Score with the existing metrics — no new metric invented after seeing results.

**Pre-registered decision rule (fix now, apply then):**

| Held-out wall-Cp shape-L2 | Outcome |
|---|---|
| **< 0.10** | Gate cleared. LE-PINN becomes the production nozzle; the PINN framing is earned; re-run P4.6 ablations with it. |
| **0.10 – 0.25** | Partial. PINN stays non-production; the paper reports a quantified near-miss against a named benchmark and says what would close it. This is a publishable negative result. |
| **> 0.25** | Failed. The nozzle PINN is retired to an appendix; the title and framing drop the accuracy claim; production stays analytic. |

No re-tuning to move a result across a band after the fact. If hyperparameters change, the
run is a new pre-registered attempt with its own record, and every attempt is reported.

---

### P4.4 — Turbine PINN: honest target, or honest retirement (F3)

1. Decide and **write down first** what the turbine PINN is for:
   - **Surrogate claim** — it approximates the work-consistent analytic expansion fast and
     differentiably. Target = analytic solution. Then the reported metric is surrogate
     fidelity (p5 within 1%, T5 within 1%), and the paper must say it is a surrogate of the
     analytic model and carries no independent accuracy.
   - **Physics claim** — it satisfies the conservation laws from residuals alone. Then it
     needs external data, which the repo does not have, and this route is closed for Phase 4.
2. Under the surrogate claim: `scripts/validation/train_turbine_surrogate.py` — sample the
   calibrated operating envelope, generate analytic work-consistent states, train, and add a
   pressure-path term so p5 is constrained rather than free.
3. Re-run `scripts/validation/adjudicate_turbine_p5.py` and
   `scripts/validation/ablate_pinn_components.py` at the v5 state.

**Gate:** surrogate fidelity < 1% on p5 and T5 across the envelope, and full-cycle thrust
and TSFC within 1% of the analytic path. Otherwise the turbine PINN is retired to an
appendix and production stays analytic — same rule as P4.3, no negotiation after the fact.

---

### P4.5 — Combustor liner heat loss and calibration v5 (F4)

1. **Source ξ before coding it.** Read a combustion text on liner wall heat transfer —
   Lefebvre & Ballal, *Gas Turbine Combustion*, is the standard starting point — and record
   in `scripts/validation/heat_loss_provenance.md` the value or range adopted, the page or
   table it comes from, and the engine class it applies to. **Do not adopt a number this
   plan invents; this plan deliberately states none.** If no defensible value is found,
   stop and escalate rather than picking one.
2. Wire the sourced ξ into `design_point['combustor_heat_loss_fraction']` as the production
   default. The hook in `combustor.py` already exists.
3. Recalibrate: `calibrate_lto.py --tag v5`. ξ is **fixed at the sourced value, not fitted** —
   fitting it reopens Reviewer 2's point 4(c) (a learnable factor absorbing model error).
   Record the sourced value under `fixed_parameters`, beside `eta_compressor`.
4. Held-out validation on v5. Compare against v4's 2.50% and report both. **A worse held-out
   MAPE on v5 is a reportable result, not a reason to revert** — it would mean the previous
   agreement depended on a compensating error, which is worth knowing and worth saying.
5. Re-run `heat_loss_sensitivity.py` around the new operating point.

**Acceptance:** `outputs/calibration_trent1000_ae3_v5.json` with ξ under `fixed_parameters`
and a provenance citation; `holdout_icao_validation_summary_v5.csv`; v4→v5 delta reported
per mode whichever direction it moves.

---

### P4.6 — Re-run every production study on v5 and regenerate the number trail (F5)

1. Re-run at the v5 state, same seeds, adjudicated component configuration from P4.3/P4.4:
   `design_point_summary.py`, `optimize_blend.py` (free-φ and `--freeze-phi`, 1000 trials,
   seed 42), `variance_decomposition.py`, `nox_holdout_validation.py`,
   `nox_dual_path.py`, `ablate_pinn_components.py`.
2. **Regenerate** `outputs/ARTIFACT_MANIFEST.md` — every row re-pointed at a v5 artifact.
3. **New** `docs/number_crosswalk_v5.md`: v4 → v5 for every manuscript-bound number, same
   table shape as `docs/number_crosswalk.md`. Archive v4 artifacts to
   `outputs/archive/pre_phase4/` — move, never delete.
4. Sweep for orphans exactly as Phase 3.5 did: any number in the preprint draft that does
   not resolve to a manifest row is an orphan and must be removed or re-sourced.

**Acceptance:** zero orphans; no v4 artifact reachable outside `outputs/archive/`;
manifest regenerated rather than edited.

---

### P4.7 — Refactor for reproduction from a clean clone

1. `REPRODUCE.md`: clone → install → one command per manifest row → expected output. Verify
   it on a clean checkout in a fresh virtualenv, not on the development machine.
2. Pin `requirements.txt` — the current file has floors (`>=`) only, so a clean install
   drifts. Record the versions the v5 numbers were produced with.
3. CPU defaults everywhere. `--device mps` defaults become `auto` with a CPU fallback.
4. Delete-or-quarantine pass on dead code. `simulation/emissions.py` is already marked
   non-production and orphaned — decide whether it stays at all. Same question for
   `pareto_visual.py`, `visualize_results.py`, `dashboard.py`, `fetch_and_build_cfd_data.py`.
5. Decide what `data/raw/cfd_datasets/` should keep. A vendored third-party SU2 GUI repo
   inside the data directory is not a data asset; if P4.1 route (b) uses it, it becomes a
   declared dependency, otherwise it goes.
6. One test run from clean clone: `pytest tests/ -v`.

---

### P4.8 — Preprint and project website

1. Preprint built from the regenerated manifest only. Rebuild every equation as a native
   object — the submitted PDF's corrupted symbols came from an exported-image equation
   pipeline, and the same export will corrupt them again.
2. Title and abstract follow the P4.3/P4.4 outcomes, not the current wording. If both gates
   failed, the title does not say "physics-informed neural network".
3. Carry forward from `docs/response_letter_draft.md` the disclosures that survive: the
   argon-cycle defect, the retired Highlights and their provenance, the withdrawal of blend
   NOx ranking, the within-family caveat on M7.
4. Website: static, self-contained, no numbers typed by hand — the figures and the tables
   are generated from manifest artifacts. One page per claim, each linking to the script and
   the CSV that produced it. Include the negative results; a site that shows the Sajben miss
   and the variance decomposition is more credible than one that shows only a Pareto front.

---

## Commands to Run

```bash
# environment
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# P4.1
python scripts/validation/sajben_data_audit.py

# P4.2 / P4.3
python scripts/validation/sajben_validation.py --model models/le_pinn_sajben.pt
python scripts/validation/train_sajben.py --epochs 5000 --device auto --seed 42
python scripts/validation/sajben_validation.py --model models/le_pinn_sajben_v5.pt

# P4.4
python scripts/validation/train_turbine_surrogate.py --seed 42 --device auto
python scripts/validation/adjudicate_turbine_p5.py

# P4.5
python scripts/optimization/calibrate_lto.py --tag v5 --n-trials 100 --beta 0.8
python scripts/validation/holdout_icao_validation.py \
    --calibration outputs/calibration_trent1000_ae3_v5.json --tag _v5
python scripts/validation/heat_loss_sensitivity.py

# P4.6
python scripts/validation/design_point_summary.py
python scripts/optimization/optimize_blend.py --n-trials 1000 --seed 42 \
    --calibration outputs/calibration_trent1000_ae3_v5.json
python scripts/optimization/optimize_blend.py --n-trials 1000 --seed 42 \
    --calibration outputs/calibration_trent1000_ae3_v5.json --freeze-phi \
    --output-csv outputs/results/optimization_results_phi_frozen.csv
python scripts/analysis/variance_decomposition.py --seed 42 --n-mc 1000
python scripts/validation/nox_holdout_validation.py --tag _v5

# after every phase
python -m pytest tests/ -v
python scripts/test_emissions.py        # after any combustor change
```

---

## Tests

- `pytest tests/ -v` → 80 passed / 1 pre-existing skip as the floor; new tests add to it.
- **New:** fine-tune regression guard (P4.2) — a fine-tuned checkpoint may not score worse
  than its initialisation on the held-out metric.
- **New:** split-leakage guard (P4.3) — `sajben_validation.py` refuses a checkpoint whose
  recorded training split intersects the evaluation set.
- **New:** checkpoint provenance test — every `models/*_v5.pt` carries seed, device, dataset
  hash, epochs, loss weights and git SHA.
- **New:** manifest integrity test — every path named in `ARTIFACT_MANIFEST.md` exists, and
  no file under `outputs/` outside `archive/` is unreferenced by it.

---

## Acceptance Criteria

1. Inherited code-repair changes committed; LE-PINN stream resolved; `phase4` branch exists.
2. `outputs/sajben_data_audit.md` states the analytic ceiling and the chosen data route.
3. Fine-tune regression fixed and test-guarded.
4. Nozzle LE-PINN retrained, scored against the **pre-registered** bands, outcome applied
   without post-hoc adjustment; every attempt reported, not just the best.
5. Turbine PINN either meets the surrogate gate and is labelled a surrogate, or is retired.
6. ξ sourced with a citation, fixed not fitted, v5 calibration and held-out validation
   produced, v4→v5 delta reported in whichever direction it went.
7. All production studies re-run on v5; manifest regenerated; `number_crosswalk_v5.md`
   complete; zero orphans; v4 artifacts archived.
8. `REPRODUCE.md` verified on a clean clone in a fresh virtualenv; requirements pinned.
9. Preprint and website built from manifest rows only; native equations; negative results
   included.

---

## Rollback Notes

- All work on `phase4`; `main` holds a reproducible v4 state until P4.6 passes.
- No `models/*.pt` overwritten — v4 checkpoints remain loadable throughout.
- v1–v4 calibration and holdout artifacts preserved; v5 added alongside, never in place.
- P4.5 is the only irreversible modelling change. If v5 held-out validation degrades badly
  and P4.5(4) says to report rather than revert, that reporting decision is the user's, not
  the executor's — escalate with both numbers in hand.
- Archive moves are `mv` into `outputs/archive/pre_phase4/`; nothing is deleted in this phase.

---

## Escalation Guidance

**Stop and escalate — do not guess:**

- P4.1 finds no affordable data route (the phase's scope premise fails).
- P4.5 finds no defensible sourced ξ value.
- A pre-registered gate is missed and there is pressure to re-tune past it. That pressure is
  the thing this plan exists to resist.
- The uncommitted LE-PINN stream conflicts with a planned edit to `le_pinn.py`.
- v5 held-out validation degrades materially against v4.

**Complexity:** P4.1 and P4.5 are judgement-heavy and short — highest capability, low token
count. P4.2 is a focused debugging task. P4.3, P4.4 and P4.6 are long mechanical runs once
their decisions are fixed. P4.7 and P4.8 are broad but low-risk.

**Suggested order:** P4.0 → P4.1 (gate) → P4.2 → P4.5 in parallel with P4.3 (independent
subsystems) → P4.4 → P4.6 → P4.7 → P4.8.

**The honest expectation, stated up front:** F1 suggests the nozzle PINN will not clear
0.10 without new data, and F3 suggests the turbine PINN's best available outcome is
"accurate surrogate of the analytic model". A preprint that reports both plainly, with the
variance decomposition showing φ dominates and blends do not, is a stronger document than
one that keeps the original claim alive. The gates in P4.3 and P4.4 exist so that this is a
finding if it happens, rather than a disappointment.
