# Phase 3 — Component Adjudication, Artifact Hygiene, and Number Freeze — July 14, 2026

Phase 2 plan archived at `docs/plan_phase2_completed.md` (commits `9588d31`…`5c91183`; 80 passed / 1 pre-existing skip). This plan is driven by a fresh post-Phase-2 audit. Two findings require action before any manuscript number is final.

## Audit findings driving this phase

**F1 — The production studies violate Phase 2's own decision rule (critical).**
`optimize_blend.py` calls `run_full_cycle` with no component flags, so both final 1000-trial studies ran on the defaults `turbine_model="pinn", nozzle_model="pinn"`. But the Phase 2 evidence rules against exactly that configuration:

- Ablation (`outputs/ablation_pinn_components.csv`): turbine PINN vs analytic disagree by ~37% in core thrust and ~1.96 bar in turbine exit pressure (2.77 vs 4.73 bar at identical work-matched T5); total thrust ±10%, TSFC ±10.7%.
- External data (`outputs/sajben_validation_errors.csv`): the nozzle LE-PINN **fails** Sajben — wall-Cp shape-L2 0.71–1.09 against the <0.10 gate.
- P2.5's decision rule: "deviation material ⇒ analytic wins by default absent external data." The only external data votes against the PINN.

Consequence: every manuscript-bound thrust/TSFC/specific-thrust number (incl. representative solution Trial 331, TSFC 8.32 mg/(N·s)) currently rests on the losing configuration. Fuel-flow validation (holdout MAPEs) is unaffected — FAR is upstream of both components.

**F2 — Stale results artifacts coexist with fresh ones (integrity hazard).**
`outputs/results/pareto_optimal_solutions.csv` still contains the *old manuscript's* numbers (first row: TSFC 29.464, spec thrust 800.58, LCA 0.7616 — last committed pre-Phase-1). It is written by `pareto_visual.py`, which (with `visualize_results.py`) carries uncommitted local modifications and was deliberately not touched in P2.7. Several `outputs/plots/*.png` (01–13 series, `final_pareto_front.png`, `pareto_3d.png`, `statistical_tests.png` — the deleted t-test figure) also predate the corrected engine. Anyone applying the Google-Docs checklists can currently grab a dead number from a live-looking file.

**F3 — Take-off bias worsened and is unexplained** (+16.5% v1 → +19.7% v3 after the physical part-power model; part-power hypothesis falsified). The leading unmodeled candidate: the model burns *all* core air at the calibrated φ, whereas real combustors pass only ~70–80% of core flow through the burner (turbine cooling + liner/dilution bypass is unmodeled as an airflow split — distinct from the ξ heat-loss hook added in P2.6).

**F4 — Two validation numbers now coexist** (v1 7.27% with inert parameters; v3 12.71% physical). The manuscript must report the one corresponding to the shipped model, with the other explained — not silently choose the better one.

## Objective

1. Adjudicate the turbine-exit-pressure discrepancy, select the production component configuration by the evidence, and make it the code default (F1).
2. Re-run both seeded studies + all figures on the adjudicated configuration; refresh representative solution, variance decomposition, and rank stability (F1).
3. Purge/archive every stale artifact and produce a canonical manifest mapping each manuscript-bound number to exactly one generating script + CSV (F2).
4. Time-boxed take-off-bias diagnostic via a combustor airflow-split parameter (F3).
5. Freeze the validation story (v3 primary) and produce the old→new number crosswalk + response-letter draft the Google-Docs edit will consume (F4).

## Constraints

- Do not modify `data/*.yaml` mechanisms; never overwrite `models/*.pt` (no retraining in this phase — the Sajben failure is disclosed, not fixed here).
- Preserve seeds; all re-runs use the same seeds as P2.7 so config change is the only diff.
- Preserve versioned calibration/holdout artifacts (v1–v3) — Phase 3 adds v4 only if P3.4 changes the engine.
- Resolve the uncommitted `pareto_visual.py`/`visualize_results.py` state with the user before editing those two files (they are user-modified; do not clobber).
- `python -m pytest tests/ -v` after each phase; `scripts/test_emissions.py` after any combustor change. No silent failures.

## Relevant Files

- Modify: `integrated_engine.py` (component defaults; P3.4 airflow-split parameter; turbine P_out investigation)
- Modify: `scripts/optimization/optimize_blend.py` (explicit component flags; re-run)
- Modify: `scripts/analysis/variance_decomposition.py` (re-run on new studies)
- Modify: `scripts/optimization/calibrate_lto.py` + `scripts/validation/holdout_icao_validation.py` (only if P3.4 adds the split parameter → v4)
- Create: `scripts/validation/adjudicate_turbine_p5.py`, `scripts/validation/airflow_split_sensitivity.py`
- Create: `outputs/ARTIFACT_MANIFEST.md`, `docs/number_crosswalk.md`, `docs/response_letter_draft.md`
- Archive (move, don't delete): stale files under `outputs/` → `outputs/archive/pre_phase2/`
- Read only: `models/*.pt`, all v1–v3 calibration/holdout artifacts

## Phase 3.1 — Adjudicate the turbine exit pressure and pick the production configuration (F1)

1. `scripts/validation/adjudicate_turbine_p5.py`: at the calibrated design point, compute turbine exit pressure three ways — (a) analytic polytropic with the imposed work (invert the work equation for the pressure ratio, η_poly=0.9); (b) the PINN's p5; (c) the value implied by the fixed `turbine_design['P_out']`. Determine which the analytic path actually uses (the 4.73 bar suggests a fixed design P_out rather than work-consistent expansion — if so, *both* paths have a defect: the analytic p5 is not work-consistent and the PINN p5 is unvalidated). Hand-verify the work-consistent value.
2. Decision (encode the rule, record the outcome in the manifest):
   - Nozzle: **analytic** — it failed external validation; no discretion here.
   - Turbine: analytic polytropic with *work-consistent* exit pressure, unless (1) shows the PINN matches the work-consistent value better than the current analytic path does.
3. Flip `run_full_cycle` defaults to the adjudicated configuration so reproduction-by-default matches the paper; keep the flags for the ablation narrative.
4. If the analytic p5 was indeed fixed-P_out (not work-consistent): fix it, and note that the ±10% thrust ablation spread must be recomputed — part of it may be a bug, not model uncertainty.

Validation: hand-calculation in the script's docstring reproduced by the code within tolerance; `pytest tests/ -v`.

## Phase 3.2 — Re-run production studies on the adjudicated configuration (F1)

1. Re-run both seeded studies (free-φ, φ-frozen; same seeds, N=1000) with explicit component flags in `optimize_blend.py` — never rely on defaults again.
2. Regenerate: variance decomposition, rank stability, all five P2.7 figures, representative balanced solution (full provenance: trial, seed, LCA draw, component config).
3. Expected: performance-neutrality conclusion survives (blend deltas were ~0.25% under both configs); absolute TSFC/thrust shift. Diff old-vs-new Pareto membership and report it — if membership churns materially, the earlier "99.4% rank stability" claim must be re-derived, not reused.
4. Take-off TSFC sanity gate re-checked (was 11.1 mg/(N·s) under PINN config).

Validation: zero failed/defaulted trials; figures regenerate from clean shell; `pytest tests/ -v`.

## Phase 3.3 — Artifact hygiene: purge stale, manifest the canonical (F2)

1. Move to `outputs/archive/pre_phase2/`: `outputs/results/pareto_optimal_solutions.csv` (contains the old paper's numbers), `statistical_tests.png` (deleted t-test), the 01–13 plot series and any plot not regenerated by the P2.7/P3.2 pipeline, old `final_pareto_front.png`/`pareto_3d.png`/`pareto_front_2d.png`/`pareto_front_3d.png` duplicates. Rule: if no Phase 2/3 script writes it, it leaves `outputs/`.
2. `outputs/ARTIFACT_MANIFEST.md`: one row per manuscript-bound number/figure → generating script, commit, output file, seed. The Google-Docs edit works *only* from this manifest.
3. Resolve the `pareto_visual.py`/`visualize_results.py` uncommitted-modification limbo with the user: either commit their local changes or supersede both with `scripts/analysis/` equivalents and archive them. One canonical figure pipeline afterward — `pareto_visual.py` must no longer be able to write `pareto_optimal_solutions.csv` with stale semantics.
4. Also archive/regenerate stale committed-but-outdated plots (`03_icao_benchmark_bars.png`, `emissions_comparison.png`, `pinn_comparison.png`, etc.) per the same rule; `git status` currently shows a mix of modified and deleted plot files — bring the tree to a clean, intentional state.

Validation: `outputs/` contains only manifest-listed or archived files; every manifest entry regenerates from a clean shell.

## Phase 3.4 — Take-off bias diagnostic: combustor airflow split (F3) [time-boxed]

1. Add optional `combustor_air_fraction` β (default 1.0 — prior results preserved) to the cycle: only β·ṁ_core enters the burner at φ; (1−β) bypasses as cooling/dilution and remixes before the turbine (enthalpy balance sets the mixed T4). Literature anchor: β ≈ 0.7–0.8 for RQL-era combustors, cited.
2. `scripts/validation/airflow_split_sensitivity.py`: sweep β ∈ {1.0, 0.9, 0.8, 0.7} → per-mode fuel-flow error against ICAO, holding v3 calibration; then one recalibration at the best sourced β → `calibration_trent1000_ae3_v4.json` + holdout v4.
3. Decision rule: if a sourced β materially shrinks the take-off bias without degrading idle/approach, adopt it (v4 becomes the reported model, provenance updated). If not, the bias is documented as a stated limitation with the falsified hypotheses listed (part-power: falsified in P2.1; airflow split: result here) — hypotheses die in public either way.
4. Time-box: this is one parameter and two scripts. No further engine surgery in Phase 3 regardless of outcome.

Validation: `scripts/test_emissions.py` + combustor tests; sensitivity CSV/plot; provenance updated.

## Phase 3.5 — Validation story freeze + number crosswalk (F4)

1. Encode the decision: **v3 (or v4 if adopted) is the reported validation** — it corresponds to the shipped physical model. v1's 7.27% is reported once, as the free-parameter upper-bound variant with inert parameters (already disclosed), not as the headline. The 5.46%-vs-11.3% seeding note carries into the response letter.
2. `docs/number_crosswalk.md`: every number in the old manuscript → its fate — corrected value + manifest row, "removed (reason)", or "replaced by band". Minimum rows: 29.46 TSFC, 800.6 specific thrust, 0.762 LCA, 11.3% MAPE, 87.65 NOx, all four Highlights numbers, 81.58 kN, T3/T4 temperatures (argon disclosure), 3.16 CO₂ factor, surrogate table.
3. `docs/response_letter_draft.md`: point-by-point reply skeleton to both reviewers, citing the crosswalk and manifest — including the voluntary disclosures (argon defect, Sajben negative result, take-off bias history, stale-Highlights provenance from Phase 1).

Validation: crosswalk cross-checked against `manuscript.txt` extraction — no orphaned old numbers.

## Execution order & dependencies

P3.1 → P3.2 (config must be adjudicated before the re-run). P3.4 → if adopted, P3.2 re-runs on v4 (do P3.4 before P3.2 to avoid running the studies twice; if the time-box expires, run P3.2 on v3). P3.3 after P3.2 (manifest lists final artifacts). P3.5 last. Commit each phase separately.

## Acceptance criteria (Phase 3 done when)

1. Turbine p5 discrepancy explained with a hand-verified work-consistent value; component defaults match the adjudicated, evidence-backed configuration.
2. Both studies re-run under explicit flags; representative solution, variance decomposition, rank stability, and figures regenerated; Pareto-membership diff reported.
3. `outputs/` is manifest-clean; `pareto_optimal_solutions.csv` stale numbers unreachable; viz-script limbo resolved with the user.
4. β sweep complete; v4 adopted or bias documented as limitation with falsified-hypothesis history.
5. Crosswalk + response-letter draft complete; every old number accounted for; every new number traceable to one manifest row.
6. `python -m pytest tests/ -v` and `scripts/test_emissions.py` pass; all results reported.

## Out of scope for Phase 3

Retraining the nozzle/turbine PINNs against Sajben or corrected-cycle data (future work, disclosed as such); chemistry EI-NOx held-out validation; cruise-point extension; the Google-Docs edits themselves (manuscript side, consuming P3.5's outputs).
