# Phase 2 — Model Upgrades, Real Emissions Axes, and Final Re-run — July 13, 2026

Phase 1 plan archived at `docs/plan_phase1_completed.md` (all acceptance criteria met, commits `f0fe401`…`57c0ae0`). Audit basis: `MANUSCRIPT_REPAIR_PLAN.md` (V-numbers) + Phase 1 findings. Code paths re-verified 2026-07-13.

## Objective

Phase 1 established integrity (correct engine identity, real held-out validation at 7.27% MAPE, honest parameters, trustworthy results pipeline). Phase 2 makes the model's claims physically defensible and regenerates all reported numbers:

1. Fix the two inert-parameter warts Phase 1 exposed (`pressure_loss` sampled but unused; per-mode π-scales written but never read → every LTO mode ran at rated OPR 43.2) and add a part-power OPR model — the leading suspect for the systematic +16.5% take-off overprediction — V1 follow-up.
2. Add a 0-D fan/bypass stream so "turbofan" outputs are engine-plausible (BPR 9.1 is defined in `design_point` and never used; turbine extracts compressor work only; TSFC 29.46 mg/(N·s) is turbojet-magnitude) — V4.
3. Replace the unsourced LCA point factors {1.0, 0.2, 0.1, 0.3} with CORSIA feedstock ranges + Monte Carlo, and compute combustion EI-CO₂ from Cantera products instead of `3.16 × LCA × ṁ_f` — V6/V7.
4. Give NOx a chemistry path (Zeldovich post-processing + HyChem-A2 anchor via `data/A2NOx.yaml`) alongside the clearly-labeled ICAO correlation — V5.
5. Ablate the PINN components against their analytic counterparts and put the existing Sajben nozzle validation into the manuscript — V10/V11.
6. Quantify combustor heat-loss sensitivity — V9.
7. Re-run the full seeded 1000-trial optimization on the upgraded model, regenerate all figures, and produce the variance decomposition that grounds the "blend-insensitivity is the finding" reframing — V12/V17/V18 (B1+B5).

## Constraints

- Do not modify `data/*.yaml` mechanisms (`A2NOx.yaml` is read-only input) or overwrite `models/*.pt`.
- Preserve seeds and seed-passing patterns; every stochastic run (calibration, Optuna study, Monte Carlo) takes an explicit seed and records it in its output artifact.
- Preserve PINN loss structure, physics constraints, and boundary-condition logic — ablations swap components at inference, never retrain.
- Preserve Phase 1 artifacts: never overwrite `outputs/calibration_trent1000_ae3.json` or the Phase 1 holdout CSVs — Phase 2 recalibrations write versioned files (`calibration_trent1000_ae3_v2.json`, etc.) so before/after is diffable.
- Run `python -m pytest tests/ -v` after each phase; run `scripts/test_emissions.py` after any combustor/emissions change (CLAUDE.md Cantera rule).
- Maintain existing logging patterns (structured dicts from `run_full_cycle`, CSVs + plots to `outputs/`). No stdout scraping — P1.2 stays dead.

## Relevant Files

- Modify: `integrated_engine.py` (part-power OPR, fan/bypass, two-stream thrust/TSFC, EmissionsEstimator CO₂/NOx paths, turbine/nozzle ablation flags)
- Modify: `scripts/optimization/calibrate_lto.py` (wire or delete inert params; v2 calibration output)
- Modify: `scripts/validation/holdout_icao_validation.py` (accept calibration-version argument; re-run post-upgrade)
- Modify: `scripts/optimization/optimize_blend.py` (CORSIA LCA sampling, new objective definitions, seeded final study)
- Modify: `scripts/visualization/pareto_visual.py` / `visualize_results.py` (uncertainty bands, variance-decomposition figure)
- Create: `simulation/fan.py`
- Create: `simulation/nox_chemistry.py` (Zeldovich post-processor)
- Create: `data/corsia_lca_values.yaml` (new data file — additive, does not modify existing mechanisms)
- Create: `scripts/validation/ablate_pinn_components.py`, `scripts/validation/heat_loss_sensitivity.py`, `scripts/validation/nox_dual_path.py`, `scripts/analysis/variance_decomposition.py`
- Create: `docs/manuscript_edits_phase2.md`
- Update: `outputs/parameter_provenance.md` (every new/changed parameter)
- Read only: `data/icao_engine_data.csv`, `data/A2NOx.yaml`, `scripts/validation/sajben_validation.py`, `models/*.pt`

## Phase 2.1 — Part-power OPR + retire inert parameters (V1 follow-up)

Phase 1 documented that `pressure_loss` is sampled but never consumed and the per-mode π-scales write to a dict the compressor never reads, so all four LTO modes ran at rated OPR 43.2. Idle at rated OPR is unphysical; the tuned φ values silently absorbed it.

1. Wire combustor pressure loss for real: apply `p_loss` between compressor exit and combustor (`p_comb = p3 × (1 − p_loss)`), or delete the parameter from calibration entirely. Wire it — Reviewer 1's heat-loss comment (V9, P2.6) needs the pressure-drop hook anyway.
2. Make `pi_c` actually mode-dependent: have `run_compressor` read `design_point['pi_c']` at call time (verify the write-path `calibrate_lto.py` uses is the one the engine reads — Phase 1 showed it is not). Implement a simple part-power law rather than free per-mode scales: π_c(mode) from the ICAO power settings (7/30/85/100% F00) via π_c ≈ 1 + (π_rated − 1)·(N-corrected)^k — a standard low-fidelity throttle model with a single calibrated exponent k, replacing 4 hand-set scales with 1 parameter. Same for `mass_flow_core`.
3. Re-run seeded calibration → `outputs/calibration_trent1000_ae3_v2.json`; re-run holdout → versioned CSVs/plots. Record before/after per-mode MAPE; hypothesis: take-off +16.5% bias shrinks. If it doesn't, say so — the bias hypothesis in the Phase 1 report must not outlive contrary evidence.
4. Note: ICAO CSV has no CLIMB rows (Phase 1 finding). Climb stays excluded from both calibration and validation; remove the untraceable 2.050 kg/s target from `calibrate_lto.py`.

Validation: `pytest tests/ -v`; holdout regenerates end-to-end; provenance table updated (p_loss now consumed; π-scales deleted; k documented).

## Phase 2.2 — Fan/bypass module (V4)

1. Create `simulation/fan.py`: 0-D fan — FPR 1.45, η_fan 0.90 (cite standard turbomachinery values; BPR 9.1 from `design_point`). Fan work = ṁ_bypass·cp·ΔT_fan + core-stream fan-root compression (or fan on total flow feeding the splitter — pick one, document it).
2. `run_turbine`: target_work_total = compressor work + fan work (currently compressor only, `integrated_engine.py` run_turbine target-work block).
3. Thrust: two streams — core nozzle (existing path) + bypass nozzle (isentropic expansion of fan stream to ambient); TSFC and specific thrust from total thrust and total airflow. Keep `thrust_core` reported separately so Phase 1 numbers remain comparable.
4. Re-run calibration+holdout (P2.1 pipeline) → `calibration_trent1000_ae3_v3.json`. Sanity gates: takeoff TSFC within ~±30% of published Trent-1000-class values (~8–11 g/(kN·s) ≈ 8–11 mg/(N·s) static SL); specific thrust drops from ~800 to BPR-9-plausible ~90–130 N·s/kg. If turbine expansion cannot supply combined work within TIT limits, stop and report — do not relax TIT_HARD_LIMIT to force it.

Validation: `pytest tests/ -v`; before/after table (thrust, TSFC, specific thrust, holdout MAPE) in `outputs/`; T4 unchanged at matched φ (fan must not silently alter the core cycle).

## Phase 2.3 — CORSIA LCA ranges + real combustion CO₂ (V6, V7)

1. Create `data/corsia_lca_values.yaml`: per pathway (HEFA, FT, ATJ) a list of feedstocks with core+ILUC LCA values in gCO₂e/MJ from ICAO Document 06, "CORSIA Default Life Cycle Emissions Values for CORSIA Eligible Fuels" (November 2025 edition — fetch and cite exact table rows; fossil jet baseline 89 gCO₂e/MJ). Store min/mode/max per pathway for triangular sampling. Every value carries its table citation in a comment.
2. Replace `LCA_FACTORS` point values in `optimize_blend.py` with draws: normalized factor f = LCA_pathway / 89. Expect roughly f_HEFA ≈ 0.15–0.7, f_FT ≈ 0.06–0.25, f_ATJ ≈ 0.25–0.9 by feedstock — the old 0.2/0.1/0.3 were near best-case; the manuscript's "0.762 blend factor" becomes a scenario band.
3. Combustion CO₂ from chemistry, not the 3.16 constant: EI-CO₂ = (44.01/12.011) × carbon_fraction per blend (fields now computed programmatically in `fuels.py` post-P1.4), cross-checked against the Cantera equilibrium product CO₂ for one case. Report combustion CO₂ (g/s) and lifecycle CO₂ (g/s CO₂e) as separate quantities; never mix in one axis.
4. Monte Carlo: for the final Pareto set, 1,000 seeded LCA draws → Pareto-membership rank-stability metric (fraction of members persisting across draws) + uncertainty bands on the LCA axis. Decision rule (AB7): membership <50% stable ⇒ all blend-selection statements become scenario-conditional in the manuscript.

Validation: `scripts/test_emissions.py`; one hand-checked EI-CO₂ (n-dodecane: 12×44.01/170.33 ≈ 3.10 kg/kg — note this differs from the old 3.16, quantify the delta); YAML values spot-checked against the ICAO PDF.

## Phase 2.4 — NOx dual path (V5)

The current NOx is a log-log regression on the same ICAO family (in-sample R² was the misreported 0.9969); it has zero fuel-chemistry dependence, so blend NOx ranking is illusory.

1. Keep the ICAO correlation, renamed/labeled "ICAO-derived correlation (fuel-flow/OPR proxy, not chemistry)" everywhere it appears.
2. Create `simulation/nox_chemistry.py`: extended Zeldovich thermal-NO post-processor on the CRECK equilibrium combustor products (CRECK C1-C16 has no N-chemistry — grep confirms zero NO species — so post-processing is the only option at fixed mechanism). Inputs: T_flame, equilibrium [O], [O2], [N2], p, residence time τ (combustor volume / volumetric flow; document τ). Standard rate constants cited (e.g., GRI/Baulch evaluations).
3. Anchor case: 0-D constant-pressure reactor with `data/A2NOx.yaml` (HyChem A-2 fuel POSF10325 + full NOx chemistry, 7,139 lines — already in repo, unused) at matched T3/p3/φ per LTO mode → reference EI-NOx for conventional fuel. Compare all three paths (correlation, Zeldovich-on-CRECK, HyChem-A2 reactor) in `scripts/validation/nox_dual_path.py`.
4. Report divergence honestly: they will disagree; the finding is the spread, which bounds NOx claim precision. Blend-level NOx differences smaller than the path spread ⇒ manuscript may not rank blends by NOx.

Validation: Zeldovich unit test against a published thermal-NO example; A2 reactor run reproducible with fixed tolerances; comparison CSV + figure in `outputs/`.

## Phase 2.5 — PINN ablations + Sajben section (V10, V11)

1. Add explicit `turbine_model={"pinn","analytic"}` and `nozzle_model={"pinn","analytic"}` flags to `run_full_cycle` (analytic nozzle fallback already exists at `run_nozzle`; analytic turbine = `scale_turbine_exit_temp` polytropic path already computes the target — expose it as the component).
2. `scripts/validation/ablate_pinn_components.py`: full cycle for all four fuels × {PINN, analytic} × {turbine, nozzle} at the calibrated design point; report max deviation in T5, thrust, TSFC, fuel flow.
3. Decision rule (AB1/AB2): deviation < holdout calibration error ⇒ manuscript claims the PINNs as physics-consistency surrogate layers (architecture contribution), not accuracy contributors; deviation material ⇒ analytic wins by default absent external data, and the discrepancy must be explained.
4. Sajben: run `scripts/validation/sajben_validation.py` (wall-Cp, velocity-profile, continuity error functions already implemented) against `models/le_pinn_sajben*.pt`; produce the manuscript's nozzle-validation figure + error table. This is the strongest unused asset — it gives the nozzle surrogate external experimental grounding.
5. Publish the test suite result (currently 71 passed / 1 skipped) as the physics-consistency table in the manuscript.

Validation: ablation matrix CSV + Sajben figure regenerate from clean shell; `pytest tests/ -v`.

## Phase 2.6 — Combustor heat-loss sensitivity (V9)

1. Add optional `heat_loss_fraction` (0–6% of heat release, default 0) to `Combustor.run`: T_out solved with (1−ξ)·heat release. One parameter, additive, default preserves all prior results.
2. `scripts/validation/heat_loss_sensitivity.py`: sweep ξ ∈ {0, 2, 4, 6}% → effect on T4, fuel flow, holdout MAPE, and blend-to-blend deltas.
3. Decision rule (AB6): if ξ=4% moves outputs more than the blend signal, the manuscript states that heat-loss treatment bounds the resolvable blend effect size. Reviewer 1's point (case/liner loss can exceed inefficiency loss) gets a direct, quantified answer; η_comb is renamed "lumped heat-delivery efficiency" in provenance + manuscript.

Validation: `scripts/test_emissions.py` + combustor tests; sensitivity CSV/plot in `outputs/`.

## Phase 2.7 — Final seeded re-run + variance decomposition (V12, V17, V18)

Runs only after P2.1–P2.4 are merged (P2.5/P2.6 inform wording, not the production model — heat loss stays at default 0 unless P2.6 changes that decision; record the choice).

1. `optimize_blend.py`: fixed seed, N=1000 trials on the upgraded engine; objectives redefined per P2.3/P2.4 (TSFC, specific thrust, lifecycle CO₂e with LCA draw index recorded, NOx clearly correlation-based unless P2.4 justifies the chemistry path). Also run the φ-frozen variant (φ fixed per calibrated mode, blends only — AB8) to separate operating-point effects from fuel effects.
2. `scripts/analysis/variance_decomposition.py`: from the trial database, variance share of each objective attributable to φ, blend fractions, and LCA draw (Sobol via `optuna` importance or sklearn regression-based). Plus the objective correlation matrix (adapt existing `pareto_visual.py` heatmap block). This is the quantitative backbone of the reframed claim: if blend fractions explain ~nothing of TSFC/thrust variance, performance-neutrality is a *finding*, stated with numbers.
3. Regenerate every results figure (Pareto with LCA bands, parallel coordinates, correlation heatmap, holdout scatter) from the final run; new representative balanced solution reported with full provenance (trial number, seed, LCA draw).
4. Cross-check: no number destined for the manuscript exists only in a printout — everything traceable to a CSV in `outputs/results/`.

Validation: study completes with 0 defaulted trials; figures regenerate from clean shell; `pytest tests/ -v` final pass, all results reported.

## Phase 2.8 — Manuscript-edit checklist, Phase 2 additions

`docs/manuscript_edits_phase2.md`, consuming P2.1–P2.7 outputs:

- §2 architecture rewrite: fan/bypass description, part-power model, two-stream thrust; all performance tables regenerated; "core-only" caveats removed where no longer true.
- §2.6 rewrite: CORSIA-cited LCA table (pathway × feedstock × value × ICAO table row), Monte Carlo method, separate combustion/lifecycle CO₂ definitions; "0.762" replaced by scenario band.
- NOx: correlation-vs-chemistry framing, footnote in every NOx figure; A2 anchor described.
- New sections: §3.x "Nozzle surrogate validation (Sajben diffuser)"; §3.y "Component ablations"; variance-decomposition results with the insensitivity finding stated as the central conclusion (Abstract + §4 rewrite per `MANUSCRIPT_REPAIR_PLAN.md` Deliverable 5 "defensible revised" claim).
- Heat-loss sensitivity paragraph answering Reviewer 1 lines 303–310 with numbers.
- Updated Highlights: only numbers verbatim from the P2.7 results set.
- Provenance updates carried into the response letter (inert parameters removed, 3.16→computed EI-CO₂ delta, take-off bias before/after).

## Execution order & dependencies

P2.1 → P2.2 (both touch cycle+calibration; recalibrate after each, keep v2/v3 JSONs) → P2.7. P2.3, P2.4 independent of engine changes but must merge before P2.7. P2.5, P2.6 anytime after P2.2; they gate wording, not the production run. P2.8 last. Commit each phase separately; validate between phases.

## Acceptance criteria (Phase 2 done when)

1. No inert calibration parameters; part-power OPR active; versioned calibration/holdout artifacts show before/after per-mode MAPE, take-off bias change quantified.
2. Fan/bypass live: BPR consumed, TSFC/specific thrust within engine-class sanity gates (or a documented stop-report explaining why not).
3. Every LCA number cites an ICAO Document 06 table row; lifecycle results presented as bands; rank-stability metric computed.
4. Three NOx paths compared; every NOx figure/table labeled by path; blend NOx ranking claimed only if blend deltas exceed path spread.
5. PINN-vs-analytic deviation quantified; Sajben validation figure + error table exist; decision rule applied and recorded.
6. Heat-loss sensitivity quantified with the ξ-sweep.
7. Final seeded 1000-trial study + φ-frozen variant complete; variance decomposition + all figures regenerate from clean shell; every manuscript-bound number traceable to `outputs/results/`.
8. `python -m pytest tests/ -v` passes; `scripts/test_emissions.py` passes; all failures reported, none skipped silently.

## Deferred beyond Phase 2 (optional upgrades)

Chemistry EI-NOx validated against held-out ICAO NOx columns (upgrades V5 from reframe to fix); 2D LE-PINN nozzle promotion (only if P2.5 shows the 1D PINN earns its place); cruise-point Breguet cross-check; measured-property blend model (density/viscosity/LHV vs SAF data).
