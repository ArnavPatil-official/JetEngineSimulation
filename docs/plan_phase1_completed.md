# Phase 1 — Reviewer-Response Repair Plan (Big Issues) — July 13, 2026

Previous plan archived at `docs/plan_2026-05-06_le_pinn_archived.md`.
Full audit basis: `MANUSCRIPT_REPAIR_PLAN.md` (V-numbers below refer to its vulnerability table). All claims re-verified against the code on 2026-07-13.

## Objective

Eliminate the fatal integrity and validation problems flagged by both reviewers before any further optimization work or resubmission drafting:

1. Engine misattribution (manuscript: Trent XWB-84EP; data + calibration: Trent 1000-AE3) — V2.
2. Highlights numbers that appear nowhere in Results (R²=0.9969, 81.58 kN, 80% CO₂ cut, 0.00% continuity) — V3.
3. "Validation" that is actually in-sample calibration on 4 points; replace with genuine held-out cross-engine validation using the 44 other Trent 1000 variants already in `data/icao_engine_data.csv` — V1.
4. False "learnable parameter" narrative and unsourced SAF efficiency penalties that manufacture blend sensitivity — V8/C4.
5. Surrogate composition mismatch between `simulation/fuels.py` and the manuscript — V15.
6. Fragile stdout-regex results pipeline with silent T4=2000 K fallback, which taints every reported optimization number — V17.
7. t-test misuse and factual errors (syngas, CO₂-in-ppm, HyChem scope, corrupted equations) — V14/V16.

Phase 1 delivers: corrected code, a real validation result, a parameter-provenance record, and a manuscript-edit checklist. Pareto re-runs, fan/bypass, LCA overhaul, and PINN ablations are Phase 2 (see `MANUSCRIPT_REPAIR_PLAN.md` Deliverable 3, tier B).

## Constraints

- Do not modify `data/*.yaml` mechanisms or overwrite `models/*.pt`.
- Preserve seeds, RNG patterns, PINN loss structure, and existing logging patterns.
- Do not change combustor/turbine/nozzle physics beyond what is listed here.
- Run `python -m pytest tests/ -v` after each phase; run Cantera validation tests after the combustor-efficiency change (P1.3).
- New scripts go in `scripts/validation/`; new results in `outputs/`.

## Relevant Files

- Modify: `scripts/optimization/optimize_blend.py` (remove regex scraper; consume structured results)
- Modify: `simulation/combustor/combustor.py` (`estimate_efficiency` SAF penalties)
- Modify: `simulation/fuels.py` (only if P1.4 shows manuscript compositions generated the reported runs; also fix the 0.857/0.847 docstring inconsistency)
- Modify: `scripts/optimization/calibrate_lto.py` (persist calibrated parameters to JSON)
- Create: `scripts/validation/holdout_icao_validation.py`
- Create: `scripts/validation/ablate_saf_penalty.py`
- Create: `outputs/parameter_provenance.md`
- Create: `docs/manuscript_edits_phase1.md`
- Read only: `integrated_engine.py`, `data/icao_engine_data.csv`

## Phase 1.1 — Held-out cross-engine validation (V1, V2) [highest value; do first]

The 11.3% MAPE is a fit residual: `calibrate_lto.py` tunes η_comb, pressure loss, and four per-mode φ values (plus hand-set airflow/π scales) against the same 4 Trent 1000-AE3 fuel-flow points reported as validation. And the manuscript attributes the data to an engine (Trent XWB-84EP) that does not appear in the CSV at all.

1. Modify `calibrate_lto.py` to write the best-trial parameters (η_comb, p_loss, φ per mode, scale factors) to `outputs/calibration_trent1000_ae3.json` with a seed. Do not change its search logic.
2. Create `scripts/validation/holdout_icao_validation.py`:
   - Load frozen calibration JSON; never re-tune on held-out engines.
   - For each of the other 44 Trent 1000 variants in `data/icao_engine_data.csv`: scale `mass_flow_core` by rated thrust ratio and set π_c from the CSV OPR column; predict fuel flow at all 4 LTO modes.
   - Output: per-mode and overall MAPE distribution (CSV in `outputs/`), predicted-vs-ICAO scatter plot with identity line, per-mode box plot.
3. Record the result whatever it is. Decision rule (from repair plan AB5): held-out MAPE ≲ 20% → report as genuine validation; ≳ 30% → manuscript drops quantitative fuel-flow claims and reframes as order-of-magnitude consistency.

Validation: script runs end-to-end from a clean shell; every quoted fuel-flow/OPR/BPR number traceable to a CSV row.

## Phase 1.2 — Kill the stdout scraper (V17)

`optimize_blend.py::scrape_log_data` regex-parses printed output and silently substitutes T4 = 2000 K on parse failure — this is the path Reviewer 2's "11.3% HEFA vs 11.3% MAPE" transcription concern points at.

1. `run_full_cycle` already returns a structured dict (`performance`, `emissions` keys). Replace all scraper usage with direct dict access; delete `scrape_log_data` and the `contextlib`/`io`/`re` capture machinery.
2. Remove the silent fallback: a failed trial must raise/prune, never inject defaults.
3. Smoke-run 20 Optuna trials with a fixed seed; confirm objective values match structured outputs exactly. (Full 1000-trial re-run and figure regeneration is Phase 2 — it depends on P1.3/P1.4 decisions.)

Validation: `grep -rn "scrape_log_data\|re.search" scripts/optimization/optimize_blend.py` returns nothing; smoke run completes with no penalized-default trials.

## Phase 1.3 — Remove unsourced SAF efficiency penalties + parameter provenance (V8, C4)

The manuscript claims η_c and η_CMB are "learnable parameters" (Reviewer 2 asked what they converged to). In code: η_c is fixed at 0.86 (`integrated_engine.py:565`), η_comb is either fixed at 0.98 in optimization or set by `Combustor.estimate_efficiency()` — a hard-coded parabola with unsourced blend-name penalties (−1.5% "Bio-SPK", −1.0% "HEFA") that manufacture the very blend sensitivity the paper reports.

1. Create `scripts/validation/ablate_saf_penalty.py`: run the full cycle for Jet-A1, HEFA, FT, ATJ blends at fixed design point with penalties ON vs OFF; report Δ(TSFC, thrust, T4, fuel flow) per blend.
2. Remove the `if 'Bio-SPK'/'HEFA'` penalty block from `estimate_efficiency` (keep the φ-parabola). Note removal + ablation numbers in provenance doc. (If the ablation shows blend ranking flips, that is itself a required disclosure.)
3. Create `outputs/parameter_provenance.md`: table of every model parameter — value, bounds, fixed vs calibrated, code location, source/citation status. Minimum rows: η_c=0.86, η_poly=0.9, η_comb (calibrated, bounds 0.96–0.999, converged value from P1.1 JSON), p_loss, φ per mode, airflow/π scales, LCA factors (flag: unsourced — Phase 2), 3.16 kg CO₂/kg fuel, LHV and carbon fractions per surrogate.
4. Run combustor/emissions validation tests after the change (per CLAUDE.md Cantera rule).

Validation: `pytest tests/ -v` + `python scripts/test_emissions.py`; ablation CSV in `outputs/`.

## Phase 1.4 — Reconcile surrogate compositions (V15)

Code: HEFA 85/15 dodecane/iso-octane, FT 50/35/15, ATJ 80/20 (mole-fraction dicts). Manuscript: HEFA 60/40, FT 45/35/20, ATJ 75/25 "by mass". Reviewers cannot tell which set produced the results, and the basis (mass vs mole) is ambiguous.

1. Check `git log -p simulation/fuels.py` against run dates to determine which set generated the reported results. If git history is inconclusive, re-run one blend case with each set and compare against reported numbers (TSFC 29.46, specific thrust 800.6).
2. Single source of truth: manuscript table must equal `fuels.py`. Change whichever is wrong; state the mole/mass basis once, consistently.
3. Fix the `FuelSurrogate` docstring/default: comment claims 144/170 = 0.857; it is 0.847 (default `carbon_fraction=0.857` contradicts the JET_A1 instance).
4. Quantify the output shift between the two composition sets (one design-point run each) — if the shift exceeds blend-to-blend differences, record that for the Phase 2 reframing.
5. Compute H/C ratio from composition programmatically for the manuscript table; drop the DCN values (uncomputed and uncited) unless a citation is added.

Validation: `pytest tests/ -v`; composition table in `outputs/parameter_provenance.md` matches `fuels.py` exactly.

## Phase 1.5 — Manuscript-edit checklist (V2, V3, V14, V16 + Reviewer 1 line edits)

Create `docs/manuscript_edits_phase1.md` — a line-by-line checklist so every edit is traceable in the response letter. No code changes in this phase. Contents:

**Integrity (fatal):**
- Replace every "Trent XWB-84EP" with "Trent 1000-AE3" (3 occurrences in PDF; also check `AGENT_CONTEXT.md` and outreach docs); add ICAO UID 02P23RR126 and a data-availability note.
- Rewrite Highlights using only numbers present verbatim in Results. Draft provenance note for the response letter explaining each deleted number: R²=0.9969 is the in-sample fit statistic printed by `EmissionsEstimator._fit_nox_model()`; 0.00% continuity is satisfied by construction (u = ṁ/ρA); "80% CO₂ cut" echoes the assumed input f_H=0.2; 81.58 kN is an untraceable printout.
- Delete the t-test (t=1.04, p=0.3734) and all "not statistically distinguishable" language; replace with per-mode % errors + P1.1 held-out MAPE.
- Rewrite §2.1–2.2: no "learnable parameter" language; describe fixed η_c=0.86 and one-time Optuna calibration with reported converged values (from P1.1 JSON); reference provenance table.
- Replace "validated" with "calibrated on Trent 1000-AE3; validated on 44 held-out variants (MAPE X%)" per P1.1 outcome. State explicitly that NOx, blend-discriminating outputs, and lifecycle CO₂ are NOT validated.

**Factual corrections (Reviewer 1 + 2):**
- Abstract & elsewhere: compressor is NOT "modeled using Cantera" chemistry — Cantera supplies thermodynamic properties for an isentropic-efficiency compression; say so precisely.
- Syngas = CO + H₂ (not CO₂) — line ~58.
- CO₂ metric: g/s (or EI g/kg), never "ppm" — line ~179.
- HyChem passage (lines 216–217): not "valid only for Jet A-1"; developed on NJFCP A-2 nominal fuel, applied to multiple NJFCP fuels; rewrite per Reviewer 1.
- Compressor losses (line 258): lead with tip-leakage/secondary flow; drop stall as a "loss mechanism" (avoided by design); blade friction minor.
- Combustor (lines 303–310): acknowledge case/liner heat loss can exceed inefficiency loss; rename η as lumped heat-delivery efficiency (full treatment = Phase 2 V9).
- Line 318: "turbine inefficiency," not "blade drag."
- Strike lines 90–91; fix "combing"→"combining" (line 131); make line 97 specific or cut; cut or make concrete lines 147–149; fix GA-predates-AI claim (line 129).
- Rebuild ALL equations as native equation objects (corruption is a Google-Docs→PDF export failure); audit the re-exported PDF on a second machine.
- Add the logic flow chart Reviewer 1 requested: objectives → tools → outputs (one figure).

## Execution order & dependencies

P1.1 → (P1.3 needs its JSON for converged values) ; P1.2 independent ; P1.4 independent ; P1.5 last (consumes results of all others). Commit each phase separately.

## Acceptance criteria (Phase 1 done when)

1. `outputs/calibration_trent1000_ae3.json` + held-out MAPE results + 2 plots exist and regenerate from clean shell.
2. No regex scraping or silent defaults in `optimize_blend.py`; seeded smoke run passes.
3. SAF penalties removed (or sourced), ablation quantified, `outputs/parameter_provenance.md` complete.
4. `fuels.py` and manuscript composition table identical; carbon-fraction docstring fixed; H/C computed.
5. `docs/manuscript_edits_phase1.md` covers every Reviewer 1 line comment and Reviewer 2 points 1, 2, 4a–d.
6. `python -m pytest tests/ -v` passes; all results reported, no silently skipped failures.

## Explicitly deferred to Phase 2

Fan/bypass module (V4), CORSIA LCA ranges + Monte Carlo (V6/V7), chemistry-based NOx (V5), PINN-vs-analytic ablations + Sajben section (V10/V11), variance decomposition & insensitivity reframing (V12/V18), heat-loss sensitivity (V9), full 1000-trial optimization re-run + figure regeneration (B1).
