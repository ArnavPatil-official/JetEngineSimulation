# Manuscript Edits — Phase 1 Checklist

Line-by-line edit checklist for the reviewer response (Phase 1 of
`docs/plan.md`, 2026-07-13). Each item is written so the response letter can
cite it directly. Supporting artifacts: `outputs/parameter_provenance.md`,
`outputs/calibration_trent1000_ae3.json`, `outputs/ablation_saf_penalty.csv`,
`outputs/holdout_icao_validation_summary.csv`.

Status legend: [ ] = pending manuscript edit (the manuscript source lives in
Google Docs, outside this repo); [x] = completed in this repo.

## A. Integrity (fatal) — V2, V3, V14, V1/V8

### A1. Engine misattribution (V2)
- [ ] Replace every "Trent XWB-84EP" with "Trent 1000-AE3" — 3 occurrences in
      the PDF (Abstract, §2.5, §3; also check Fig. 1 caption).
- [ ] Add the ICAO databank identifiers wherever the engine is named:
      UID **02P23RR126**, OPR 43.2, BPR 9.1, rated thrust 310.9 kN — all
      traceable to `data/icao_engine_data.csv` rows.
- [ ] Add a data-availability note: ICAO Aircraft Engine Emissions Databank,
      Trent 1000 family records, as shipped in `data/icao_engine_data.csv`.
- [x] `AGENT_CONTEXT.md` corrected (XWB → Trent 1000-AE3, calibration-vs-
      validation language, composition table) — 2026-07-13.
- [ ] Outreach two-pager: purge the XWB attribution before any wave-1 email
      (per `MANUSCRIPT_REPAIR_PLAN.md` outreach note).

### A2. Highlights numbers absent from Results (V3)
Rewrite Highlights using only numbers that appear verbatim in Results.
Response-letter provenance for each deleted number:
- [ ] **R² = 0.9969** — the in-sample fit statistic printed by
      `EmissionsEstimator._fit_nox_model()` at engine startup (NOx power-law
      refit to the full ICAO CSV on every run). It is not a validation result
      and never appeared in Results. Delete.
- [ ] **0.00% continuity error** — satisfied by construction: the nozzle
      velocity is computed as u = ṁ/(ρA), so mass conservation is an identity
      of the post-processing, not evidence. Delete (or state "by
      construction" explicitly).
- [ ] **"80% CO₂ cut"** — echoes the assumed HEFA LCA input factor
      (1 − 0.2 = 80%); `scripts/test_emissions.py` prints 80.1% for
      Bio-SPK because the 0.2 factor is an input. Not a finding. Delete.
- [ ] **81.58 kN** — untraceable console printout; no Results table contains
      it. Delete.

### A3. t-test misuse (V14)
- [ ] Delete the t-test (t = 1.04, p = 0.3734) and all "not statistically
      distinguishable" language: 4 paired points with tuned parameters do not
      support a t-test, and failure to reject H₀ is not evidence of accuracy.
- [ ] Replace with per-mode % errors (from the calibration record) plus the
      held-out cross-engine MAPE distribution
      (`outputs/holdout_icao_validation_summary.csv`).

### A4. "Learnable parameter" narrative (V8/C4; Reviewer 2 point 2)
- [ ] Rewrite §2.1–2.2: remove all "learnable parameter" language.
      State precisely: η_c = 0.86 **fixed** (`integrated_engine.py:565`);
      η_poly = 0.9 **fixed**; η_comb was **fixed at 0.98** during all blend
      optimization; in LTO calibration η_comb was fit once by seeded Optuna
      (bounds 0.96–0.999) and converged to **0.9765**, with per-mode φ
      converging to Idle 0.295 / Approach 0.340 / Climb 0.457 / Take-off 0.507
      (`outputs/calibration_trent1000_ae3.json`). This answers Reviewer 2's
      "what did they converge to?" directly.
- [ ] Reference the parameter provenance table
      (`outputs/parameter_provenance.md`) — include as supplementary material.
- [ ] Disclose the two inert calibration parameters found during repair:
      the sampled "pressure_loss" was never consumed by any cycle equation,
      and the per-mode π-scales were written to a dict the compressor never
      reads (all LTO modes ran at rated OPR 43.2). Neither may be described
      as a model parameter.
- [ ] Disclose that the SAF combustor-efficiency penalties (−1.5% Bio-SPK,
      −1.0% HEFA) were unsourced and have been **removed**; per the ablation
      (`outputs/ablation_saf_penalty.csv`) they shifted TSFC by up to +0.50%
      and **flipped the TSFC fuel ranking** — any blend-sensitivity claims
      derived from them are withdrawn pending the Phase 2 re-run.

### A5. Calibration vs validation (V1; Reviewer 2 point 1)
- [ ] Replace every "validated" describing the LTO comparison with:
      "calibrated on the Trent 1000-AE3 certification record (UID 02P23RR126);
      validated on the 57 held-out Trent 1000 certification records (19 other
      model variants × 3 certifications, excluding AE3 re-certifications) with
      overall fuel-flow MAPE **7.27%** — Approach 2.1%, Idle 3.2%,
      Take-off 16.5%". Per the plan's decision rule (≲ 20% → genuine
      validation), this QUALIFIES as genuine validation and may be reported
      as such.
- [ ] Report the take-off error honestly as a **systematic overprediction**
      (all 57 held-out take-off points sit ~11–23% above the identity line,
      median +16.5%), not random scatter — consistent with the missing
      part-power/OPR model (the inert π-scales) and absent fan/bypass split;
      cite Phase 2 scope. Figures:
      `outputs/plots/holdout_pred_vs_icao_scatter.png`,
      `outputs/plots/holdout_mode_error_boxplot.png`.
- [ ] Correct the previously reported 11.3% figure: it was the in-sample fit
      residual of an unseeded calibration run. The seeded, reproducible
      calibration converges to **5.46%** in-sample mean absolute error — and
      being in-sample, neither number is validation evidence.
- [ ] Disclose that the ICAO CSV contains **no Climb-out rows** (Idle,
      Approach, Take-off only): the climb calibration target (2.050 kg/s) is
      not traceable to the shipped dataset, and climb is excluded from
      held-out validation.
- [ ] State explicitly that NOx (in-sample correlation), CO (single-anchor
      calibration), blend-discriminating outputs, and lifecycle CO₂
      (assumed point factors) are **NOT validated**.

## B. Factual corrections (Reviewer 1 + Reviewer 2 points 4a–d)

- [ ] **Abstract & throughout**: the compressor is NOT "modeled using
      Cantera" chemistry — Cantera supplies real-gas thermodynamic properties
      (entropy/enthalpy states) for an isentropic-efficiency compression;
      no kinetics are involved. Say so precisely. (Also fixed in
      `AGENT_CONTEXT.md`.)
- [ ] **Line ~58**: syngas = CO + H₂ (not CO₂).
- [ ] **Line ~179**: CO₂ reported in g/s (or EI g/kg fuel), never "ppm".
- [ ] **Lines 216–217 (HyChem scope)**: not "valid only for Jet A-1" —
      HyChem was developed on the NJFCP A-2 nominal fuel and has been applied
      across multiple NJFCP fuels; rewrite per Reviewer 1's wording.
- [ ] **Line 258 (compressor losses)**: lead with tip-leakage and secondary
      flow; remove stall as a "loss mechanism" (it is avoided by design, not
      a steady loss); blade friction is minor.
- [ ] **Lines 303–310 (combustor)**: acknowledge case/liner heat loss can
      exceed combustion-inefficiency loss; rename η as a lumped
      heat-delivery efficiency (full heat-loss treatment deferred to Phase 2,
      V9).
- [ ] **Line 318**: "turbine inefficiency," not "blade drag."
- [ ] **Lines 90–91**: strike.
- [ ] **Line 97**: make specific or cut.
- [ ] **Line 129**: fix the claim that genetic algorithms predate AI (GAs are
      a subfield of AI; they predate deep learning, not AI).
- [ ] **Line 131**: "combing" → "combining".
- [ ] **Lines 147–149**: cut or make concrete.
- [ ] **Equations**: rebuild ALL equations as native equation objects — the
      corruption is a Google-Docs→PDF export failure; after re-export, audit
      the PDF on a second machine before submission.
- [ ] **Logic flow chart** (Reviewer 1): add one figure mapping
      objectives → tools (Cantera / PINN / Optuna) → outputs.

## C. Consistency with repaired code (this repo, done)

- [x] Surrogate composition single source of truth = `simulation/fuels.py`
      (mole fractions): HEFA 85/15 dodecane/iso-octane, FT 50/35/15
      dodecane/decane/iso-octane, ATJ 80/20 iso-octane/dodecane. Git history
      proves these values (unchanged since 2025-12-10) generated all reported
      results; the manuscript table (HEFA 60/40, FT 45/35/20, ATJ 75/25
      "by mass") is wrong on both values and basis.
  - [ ] Manuscript: replace the composition table with the `fuels.py` values
        and state the **mole-fraction** basis once.
  - [ ] Manuscript: drop DCN values (uncomputed, uncited) unless a citation
        is added; report computed H/C instead: Jet-A1 2.167, HEFA 2.175,
        FT 2.187, ATJ 2.227 (`FuelSurrogate.h_over_c_ratio()`).
- [x] Stdout-scraper pipeline removed from
      `scripts/optimization/optimize_blend.py` (V17): objectives now read
      only the structured result dict; failed trials FAIL loudly (no
      T4 = 2000 K or penalty-default injection); 20-trial seeded smoke run
      verified objective values ≡ structured outputs × TIT penalty.
  - [ ] Response letter: note the transcription-integrity concern
        (Reviewer 2, "11.3% HEFA vs 11.3% MAPE") is addressed structurally —
        reported numbers can no longer originate from regex-parsed logs.
        Full 1000-trial re-run + figure regeneration is scheduled (Phase 2).
- [x] SAF efficiency penalties removed from
      `simulation/combustor/combustor.py::estimate_efficiency` (φ-parabola
      retained); ablation quantified in `outputs/ablation_saf_penalty.csv`.
- [x] `FuelSurrogate` default carbon fraction corrected 0.857 → 0.847
      (comment matched to arithmetic); attribute is inert (nothing consumes
      it — CO₂ uses the flat 3.16 kg/kg factor).

## D. Deferred to Phase 2 (do not promise as done in the response letter)

Fan/bypass module (V4), CORSIA LCA ranges + Monte Carlo (V6/V7),
chemistry-based NOx (V5), PINN-vs-analytic ablations + Sajben section
(V10/V11), variance decomposition & insensitivity reframing (V12/V18),
heat-loss sensitivity (V9), full 1000-trial optimization re-run + figure
regeneration (B1).
