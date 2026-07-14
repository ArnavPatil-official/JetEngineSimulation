# Manuscript Edits — Phase 2 Checklist

Line-by-line edit checklist consuming the Phase 2 outputs (see `docs/plan.md`,
executed 2026-07-13/14). Supporting artifacts referenced inline; every number
here is traceable to a CSV/JSON in `outputs/`.

Status legend: [ ] = pending manuscript edit (Google Docs source, outside this
repo); [x] = repo-side artifact complete.

## A. §2 Architecture rewrite (P2.1, P2.2)

- [ ] Describe the two-stream engine: 0-D fan (FPR 1.45, η_fan 0.90 —
      design-class textbook values, not Trent measurements), bypass stream
      BPR 9.1 (ICAO CSV), turbine work = compressor + fan, thrust = core
      nozzle + bypass nozzle, TSFC/specific thrust on total thrust and total
      airflow. Remove all "core-only" caveats that no longer apply; keep
      `thrust_core` in tables for Phase 1 comparability.
- [ ] Describe the part-power throttle model (π_c = 1+(π_rated−1)x^k_pi,
      ṁ = ṁ_rated x^k_mdot, x = ICAO power setting). State plainly:
      k_mdot = 0.697 is calibrated; **k_pi is not identifiable from fuel flow**
      (FAR depends on φ only) and functions as an assumption that matters for
      NOx (via OPR) and T3.
- [ ] State the combustor pressure loss is now physically applied
      (p_comb = p3(1−p_loss), p_loss = 0.034 calibrated).
- [ ] **Disclose the argon-compression defect and its fix**: all temperatures
      in the previous submission were computed for argon (T3 1464 K vs 901 K
      at rated OPR); fuel-flow results were unaffected; every temperature,
      thrust, and TSFC in the revision comes from the corrected air cycle.
      This is a required integrity disclosure, not optional.
- [ ] Replace all performance tables with regenerated numbers (take-off
      design point, v3 calibration): thrust 250.7 kN (core 64.5 + bypass
      186.2), TSFC 11.12 mg/(N·s), specific thrust 311 N·s/kg at static SL,
      T3 901 K, T4 2031 K. Note TSFC now sits in the published
      Trent-1000-class band (~8–11 mg/(N·s) static SL) instead of the
      turbojet-magnitude 29.46.
- [ ] Validation numbers: calibrated on Trent 1000-AE3 (UID 02P23RR126,
      3 traceable LTO points; Climb absent from the databank); held-out
      cross-engine fuel-flow MAPE **12.71%** (57 records excl. AE3
      re-certifications; Idle 6.5%, Approach 12.0%, Take-off 19.7%).
      State explicitly that the Phase 1 free-scale model achieved 7.27% and
      the physically-constrained model is *worse on fuel flow* — constraint
      cost reported, not hidden — and that the take-off systematic
      overprediction was NOT explained by part-power OPR (hypothesis
      falsified).

## B. §2.6 Lifecycle emissions rewrite (P2.3)

- [ ] Replace the LCA table with the CORSIA-cited one from
      `data/corsia_lca_values.yaml`: pathway × feedstock × core LCA × ILUC ×
      L_CEF, each citing its ICAO Document 06 (8th ed., November 2025) table
      row (e.g., HEFA-UCO 13.9 = Table 2 row 2.6; FT-MSW 5.2 = Table 1 row
      1.3; ATJ corn grain 55.8 + ILUC 25.6 = Table 3 row 3.4 + Table 9 row
      9.15). Fossil baseline 89 gCO₂e/MJ.
- [ ] State the Monte Carlo method: triangular(min, mode, max) per pathway,
      seeded draws recorded per trial; common-scenario draws for
      rank-stability.
- [ ] Define combustion CO₂ (EI = 3.664·w_C, chemistry-based; 3.100 kg/kg for
      the Jet-A1 surrogate, −1.9% vs the old flat 3.16) and lifecycle CO₂e
      (L_CEF·LHV·ṁ_f) as SEPARATE quantities; never on one axis. The old
      "0.762 blend LCA factor" becomes a scenario band.
- [ ] Delete every use of the retired point factors {1.0, 0.2, 0.1, 0.3} and
      any "80% CO₂ cut" phrasing (input echo).

## C. NOx framing (P2.4)

- [ ] Rename the NOx model everywhere: "ICAO-derived correlation
      (fuel-flow/OPR proxy, not chemistry)". Footnote in EVERY NOx figure and
      table.
- [ ] Add the three-path comparison (`outputs/nox_dual_path.csv`, figure
      `outputs/plots/nox_path_comparison.png`): at idle/approach the thermal-
      chemistry paths give ~0 vs certification 5.5/13.7 g/kg; at take-off
      Zeldovich-on-CRECK gives 11.3 and the HyChem-A2 kinetic anchor 8.6 vs
      certification 48.3 (single lean zone omits the near-stoichiometric
      primary zone). Path spread ≫ any blend-to-blend NOx difference ⇒ the
      manuscript may NOT rank blends by NOx. State this as a scoping
      limitation, and describe the A-2 anchor method (equilibrium products,
      N-oxides zeroed, τ-integrated full N-chemistry).
- [ ] Document the residence-time assumption (V = 0.207 m² × 0.5 m, τ ≈ 5–9 ms).

## D. New sections (P2.5)

- [ ] §3.x "Nozzle surrogate vs Sajben diffuser": candid negative result.
      Both checkpoints fail quantitative validation (wall-Cp shape-L2
      0.71–1.09 against the <0.10 threshold; the fine-tuned checkpoint is
      *worse* than the base). Figure
      `outputs/plots/sajben_wall_cp_validation.png`, table
      `outputs/sajben_validation_errors.csv`. Frame as: the LE-PINN is a
      physics-consistency architecture demonstration; 2-D retraining is
      future work. Do NOT present Sajben as successful external validation.
- [ ] §3.y "Component ablations": from
      `outputs/ablation_pinn_components.csv` — swapping turbine PINN→analytic
      changes total thrust −10%, nozzle PINN→analytic +10% (core-stream
      deviations to 37%), with partial cancellation in the production
      combination (−1.6%). T5 and fuel flow are identical by construction
      (work-matched). Claim the PINNs as physics-consistency surrogate
      layers, not accuracy contributors, and disclose the ±10% model-choice
      sensitivity on thrust-derived quantities.
- [ ] Physics-consistency table: publish the suite result (80 passed,
      1 skipped — the skip is a benchmark comparison test, named).

## E. Heat loss (P2.6) — answers Reviewer 1 lines 303–310

- [ ] Add the quantified paragraph: with ξ = 4% of heat release lost through
      the case/liner, TSFC shifts by 0.059 mg/(N·s) — more than the entire
      blend-to-blend TSFC spread (0.052) — and T4 drops 45 K
      (`outputs/heat_loss_sensitivity.csv`, figure
      `outputs/plots/heat_loss_sensitivity.png`). Therefore heat-loss
      treatment bounds the resolvable blend effect size; η_comb is renamed
      "lumped heat-delivery efficiency" throughout. Production results use
      ξ = 0 (recorded choice).

## F. Results regeneration + variance decomposition (P2.7)

- [ ] All results figures regenerated from the final seeded study
      (`outputs/results/optimization_results.csv`, seed 42, N=1000, v3
      calibration, corrected air cycle): Pareto 3-D, parallel coordinates,
      correlation heatmap, Pareto-with-CORSIA-bands, variance-decomposition
      figure, holdout scatter (v3).
- [ ] Report the φ-frozen companion study
      (`outputs/results/optimization_results_phi_frozen.csv`) separating
      operating-point effects from fuel effects.
- [ ] Variance decomposition (`outputs/results/variance_decomposition.csv`) —
      the quantitative backbone of the reframed conclusion:
      * Free-φ study (N=1000, 0 failed/pruned): φ explains ~100% of TSFC,
        specific-thrust, and NOx variance; **blend fractions explain ≈ 0%**.
        Lifecycle CO₂e: φ 75.7% (through fuel flow), CORSIA draw 22.8%,
        blend fractions ~1.6%.
      * φ-frozen study (AB8, N=1000, 0 failed/pruned): with the operating
        point fixed, blends move TSFC by **0.25%** (12.09–12.12 mg/(N·s)),
        specific thrust 0.03%, NOx 0.49% — while lifecycle CO₂e spans
        **84%** (6477–11910 g/s). The blend TSFC spread is SMALLER than the
        quantified model uncertainties (heat-loss ξ=4%: 0.5%; PINN choice:
        ~10% on thrust; composition-set ambiguity ~0.1%), so
        **performance-neutrality within model resolution is the finding**,
        stated with these numbers (Abstract + §4 rewrite).
      * Caveat to state: 664/1000 free-φ trials are 4-objective
        non-dominated — the free-φ "Pareto front" is weakly discriminating;
        blend selection is meaningful only in the φ-frozen framing (9-member
        front spanning the FT-heavy ↔ ATJ-heavy lifecycle trade).
- [ ] CORSIA rank stability (`outputs/results/lca_rank_stability.csv`,
      1000 seeded common-scenario draws): 99.4% of baseline Pareto members
      remain Pareto-optimal in ≥50% of draws (median persistence 1.00) ⇒ per
      decision rule AB7, blend-selection statements may be made with scenario
      bands (figure `outputs/plots/pareto_lca_bands.png`) rather than being
      fully scenario-conditional.
- [ ] New representative balanced solution (free-φ Pareto member closest to
      the normalized ideal point) with full provenance: **Trial 331, seed
      42** — TSFC 8.32 mg/(N·s), specific thrust 298.6 N·s/kg, lifecycle
      CO₂e 4701 g/s, NOx (correlation) 83.1 g/s at φ=0.374, SAF 48.1%
      (HEFA 19.9% / FT 23.3% / ATJ 5.0%), CORSIA draw recorded in-CSV
      (L_CEF: HEFA 18.24, FT 1.08, ATJ 64.16 gCO₂e/MJ; blend 53.3), T4
      1750 K. Replaces the old "29.6% SAF, TSFC 29.46, 0.762" solution.
- [ ] Updated Highlights: only numbers verbatim from the P2.7 results set.
- [ ] Cross-check before submission: every manuscript-bound number exists in
      a CSV under `outputs/results/` (no printout-only numbers).

## G. Response-letter provenance carryovers

- [ ] Inert parameters (p_loss, π-scales) — now wired or deleted; k_pi
      unidentifiability disclosed.
- [ ] 3.16 → 3.664·w_C EI-CO₂ (−1.9%) with equilibrium cross-check.
- [ ] Take-off bias before/after (16.5% → 19.7%): hypothesis falsified,
      reported as such.
- [ ] Argon-compression defect disclosure (see §A).
- [ ] Updated `outputs/parameter_provenance.md` (Phase 2 section) shipped as
      supplementary material.
