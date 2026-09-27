# Response Letter — Draft Skeleton (Phase 3.5)

Point-by-point reply to both reviewers. Every number cites
`docs/number_crosswalk.md` (old → new mapping) and
`outputs/ARTIFACT_MANIFEST.md` (each new number's generating script, seed,
and commit). Text in [brackets] is where the final Google-Docs edit inserts
manuscript section/line references.

> **SUPERSEDED IN PART BY PHASE 6 (2026-09-26) — do not send as written.**
> The Reviewer 2 point 1, point 2 and point 4d sections below rest on v4
> results that Phase 6 withdrew or replaced (marked inline). Take every number
> from the regenerated `outputs/ARTIFACT_MANIFEST.md` (row ids below) and the
> v4 → v5 mapping in `docs/number_crosswalk_v5.md`; the reply to each reviewer
> item is mapped in `docs/manuscript_checklist_final.md`. The other sections
> still stand as drafted.

---

## Preamble — voluntary disclosures

We made four disclosures the reviewers did not ask for, because they changed
reported numbers. We believe stating them plainly is the fastest route to a
trustworthy revision.

1. **Engine misattribution.** The submitted manuscript attributed the ICAO
   validation data to the Trent XWB-84EP. The data are, and always were, the
   Rolls-Royce **Trent 1000-AE3** (ICAO UID 02P23RR126); no XWB record
   appears in the databank extract we use. Every mention is corrected, and a
   data-availability note now identifies the exact certification record.
2. **Working-fluid defect.** During revision we discovered the compressor
   had been computing its state for **pure argon** (an uninitialized
   Cantera object defaulting to the mechanism's first species) rather than
   air. Compressor exit temperature was ~500 K too high at rated OPR.
   Fuel-flow results were unaffected (the fuel-air ratio was computed on a
   separately initialized gas), but every temperature, thrust, and TSFC in
   the original submission was invalid. All such numbers in the revision
   come from the corrected air cycle, and the defect plus its fix is stated
   in [§2.x].
3. **Nozzle surrogate external validation is negative.** Validating the
   LE-PINN nozzle against the Sajben transonic-diffuser experiment produces
   wall-pressure shape-L2 errors of 0.71–1.09 against a <0.10 good-match
   gate. We report this as a negative result; the PINN components are
   presented as a physics-consistency architecture demonstration, not as
   validated accuracy contributors, and the production results use the
   analytic components (see reply to R2-4 below).
4. **Highlights provenance.** Four Highlights numbers in the submission
   were not traceable to Results (R² = 0.9969 was an in-sample fit
   statistic; the 0.00% continuity error is satisfied by construction;
   "80% CO₂ cut" restated an input assumption; 81.58 kN was a console
   printout). All are removed; the revised Highlights contain only numbers
   present verbatim in Results (crosswalk rows H1–H4).

---

## Reviewer 2, point 1 — "validation" vs calibration

> **SUPERSEDED (Phase 6).** The 2.50 % "held-out validation" below is
> **withdrawn**: it was a rated-thrust rescaling rule, not a test of the cycle
> (finding F-A, manifest V4). The replacement is the thrust-matched held-out
> test against two naive baselines (V3): the model beats constant TSFC but
> **does not beat rated-thrust rescaling**, and the TSFC–OPR sign check fails
> at approach. β = 0.8 is dropped (single-zone combustor, V7). Calibration:
> V1; identifiability: V2.

**Was:** 11.3% MAPE on 4 LTO points presented as validation, with a t-test
(t = 1.04, p = 0.3734) supporting "not statistically distinguishable."

**Now:** The 11.3% was an in-sample calibration residual (the same 4 points
tuned the parameters), and the t-test was statistically inappropriate; both
are removed. The revision separates:

- **Calibration:** one seeded Optuna fit against the 3 LTO points of the
  Trent 1000-AE3 certification record that are traceable to the databank
  (the previously used climb value, 2.050 kg/s, does not appear in the
  record and is dropped). In-sample error 1.62%.
- **Validation:** the frozen model predicts fuel flow for the **57 held-out
  Trent 1000 certification records** (19 other variants) — overall MAPE
  **2.50%** (take-off 1.81%, approach 1.09%, idle 4.60%)
  [manifest: holdout_icao_validation_v4].
- **Scope:** NOx, CO, lifecycle CO₂e, and blend-discriminating outputs
  remain unvalidated and are labeled accordingly throughout.

The validation-model history is reported, not hidden: a free-parameter
Phase-1 variant achieved 7.27% with two inert parameters; constraining the
throttle physics initially *worsened* generalization to 12.71%; adding the
missing combustor airflow split (β = 0.8 of core air burned, the remainder
liner-cooling/dilution air — Lefebvre & Ballal) resolved a systematic
take-off overprediction (+19.7% → +1.8%) and produced the 2.50% figure.
Hypotheses that failed along the way (part-power OPR as the bias cause) are
stated as falsified in [§3.x].

## Reviewer 2, point 2 — "learnable parameters": what did they converge to?

> **SUPERSEDED (Phase 6).** The fitted set is now W_ref, a, k_π, k_ṁ, all
> identified (V1, V2); η_b and pressure loss are fixed with cited ranges, β is
> dropped, and every fixed value carries a cited range and a band (V7, V8;
> `outputs/parameter_provenance.md` Phase 6 P6.2). φ is solved from thrust, not
> fitted. The v4 values quoted below are archived.

**Was:** η_c and η_CMB described as learnable, with no converged values.

**Now:** Nothing in the model is "learnable" in the ML sense; the language
is removed. Fixed: η_c = 0.86, η_poly = 0.9, FPR = 1.45, η_fan = 0.90,
BPR = 9.1, β = 0.8 (each with source status in the provenance table,
supplementary). Calibrated once (seed 42, converged values): η_comb =
0.9963, combustor pressure loss = 0.0442, throttle exponents k_π = 0.562
and k_ṁ = 0.622, φ = {0.295, 0.305, 0.541} per mode. One honest caveat is
stated: k_π is not identifiable from fuel-flow data (the fuel-air ratio
depends on φ only) and functions as an assumption that matters for OPR-
dependent quantities.

## Reviewer 2, point 4a — compressor "modeled using Cantera"

Corrected: Cantera supplies real-gas thermodynamic properties for an
isentropic-efficiency compression of air; no chemical kinetics are involved
in the compressor. (Abstract and [§2.1] rephrased.)

## Reviewer 2, point 4b — SAF efficiency penalties

The −1.5%/−1.0% name-triggered combustor-efficiency penalties were
unsourced and have been removed. Ablation showed they *created* the blend
sensitivity the manuscript reported — including flipping the TSFC fuel
ranking [manifest: ablation_saf_penalty]. Blend-sensitivity claims built on
them are withdrawn; the revised claim is the opposite and is now the paper's
central finding (see R2-4d).

## Reviewer 2, point 4c — surrogate compositions

The code's mole-fraction surrogates (HEFA 85/15 n-dodecane/iso-octane,
FT 50/35/15, ATJ 80/20) generated all results (verified by git history);
the manuscript table was wrong on both values and basis and is replaced.
H/C ratios are now computed from composition (2.167–2.227); uncited DCN
values are dropped.

## Reviewer 2, point 4d — do the blends actually differ?

> **SUPERSEDED (Phase 6).** The comparison below was at fixed φ (unequal
> thrust). Replace with the matched-thrust results: B1 (blends move fuel flow
> and TSFC by at most 0.16 %, below the V8 bands, so no performance ranking;
> each 50 % SAF blend is lower than Jet-A1 on lifecycle CO₂e), B2 (variance
> shares), B3 (CORSIA rank stability), E4 (heat loss), E3 (NOx paths at v5
> states, now low-side proxies).

The revision reframes this honestly with a variance decomposition
[manifest: variance_decomposition]: at a fixed operating point, blend
composition moves TSFC by 0.25%, specific thrust by
0.03%, and correlation-NOx by 0.48% — below the model's
own quantified uncertainties (heat-loss treatment: 0.5% TSFC; component
model choice: ~10% thrust) — while lifecycle CO₂e spans ~90% across CORSIA
feedstock scenarios. **Performance-neutrality within model resolution,
combined with large lifecycle differences, is the finding.** Blend NOx
rankings are not claimed: the spread between NOx model paths (ICAO-derived
correlation vs thermal-chemistry paths) exceeds any blend difference
[manifest: nox_dual_path].

## Reviewer 1 — line-by-line corrections

All accepted and applied (crosswalk section C for each):
syngas = CO + H₂ [line ~58]; CO₂ in g/s / EI, never ppm [line ~179];
HyChem developed on NJFCP A-2 and applied across NJFCP fuels [lines
216–217]; compressor losses led by tip leakage/secondary flow, stall
removed as a "loss mechanism" [line 258]; combustor case/liner heat loss
acknowledged and *quantified* — a 4% heat-loss fraction moves TSFC by more
than the entire blend-to-blend spread, so η_comb is renamed a lumped
heat-delivery efficiency and heat loss is stated to bound the resolvable
blend effect size [lines 303–310; manifest: heat_loss_sensitivity];
"turbine inefficiency" replaces "blade drag" [line 318]; lines 90–91
struck; line 97 made specific; "combing"→"combining" [131]; GA/AI claim
corrected [129]; lines 147–149 cut; all equations rebuilt as native
objects and the re-exported PDF audited on a second machine; the requested
objectives→tools→outputs flow chart added as [Fig. 1].

## Reviewer 1 — combustor heat loss (substantive)

Beyond the wording fix: the revision adds an explicit heat-loss hook
(ξ ∈ [0, 6%] of heat release) and reports the sweep. ξ = 4% shifts TSFC by
0.059 mg/(N·s) versus a total blend spread of 0.052 mg/(N·s), and T4 by
45 K. Production results use ξ = 0 with this sensitivity disclosed
[manifest: heat_loss_sensitivity].

---

## Note on component models (anticipating the obvious question)

The turbine/nozzle PINN surrogates are retained in the architecture and
ablated, but the production numbers use the analytic components. Reason,
in the order the evidence arrived: (1) the LE-PINN nozzle failed external
validation (Sajben); (2) the turbine PINN's exit pressure is a raw network
output that deviates −41.5% from the hand-verified work-consistent
polytropic value [manifest: turbine_p5_adjudication]; (3) swapping either
component moves thrust ~±10%, which we report as model-choice sensitivity.
Retraining the surrogates on corrected-cycle data is future work.
