# Number crosswalk — v4 (Phase 3–5 manifest) → v5 (Phase 6)

Every manuscript-bound number in the v4 manifest (commit `4308ffa`,
`outputs/ARTIFACT_MANIFEST.md` before regeneration) and its fate. New values
come from the regenerated manifest (`outputs/ARTIFACT_MANIFEST.md`, row ids
V/B/E/P), which reads them from the artifacts. v4 artifacts are archived in
`outputs/archive/pre_phase6/` (`MAPPING.json`). Older numbers (submitted
manuscript → v4) remain in `docs/number_crosswalk.md`.

## Formulation change behind every row

v4 ran each LTO mode at a calibrated φ; v5 solves φ so that static thrust equals
the ICAO thrust of the mode, and fuel flow becomes a model output (plan §3).
β (burner air fraction) is dropped (single-zone combustor); η_b and pressure
loss are no longer fitted (V7); the fitted set is W_ref, a, k_π, k_ṁ (V1, V2).

## Model and validation

| v4 row | v4 number | Fate | v5 value + row |
|---|---|---|---|
| M1 | η_comb 0.9963, p_loss 0.0442, k_π 0.5618, k_ṁ 0.6218, φ = {0.2953, 0.3053, 0.5409}; in-sample 1.62 % | **Replaced.** η_b and p_loss were unidentified; φ is no longer a parameter | W_ref 102.40 kg/s, a 1.109, k_π 1.348, k_ṁ 0.414; in-sample 1.80 % (V1); all four identified (V2); η_b per mode and p_loss fixed with cited ranges (V7) |
| M2 | Held-out MAPE **2.50 %** (TO 1.81 / APP 1.09 / IDLE 4.60) | **Withdrawn** — finding F-A: a rated-thrust rescaling rule, not a model test (V4) | Held-out model 1.82 % vs B0 2.19 % vs B1 1.08 % (V3); **no demonstrated skill over rated-thrust rescaling** (A2 fail); A3 fail at approach |
| M3 | Validation history v1 7.27 / v2 14.64 / v3 12.71 / v4 2.50 % | **Withdrawn** — every version used the same rescaling design | V3 is the only held-out fuel-flow result; history kept in the archive |
| M4 | Take-off: thrust 241.6 kN (core 55.4 + bypass 186.2), TSFC 9.59, specific thrust 299, T3 901.5 K, T4 1894.9 K, T5 1174.4 K, fuel flow 2.318 vs 2.327 kg/s | **Replaced.** v4 missed ICAO take-off thrust by 17 % (52.8 kN after the accounting repair) | Thrust matched at 310.9 kN (core 72.3 + bypass 238.6), TSFC 7.70, specific thrust 300.6, T3 901.5 K, T4 1718.5 K, T5 983.2 K, fuel flow 2.395 vs 2.327 kg/s (+2.9 %, in-sample) (V5). The 17 % gap is now the fitted airflow W_ref = 1.28 × the hand-set 79.9 kg/s (V1) |
| M5 | Airflow-split (β) sweep | **Retired** — β dropped (no citable range) | — |
| M6 | Turbine p5 adjudication | **Kept** as decision evidence | E11 |
| M7 | NOx LOEO MAPE 4.17 % | **Kept** as supplementary | V6b (4.17 %); primary NOx held-out now on the P6.1 split: 3.77 % vs naive 10.60 % (V6) |
| — | — | **New** | Fixed-parameter bands: held-out MAPE 1.82–1.91 %, TO TSFC 7.64–7.75, T4 1667–1795 K; rated FPR is the limiting assumption (V8, `parameter_provenance.md`) |

## Blend studies

| v4 row | v4 number | Fate | v5 value + row |
|---|---|---|---|
| S1 | Free-φ study, 642 Pareto members | **Withdrawn** — blends compared at unequal thrust (φ a design variable) | B1: blends at matched thrust |
| S2 | φ-frozen: blends move TSFC 0.25 %, NOx 0.48 %; lifecycle spans 89.8 % | **Replaced** — fixed φ is still unequal thrust | At matched thrust: fuel flow/TSFC ≤ 0.16 % (ATJ-50), T4 < 0.5 K; below the V8 bands, so **no cycle-quantity ranking**; lifecycle CO₂e −34 to −46 % vs Jet-A1 (claimed) (B1) |
| S3 | φ ≈ 100 % of TSFC/NOx variance; lifecycle φ 68.5 % / CORSIA 29.3 % | **Replaced** (φ is no longer free) | Cycle outputs vary ≤ 0.26 % across 256 blends; lifecycle varies 47 %, blend fractions 97 % / CORSIA draw 3 % of explained variance (B2) |
| S4 | Objective correlations | **Replaced** | `objective_correlation_v5.csv` (B2) |
| S5 | CORSIA rank stability **96.7 %** | **Replaced** | 80 % of 5 central Pareto members stable in ≥ 50 % of draws (B3) |
| S6 | Representative solution Trial 900 (TSFC 8.14, SAF 49.4 %, φ 0.4155 …) | **Withdrawn** — depended on free φ; v5 reports blend differences against the ranking rule instead of one representative point | B1 |
| S7 | Pareto-membership diff vs Phase 2 | **Retired** with S1 | — |

## Evidence

| v4 row | v4 content | Fate | v5 |
|---|---|---|---|
| E3 | NOx three-path spread (v3 states; "upper-bound proxies") | **Replaced** at v5 states; the chemistry paths are now **low-side** proxies (single zone at overall φ) | E3: TO correlation 49.3, Zeldovich 0.03, HyChem-A2 0.09 g/kg vs certification 48.25 |
| E4 | ξ = 4 % > blend signal (fixed φ: fuel flow could not respond) | **Replaced** at matched thrust | E4: ξ = 4 % → +0.327 mg/(N·s) TSFC, 13× the neat-fuel spread |
| E1, E2, E5–E8 | — | **Kept** | same ids |
| — | — | **New** | E9 mechanism/surrogate sensitivity (+2.55 / +3.42 % fuel for HyChem A1/A2 at fixed calibration, mostly heating value); E10 computed surrogate LHVs; E11 turbine adjudication |
| P1–P8 | PINN record | **Kept** (closed, P6.5) | same ids |
