# Number Crosswalk — old manuscript → revision (Phase 3.5)

Every number in the submitted manuscript, and its fate. "Manifest Mx/Sx/Ex"
refers to `outputs/ARTIFACT_MANIFEST.md` rows. Old values are taken from the
submitted PDF as inventoried in `MANUSCRIPT_REPAIR_PLAN.md` and from the
archived artifacts (`outputs/archive/pre_phase2/results/pareto_optimal_solutions.csv`,
whose first row is the old representative solution).

**Cross-check status:** no `manuscript.txt` extraction exists in this repo;
this crosswalk covers every number inventoried by the audit documents. Final
step for the Google-Docs edit: grep the manuscript export against column 1
of this table and confirm zero orphans (user-side step; flagged, not done).

## A. Performance numbers

| Old number | Where it was | Fate | New value + source |
|---|---|---|---|
| TSFC 29.46 mg/(N·s) | representative solution, Results | **Replaced.** Old value was turbojet-magnitude (no bypass stream) computed on the argon cycle with PINN components | Representative solution TSFC **8.14 mg/(N·s)** (manifest S6); take-off design point **9.59 mg/(N·s)** (M4) |
| Specific thrust 800.6 N·s/kg | representative solution | **Replaced.** Old value divided by core airflow only | **270.9 N·s/kg** on total airflow (S6); design point 299 (M4) |
| 81.58 kN thrust | Highlights | **Removed** — untraceable printout (Phase 1 finding). Thrust now reported per design-point table | 241.6 kN take-off, core 55.4 + bypass 186.2 (M4) |
| T3/T4 temperatures (implied ~1464/2500 K class) | §2/Results | **Replaced — argon defect disclosure.** All old temperatures computed for argon working fluid | T3 901.5 K, T4 1894.9 K, T5 1174.4 K (M4); disclosure in response letter preamble |
| "0.762 blend LCA factor" | representative solution | **Removed concept.** Point LCA factors retired; lifecycle is a CORSIA scenario quantity | Representative lifecycle CO₂e 4177 g/s under recorded draw; P5–P95 bands (S5, S6) |
| NOx 87.65 g/kg (old representative) | Results | **Replaced + relabeled.** NOx is the ICAO-derived correlation (proxy, not chemistry); blend NOx ranking withdrawn | Representative NOx(corr) 65.9 g/s (S6); path-spread evidence E3 |
| SAF 29.6% (11.3 H / 9.5 F / 8.8 A) representative blend | Results | **Replaced** | SAF 49.4% (HEFA 8.6 / FT 36.3 / ATJ 4.5), Trial 900, seed 42, draw recorded (S6) |

## B. Validation numbers

| Old number | Where it was | Fate | New value + source |
|---|---|---|---|
| 11.3% MAPE "validation" (4 LTO points, Trent XWB-84EP) | Abstract, §2.5, §3 | **Removed as stated.** It was an in-sample fit residual on misattributed engine identity; also unseeded (a seeded rerun of the same procedure gave 5.46%) | Calibration: Trent 1000-AE3 (UID 02P23RR126), in-sample 1.62% (M1). Validation: held-out MAPE **2.50%** across 57 records (M2). History v1→v4 reported (M3) |
| t = 1.04, p = 0.3734 ("not statistically distinguishable") | §3 | **Removed** — inappropriate test; figure `statistical_tests.png` archived | Per-mode MAPE distribution + scatter/boxplot figures (M2) |
| Climb fuel-flow point 2.050 kg/s | calibration/validation | **Removed** — not present in the ICAO databank record (no CLIMB rows) | 3-mode calibration, documented (M1) |
| "Trent XWB-84EP" (3 occurrences) | Abstract/§2.5/§3 | **Replaced** by Trent 1000-AE3 + UID + data-availability note | M1/M2 |

## C. Highlights (all four)

| Old | Fate |
|---|---|
| R² = 0.9969 | **Removed** — in-sample NOx-correlation fit statistic printed at startup, never a result |
| 0.00% continuity error | **Removed** — satisfied by construction (u = ṁ/ρA) |
| "80% CO₂ cut" | **Removed** — echo of the assumed HEFA LCA input (1 − 0.2); lifecycle is now a CORSIA band |
| 81.58 kN | **Removed** — untraceable printout (see A) |
| (New Highlights) | Drawn verbatim from manifest rows M2, M4, S2, S6 only |

## D. Emissions & fuels

| Old number | Fate | New value + source |
|---|---|---|
| 3.16 kg CO₂/kg fuel (flat) | **Replaced** by chemistry EI = 3.664·w_C | 3.100 kg/kg (Jet-A1 surrogate), −1.9%; Cantera cross-check 3.098 (E2) |
| LCA factors {1.0, 0.2, 0.1, 0.3} | **Removed** — unsourced, near best-case | CORSIA L_CEF per feedstock, cited to ICAO Doc 06 Nov-2025 table rows; triangular ranges HEFA 13.9/28.6/74.0, FT −3.4/7.7/21.1, ATJ 18.2/29.3/81.4 gCO₂e/MJ (E1) |
| Surrogate table (HEFA 60/40, FT 45/35/20, ATJ 75/25 "by mass") | **Replaced** — never generated any result (git history) | Mole-fraction table: HEFA 85/15, FT 50/35/15, ATJ 80/20; computed H/C 2.167–2.227 (provenance E8); DCN values dropped (uncited) |
| SAF efficiency penalties (−1.5%/−1.0%) | **Removed** — unsourced; created the reported blend sensitivity (flipped TSFC ranking) | Ablation E7; blend-sensitivity claims withdrawn |
| "not statistically distinguishable" blend claims | **Replaced** by the variance-decomposition finding | Blends: TSFC 0.25% / spec thrust 0.03% / NOx 0.48% at fixed φ, below model uncertainty; lifecycle CO₂e spans 89.8% (S2, S3) |

## E. Model-description claims

| Old claim | Fate |
|---|---|
| Compressor "modeled using Cantera" (chemistry) | **Rephrased**: Cantera thermodynamic properties for isentropic-efficiency compression of air; no kinetics |
| "Learnable parameters" η_c, η_CMB | **Removed.** Fixed vs calibrated split with converged values (M1) and the k_π-unidentifiability caveat |
| "Digital twin", "validated model" | **Rephrased** per response letter: reduced-order screening framework; calibrated + held-out-validated for LTO fuel flow only |
| Syngas = CO₂ (line ~58) | **Corrected**: CO + H₂ |
| CO₂ in ppm (line ~179) | **Corrected**: g/s and EI g/kg |
| HyChem "valid only for Jet A-1" | **Corrected**: developed on NJFCP A-2, applied across NJFCP fuels |
| Stall as compressor "loss mechanism" (line 258) | **Corrected**: tip leakage / secondary flow lead; stall avoided by design |
| "Blade drag" (line 318) | **Corrected**: turbine inefficiency |
| PINN turbine/nozzle as production models | **Reframed**: architecture demonstration; production numbers analytic (M6, E5, E6); PINN retraining = future work |
