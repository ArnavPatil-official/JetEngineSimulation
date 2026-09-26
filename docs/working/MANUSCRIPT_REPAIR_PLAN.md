# SAF Turbofan Project — Technical Repair Plan
**Prepared 2026-07-05. Basis: Patil_Manuscript.pdf (30 pp), two reviewer reports, SAF_Turbofan_Outreach_Emails (37 pp), and direct inspection of the JetEngineSimulation repository (code, data, models, tests).**

This document is blunt by request. Everything below is grounded in specific files. Where the code contradicts the manuscript, the code is quoted, because reviewers of a resubmission will run it.

---

## DELIVERABLE 1 — Source inventory and evidence map

### 1.1 Materials inspected

**Manuscript (Patil_Manuscript.pdf):**
- Abstract/§1 Introduction (pp. 1–8): claims "turbofan digital twin," compressor+combustor "modeled using Cantera," PINN turbine/nozzle, Bayesian 4-objective optimization, ICAO validation MAPE 11.3% against "Trent XWB-84EP."
- §2 Methodology (pp. 8–17): surrogate definitions (HEFA 60/40 dodecane/iso-octane; FT 45/35/20; ATJ 75/25), CRECK C1–C16 single mechanism, HyChem for validation only, Eqs. 3–5 compressor, Eqs. 6–8 combustor with "learnable" N_CMB, Eqs. 9–13 turbine PINN with loss K(x) and work-balance loss, Eqs. 14–18 nozzle, §2.5 ICAO validation approach, §2.6 optimization with LCA factors f = {1.0, 0.2, 0.1, 0.3}.
- §3 Results (pp. 17–21): Fig. 1 LTO validation (MAPE 11.3%, idle −14.4%, takeoff +26.6%, t=1.04/p=0.3734), Fig. 2 nozzle profiles, Figs. 3–5 Pareto/LCA plots; balanced solution 29.6% SAF, TSFC 29.46 mg/(N·s), specific thrust 800.6 N·s/kg, NOx 87.65 g/kg, LCA 0.762.
- §4 Discussion (pp. 21–27): concedes n=4, concedes near-identical thrust/TSFC across blends, concedes LCA factors are "normalized scenario assumptions" — then still concludes "meaningful lifecycle carbon reductions."
- Note: the Highlights that Reviewer 2 quotes (R²=0.9969 NOx, 81.58 kN, 80% CO₂ cut, 0.00% mass continuity) do **not appear anywhere in the PDF body**. Their provenance is traceable to code artifacts (see evidence map rows H1–H4).

**Reviewer concerns:** R1: corrupted equations, "compressor modeled in Cantera" inaccuracy, HyChem/NJFCP factual error, wrong compressor-loss taxonomy (tip leakage missing), combustor case heat loss > inefficiency loss, "blade drag" wording, no coherent objective→tool logic chart. R2: Highlights contradict body; n=4 t-test is not validation; unsourced surrogates and LCA factors; syngas = CO+H₂ not CO₂+H₂; CO₂ "ppm" error; learnable efficiencies unreported and able to absorb error; 11.3% HEFA vs 11.3% MAPE coincidence; model predicts no blend sensitivity yet ranks blends.

**Repository / database (ground truth):**
- `data/icao_engine_data.csv`: **180 rows = 45 Trent 1000 variants × 4 LTO modes** (fuel flow, HC, CO, NOx, smoke, OPR, BPR, rated thrust). **No Trent XWB of any kind is present** (`grep XWB` → 0 hits). `data/raw/ICAO_RR_TRENT_1000/` confirms the source.
- `scripts/optimization/calibrate_lto.py`: Optuna **tunes** η_comb (0.96–0.999), pressure loss, and four per-mode equivalence ratios (φ_idle…φ_takeoff), with hand-set per-mode airflow and pressure-ratio scale factors (0.15/0.35/0.85/1.0), to minimize error against the four ICAO fuel-flow targets — which are labeled `# Targets: Trent 1000-AE3` and match the AE3 rows in the CSV exactly (0.244 / 0.643 / 2.050 / 2.327 kg/s).
- `integrated_engine.py` (1,583 lines): η_c **fixed at 0.86**; combustor efficiency either fixed (0.98 in optimization) or from `Combustor.estimate_efficiency()` — a hard-coded parabola in φ with **unsourced SAF penalties (−1.5% "pure synthetic", −1.0% blended HEFA)**. NOx = `EmissionsEstimator._fit_nox_model()`: a log-log regression NOx = A·OPR^B·ṁ_f^C **fitted to the 180 ICAO rows**. CO₂ = `3.16 × lca_factor × ṁ_fuel × 1000` (g/s). `bypass_ratio: 9.1` is defined in `design_point` and **never used** — the cycle is core-only; no fan, no bypass thrust, turbine work target = compressor work only. Turbine "PINN" call path adjusts outlet T to match target work (`run_turbine`, `scale_turbine_exit_temp`, η_poly=0.9). Optimizer (`optimize_blend.py`) scrapes results by **regex on printed stdout** with a silent fallback T4=2000 K.
- `simulation/fuels.py`: surrogates **differ from the manuscript** — HEFA 85/15 (ms: 60/40), FT 50/35/15 (ms: 45/35/20), ATJ 80/20 (ms: 75/25). Docstring default `carbon_fraction=0.857` with comment claiming 144/170 = 0.857 (it is 0.847; the JET_A1 instance uses 0.847).
- Assets the manuscript never uses: **Sajben transonic diffuser validation** for the LE-PINN nozzle (`data/raw/sajben.x.fmt`, `scripts/validation/sajben_validation.py`, `models/le_pinn_sajben*.pt`); HyChem-vs-CRECK validation mode (`documentation/HYCHEM_VALIDATION_SUMMARY.md`); an A2 NOx mechanism (`data/A2NOx.yaml`); a pytest suite (`tests/test_physics_conservation.py`, `test_choking_detection.py`, `test_nozzle_regression.py`, `test_le_pinn_benchmark.py`).

**Professor outreach doc:** 25+ drafted emails in 3 waves, organized by category (A architecture, B Cantera framing, C heat loss/NOx, D surrogates/SAF chemistry, E validation targets, F LCA/CORSIA, G PINN/UQ). Framing is already honest ("reduced-order screening framework under revision"). Its weakness: it asks professors about problems whose answers are already determinable from your own data (e.g., held-out validation), and one email repeats the false "XWB" attribution indirectly by describing the benchmark engine.

### 1.2 Evidence map

| # | Claim | Where | Current evidence | Weakness | Verdict |
|---|---|---|---|---|---|
| C1 | "Validated against ICAO LTO data for Trent XWB-84EP, MAPE 11.3%" | Abstract, §2.5, §3, Fig. 1 | 4 fuel-flow points | **The engine is misattributed** (data = Trent 1000-AE3; XWB not in database) and the 11.3% is a **fit residual, not validation**: φ per mode, η_comb, pressure loss, and airflow scales were tuned against these same 4 points (`calibrate_lto.py`) | **Remove as stated; rebuild** as calibration + held-out cross-engine validation |
| C2 | "Turbofan digital twin" | Title-adjacent, Abstract | None | No fan, no bypass stream, BPR unused; TSFC 29.46 mg/(N·s) and specific thrust 800.6 N·s/kg are turbojet-magnitude, ~2–3× a real BPR-9 engine | **Remove "digital twin"; reframe** as core-flow screening model or add fan model |
| C3 | "η_c and N_CMB treated as learnable parameters" (Nath et al. inverse method) | §2.1, §2.2 | None in code | η_c fixed 0.86; η_comb fixed 0.98 in optimization or heuristic parabola; nothing is learned inside a PINN; Optuna calibration ≠ inverse PINN | **Rewrite** to describe actual calibration; report all values |
| C4 | Blend-specific combustor efficiency differences | `estimate_efficiency()` | Hard-coded −1.5%/−1.0% penalties | Unsourced; manufactures the very blend sensitivity the paper reports | **Remove or source**; ablate |
| C5 | NOx = 87.65 g/kg per blend; NOx as optimization objective | §2.6, §3 | ICAO-fitted regression NOx=A·OPR^B·ṁ^C | Zero fuel-chemistry dependence → cannot discriminate blends except via fuel flow; any "R² vs ICAO" is in-sample | **Reframe** as ICAO-derived correlation (fuel-flow proxy); or add chemistry NOx via A2NOx.yaml/Zeldovich |
| C6 | Net CO₂ objective (g/s) | §2.6, Figs. 3–4 | `3.16 × LCA × ṁ_fuel` | Affine in fuel flow × assumed factor; not a chemistry output; makes 2 of 4 objectives redundant re-expressions of fuel flow | **Reframe**; compute EI-CO₂ from Cantera products; separate combustion vs lifecycle |
| C7 | LCA factors 1.0/0.2/0.1/0.3 | §2.6 Eq. 20 | None | Unsourced; CORSIA pathway values vary ~5× by feedstock; ATJ can exceed 0.5 | **Replace** with CORSIA ranges + uncertainty propagation |
| C8 | Surrogate compositions & H/C, DCN values | §2 pp. 10 | "consistent with … recent studies" (no refs) | Manuscript numbers ≠ `fuels.py` numbers (60/40 vs 85/15 etc.); mass vs mole ambiguity; DCN not computed anywhere | **Fix mismatch, cite NJFCP/Heyne-style sources, state basis** |
| C9 | t=1.04, p=0.3734 → "not statistically distinguishable" | §3, §4 | Paired t-test, n=4 | No power; absence of significance ≠ equivalence | **Remove** the inferential claim entirely |
| C10 | PINN turbine enforces PDEs + work balance | §2.3 | Loss terms exist in training code; tests exist | At inference, outlet T is adjusted to match imposed compressor work → analytic polytropic relation may reproduce it; PDE claims untested against analytic baseline in the paper | **Keep but ablate** (PINN vs analytic); publish `tests/` results |
| C11 | Nozzle PINN quasi-1D isentropic + choke | §2.4, Fig. 2 | Model runs; **Sajben validation exists in repo but is absent from the manuscript** | Quasi-1D isentropic is analytically solvable; PINN adds risk without the Sajben evidence | **Keep, add Sajben section** — this is your strongest unused asset |
| C12 | "Optimization produced Pareto-optimal blends that reduce lifecycle carbon while maintaining thrust" | Abstract, §4 | Optuna runs, Pareto plots | Discussion admits near-zero blend sensitivity; discriminating signal comes from assumed LCA + fitted NOx + invented penalties | **Weaken** to scenario analysis; make the insensitivity a *finding* |
| H1 | Highlights: "R²=0.9969 for NOx" | Highlights only | In-sample fit R² printed by `_fit_nox_model()` | Training-fit statistic misreported as validation | **Delete** |
| H2 | Highlights: "mass continuity error 0.00%" | Highlights only | Turbine enforces u=ṁ/(ρA) exactly by construction | A constraint satisfied by construction is not a result | **Delete** |
| H3 | Highlights: "80% CO₂ cut" | Highlights only | The assumed f=0.2 in the `estimate_co2` docstring ("Bio-SPK: 0.2") | An input echoed as a finding | **Delete** |
| H4 | Highlights: "81.58 kN thrust" | Highlights only | A core-thrust printout from some run; not in Results | Untraceable to reported results | **Delete** |

---

## DELIVERABLE 2 — Technical vulnerability table

Columns: (1) issue, (2) location, (3) why it matters, (4) reviewer risk, (5) type, (6) exact fix, (7) test/ablation, (8) acceptance criterion, (9) manuscript edit, (10) professor validation needed?

| # | Issue | Location | Why it matters | Risk | Type | Exact fix | Test/ablation | Acceptance criterion | Manuscript edit | Prof? |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | **Validation is calibration.** φ per LTO mode, η_comb, p_loss, airflow/π scale factors all tuned against the same 4 fuel-flow points reported as validation | `scripts/optimization/calibrate_lto.py`; ms §2.5, §3, Fig. 1 | ~6 free parameters + 8 hand-set scales vs 4 targets → 11.3% MAPE is a residual of fitting, carries no predictive content | **Fatal** | Validation | Freeze calibration on one engine (Trent 1000-AE3); predict fuel flow for the other **44 Trent 1000 variants** already in `icao_engine_data.csv` with no retuning (scale by rated thrust/OPR/BPR from CSV columns) | Leave-one-engine-out CV across the family; report MAPE distribution | Held-out MAPE reported per mode with spread; calibration/validation split stated explicitly | New §3.1 "Calibration" + §3.2 "Held-out validation"; new figure: predicted vs ICAO fuel flow, 45 engines | No — you have the data |
| V2 | **Engine misattribution**: manuscript says Trent XWB-84EP; database and calibration targets are Trent 1000-AE3 | Abstract, §2.5, §3 vs `calibrate_lto.py` + CSV | If a reviewer checks the ICAO databank, XWB-84EP fuel flows will not match your Fig. 1 → integrity finding, same class as the Highlights problem | **Fatal** | Data/integrity | Correct every mention to Trent 1000-AE3 (OPR 43.2, BPR 9.1, 310.9 kN — matches `design_point`); state ICAO UID (02P23RR126) | Diff manuscript numbers vs CSV row | Every quoted fuel-flow/OPR/BPR value traceable to a CSV row | Global find-replace + data-availability note | No |
| V3 | **Highlights fabrication-adjacent numbers** (R² 0.9969, 81.58 kN, 80% cut, 0.00% continuity) | Highlights (submission field) | Reviewer 2 flags integrity; each number is a training artifact, constraint, or input, not a result | **Fatal** | Integrity/wording | Rewrite Highlights only from numbers present in Results; add provenance note in response letter explaining each old number's origin | n/a | Every Highlight number appears verbatim in Results | New Highlights | No |
| V4 | **No fan/bypass** despite "turbofan": BPR defined, unused; turbine extracts compressor work only | `integrated_engine.py` (`design_point`, `run_full_cycle`) | TSFC 29.46 mg/(N·s) and specific thrust 800.6 N·s/kg are ~2–3× real BPR-9 values; takeoff +26.6% error plausibly partly from missing fan-work extraction; claims of engine-level TSFC are physically wrong | **Fatal** for "turbofan performance", Major otherwise | Modeling | Add a 0-D fan/bypass: FPR≈1.4–1.5, η_fan≈0.9, turbine work = compressor + fan work, thrust = core + bypass streams; ~100 lines, no new dependencies. Alternative: rebrand all outputs as core-stream quantities and stop reporting engine TSFC | Before/after comparison of TSFC, specific thrust vs Trent-class published values | TSFC within ~±30% of engine-class values at takeoff; or explicit "core-only" labeling everywhere | Rewrite §2 architecture + all performance tables | Optional (Emerson/Lieuwen category A) |
| V5 | **NOx has no fuel-chemistry dependence** (ICAO-fitted power law) | `EmissionsEstimator._fit_nox_model`; ms §2.6 | Optimizer "trades off NOx" that is a deterministic function of OPR and fuel flow → blend NOx ranking is illusory; any NOx-vs-ICAO comparison is circular | **Major** | Modeling+validation | Declare it a P3/T3-style ICAO correlation used for magnitude anchoring only; optionally add chemistry NOx: post-process combustor Cantera solution with thermal-NO (Zeldovich) or `data/A2NOx.yaml`; compare both | Run both NOx paths across blends at fixed cycle state; report divergence | NOx labeled "correlation-based" in every figure; or chemistry-based EI-NOx with mechanism cited | §2.6 rewrite; footnote in Figs. 3–5 | Yes (W. Sun / Khandelwal: minimum credible NOx treatment) |
| V6 | **CO₂ objective = 3.16·LCA·ṁ_f** (affine in fuel flow) | `estimate_co2`; ms Eq. 20 | Two of four objectives (CO₂, and largely TSFC) are monotone functions of fuel flow → "4-objective" claim inflated; Pareto structure driven by inputs | **Major** | Modeling | Compute EI-CO₂ from Cantera product composition per blend (uses `carbon_fraction` already in `fuels.py`); report combustion CO₂ and lifecycle CO₂ as separate axes; state objective coupling explicitly | Correlation matrix of objectives across trials (already have `pareto_visual.py` correlation code) | Reported objective count matches independent objectives; coupling stated | §2.6 + Results | No |
| V7 | **LCA factors unsourced** (1.0/0.2/0.1/0.3) | `optimize_blend.py` LCA_FACTORS; ms Eq. 20 | These four numbers determine the headline "0.762" and the entire LCA axis; real CORSIA core LCA values span ~0.1–1.2 of fossil baseline by feedstock (some ATJ > fossil parity is possible) | **Major** | Data | Replace with CORSIA default life-cycle emission values (ICAO CAEP tables): map each pathway to a feedstock **range**; propagate via Monte Carlo (uniform/triangular over range) into Pareto analysis | Rank-stability test: fraction of Pareto members that persist across 1,000 LCA draws | Every f value cited to CORSIA table + feedstock; results shown as bands, not points | §2.6 rewrite; new figure: Pareto with LCA uncertainty bands; reframe "0.762" as scenario output | Yes (Malina/Prussi category F) |
| V8 | **"Learnable efficiency" narrative false**; SAF efficiency penalties invented | ms §2.1–2.2 vs `Compressor(eta_c=0.86)`, `estimate_efficiency()` penalties | Reviewer 2(c) asked what learned values converged to — nothing converged; the −1.5%/−1.0% penalties directly manufacture blend differences in the one place chemistry could matter | **Major** | Wording+modeling | Rewrite: η_c fixed (0.86, cited to turbomachinery texts), η_comb calibrated once via Optuna (report value + bounds 0.96–0.999); delete SAF penalties or cite combustor-rig data | Ablation: penalties on/off → does blend ranking change? | All efficiency values, bounds, and provenance in a table; penalties removed or sourced | New Table: "Model parameters, values, provenance, calibration status" | Optional |
| V9 | **Combustor heat loss absent**; η lumps everything | `combustor.py` `T_out = T_in + η(T_ideal−T_in)`; ms §2.2, lines 303–310 | R1: case/liner heat loss can exceed inefficiency loss at many conditions; lumping biases T4, hence turbine inlet state and NOx inputs | Moderate | Modeling | Either add explicit liner heat-loss fraction (2–5% of heat release, sourced) or rename η as "lumped heat-delivery efficiency" and say exactly what it absorbs | Sensitivity: T4 and fuel flow vs heat-loss fraction 0–6% | Effect on validation MAPE quantified; wording no longer implies pure incomplete combustion | §2.2 rewrite | Yes (Khandelwal category C) |
| V10 | **Turbine PINN may be decorative**: outlet T adjusted to match imposed work target | `run_turbine` + `scale_turbine_exit_temp` | If analytic polytropic expansion reproduces the PINN output, the PINN novelty claim collapses; K(x) "blade drag" loss is arbitrary (R1: tip leakage dominates, stall avoided by design) | **Major** for novelty | Modeling+wording | Ablation PINN vs analytic polytropic turbine (η_p=0.9) in the full loop; fix loss taxonomy wording (tip-leakage/secondary flow, "turbine inefficiency" not "blade drag") | Full-cycle outputs for all blends, both turbine paths; report max deviation | If deviation < calibration error: reframe PINN as physics-consistency layer, not accuracy claim | §2.3 rewrite; new ablation table | Yes (Lu / J.-X. Wang / B.J. Lee category G) |
| V11 | **Nozzle PINN unvalidated in ms** while Sajben validation exists unused | ms §2.4/Fig. 2 vs `scripts/validation/sajben_validation.py`, `models/le_pinn_sajben*.pt` | Strongest unused evidence in the project; Fig. 2 currently shows only self-consistency (trends), which R1/R2 discount | Moderate (opportunity) | Validation | Run Sajben benchmark, report error vs experimental/CFD wall data; also ablate PINN vs analytic isentropic-with-choke | `pytest tests/ -v` + Sajben error metrics | Quantified nozzle error vs external data in the manuscript | New §3.x "Nozzle surrogate validation (Sajben diffuser)" + figure | No |
| V12 | **Blend insensitivity vs blend ranking** (R2 point 5) | §4 Discussion vs Abstract | The model's honest output is "within ASTM limits, cycle performance is nearly blend-invariant"; the paper instead sells blend optimization | **Fatal** to current claim; salvageable | Framing | Invert the claim: the finding is performance-insensitivity + LCA-dominance; the optimizer then explores φ and lifecycle trade-offs, not fuel-performance trade-offs | Variance decomposition: fraction of each objective's variance explained by φ vs blend fractions vs LCA | Stated variance shares in Results | Rewrite Abstract/§4 conclusion around insensitivity finding | No |
| V13 | **φ ∈ [0.35, 0.65] as free design variable** | `optimize_blend.py`, ms §2.6 | Conflates throttle/operating point with fuel design; global φ at combustor inlet ≠ primary-zone φ; stability limits unmodeled (ms admits) | Moderate | Modeling/wording | Fix φ per operating mode (from calibration) when ranking blends; or present φ explicitly as operating-condition co-optimization with stability caveat | Re-run optimization at fixed φ per mode; compare Pareto membership | Blend comparisons reported at matched operating condition | §2.6 + Results | Optional |
| V14 | **t-test misuse** | §3/§4 | n=4 non-significance presented as agreement | Major (easy) | Wording/stats | Delete the t-test; if any statistic is kept, report per-mode % error and held-out MAPE only | n/a | No hypothesis-test language anywhere | Strike sentences | No |
| V15 | **Surrogate mismatch code vs manuscript**; unsourced H/C & DCN | `fuels.py` vs ms p.10 | Reviewer can't tell which composition produced the results; DCN values asserted, never computed/cited | **Major** | Data/integrity | Determine which compositions generated the reported runs; make code and ms identical; cite surrogate sources (NJFCP-style literature); state mole vs mass basis once and use consistently; compute H/C from composition (trivial) and drop DCN or cite it | Re-run one blend sweep with both composition sets; report output shift vs blend-to-blend differences | Single composition table, matching code, with citations; mismatch shift quantified | New Table: surrogate compositions + properties + sources | Yes (Heyne category D) |
| V16 | **Factual/consistency errors**: syngas "CO₂ and H₂"; CO₂ "in ppm"; HyChem "valid only for Jet A-1"; "combing"; corrupted equations; exergy efficiency promised (§2 p.8) never reported; compressor loss list | Various | Each is small; jointly they produced R1's "lazy" verdict — the credibility tax exceeds the technical cost | Major (cumulative) | Wording | Full correctness pass: syngas=CO+H₂; CO₂ in g/s or EI g/kg; HyChem rewritten per R1 (A-2 nominal fuel, multiple NJFCP fuels); remove exergy or compute it; compressor losses: tip-leakage/secondary flow first; re-export all equations (the corruption is a Google-Docs→PDF equation-object export failure — rebuild in LaTeX/Word native equations) | Fresh PDF equation audit on a different machine | Zero unreadable symbols; every promised metric reported | Line edits throughout | No |
| V17 | **Optimizer scrapes stdout with regex**, silent T4=2000 K fallback | `optimize_blend.py` `scrape_log_data` | Parse failures silently inject default values into "results"; the 11.3%-HEFA/11.3%-MAPE coincidence R2 flagged deserves an audit of exactly this path | **Major** (implementation) | Implementation | Return a structured dict from `run_full_cycle` (it already returns one — use it); delete the scraper; re-run the 1,000-trial study; verify the balanced-solution numbers | Re-run study with fixed seed; diff old vs new Pareto set | No regex parsing anywhere; reported representative solution reproduced from structured output | Regenerate Figs. 3–5 + reported numbers | No |
| V18 | **"Screening model" defensibility** | Whole ms | Screening requires the screen to discriminate on real signal; today the discriminating signal is assumed LCA + fitted NOx + invented penalties | **Fatal as "performance predictor," defensible as scoped screen** | Framing | Define exactly what is screened: (a) lifecycle-carbon scenarios under performance-neutrality, (b) operating-condition (φ) trade-offs, (c) thermo-property propagation consistency. Explicitly disclaim fuel-performance ranking | Variance decomposition (V12) is the test | Claim boundary section present (see Deliverable 5) | New Limitations structure | Yes (Owoyele/S. Menon: what a credible screen requires) |

---

## DELIVERABLE 3 — Prioritized repair plan

Dependencies noted as →. Time costs assume evenings/weekends, existing repo, no new data purchases (none needed).

### A. Fix immediately, before any professor outreach (order matters)

**A1. Correct the engine identity everywhere (V2).**
Task: replace XWB-84EP with Trent 1000-AE3 in manuscript, outreach two-pager, and `AGENT_CONTEXT.md`; verify Fig. 1 numbers against CSV rows. Time: 2 h. Needs: CSV. Benefit: removes the most checkable integrity error. Risk reduction: prevents a professor from discovering it in 5 minutes on the ICAO databank. Depends on: nothing.

**A2. Rewrite Highlights and abstract numbers from Results only (V3).**
Task: delete R²=0.9969 / 81.58 kN / 80% cut / 0.00% continuity; draft a provenance note (training-fit R², by-construction constraint, input echo, untraceable printout) for the response letter. Time: 2 h. Benefit: directly answers Reviewer 2's integrity objection with an explanation instead of silence. Depends on: nothing.

**A3. Kill the false "learnable parameter" narrative (V8).**
Task: rewrite §2.1–2.2 to describe fixed η_c=0.86 and one-time Optuna calibration of η_comb/p_loss/φ-per-mode; add the parameter-provenance table. Time: 3 h. Benefit: converts "hidden fudge factors" into "documented calibration," which is respectable. Depends on: A4 (know final calibrated values).

**A4. Relabel calibration vs validation and run the held-out cross-engine study (V1).**
Task: freeze calibration on Trent 1000-AE3; write ~80-line script to sweep the other 44 CSV engines (scale mass flow by rated thrust, set π from CSV OPR), predict fuel flow, report per-mode MAPE distribution. Time: 1–2 days incl. debugging. Needs: CSV, `integrated_engine.py`. Benefit: **this is the single highest-value fix in the project** — real validation from data you already own. Risk reduction: converts R2's strongest objection into your strongest figure. Depends on: nothing; blocks B-tier claims.

**A5. Delete the t-test (V14), fix factual errors and equations (V16).**
Time: 1 day for the full correctness pass; equations must be rebuilt as native objects (the corruption came from exported equation images). Depends on: nothing.

**A6. Reconcile surrogate compositions (V15, first half).**
Task: find which composition set generated the reported runs (`git log` on `fuels.py` vs run dates; or re-run one case with each set); make manuscript = code. Time: 3–4 h. Depends on: nothing.

### B. Fix before resubmission

**B1. Replace stdout-scraping and re-run the optimization study (V17).** 1 day + compute overnight. Depends on A6 (fixed compositions). All reported Pareto numbers regenerate from structured output with a seed.

**B2. Fan/bypass module (V4).** 2–3 days. 0-D fan (FPR 1.45, η 0.9), turbine work = compressor + fan, two-stream thrust. Re-calibrate (A4 pipeline) afterward. Benefit: TSFC/specific thrust become engine-plausible; the takeoff +26.6% error likely shrinks. Depends on: A4 pipeline in place so you can show before/after on held-out engines.

**B3. LCA overhaul (V7 + V6).** 2 days. CORSIA default values by feedstock → ranges; Monte Carlo over LCA; EI-CO₂ from Cantera products; separate combustion vs lifecycle axes; Pareto rank-stability metric. Depends on: B1.

**B4. PINN ablations (V10, V11).** 2–3 days. Analytic turbine & nozzle drop-ins already effectively exist (`run_nozzle`, `scale_turbine_exit_temp`); run full-cycle for all fuels both ways; add Sajben nozzle validation section; run `pytest tests/ -v` and report. Depends on: B1.

**B5. Variance decomposition of objectives (V12, V18).** 1 day. From the B1 trial database: regression/Sobol share of each objective's variance attributable to φ, blend fractions, LCA. This is the quantitative backbone of the reframed claim. Depends on: B1, B3.

**B6. NOx dual-path (V5).** 2–3 days. Keep ICAO correlation (labeled); add Zeldovich/A2NOx post-processing; report divergence honestly (they will disagree — that is a finding about correlation-based NOx, not a failure). Depends on: B1.

**B7. Heat-loss sensitivity (V9).** 1 day. Parametric liner-loss fraction 0–6%; effect on T4/fuel flow/held-out MAPE. Depends on: A4.

### C. Reframe instead of fix
- Blend-performance insensitivity (V12): make it the central finding, not an embarrassment.
- NOx/CO₂ objectives (V5, V6): reframe as "correlation-anchored emissions proxies," never validated chemistry.
- φ optimization (V13): reframe as operating-condition co-exploration with stability caveat, or freeze per mode.
- "Screening model" (V18): scope it to lifecycle-scenario screening under performance-neutrality (Deliverable 5).

### D. Remove because not defensible
- "Digital twin" (everywhere). — "Validated model" language. — t-test inference. — All four Highlights numbers. — SAF combustor-efficiency penalties (unless sourced). — Exergy efficiency (promised, never delivered — remove or implement). — DCN values (uncomputed, uncited). — XWB-84EP. — "80% CO₂ cut" in any form.

### E. Optional upgrades if time allows
- Chemistry-based EI-NOx validated against the ICAO NOx columns on *held-out engines* (would upgrade V5 from reframe to fix). 1 week.
- 2D LE-PINN nozzle already staged in repo (`le_pinn.py`, unified models) — only if B4 shows the 1D PINN earns its place. 
- Cruise-point extension via Breguet-style checks against public Trent 1000 cruise TSFC estimates. 
- Property-based blend model (density, viscosity, LHV from blending rules vs measured SAF data) to give the screen a real fuel-property axis — this is the S. Menon/Heyne direction and the best long-term upgrade.

---

## DELIVERABLE 4 — Minimum viable validation and ablation plan

**Baseline model (B0):** post-A6/B1 pipeline — fixed surrogate set (code-matched), η_c=0.86, calibrated η_comb/p_loss/φ-per-mode (calibrated on Trent 1000-AE3 only), turbine PINN, nozzle PINN, ICAO NOx correlation, CORSIA-range LCA. Everything below is CPU-feasible; the only training-free runs are full-cycle evaluations (~seconds each).

**Ablation variants:**

| ID | Variant | Tests dependence on |
|---|---|---|
| AB1 | Analytic polytropic turbine replaces turbine PINN | PINN turbine |
| AB2 | Analytic isentropic-choke nozzle replaces nozzle PINN | PINN nozzle |
| AB3 | η_comb fixed at 0.98 / 0.96 / 0.995 (no calibration); SAF penalties off | Learnable/calibrated efficiencies + invented penalties |
| AB4 | Manuscript surrogate set vs code surrogate set (HEFA 60/40 vs 85/15, etc.) | Blend-property approximations |
| AB5 | Leave-one-mode-out calibration (calibrate on 3 LTO points, predict 4th); plus 44-engine holdout | Sparse ICAO operating points |
| AB6 | Liner heat loss 0/2/4/6% of heat release | Heat-loss treatment |
| AB7 | LCA Monte Carlo, 1,000 draws over CORSIA feedstock ranges | LCA assumptions |
| AB8 | φ frozen per mode vs φ free | Operating-point confound |

**Sensitivity analysis:** one-at-a-time ±5% on η_c, η_poly, p_loss, P_out(turbine), A_e; tornado plot of effect on fuel flow, TSFC, T4.

**Validation targets (all already in your possession):**
1. Fuel flow, 4 modes × 44 held-out Trent 1000 variants (`icao_engine_data.csv`).
2. Nozzle: Sajben transonic diffuser wall pressure/Mach data (`data/raw/sajben*`, existing scripts).
3. Combustor chemistry: HyChem A1 vs CRECK n-dodecane surrogate — adiabatic flame T and ignition delay over φ ∈ [0.3, 1.2], p ∈ {10, 30, 43} bar (mode exists: `mechanism_profile="validation"`).
4. Physics self-consistency: `pytest tests/ -v` (conservation, choking, regression) — publish the counts.
5. Optional: chemistry EI-NOx vs held-out ICAO NOx columns.

**Plots to generate:** (P1) predicted vs ICAO fuel flow, 45 engines, colored by mode, identity line; (P2) MAPE distribution per mode (box plot); (P3) PINN vs analytic component outputs across blends (bar deltas); (P4) tornado sensitivity; (P5) Pareto front with LCA uncertainty bands (replaces current Fig. 3); (P6) variance-decomposition stacked bars per objective; (P7) Sajben validation profile overlay; (P8) HyChem-vs-CRECK Tad and τ_ign overlays.

**Tables:** (T1) parameter provenance (value, bounds, fixed/calibrated, source); (T2) surrogate compositions + computed H/C + LHV + citations; (T3) ablation matrix: ΔTSFC, Δfuel-flow-MAPE, ΔPareto-membership per variant; (T4) CORSIA LCA ranges by pathway/feedstock; (T5) test-suite results.

**Expected interpretation and decision rules:**
- AB1/AB2 ≈ B0 (likely): PINNs are physics-consistency layers, not accuracy contributors → keep them but claim architecture/consistency, not superiority. If they differ materially, you must explain which is right — analytic wins by default absent external data.
- AB3 changes blend ranking (likely for penalties-off): the penalties were manufacturing signal → remove them, report that blend ranking within ASTM limits is efficiency-assumption-dominated.
- AB4 shift > blend-to-blend differences (plausible): surrogate uncertainty exceeds the signal → you cannot rank pathways on combustion performance; performance-neutrality framing becomes mandatory.
- AB5 held-out MAPE ≤ ~15–20%: genuinely strengthens the paper — a calibrated 0-D core model transferring across a 45-variant family is a publishable, honest result. Held-out MAPE ≥ ~30%: weaken to "order-of-magnitude consistency" and drop quantitative fuel-flow claims.
- AB7: if Pareto membership is <50% stable across LCA draws, all blend-selection conclusions must be presented as scenario-conditional (they likely are).
- AB6: if 4% heat loss moves fuel flow more than the blend signal, say so — it bounds the resolvable effect size.

**What strengthens the manuscript:** AB5 success + Sajben + HyChem/CRECK agreement + honest variance decomposition. **What forces weakening:** AB5 failure (drop predictive claims entirely → framework paper), or AB4 dominance (drop pathway ranking → scenario explorer).

---

## DELIVERABLE 5 — Defensible claim boundary

**1. Too strong (current):** "A validated turbofan digital twin combining Cantera and PINNs identifies Pareto-optimal SAF blends that cut lifecycle CO₂ by up to 80% while maintaining thrust, validated against ICAO data (MAPE 11.3%, NOx R²=0.9969)."

**2. Defensible revised:** "A reduced-order, chemistry-consistent turbofan-core screening framework couples Cantera kinetics/thermodynamics with physics-constrained surrogate components and multi-objective search. Calibrated on one engine's four ICAO LTO fuel-flow points, it predicts fuel flow across 44 held-out Trent 1000 variants with X% median error. Within ASTM blend limits the model predicts near-neutral thrust and TSFC across HEFA/FT/ATJ blends, implying that blend selection is dominated by lifecycle-carbon assumptions rather than engine performance; we therefore present blend optimization as a scenario analysis over CORSIA-range carbon intensities, with NOx represented by an ICAO-derived correlation rather than validated chemistry."

**3. Very conservative (if AB5 fails):** "An open-source computational framework demonstrating the coupling of detailed kinetics (Cantera/CRECK), physics-constrained neural surrogates, and Bayesian multi-objective search for SAF blend exploration. Quantitative outputs are illustrative; the contribution is the reproducible architecture and the demonstration that, under this model class, ASTM-limit SAF blending is performance-neutral and lifecycle-assumption-dominated."

**Claims you can still make:** chemistry-consistent single-mechanism blend comparison (CRECK) is a sound design choice; fuel-dependent γ, R, cp propagate through the whole cycle; nozzle surrogate error quantified against Sajben data; physics constraints (mass continuity, work balance, choking) enforced and unit-tested; fuel-flow calibration + (pending) held-out family validation; performance-neutrality of ASTM-limit blends *within this model class*; Pareto structure of the scenario problem.

**Claims you cannot make:** validated engine performance prediction; any NOx validation; any lifecycle-carbon *finding* (they are inputs); blend-specific combustion-efficiency effects; turbofan-level TSFC/specific thrust (until fan added); PINN superiority over analytic components (until ablated); "digital twin."

**Require professor validation (judgment calls, not data):** whether one CRECK mechanism distorts blend differences (Xu/Pepiot/Burke); minimum credible NOx treatment (W. Sun); surrogate strategy defensibility (Heyne); heat-loss treatment floor (Khandelwal); CORSIA range selection (Malina/Prussi); PINN-vs-analytic framing (Lu/J.-X. Wang).

**Require additional simulations (all feasible locally):** held-out validation, all ablations AB1–AB8, chemistry EI-NOx, fan module.

**Require experimental/literature data you may lack:** blend-specific combustor efficiency (rig data — you will not get this; do not claim it); real-fuel DCN/flame-speed for your exact surrogates (literature exists — NJFCP/Heyne datasets — cite rather than claim); cruise-condition fuel burn (only public estimates; use for order-of-magnitude only).

**Category placement:** this project is a **reduced-order screening framework + optimization scaffold + exploratory SAF blend analysis**. It is *not* a predictive performance model (no held-out accuracy yet at engine level beyond fuel flow), and only partially a physics-informed surrogate paper (surrogates unablated). Position it as the first category and the reviewers lose most of their ammunition.

---

## DELIVERABLE 6 — Manuscript repair outline

**Highlights:** rewrite from scratch; only numbers present in Results; lead with held-out validation MAPE and the insensitivity finding.

**Abstract:** Keep: motivation (cost gap between cycle models and CFD), architecture summary, single-mechanism rationale. Rewrite: replace "digital twin"→"reduced-order core screening framework"; "validated"→"calibrated on 4 LTO points (Trent 1000-AE3) and evaluated on 44 held-out variants"; state performance-neutrality finding and scenario-based LCA explicitly. Delete: XWB, 0.762 as a finding, "Pareto-optimal blends reduced lifecycle carbon" phrasing. Unsafe language: "validated," "digital twin," "demonstrates … can enable" (→ "suggests").

**Introduction:** Keep: SAF pathway background (fix syngas = CO + H₂), PINN/Cantera literature review (it is decent). Rewrite: strike the vague sentence flagged at lines 147–149; strike lines 90–91 per R1; answer R1's line-97 question or delete the sentence; fix genetic-algorithms history (line 129) — they are decades old and not "AI" in the modern sense; fix "combing"→"combining". Add: the logic flow chart R1 demanded — objectives → required outputs → tool per output → validation per output. One figure; this single figure answers R1's core complaint. Delete: any implication that this predicts engine performance.

**Methods:** Keep: single-mechanism CRECK strategy and its justification; HyChem-as-benchmark design (rewrite the HyChem paragraph: developed on NJFCP A-2 nominal fuel, applied to multiple fuels — per R1); equations (rebuilt, uncorrupted). Rewrite: §2.1 "Cantera modeled the compressor" → isentropic relations with Cantera state/property evaluation (Schoegl framing); compressor loss list → tip-leakage/secondary flows primary; §2.2 define η_comb as lumped heat-delivery factor OR add explicit liner loss; delete "learnable parameter" language, add calibration subsection with parameter-provenance table (T1); §2.3 turbine: "blade drag"→"turbine inefficiency," describe work-matching honestly, present K(x) as tunable loss with stated value; §2.4 nozzle unchanged + reference Sajben validation; §2.5 split into Calibration (what was tuned, on what) and Validation (held-out protocol); §2.6: surrogate table (T2) matching code, mole/mass basis stated, CORSIA LCA ranges (T4), objective-coupling statement, EI-CO₂ from products. Add figures: pipeline flow chart; calibration/validation split schematic. Disclose: core-only vs fan-added status; NOx is correlation-based.

**Validation (new standalone section):** Fig. P1/P2 held-out fuel flow; Sajben nozzle section (P7); HyChem-vs-CRECK Tad/τ_ign (P8); test-suite table (T5). Disclose: no NOx validation; no cruise/transient validation; fuel flow only.

**Results:** Keep: Pareto machinery, parallel-coordinates idea. Rewrite: all numbers regenerated from structured-output rerun (B1); Fig. 3 replaced by LCA-uncertainty-band version (P5); add variance decomposition (P6) — this is the new centerpiece; representative solution reported with uncertainty and explicitly scenario-conditional; check and resolve the 11.3%/11.3% coincidence in the rerun. Delete: t-test; any NOx g/kg presented as chemistry.

**Discussion:** Keep: the honest paragraphs (idle/takeoff error mechanisms, n=4 caveat, LCA caveat — currently buried). Rewrite: lead with the insensitivity finding as the result; interpret Pareto structure as LCA-driven by construction; the "particularly useful moderate-SAF region" paragraph must be conditioned on LCA draws. Delete: "validated model" callbacks.

**Limitations (promote to numbered section):** core-only architecture (if fan not added); calibration/validation scope; correlation NOx; surrogate uncertainty > blend signal (if AB4 confirms); LCA as scenario inputs; no stability/operability modeling; single-engine-family validation.

**Future Work:** Keep: cruise validation, better idle/takeoff, richer surrogates, added objectives (already reasonable). Add: chemistry NOx vs held-out ICAO, property-based merit functions (Menon/Heyne direction), fan/booster model if not done. Delete: 2D LE-PINN promise unless B4 justifies it.

---

## DELIVERABLE 7 — Professor-facing technical questions

Your outreach doc's structure (waves, one question per email) is good — keep it. But two fixes first: (1) purge the XWB attribution from the two-pager; (2) do A4/AB5 *before* wave 1, because "I ran held-out validation across 44 variants, here's the MAPE distribution" transforms you from student-with-a-problem to researcher-with-a-result. Ten questions worth asking:

**Q1. Held-out validation design.** Weakness: V1. Ask: multi-fidelity ML/validation people (Owoyele). "I calibrated on one engine's 4 LTO points and now predict fuel flow across 44 same-family variants without retuning. Is within-family holdout meaningful validation for a 0-D model, or does shared architecture make it too easy — and what would you add before trusting blend-level outputs?" Matters: determines whether AB5 success is a headline or a footnote. If "too easy": seek a different engine family in the ICAO databank as a second holdout. If "meaningful": lead the revision with it.

**Q2. Property-grounded screening.** Weakness: V15/V18. Ask: S. Menon (LSU merit functions). "If engine outputs are nearly blend-invariant within ASTM limits, should a screening model rank blends on measured property–performance relationships (DCN, LHV, density, viscosity) instead of surrogate combustion differences — and what is the minimum property set?" Matters: defines the E-tier upgrade path. If yes: add a property-based merit axis; if no: keep chemistry-only and say so.

**Q3. Cantera framing.** Weakness: R1's abstract objection. Ask: Schoegl or Weber (Cantera devs). "Is it accurate to say Cantera supplies EOS/thermo/transport evaluation plus the constant-pressure reactor, while the compressor is isentropic relations calling Cantera for state properties? And is one CRECK C1–C16 mechanism across Jet-A1 + SAF surrogates a defensible consistency choice?" Matters: one-sentence wording fix with an authoritative citation-by-conversation. Answer changes only text.

**Q4. Single-mechanism distortion.** Weakness: V15/C8. Ask: Xu, Pepiot, or Burke. "Does forcing Jet-A1 and all SAF surrogates into one CRECK mechanism compress real blend-to-blend differences in flame temperature and heat release — i.e., could my observed blend insensitivity be a mechanism artifact rather than physics?" Matters: directly tests whether the central reframed finding (insensitivity) is itself defensible. If artifact: the insensitivity claim must be weakened to model-class-conditional. 

**Q5. Surrogate defensibility.** Weakness: V15. Ask: Heyne (NJFCP/ASCENT). "Are 2–3-component n-alkane/iso-octane surrogates (HEFA 85/15 dodecane/iso-octane, ATJ 80/20 iso-octane/dodecane) adequate for *relative* blend screening of Tad and heat release, and which published surrogate formulations should I anchor to?" Matters: yields the citations Reviewer 2 demanded. Answer determines whether compositions change before rerun (do this early — it gates B1).

**Q6. Combustor heat loss floor.** Weakness: V9. Ask: Khandelwal. "For a 0-D constant-pressure reactor combustor in a screening model, is a fixed liner heat-loss fraction (2–5% of heat release) plus a separate incomplete-combustion efficiency the minimum credible treatment, or is lumping both into one factor acceptable if labeled honestly?" Matters: decides B7 scope. Lumped-OK: wording fix; not-OK: add the term.

**Q7. Minimum credible NOx.** Weakness: V5. Ask: W. Sun or Steinberg. "Given no NOx measurements, which is more honest for a screening paper: an ICAO-fitted OPR/fuel-flow correlation clearly labeled as such, or Zeldovich post-processing of the reactor solution — and can either support blend-to-blend NOx comparison at all?" Matters: decides whether NOx stays an objective or becomes a reported diagnostic. If neither supports blend comparison: drop NOx from the Pareto axes.

**Q8. PINN vs analytic.** Weakness: V10/V11. Ask: Lu Lu or J.-X. Wang. "My turbine/nozzle PINNs are boundary-constrained with work-matching; ablation shows [X]% deviation from analytic polytropic/isentropic modules. When is a PINN justified over the analytic solution in a quasi-1D regime — only with 2D/data extensions, or does the constraint-enforcement layer itself have value?" Matters: decides whether PINNs stay in the headline or move to "architecture" framing. (Run AB1/AB2 first so [X] is a number.)

**Q9. CORSIA LCA ranges.** Weakness: V7. Ask: Malina or Prussi. "I'm replacing fixed LCA factors with CORSIA default core-LCA values by pathway/feedstock and propagating ranges via Monte Carlo into the Pareto analysis. Which feedstock groupings and which uncertainty treatment would you consider minimally credible for a scenario analysis (not an LCA study)?" Matters: converts the weakest inputs into a citable method. Answer sets T4's structure.

**Q10. Architecture triage.** Weakness: V4/V18. Ask: Emerson (or Lieuwen if he replies). "For a core-only 0-D cycle with surrogate components, which single addition most improves credibility for SAF screening: a fan/bypass model, explicit combustor heat loss, or off-design component maps?" Matters: prioritizes B2 vs B7 vs E-tier. Their pick reorders your 14-day plan.

(Questions on validation observables — Blunck — and emissions accounting — Dedoussi — from your doc are fine as wave-2 spares; the ten above cover every fatal/major row in Deliverable 2.)

---

## DELIVERABLE 8 — Execution plans

### 7-day plan (goal: coherent enough for outreach)

| Day | Task | Output | Decision point | Risk | Save/document |
|---|---|---|---|---|---|
| 1 | A1 + A2: fix engine identity; rewrite Highlights; draft provenance note for response letter | Corrected ms + note | Do the Fig. 1 numbers match Trent 1000-AE3 CSV rows exactly? If not, find the run that generated them | Old runs unreproducible → regenerate Fig. 1 now | `git commit` per fix; provenance note in `docs/` |
| 2 | A6: reconcile surrogates (git history vs run dates); pick the code set unless a professor (Q5) says otherwise | Single composition table (T2 draft) | Did reported results use ms or code compositions? | Ambiguity → rerun one blend both ways, keep whichever matches reported numbers, then standardize | Composition-audit note |
| 3–4 | A4: held-out cross-engine validation script; run 44 engines × 4 modes | P1, P2 figures; MAPE table | Held-out MAPE ≤ 20%? → validation section. ≥ 30%? → conservative claim tier (D5.3) | Per-engine scaling (thrust→airflow) may need a simple fan-less correction; document whatever scaling rule you use | Script in `scripts/validation/`; CSV of results |
| 5 | A3: rewrite §2.1–2.2 calibration story; build parameter-provenance table (T1) from `calibrate_lto.py` best trial | New Methods text + T1 | Are calibrated values physically plausible (η_comb 0.96–0.999, φ_idle 0.22–0.30)? Report regardless | Best-trial params not saved → rerun calibration (50 trials, fast) with seed and save JSON | Calibration JSON committed |
| 6 | A5: correctness pass (syngas, ppm, HyChem/NJFCP, combing, loss taxonomy, exergy removal, t-test deletion); rebuild equations natively | Clean ms draft | Any equation still an image? Rebuild it | Export pipeline corrupts again → export test on another machine | Equation-audit checklist |
| 7 | Update two-pager + wave-1 emails with held-out result and corrected engine; send nothing yet — re-read cold | Outreach-ready package | Does every number in the two-pager trace to a file? | Overclaiming relapse under deadline pressure | Traceability list (number → file/line) |

### 14-day plan (goal: resubmission-quality)

| Day | Task | Output | Decision point | Risk | Save |
|---|---|---|---|---|---|
| 8 | B1: replace stdout scraper with structured returns; seed everything | Refactored `optimize_blend.py` | Do old headline numbers reproduce? If not, report new ones and say why in response letter | The 11.3%/11.3% coincidence resolves here — audit it explicitly | Diff of old vs new Pareto CSV |
| 9 | B1 rerun: 1,000 trials overnight | New trials database | Pareto set stable vs old? | Cantera runtime → reduce to 500 trials if needed; state count | `outputs/` CSV + seed |
| 10 | AB1/AB2: analytic turbine/nozzle ablation across fuels | T3 rows; P3 | Deviation < held-out error? → reframe PINN as consistency layer (§2.3/2.4 language) | PINN checkpoint incompatibilities → use the fallback path already in `run_full_cycle` | Ablation CSV |
| 11 | AB3 + AB8: efficiency/penalty and φ-frozen ablations | T3 rows | Does blend ranking survive penalties-off? (Expect: no) → delete penalties, adopt insensitivity framing | None serious | Ablation CSV |
| 12 | B3: CORSIA LCA ranges (T4), Monte Carlo, EI-CO₂ from products; P5 + rank-stability | New Fig. 3 replacement | Pareto membership stability ≥ 50%? Report either way | CORSIA table reading errors — cite exact document/version | LCA-draws CSV |
| 13 | B5: variance decomposition (P6); B7 heat-loss sweep (AB6) | P6, sensitivity figure | Which factor dominates each objective? Write it in Results verbatim | None | Analysis notebook |
| 14 | B4: Sajben validation section + `pytest tests/ -v` results (T5); assemble revised ms per Deliverable 6; write response-to-reviewers skeleton | Full revised draft + response letter skeleton | Every reviewer point mapped to a change or a rebuttal? | Response letter tone — concede everything true, rebut only with evidence | Point-by-point response table |

Fan/bypass (B2) and chemistry NOx (B6) are the two items that don't fit in 14 days at this pace; schedule them for days 15–21, and say in the response letter that core-only scope is disclosed and the fan extension is in progress — unless Q10 answers say fan-first, in which case swap B2 with days 10–11.

---

## DELIVERABLE 9 — Brutal final diagnosis

**Is the project salvageable?** Yes — but not as the paper you submitted. The codebase is more honest than the manuscript: it contains a real 45-engine ICAO dataset you barely used, a real external nozzle benchmark (Sajben) you never mentioned, a physics test suite you never reported, and a HyChem cross-check mode you undersold. The manuscript's failure mode was dressing a calibrated exploratory pipeline as a validated predictive twin. The repair is mostly subtraction plus one addition (held-out validation).

**Strongest version of the project:** "An open, chemistry-consistent, reduced-order turbofan-core screening framework, calibrated on four LTO points and shown to transfer across 44 held-out engine variants, which demonstrates quantitatively that within ASTM blend limits SAF selection is performance-neutral and dominated by lifecycle-carbon assumptions — and therefore reframes SAF blend optimization as a lifecycle-scenario problem." That is a real, defensible, mildly contrarian contribution.

**Weakest part reviewers will attack:** provenance integrity — Highlights numbers, the XWB/Trent-1000 misattribution, and manuscript-vs-code surrogate mismatches. These are checkable in minutes against your own public repo, and any one of them ends a review. Fix all three before anything else touches the science.

**One fix with the most credibility gain:** the held-out cross-engine validation (A4/AB5). It costs two days, needs zero new data, and converts Reviewer 2's strongest objection ("four points, calibrated, no validation") into your strongest figure.

**One claim to stop making:** that the model ranks SAF blends by engine performance and emissions. Your own Discussion admits the outputs are blend-insensitive; your own code shows NOx and CO₂ objectives contain no fuel chemistry. Stop saying "optimized blends"; start saying "scenario-dominant lifecycle trade-offs under performance neutrality."

**Core framing sentence for the revised project:** *"Within ASTM blending limits, a chemistry-consistent reduced-order engine model predicts that SAF blend choice is nearly performance-neutral — so the blend-optimization problem is governed by lifecycle-carbon assumptions, which we treat as explicit, uncertainty-banded scenarios rather than findings."*
