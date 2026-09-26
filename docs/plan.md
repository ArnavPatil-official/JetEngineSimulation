# Phase 6 — Final repair plan: close every item in the DTE-D-26-00365 review — September 26, 2026

This is intended to be the last repair plan. It supersedes the unfinished part of
Phase 5 (archived at `docs/plan_phase5_superseded.md`; review and nozzle-repair
plans remain at `docs/plan_phase5_review.md`, `docs/plan_phase5_nozzle_repair.md`).
Branch `phase4`, head `0f3321b`. Tests at head: 132 passed / 1 skipped.

The plan is organised around one question per reviewer item: **is it fixed in the
repo, and if not, what closes it?** Section 1 is the full traceability table. Sections
2–3 record the two findings that change the approach. Sections 4 onward are the phases.

---

## 1. Every reviewer item, and its state at `0f3321b`

"Repo" = closed by code, data or an artifact. "Text" = closes only in the manuscript,
but the manuscript must draw on a named repo artifact (P6.8 does that mapping).

### Reviewer 1

| # | Item | State | Closes in |
|---|---|---|---|
| R1.1 | Corrupted equation symbols (L226–368, Eqs 1–2 …) | open | P6.8 — native equations; audit on a second machine |
| R1.2 | "Compressor modelled using Cantera" | code docstring fixed (`592cc20`) | P6.8 wording |
| R1.3 | L90–91 strike | open | P6.8 |
| R1.4 | L97: which aspects of engine performance are affected | open | P6.8, sourced from P6.3 matched-thrust results |
| R1.5 | L129: GAs predate AI | open | P6.8 |
| R1.6 | L131 "combing" | open | P6.8 |
| R1.7 | L147–149 vague | open | P6.8 |
| R1.8 | L216–217: HyChem not "valid only for Jet A-1"; A-2 nominal | open in text; **repo question open**: does the mechanism choice affect any reported number? | P6.4 mechanism-sensitivity test, then P6.8 |
| R1.9 | L258: tip leakage is the main compressor loss; stall avoided by design | code fixed | P6.8 |
| R1.10 | L303–310: structural heat loss vs inefficiency | decided — ξ = 0 by structural argument (`847857c`) | P6.8 wording from `heat_loss_provenance.md` |
| R1.11 | L318 "turbine inefficiency", not "blade drag" | code fixed | P6.8 |
| R1.12 | Logic flow chart connecting outputs to tools; objectives unclear | **open** | P6.7 claim-to-evidence map |
| R1.13 | "Splitting maul" — kinetics used for flame temperature and exhaust stoichiometry | **partly false of the code**: the combustor already uses `equilibrate('HP')`, no reactor; kinetics appear only in the Zeldovich NOx evidence path. The *claim* "kinetics-informed" is what is wrong | P6.4 + title change in P6.8 |
| R1.14 | "Black box making ill-defined connections" | PINNs retired from production; NOx labelled a correlation | P6.7 map makes every connection explicit |

### Reviewer 2

| # | Item | State | Closes in |
|---|---|---|---|
| R2.1 | Highlights contradict body | numbers retired and crosswalked (Phase 3) | P6.8 regenerate Highlights from manifest |
| R2.2a | n = 4, paired t-test | removed | done |
| R2.2b | Fuel-flow agreement does not validate what the optimisation depends on | **reopened — see F-A**: the held-out fuel-flow validation carries no information about the cycle model | **P6.1** |
| R2.3a | Surrogate compositions uncited | table reconciled with code (Phase 1); **no literature citation** | P6.2 |
| R2.3b | LCA factors unsourced | closed — CORSIA Doc 06 triangular ranges + Monte Carlo | done |
| R2.4a | Syngas is CO + H₂ | text | P6.8 |
| R2.4b | CO₂ in ppm | code fixed | P6.8 |
| R2.4c | "Learnable" efficiencies absorb model error | **open in a sharper form**: η_b, pressure_loss, k_pi unidentified; k_mdot/φ ridge; η_c, η_poly, η_fan, FPR unsourced | P6.1 + P6.2 |
| R2.4d | 11.3 % HEFA / 11.3 % MAPE coincidence | closed — stdout scraper removed | done |
| R2.5 | Model barely discriminates blends, yet ranks them | reframed via variance decomposition, but blends are still compared at **fixed φ**, not fixed thrust | **P6.3** |

---

## 2. Finding F-A — the held-out fuel-flow validation is a rescaling rule (critical)

`scripts/validation/holdout_icao_validation.py` sets each held-out engine's core
airflow to `base_airflow × thrust_ratio` (line 96), where `thrust_ratio` is that
engine's **ICAO rated thrust** over AE3's. With φ fixed per mode, fuel flow is
φ·f_st·β·ṁ_core·x^k_mdot — OPR and BPR never enter it. Measured on
`outputs/holdout_icao_validation_v4.csv` (2026-09-26):

- Predicted fuel flow ÷ thrust ratio is **constant to 1e-16 within each mode**
  (take-off 2.318207, approach 0.618863, idle 0.242230 kg/s, 59 rows each).
  The held-out prediction is AE3's calibrated fuel flow multiplied by the held-out
  engine's rated-thrust ratio. The Cantera cycle contributes nothing to it.
- A model-free baseline — AE3's *ICAO* fuel flow × thrust ratio — scores **3.22 %**
  MAPE against the model's **2.46 %** on the same rows. At take-off the baseline is
  *better* (1.78 % vs 1.81 %). The model's edge at approach and idle comes only from its
  calibrated per-mode values sitting nearer the family mean than AE3's own record.

So manifest row M2 ("held-out validation MAPE 2.50 %") measures whether ICAO fuel
flow is proportional to rated thrust across Trent 1000 variants. That is true, but it
is a property of the data, not of the model — and it is the validation the response
to Reviewer 2 point 2 currently rests on. The same design makes held-out *thrust*
validation (Phase 5 P5.2 Step 4) circular by construction, because airflow would again
be scaled by the answer.

## 3. Finding F-B — the calibration cannot be repaired without changing what is fixed and what is solved

Phase 5 established that the fuel-flow objective identifies φ_to only, and that the
take-off thrust gap (now 52.79 kN, 17.0 %, after the `0be8227` accounting repair) can
be closed only by an unsourced input (total airflow ×1.2045 or BPR 12.29). The Phase 5
gate "no sourced value, no fixed value" then stops everything, because engine-specific
quantities (rated airflow, component efficiencies, pressure loss) are proprietary and
will not be found as single page-cited numbers.

Both findings have one root cause: **the model is driven by φ, but ICAO defines LTO
modes by thrust.** Driving it by φ is what makes fuel flow independent of the cycle
(F-A), what leaves η_b and k_pi with nothing to constrain them (F-B), and what makes
blend comparisons happen at unequal thrust (R2.5).

**The fix is to run the cycle at the thrust ICAO specifies and predict fuel flow.**
At matched thrust, fuel flow depends on OPR, BPR, airflow, every component
efficiency and the fuel's LHV — so it becomes a genuine model output, the 17 % gap
becomes a fitted airflow reported against its hand-set value, and blend comparisons
happen at equal thrust, which is the comparison Reviewer 2's point 5 asks for.

---

## 4. Decisions required from the user before P6.1

**Approved 2026-09-26:** Arnav explicitly approved all three decisions in this
session. Execute the approved repo work through the dispatcher; exact title wording
can wait. The escalation conditions in §9 still apply to actual findings.

1. **Adopt thrust-matched operation** as the production formulation (φ solved per
   mode from a thrust target, fuel flow predicted). Recommended — it is the only change
   that closes R2.2b, R2.4c and R2.5 together.
2. **Replace the single-value sourcing rule with a range rule** for engine-proprietary
   parameters: a fixed parameter needs a *cited range* (textbook or public source, page
   or table), a central value inside it, and its effect propagated to every reported
   number as a band. The purpose of the old rule — no invented number presented as a
   result — is kept, because the spread is reported rather than hidden. A parameter with
   no citable range must be data-identified (P6.1) or dropped.
3. **Title and claims.** With both PINN gates missed and no kinetics in the production
   path, "physics- and kinetics-informed" and any PINN-accuracy framing come out. The
   supportable description is a *thermodynamic reduced-order turbofan screening model,
   calibrated and held-out-tested on ICAO LTO data, with a documented PINN surrogate
   study*. Final wording is the user's; the repo work below does not depend on it.

---

## 5. What the supplied papers change

- **Ma et al. 2025 (LE-PINN, Aerosp. Sci. Technol. 168, 111002)** is the source of the
  repo's nozzle architecture: same 6 inputs (x, y, A5, A6, P_in, T_in), same 9 outputs
  (ρ, u, v, P, T, u′u′, v′v′, u′v′, μ_eff), dual network, **ReLU** with Xavier init,
  AdamW + ReduceLROnPlateau (patience 10, factor 0.5, floor 1e-8 — the scheduler whose
  collapse attempt 1 diagnosed). Three differences matter for the write-up and belong
  in P6.5: (a) Ma's momentum residual keeps the Reynolds-stress divergence and writes
  viscous terms as ∂/∂x(μ∂u/∂x), so under ReLU only the μ∂²u part vanishes, whereas the
  repo's Laplacian form loses all of it; (b) Ma validates against a held-out split of its
  *own* CFD database inside the trained NPR 4–14.5 range, while the repo tests on a
  different geometry against experiment — a harder, out-of-distribution test; (c) Ma uses
  sigmoid dynamic loss weights (Eqs 30–33). The repo's negative result is therefore a
  finding about transfer and residual formulation, not a contradiction of Ma's.
- **Nath et al. 2023 (Sci. Rep. 13, 13683)** identify unknown engine parameters with a
  PINN from *measured states that depend on them*, and give a flowchart of the inverse
  problem. That is the identifiability lesson of F-B in general form, and the model for
  R1.12's flowchart.
- **Wang 2024 (NNICE, data-efficient PINNs)** uses tanh because the Navier–Stokes residual
  needs second derivatives — consistent with the attempt-3 fix.
- **Kuzhagaliyeva et al. 2022 (Commun. Chem. 5, 111)** benchmark their mixture predictions
  against a naive linear-by-mole mixing baseline. P6.1 adopts the same discipline: every
  predictive claim is reported beside a naive baseline, which is exactly what exposed F-A.
- **Gal et al. 2024 (Energies 17, 5543)** use Cantera 0D reactors where flame speed and
  ignition delay matter, then carry that into engine simulations checked against measurements — the
  "right tool for the quantity" argument R1.13 asks for.
- **Uy & San Juan 2024 (J. Clean. Prod. 482, 144241)** quote "up to 80 %" SAF CO₂
  reduction from an industry source (Shell 2020). That is a plausible origin of the retired
  HEFA factor 0.2 and the "80 % CO₂ cut" Highlight. Cite it for context only; CORSIA
  Doc 06 stays the LCA input.
- **Şahin 2023 (Heliyon 9, e21365)** reports ML regression quality mainly as R², RMSE and
  MAE. Useful as related work on ML for blend performance; before citing any score, check
  what data it was computed on — an in-sample R² is what became the retracted 0.9969.

---

## 6. Constraints (carried forward, with the one change from decision 2)

- Pre-register before running: objectives, splits, fitted-vs-fixed lists, baselines and
  acceptance thresholds are committed before the first run of each phase.
- Every fixed parameter has a cited range (decision 2); every fitted parameter passes
  the identifiability test on the objective it is fitted to.
- Every predictive claim is reported beside a naive baseline.
- Never overwrite `models/*.pt` or any v1–v4 artifact; the protected-hash check from
  `outputs/logs/static_thrust_accounting_protected_sha256.json` runs at the end of each phase.
- Nothing deleted; removals are `mv` into `archive/`.
- `python -m pytest tests/ -v` (floor 132 / 1) and `scripts/test_emissions.py` after
  each phase.

---

## 7. Phases

### P6.1 — Thrust-matched calibration and a validation that tests the cycle (F-A, F-B; R2.2b, R2.4c)

**Step 1 — Record F-A as a failing test.** `tests/test_holdout_informativeness.py`:
on the v4 holdout CSV, assert that predicted fuel flow ÷ thrust ratio varies within a
mode (it currently does not), and compute the naive baseline of §2. The test documents
F-A and must pass only once the new validation exists. Add F-A to
`outputs/parameter_provenance.md` and flag manifest M2 as superseded.

**Step 2 — Thrust-matched cycle.** `integrated_engine.run_at_thrust(target_kN, …)`: solve
φ with a bracketed root-finder so static thrust equals the target; return the same
structured result as `run_full_cycle` plus the solved φ and solver status. Tests:
round-trip (φ from `run_full_cycle` → thrust → recovered φ within 1e-6); monotonicity of
thrust in φ over the working range; clean failure (not a silent default) when the target
is unreachable below the T4 guard.

**Step 3 — Grouped split and baselines (register before any fit).**
- Split the 28 engine models (not records) into calibration and held-out groups with a
  seeded, rated-thrust-stratified grouped split; AE3 in the calibration group. Commit
  the list.
- Targets per record and mode: thrust = `Power (%)` × `Rated Thrust (kN)` (input) and
  ICAO fuel flow (output).
- Per-record inputs from the CSV: rated OPR, BPR, rated thrust.
- Baselines: **B0** constant TSFC — calibration-group mean TSFC per mode × held-out
  thrust; **B1** the F-A rule — nearest-rated-thrust calibration engine's fuel flow ×
  thrust ratio.
- Fitted parameters (shared across the calibration group): reference airflow and its
  rated-thrust scaling exponent, k_pi, k_mdot. Everything else fixed per P6.2.
- Acceptance thresholds, stated as numbers in the registration: the identifiability
  test must pass for all four; held-out fuel-flow MAPE must beat **both** baselines by a
  stated margin; the sign of the model's d(TSFC)/d(OPR) across held-out engines must match
  the data's.

**Step 4 — Identifiability first.** Extend `scripts/validation/identifiability_profile.py`
to the new objective; run it with a cheap pilot. Any parameter that fails is moved to
"fixed with cited range" (P6.2) or dropped before the full fit — not after.

**Step 5 — Calibrate v5, validate.** `calibrate_lto.py --tag v5` on the calibration
group only; `holdout_icao_validation.py --tag _v5` on the held-out group, reporting model,
B0 and B1 side by side per mode, plus the fitted reference airflow against the hand-set
79.9 kg/s core value (this is where the old 17 % gap now appears, as a number with an
explanation rather than a residual).

**Report whatever comes out.** If the model does not beat B0, the honest result is that,
within one engine family whose variants differ mainly in rating, the cycle model adds no
fuel-flow skill over a constant-TSFC rule. That is publishable and it answers R2.2b
directly. Do not tune toward the baseline after seeing the held-out numbers.

**Also:** refit the NOx correlation on the calibration group only
(`nox_fit_exclude_models` already supports it) so NOx and fuel flow share one split,
and re-run `nox_holdout_validation.py` against the held-out group.

**Acceptance:** F-A test passes; identifiability passes for every fitted parameter;
held-out table reports model vs B0 vs B1; thresholds applied as registered.

---

### P6.2 — Sourcing pass under the range rule (R2.3a, R2.4c)

For each fixed parameter, record in `outputs/parameter_provenance.md`: cited range
(source, page/table), central value, and the sensitivity of the v5 design point and
held-out MAPE across the range.

| Parameter | Now | Needs |
|---|---|---|
| η_b | 0.9963 (unidentified sampler value) | cited high-power combustion-efficiency range |
| combustor pressure loss | 0.0442 (unidentified) | cited range for annular combustors |
| η_c | 0.86 (unsourced) | cited range for modern HP compressors |
| η_poly (turbine) | 0.90 (unsourced) | cited range |
| η_fan, FPR | 0.90, 1.45 | cited ranges |
| combustor air fraction β | 0.8 (Phase 3.4) | cited range or data identification |
| surrogate compositions | HEFA 85/15, FT 50/35/15, ATJ 80/20 (mole) | literature surrogate sources; if none, declare "H/C-matched illustrative binaries" and keep the Phase 1 composition-sensitivity result beside every blend number |
| LHV per surrogate | fuels.py values | cited or computed from Cantera thermo, with method stated |

Then one Monte Carlo over the fixed-parameter ranges (seeded) to produce bands on the v5
design point and held-out MAPE. A parameter whose band dominates a reported conclusion is
named in the results as the limiting assumption.

**Acceptance:** no fixed parameter without a cited range or an explicit "illustrative"
label; bands produced; the Phase-5 citation gate is closed by this rule, not bypassed.

---

### P6.3 — Blend comparison at matched thrust (R2.5, R1.4)

1. `optimize_blend.py`: blends evaluated at matched thrust per LTO mode (take-off
   primary), with fuel flow, TSFC, T4, NOx(corr) and lifecycle CO₂e as outputs. φ is no
   longer a design variable; the free-φ study is archived as superseded. Remaining design
   variables: blend fractions (and CORSIA scenario draw).
2. Re-run `variance_decomposition.py`. Expected — to be confirmed, not assumed — that
   blend effects on TSFC are set by LHV and are small, while lifecycle CO₂e is dominated
   by pathway and CORSIA draw. Whatever comes out is the answer to R2.5 and R1.4.
3. Report every blend difference beside (a) the seed/Monte-Carlo spread and (b) the P6.2
   parameter bands. A blend ranking is claimed only where the difference exceeds both.

**Acceptance:** no blend comparison anywhere at unequal thrust; every ranking claim
carries its comparison against the two spreads.

---

### P6.4 — Combustion fidelity and mechanism dependence (R1.8, R1.13)

1. Document what the combustor does: adiabatic equilibrium (`equilibrate('HP')`) at the
   combustor-inlet state, scaled by η_b(1−ξ). No kinetics in the production path.
2. `scripts/validation/mechanism_sensitivity.py`: T4, product composition and fuel flow
   at matched take-off thrust with each shipped mechanism profile (CRECK, HyChem A1, A2).
   If differences are below the P6.2 bands, state that mechanism choice does not affect
   any reported number — which also removes the HyChem-validity question from the
   critical path, whatever the text says about A-1 vs A-2.
3. Keep kinetics exactly where they matter: the Zeldovich NOx post-processor stays an
   evidence path (E3) and is described as such.

**Acceptance:** mechanism-sensitivity table in the manifest; production path described
as equilibrium thermodynamics everywhere in code docstrings.

---

### P6.5 — Close the PINN record (title claim, R1.14)

No new training. P4.3 is closed at attempt 3 (0.159 ± 0.016, partial; physics-on
worse than matched data-only by 0.044 > seed spread 0.038). P4.4 is closed (turbine
surrogate retired).

1. `docs/le_pinn_vs_ma2025.md`: the reimplementation audit from §5 — what matches Ma
   et al., the three differences, and which of them each negative result bears on.
2. Manifest rows for: Sajben attempts 1–3 with matched ablations; turbine surrogate
   attempts 1–2; `outputs/physics_residual_defect.md`; P4.1 data audit and ceiling.
3. Future-work note (not a registered attempt): Ma-form residual with Reynolds-stress
   divergence, dynamic loss weights, CFD-split evaluation in the Ma style *and* the
   Sajben experiment test, reported separately.

---

### P6.6 — Re-run production and regenerate the number trail

1. Re-run at v5, matched thrust, analytic turbine and nozzle, ξ = 0:
   `design_point_summary.py`, `optimize_blend.py`, `variance_decomposition.py`,
   `nox_holdout_validation.py`, `nox_dual_path.py`, `heat_loss_sensitivity.py`,
   `takeoff_thrust_gap.py`, P6.2 Monte Carlo, P6.4 mechanism sensitivity.
2. Regenerate `outputs/ARTIFACT_MANIFEST.md` from scratch; M2 replaced by the P6.1
   table (model vs B0 vs B1).
3. `docs/number_crosswalk_v5.md`: v4 → v5 for every manuscript-bound number.
4. v4 production outputs → `outputs/archive/pre_phase6/`; orphan sweep to zero.

---

### P6.7 — Claim-to-evidence map (R1.12, R1.14)

`docs/model_map.md` with one diagram (Mermaid, rendered in the repo) and one table:
every reported quantity → the model that computes it → the inputs it depends on → the
data that constrains it (or "none — assumption, range cited") → the manifest row. This
is the flow chart Reviewer 1 asked for, and it makes any remaining "black box"
connection visible by construction. Generate the table from a small registry in code
so it cannot drift from the pipeline; add a test that every manifest row appears in it.

---

### P6.8 — Reproducibility, then the manuscript mapping

**Repo (carried from P5.4):** pin `requirements.txt` to the v5 environment and document
the working CPU torch install; `REPRODUCE.md` verified on a fresh clone in a fresh
virtualenv; archive code outside the manifest pipeline (`simulation/emissions.py`,
`pareto_visual.py`, `visualize_results.py`, `dashboard.py`, `fetch_and_build_cfd_data.py`,
vendored SU2 repo → `archive/`); `parse_sajben_cfd.py` zeros not NaN; remove
`turbine.py` import-time print; convert `tests/test_nozzle_pinn_fix.py` bool returns to
asserts; add `test_manifest_integrity.py` and `test_checkpoint_provenance.py`.

**Manuscript mapping:** `docs/manuscript_checklist_final.md` — one row per item in §1
with the exact replacement text source (manifest row or provenance section). It covers
the text-only items (R1.1, R1.3–R1.7, R2.4a), the equations rebuilt as native objects
and checked in a PDF opened on a second machine, Highlights regenerated verbatim from
manifest rows, the title per decision 3, and the §5 citations.

---

### P6.9 — Freeze

Final pytest, emissions check, protected-hash check; merge `phase4` → `main`; tag
`v5.0`. Every row of §1 marked closed with its commit.

---

## 8. Order, and what each phase unblocks

P6.0 decisions → **P6.1** (the core; unblocks everything numeric) and in parallel
**P6.2** (sourcing, needed by P6.1's fixed list) and **P6.5** (no dependencies) →
P6.3 and P6.4 → P6.6 → P6.7 → P6.8 → P6.9.

## 9. Escalation

Stop and bring back to the user:
- `run_at_thrust` cannot reach take-off thrust for calibration-group engines inside the
  T4 guard at any airflow in the searched range — that is a structural cycle defect.
- Any fitted parameter still fails identifiability after moving others to fixed.
- No citable range exists for a parameter that dominates a conclusion.
- The model loses to B0 by more than the registered margin — report it, but confirm the
  framing with the user before P6.3 builds on v5.

## 10. Rollback

`main` stays at v4 until P6.9. `phase4` resets to `0f3321b` to discard Phase 6. v1–v4
artifacts and all checkpoints are untouched; every removal reverses with the opposite `mv`.


## 11. Executor handoff and review contract (2026-09-26)

### Objective and Repo Context

Execute the approved Phase 6 phases in §7, with the dependency order in §8. The
planner independently reproduced F-A: 177 rows give model MAPE 2.46016229% and
AE3-ratio baseline 3.21571920%; the 171-row strict subset gives 2.50278929% and
3.24190221%. These subsets must stay distinct in reports.

### Relevant Files and File-Level Edits

The exact phase-specific paths and edits are specified in §7. The production engine
is `integrated_engine.py` at repository root. Before each phase, enumerate any
additional supporting paths in that phase's committed registration. Create
`outputs/phase6_execution_status.md` for completed work, commands, results, commits,
and outstanding gates. Leave unrelated `.DS_Store` edits alone.

### Implementation Phases

Carry out §7 through the first actual §9 escalation or completion. Commit numeric
registrations before pilot/fitting runs, including the grouped model split, weighting
of recertifications, pilot budget, identifiability diagnostic/threshold, baseline
margin, and the precise per-mode OPR trend statistic. Do not inspect held-out target
values to choose these decisions. P6.2 fixed values and source ranges must be settled
before fitting; document which assumptions are provisional in any feasibility probe.
Specify how parameter bands condition on or refit calibration; do not call simple
range propagation a statistical confidence interval.

Review clarifications that preserve the intended evidence requirements:
- Keep the v4 CSV immutable. Demonstrate its informativeness failure first, then
  retain that historical diagnosis while separately requiring the new held-out
  predictions to depend on cycle inputs. Do not make an assertion about frozen v4
  predictions pass by modifying the frozen artifact.
- Thrust matching is a formulation repair; it does not itself establish parameter
  identifiability, predictive skill, or blend ranking. Apply all registered gates.
- Check whether stored LHV is actually consumed by the production thermal path.
  Equilibrium fuel thermochemistry and a reporting-only LHV scalar are distinct;
  descriptions of causal dependencies must follow code evidence.
- In P6.4, differences smaller than propagated bands mean they are unresolved at
  that uncertainty level, not literally zero or absent from every reported number.
- Preserve original protected hashes. Archive moves may be verified by content at
  their new paths with an explicit old-to-new mapping; never regenerate the baseline
  hashes from modified files to make the check pass.
- P6.8 is a repo manuscript checklist, not authorization to modify manuscript text,
  contact reviewers, or claim an unavailable second-machine PDF check happened.
  Text-only reviewer items remain explicitly external/pending until verified.
- No further PINN training. No merge/tag or blanket closure while scientific gates
  or required repo acceptance checks are outstanding. Scope the freeze accurately
  if only external manuscript work remains; do not mark it done without evidence.

### Commands to Run and Tests

Use `.venv/bin/python` (plain `python` is not on the default shell PATH).
Run `.venv/bin/python -m pytest tests/ -v` and
`.venv/bin/python scripts/test_emissions.py` after each phase, plus the registered
phase commands and protected-hash verification. Validate shipped Cantera mechanisms
after combustor/emissions changes. Store concise logs and exact commands.

### Acceptance Criteria

Apply each phase's acceptance criteria and §9 verbatim, subject to the scientific
clarifications above. Record failed predictive gates as results, not software bugs to
optimize against the held-out set. Review both source code and generated evidence.

### Rollback Notes

Retain the original §10 baseline and all artifacts. Do not run a destructive reset
or overwrite unrelated user work. Revert named commits or reverse archive moves if
rollback is actually requested.

### Escalation Guidance

High complexity across calibration, thermodynamic integration, and evidence
provenance: use Claude Opus via `scripts/run_claude_from_plan.sh`. Follow §9 when
an actual finding triggers a gate. Finish independent authorized evidence work where
possible, then report the precise result and decision needed; do not invent sources
or thresholds after looking at held-out outcomes.
