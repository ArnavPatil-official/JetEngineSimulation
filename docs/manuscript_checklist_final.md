# Manuscript checklist — final (Phase 6, P6.8)

One row per reviewer item in `docs/plan.md` §1. For each item: what closes it
in the repo, and the exact source to take replacement text or numbers from.
Take every number from the named manifest row (`outputs/ARTIFACT_MANIFEST.md`,
generated from artifacts), never from this file or from older checklists.

**Scope.** This is a repo checklist. It does not edit the manuscript (the
Google Docs source is outside this repo), contact reviewers, or claim checks
that have not happened. Status legend: **repo: closed** = the artifact exists
and is in the manifest; **external: pending** = the manuscript edit (or a
check on a second machine) has not been done or verified from this repo.

## Reviewer 1

| # | Item | Repo | Replacement text source | Manuscript |
|---|---|---|---|---|
| R1.1 | Corrupted equation symbols (L226–368, Eqs 1–2 …) | n/a | Rebuild every equation as a native equation object; symbols per `docs/model_map.md` and the V1/V7 rows. Verify in a PDF opened on a second machine | external: pending (second-machine check not done) |
| R1.2 | "Compressor modelled using Cantera" | closed (`592cc20`: docstring) | Compressor = isentropic-efficiency temperature rise on Cantera air thermo (V7 η_c row) | external: pending |
| R1.3 | L90–91 strike | n/a | delete | external: pending |
| R1.4 | L97: which aspects of engine performance are affected | closed (P6.3) | B1 (fuel flow, TSFC, T4, NOx(corr), lifecycle at matched thrust) and E4 (heat loss bounds the resolvable blend effect) | external: pending |
| R1.5 | L129: genetic algorithms predate AI | n/a | reword (text only) | external: pending |
| R1.6 | L131 "combing" | n/a | "combining" | external: pending |
| R1.7 | L147–149 vague | n/a | restate objectives as the rows of `docs/model_map.md` | external: pending |
| R1.8 | HyChem not "valid only for Jet A-1"; A-2 nominal | closed (P6.4) — **the question stays in the text** | E9: at fixed calibration, HyChem A1/A2 need 2.5 / 3.4 % more fuel than CRECK n-dodecane (above the V8 band), mostly because the n-dodecane surrogate's heating value is ~2 % higher; T4 and A1-vs-A2 differences are within the band. Correct the A-1/A-2 description and state that the surrogate's heating value, not rate chemistry, sets this sensitivity | external: pending |
| R1.9 | Tip leakage main compressor loss; stall avoided by design | closed (code) | V7 η_c row (single lumped isentropic efficiency; no loss breakdown claimed) | external: pending |
| R1.10 | Structural heat loss vs inefficiency | closed (`847857c`) | `scripts/validation/heat_loss_provenance.md`; E4 for the size of the effect | external: pending |
| R1.11 | "Turbine inefficiency", not "blade drag" | closed (code) | V7 turbine η_poly row | external: pending |
| R1.12 | Logic flow chart; objectives unclear | closed (P6.7) | `docs/model_map.md` (diagram + claim-to-evidence table, generated) | external: pending (render the Mermaid diagram as a figure) |
| R1.13 | "Splitting maul" — kinetics for flame temperature and exhaust | closed (P6.4) | Production combustor is HP equilibrium, no kinetics (`integrated_engine.py`, `combustor.py` docstrings); kinetics only in the E3 evidence path; drop "kinetics-informed" (decision 3) | external: pending |
| R1.14 | "Black box making ill-defined connections" | closed (P6.5, P6.7) | `docs/model_map.md`; PINN record P1–P8 (all negative/partial; retired from production) | external: pending |

## Reviewer 2

| # | Item | Repo | Replacement text source | Manuscript |
|---|---|---|---|---|
| R2.1 | Highlights contradict body | closed (manifest regenerated) | Write Highlights verbatim from V1, V3, B1, E4 rows only | external: pending |
| R2.2a | n = 4 paired t-test | closed (removed, Phase 3) | — | external: pending (confirm removal) |
| R2.2b | Fuel-flow agreement does not validate what the optimisation depends on | closed (P6.1) | V4 (F-A: the old validation was a rescaling rule), V3 (thrust-matched held-out test vs B0 and B1: beats constant TSFC, **does not beat rated-thrust rescaling**; A2 and A3 fail as registered), V2 (all fitted parameters identified) | external: pending |
| R2.3a | Surrogate compositions uncited | partly closed | E10 (LHVs computed, method stated); compositions remain **H/C-matched illustrative binaries** — say so beside every blend number (P6.2 rule) | external: pending |
| R2.3b | LCA factors unsourced | closed | E1 | external: pending |
| R2.4a | Syngas is CO + H₂ | n/a | text | external: pending |
| R2.4b | CO₂ in ppm | closed (code) | E2 | external: pending |
| R2.4c | "Learnable" efficiencies absorb model error | closed (P6.1, P6.2) | V7 (fixed values with cited ranges; β dropped), V2 (the four fitted parameters are identified), V8 (bands, not CIs) | external: pending |
| R2.4d | 11.3 % coincidence | closed (Phase 3) | — | external: pending |
| R2.5 | Model barely discriminates blends, yet ranks them | closed (P6.3) | B1 (comparisons at matched thrust; rankings only where the registered rule allows), B2 (variance shares), B3 (CORSIA rank stability) | external: pending |

## Title and claims (decision 3)

Drop "kinetics-informed" and any PINN-accuracy framing. Supportable description:
a thermodynamic reduced-order turbofan screening model, calibrated and
held-out-tested on ICAO LTO data, with a documented PINN surrogate study.
Final wording is the author's. external: pending.

## Citations from `docs/plan.md` §5

Ma et al. 2025 (P8); Nath et al. 2023 (inverse-problem identifiability, model
for the R1.12 flowchart); Wang 2024 (tanh for second derivatives); Kuzhagaliyeva
et al. 2022 (naive-baseline discipline, V3); Gal et al. 2024 (right tool for the
quantity, R1.13); Uy & San Juan 2024 (context for the retired 80 % claim; CORSIA
stays the input); Şahin 2023 (related work — check what data any quoted score
used). external: pending.
