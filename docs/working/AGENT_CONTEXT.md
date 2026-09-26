# Agent Context: JetEngineSimulation

## What this project is

This repository is the code behind the research paper *"Multi-objective optimization of
sustainable aviation fuel blends using a physics- and kinetics-informed turbofan
simulation"* (Arnav Patil, Cos Fi; Academy of Science, Academies of Loudoun). It is a
turbofan "digital twin" that predicts jet engine performance and emissions for Jet-A1 and
sustainable aviation fuel (SAF) blends, then runs multi-objective Bayesian optimization
over fuel composition to find Pareto-optimal blends. Source is public at
github.com/ArnavPatil-Official/JetEngineSimulation, archived at
doi.org/10.5281/zenodo.20336725.

The goal is scientific, not production software: evaluate hundreds of candidate SAF
blends (HEFA-SPK, FT-SPK, ATJ-SPK, and mixes with Jet-A1) across thrust, thrust specific
fuel consumption (TSFC), NOx emissions, and lifecycle CO2 — without running full CFD for
every candidate.

## Core modeling approach

The engine cycle (compressor → combustor → turbine → nozzle, i.e. the Brayton cycle) is
split into a hybrid grey-box model:

- **Compressor** — modeled in Cantera as isentropic compression with a learnable
  efficiency factor (η_c). Outputs pressure/temperature/enthalpy state.
- **Combustor** — modeled in Cantera using `IdealGasConstPressureReactor` (constant
  pressure combustion). Solves species evolution and energy balance to get adiabatic
  flame temperature, product species distribution, and outlet enthalpy. Combustion
  efficiency is also a learnable parameter.
- **Turbine** — a physics-informed neural network (PINN), not Cantera. Enforces mass
  conservation, momentum conservation (with a blade-drag loss term), enthalpy/shaft-work
  balance, and the ideal gas law along the axial coordinate. 5-layer MLP (64,64,64,32,32,
  tanh activations, ~20k params), Softplus outputs for ρ/p/T (positivity), ReLU for
  velocity (no backflow). Loss = L_PDE + L_BC + L_phys + L_work, trained with Adam in
  PyTorch.
- **Nozzle** — also a PINN, enforcing quasi-1D isentropic expansion with a choke check,
  converting turbine-exit enthalpy into exit velocity/thrust. Same MLP architecture
  pattern as the turbine, different governing equations/boundary conditions. There is
  also a "locally-enhanced" PINN variant (LE-PINN, `le_pinn.py`) inspired by literature
  showing dual-network (global + near-wall) architectures improve accuracy near walls.

Rationale for the split: Cantera chemical kinetics are used where combustion chemistry
matters (the combustor); for the compressor, Cantera supplies only real-gas
thermodynamic properties for an isentropic-efficiency compression (no kinetics).
PINNs are used where the physics is governed by flow PDEs
with sparse/no labeled data (turbine, nozzle) — PINNs don't need large labeled datasets,
just governing equations and boundary conditions.

**Chemistry**: all combustion in the integrated engine uses one detailed mechanism, the
CRECK C1–C16 mechanism (`data/creck_c1c16_full.yaml`), for thermodynamic consistency
across all fuels. Jet-A1 is represented via HyChem-style validation against
`data/A1highT.yaml` (used only for validation, not inside the main sim, since HyChem
isn't built for blending). SAF pathways are represented as surrogate hydrocarbon
mixtures matching their dominant molecular classes:
- HEFA-SPK: 85% n-dodecane / 15% iso-octane (mole fractions)
- FT-SPK: 50% n-dodecane / 35% n-decane / 15% iso-octane (mole fractions)
- ATJ-SPK: 80% iso-octane / 20% n-dodecane (mole fractions)

(These are the `simulation/fuels.py` values that generated all reported results;
an earlier version of this document echoed the manuscript's incorrect table.)

Fuel blends are parameterized by a composition vector `{p_J, p_H, p_F, p_A}` (Jet-A1,
HEFA, FT, ATJ proportions) constrained per ASTM-style rules (Jet-A1 ≥ 50%, other three
≤ 50% combined). Each blend also carries a lifecycle-assessment (LCA) carbon-intensity
factor relative to Jet-A1 (Jet-A1=1.0, HEFA=0.2, FT=0.1, ATJ=0.3).

**Calibration/validation**: the integrated model is *calibrated* (one-time seeded
Optuna fit of η_comb and per-mode φ) against ICAO Landing-Takeoff (LTO) fuel-flow
data for the Rolls-Royce **Trent 1000-AE3** (ICAO UID 02P23RR126) — an earlier
version of this document repeated the manuscript's incorrect "Trent XWB-84EP"
attribution; no XWB engine appears in `data/icao_engine_data.csv`. The previously
quoted 11.3% MAPE was the in-sample fit residual of that calibration, not
validation. Genuine held-out cross-engine validation against the other Trent 1000
certification records is produced by `scripts/validation/holdout_icao_validation.py`
(results: `outputs/holdout_icao_validation_summary.csv`). The ICAO CSV contains
Idle/Approach/Take-off rows only (no Climb), and this remains a comparative
SAF-screening tool, explicitly **not** certification-grade (no cruise, transients,
or altitude effects).

**Optimization**: Optuna-based Bayesian optimization (TPE sampler) over 5 continuous
variables (total SAF fraction, unnormalized HEFA/FT/ATJ weights, combustor equivalence
ratio Φ) against 4 objectives (min TSFC, max specific thrust, min net CO2, min NOx),
producing a Pareto front. A representative balanced Pareto solution: ~29.6% total SAF
(11.3% HEFA, 9.5% FT, 8.8% ATJ), TSFC=29.46 mg/(N·s), specific thrust=800.6 N·s/kg,
NOx=87.65 g/kg, LCA=0.762.

## Key finding / thesis

Within the model's ASTM-style blend limits, bulk thrust/TSFC are controlled more by
thermodynamic cycle state than by which SAF is used — so the real question isn't "can
SAF maintain thrust" but "which blend best trades off lifecycle carbon, NOx, and fuel
consumption." HEFA/FT/ATJ are chemically distinct (different H/C ratio, heating value,
LCA factor) and should be treated as a multi-component design space, not a single
"SAF fraction" knob. Equivalence ratio is a major TSFC/NOx control lever independent of
fuel choice.

## Repository layout

```
simulation/                 Core physics modules
  combustor/combustor.py    Cantera IdealGasConstPressureReactor combustion model
  compressor/compressor.py  Cantera isentropic compressor model
  nozzle/nozzle.py           Nozzle PINN (production path)
  nozzle/nozzle_pinn_v2.py   Nozzle PINN variant
  nozzle/le_pinn.py           Locally-enhanced PINN (dual-network, near-wall accuracy)
  nozzle/nozzle_conditions.py Boundary-condition helpers
  turbine/turbine.py          Turbine PINN
  turbine/turbine_boundary.py Turbine boundary conditions
  emissions.py                 NOx/CO2 emissions estimation
  fuels.py                     Fuel property + blending definitions
  thermo_utils.py              Thermodynamic property extraction (from Cantera states)
integrated_engine.py        Full engine integration (LocalFuelBlend, EmissionsEstimator,
                             IntegratedTurbofanEngine — the compressor→combustor→
                             turbine→nozzle pipeline orchestrator)
data/
  *.yaml                     Chemical kinetic mechanisms (CRECK C1-C16, HyChem A1highT,
                              A2NOx, isooctane, n-dodecane HyChem) — treat as read-only
                              authoritative config, don't inline values elsewhere
  icao_engine_data.csv        ICAO LTO validation data
  raw/ICAO_RR_TRENT_1000/     Raw ICAO emissions databank PDFs (Trent 1000 certs)
  raw/cfd_datasets/           External CFD reference datasets (NASA sajben nozzle,
                              Kaggle, GitHub nozzle_flow_cfd) used for PINN
                              validation/fine-tuning, not for the main engine sim
  processed/                  Preprocessed shock-tube dataset for PINN training
  processors/                 Scripts to convert raw mechanisms/engine data to usable form
models/*.pt                 Trained PINN weights (turbine_pinn.pt, nozzle_pinn.pt,
                             several le_pinn_*.pt variants incl. sajben-finetuned and
                             "unified" versions) — protected, never overwrite casually
scripts/
  optimization/               Optuna blend optimization + LTO calibration
  validation/                 PINN training/fine-tuning/comparison scripts (incl.
                               Sajben nozzle CFD benchmark used to validate/finetune
                               the nozzle PINN)
  visualization/               Plotting (Pareto fronts, nozzle geometry, results)
  dispatch_claude.py / run_claude_from_plan.sh  Planner→executor dispatch (see AGENTS.md)
evaluation/                 EDA scripts (Cantera mechanism EDA, ICAO EDA, network viz)
tests/                      pytest suite — physics conservation, choking detection,
                             PINN benchmarks/regression, thermo fixes
outputs/                    Generated plots, results, archived experiment runs
dashboard.py                 Streamlit/Plotly interactive dashboard over results
docs/plan.md                 Current/latest execution plan (see workflow below)
```

## Operating workflow (already encoded in AGENTS.md / CLAUDE.md — follow these)

This repo uses a planner/executor split:
- **Gemini (Antigravity)** is the planner/reviewer. It writes `docs/plan.md` for any
  non-trivial change (Objective, Constraints, Repo Context, Relevant Files,
  Implementation Phases, File-Level Edits, Commands to Run, Tests, Acceptance Criteria,
  Rollback Notes, Escalation Guidance), and reviews diffs/logs/tests afterward.
- **Claude Code is the executor.** It reads `docs/plan.md`, follows it exactly (no
  unscoped refactors or "improvements"), makes minimal localized edits, and runs
  `python -m pytest tests/ -v` (plus any plan-specified validation) between phases.
- If a plan is unclear, references nonexistent files, needs a missing dependency, or
  tests fail for reasons outside plan scope — **stop and explain**, don't guess.

Hard rules to respect as an agent working in this repo:
- Never change RNG seeds/reproducibility patterns.
- `data/*.yaml` mechanisms and model configs are authoritative — don't hardcode/inline
  values that belong there.
- Never overwrite `models/*.pt` without the plan explicitly requiring it (back up first).
- Any combustor/emissions change requires re-running Cantera validation tests.
- PINN changes must preserve loss function structure, physics constraint formulations,
  and boundary condition logic.
- Preserve existing experiment logging (output dirs, CSV results, plot generation).
- Don't touch files outside the plan's "Relevant Files" list without justifying why.

## Practical notes for an agent picking up work here

- Python stack: numpy/pandas/scipy, PyTorch (PINNs), Cantera 3.0+ (kinetics/thermo),
  Optuna (Bayesian multi-objective optimization), matplotlib/seaborn/plotly (viz),
  Streamlit (dashboard), pytest (tests), fpdf2 (report generation).
- Entry point for running the full cycle is `integrated_engine.py`
  (`IntegratedTurbofanEngine` class, `main()` function).
- `simulation/nozzle/` and `simulation/turbine/` contain multiple PINN
  variants/checkpoints (v2, LE-PINN, sajben-finetuned, "unified") reflecting iterative
  validation work — check `scripts/validation/` (e.g. `compare_pinn_le_pinn.py`,
  `sajben_validation.py`, `train_le_pinn_unified.py`) to see how these relate before
  assuming one is "the" current model.
- The manuscript itself (`Patil_Manuscript.pdf`, uploaded separately) is the authoritative
  description of methodology/equations/results; this file summarizes it for quick agent
  orientation but the paper (and `docs/` plan files) should be treated as ground truth for
  any question about exact numerical results or narrative framing.
- Known limitations acknowledged by the author (useful context before "fixing" things
  that are actually intentional simplifications): reduced-order component modeling
  (no ducts/inlets/cooling flows/blade losses modeled explicitly), surrogate (not
  real-composition) SAF chemistry, LCA factors are normalized scenario assumptions not
  universal constants, and validation covers only 4 ICAO LTO points (no cruise/transient/
  altitude data).
