# CLAUDE.md — Executor Operating Rules

## Before Any Change

1. Read `docs/plan.md` in full.
2. Confirm you understand the objective, constraints, and acceptance criteria.
3. If the plan is unclear or underspecified, **stop and explain the blocker** — do not guess.

## Execution Rules

- **Follow the plan exactly.** Do not add features, refactor code, or "improve" things outside the plan scope.
- **Prefer minimal, localized edits.** Change only what the plan specifies.
- **Run validation commands** listed in the plan after each major phase.
- **Commit logical units.** If the plan has multiple phases, validate between phases.
- **Never modify files not listed** in the plan's "Relevant Files" section unless absolutely necessary (and document why).

## ML/Simulation-Specific Rules

- **Reproducibility**: Never remove or change random seeds, RNG initialization, or seed-passing patterns.
- **Config integrity**: YAML data files (`data/*.yaml`) and model configs are authoritative. Don't inline values.
- **Model weights**: Never overwrite `models/*.pt` without the plan explicitly requiring it. Back up first.
- **Cantera mechanisms**: After any combustor or emissions change, run validation tests.
- **PINN training**: Preserve loss function structure, physics constraint formulations, and boundary condition logic.
- **Experiment logging**: Maintain any existing logging patterns (output dirs, CSV results, plot generation).

## Testing

- Run `python -m pytest tests/ -v` after implementation.
- If the plan specifies additional validation scripts, run those too.
- Report all test results — do not silently skip failures.

## When Blocked

If you encounter any of these, **stop and explain clearly**:
- Plan references files that don't exist
- Plan requires a dependency not in `requirements.txt`
- Plan's edits conflict with each other
- Tests fail and the fix is outside plan scope
- Ambiguous instructions with multiple valid interpretations

## Project Structure Reference

```
simulation/           # Core physics modules
  combustor/          # Combustion chamber simulation
  compressor/         # Compressor stage
  nozzle/             # Nozzle flow (includes PINN surrogate)
  turbine/            # Turbine stage (includes PINN surrogate)
  emissions.py        # Emissions estimation
  fuels.py            # Fuel properties and blending
  thermo_utils.py     # Thermodynamic utilities
integrated_engine.py  # Full engine integration
data/                 # Chemical mechanisms, ICAO data (read-only by default)
models/               # Trained PINN weights (protected)
tests/                # pytest test suite
scripts/              # Optimization, visualization, utilities
evaluation/           # EDA and analysis scripts
outputs/              # Generated plots, results, reports
```
