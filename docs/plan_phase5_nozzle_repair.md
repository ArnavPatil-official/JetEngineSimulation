# Proposed P5.2 structural repair — awaiting user decision

## Objective

Correct the static engine thrust accounting exposed by the P5.2 decomposition,
then repeat the diagnosis before any v5 calibration. This proposal is not yet
approved: `docs/plan.md`, P5.2 Step 2 says to stop and escalate on a structural gap.

Independent review reproduces 241.6099946 kN = 55.3914545 kN core +
186.2185401 kN bypass. The core calculation subtracts 16.5003810 kN of internal
turbine-exit momentum. Removing that subtraction gives 258.1103756 kN, leaving
52.7896244 kN below the 310.9 kN target. The remaining gap must stay visible.

## Constraints

- No parameter fitting, new input values, or changes to registered Sajben runs.
- Preserve checkpoints, v1–v4 calibration/holdout artifacts, and original diagnostic
  outputs. Add corrected-case diagnostics beside them.
- Keep the analytic nozzle's ideal, fully expanded approximation explicit. Do not
  silently change to a convergent nozzle, impose an unsourced throat geometry, or
  treat a calculated effective area as a measured engine dimension.
- Chemical YAML stays read-only. Existing seed and component-selection behavior stays.
- Work on `phase4`; no merge/tag until the full original plan passes.

## Repo Context

`IntegratedTurbofanEngine.run_nozzle()` returns thrust used as the core contribution
in `run_full_cycle()`. It currently subtracts its own inlet velocity, an internal
station. The engine-level equation subtracts freestream inlet momentum, which is
zero for the modeled static test stand. Reference: [NASA general thrust equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-force/).

The bypass path already uses static gross momentum. PINN nozzle paths already
receive `thrust_model='static_test_stand'`. This proposal repairs the analytic path.

## Relevant Files

- Modify `integrated_engine.py` (analytic nozzle thrust accounting and documentation).
- Modify `scripts/validation/takeoff_thrust_gap.py` (separate pre/post diagnostic outputs).
- Create `tests/test_static_thrust_accounting.py`.
- Read `tests/test_nozzle_regression.py`, `simulation/fan.py`, and
  `simulation/nozzle/nozzle.py` to preserve the existing component contracts.
- Create `outputs/takeoff_thrust_gap_after_accounting.{json,md}` and update
  `outputs/phase5_execution_status.md` with the remaining gate status.

## Implementation Phases

1. Add regression tests for the static engine-level momentum balance. Demonstrate
   failure with the current internal-inlet subtraction before modifying production code.
2. Correct analytic core momentum thrust to exhaust mass flow times exit velocity
   for the existing static-test-stand model. Retain its current thermodynamic
   expansion and pressure term; do not introduce a flight-speed model in this repair.
3. Document that ideal expansion determines an effective exit area through
   continuity; the configured PINN geometry does not constrain the analytic flow.
   Expose the calculated area as diagnostic metadata if useful to consumers.
4. Rerun the v4-point decomposition with all frozen inputs unchanged. Preserve the
   pre-repair results and report the remaining shortfall, input sensitivities, and
   convergent-nozzle counterfactual with their assumptions.
5. Run the full suite and emissions/mechanism checks. Resume P5.2 registration only
   when the accounting defect is resolved and the remaining model assumptions and
   inputs satisfy the original plan. Unresolved structural issues or missing sources
   still require escalation; this approval would not waive those gates.

## File-Level Edits

- `integrated_engine.py`: repair the analytic static momentum term and clarify its
  modeling boundary. Leave the flow-state velocity available for diagnostics.
- `tests/test_static_thrust_accounting.py`: test the static momentum equation,
  independence from internal inlet velocity at fixed nozzle total state, core-plus-
  bypass summation, and unchanged fuel flow/thermal states at the frozen v4 point.
- `takeoff_thrust_gap.py`: allow a distinct output tag and compare to the preserved
  pre-repair diagnostic. Prevent accidental overwrite of that evidence.
- New reports/status: distinguish the corrected accounting from the unexplained
  residual and from merely illustrative input changes.

## Commands to Run

```bash
.venv/bin/python -m pytest tests/test_static_thrust_accounting.py -v
.venv/bin/python -m pytest tests/ -v
.venv/bin/python scripts/test_emissions.py
# Run the decomposition with the separate post-accounting output tag added above.
```

## Tests

Use equation-based checks and a frozen-v4 integration regression. At the baseline
state, expect approximately 71.8918355 kN core and 258.1103756 kN total, within
0.001 kN; fuel flow remains 2.3182068896 kg/s within relative tolerance 1e-9.
The thrust change is exactly the previously subtracted internal momentum within
numerical precision. Temperature, fuel-flow and emissions inputs remain unchanged.

## Acceptance Criteria

1. Static thrust uses freestream momentum consistently across core and bypass.
2. The 16.5003810 kN accounting effect is reproduced; the remaining 52.7896244 kN
   shortfall is reported openly and is not eliminated by an unsourced input change.
3. Nozzle geometry/expansion assumptions are explicit; no unapproved physics model
   or calibrated parameter is introduced.
4. Required tests and emissions validation pass; protected artifacts are unchanged.
5. Original P5.2 scientific gates still govern whether calibration can resume.

## Rollback Notes

Revert the isolated accounting repair commit. Preserve both diagnostic reports;
no model weights or historical calibration files are replaced.

## Escalation Guidance

Medium-to-high complexity; Claude executor via the dispatcher, followed by review.
User approval is needed because the original plan explicitly requires escalation
when the thrust gap includes a structural defect. Any additional nozzle physics,
geometry choice, or adoption of a new airflow/FPR value needs separate justification.
