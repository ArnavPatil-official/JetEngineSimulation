# Phase 6 interim review and continuation — 2026-09-26

## Objective

Correct the issues found in the first execution pass, then continue the already
approved `docs/plan.md` through completion or an actual §9 escalation. The user
approved all three Phase 6 decisions; no renewed approval is needed for this work.
The planner stopped the first dispatcher before any pilot/full fit because its
registration allowed an unsourced engine parameter contrary to decision 2.
HEAD at review: `59d9665`. That registration is rejected for use in fitting until
corrected. No pilot, v5 calibration, or held-out evaluation has run.

## Constraints

Retain the approved plan, split, historical artifacts, chemical YAML and models.
Preserve the rejected registration as evidence; amend it explicitly before the
first pilot. Do not hide the rejected version or tune thresholds after outcomes.
Leave `.DS_Store` alone. The first dispatch log ends in `Execution error` despite
its wrapper printing DONE; it was reviewer-interrupted, not completed.

## Repo Context

`run_at_thrust` and F-A tests are implemented. Planner independently reproduced
F-A, verified 40 protected hashes, and ran the first seven solver tests (all pass).
The subsequent warm-start change has a reproduced failure-path defect. The split
uses metadata only and merges input-identical model names: 9 calibration groups
(15 names,31 records), 9 held-out groups (13 names,29 records). Retain this split.

## Relevant Files

- MODIFY `integrated_engine.py`, `tests/test_run_at_thrust.py`.
- MODIFY `scripts/optimization/lto_v5.py`, `scripts/validation/phase6_register_p61.py`.
- ARCHIVE rejected `outputs/phase6/p61_registration.json` and
  `docs/phase6_p61_registration.md` under a clearly named superseded path; write
  corrected active registration and link both versions in the status record.
- CREATE `outputs/phase6_execution_status.md` and committed cache regression
  evidence/tests as appropriate under `tests/` and `outputs/logs/`.
- Continue other exact paths/phases in the approved plan after these corrections.

## Implementation Phases

1. Fix and test the solver failure path, and verify caching is order-independent.
2. Correct the sourcing/definition issues and commit the amended registration
   before any pilot. Preserve its split, objective, margins and thresholds.
3. Resume P6.1 and dependent Phase 6 work. Finish independent P6.5 evidence work
   even if a numerical gate blocks dependent studies. No additional PINN training.
4. Run required checks and record pass/fail against every reached gate. Do not
   claim the whole plan complete or freeze with unresolved required repo checks.

## File-Level Edits / Required corrections

### R6-A: unsourced beta is not permitted by the approved rule

The registration and `FIXED_RANGES` currently fix beta=0.8 with [0.7,0.9], marked
ILLUSTRATIVE and no source. Decision 2 explicitly requires an engine parameter
without a citable range to be data-identified or dropped. The illustrative
exception in P6.2 is for surrogate compositions, not arbitrary engine parameters.

Find a source for the parameter actually implemented (air bypassing equilibrium
and remixed BEFORE the turbine); a turbine cooling-flow fraction injected INTO
the turbine is not automatically the same quantity. If no suitable range is
available, exercise the plan's allowed drop option explicitly: consider removing
the burner/dilution split from the v5 formulation (single-zone equilibrium using
all core air), document the change in model structure, and preserve legacy beta
behavior for frozen/reproduction paths. Do not disguise an unsourced fitted/fixed
beta as a different named constant. If neither a sourced range nor an honest
reduced formulation is defensible, record the precise scientific gate and stop
its dependent work. Do not proceed with the rejected illustrative range.

### R6-B: warm-started guard violation raises the wrong failure

Reproduction at v4 design values:
`pi_c=43.2, mass_flow_core=79.9, pressure_loss=.0442, beta=.8, fpr=1.45,
eta_b=.9963`, Jet-A1. `run_full_cycle(phi=.9)` gives thrust291.72574888kN,
T4=2366.771K. Cold `run_at_thrust(target)` raises ThrustTargetUnreachable with
maximum274.343kN at the guard. Warm `phi_guess=.89` instead raises generic
ValueError('Computed non-positive nozzle exit velocity').

Cause: after a warm root exceeds T4, `fail_high(lo)` passes the original .05
lower endpoint to `guard_phi`, although that endpoint does not close the cycle.
Use a valid, closing bracket to diagnose guarded failures; cold and warm paths
must agree on reachability and the guarded bound. Add this regression. Validate
finite inputs and do not conceal unrelated configuration/programming exceptions
as normal unreachable rows. In the calibration driver, unexpected errors must
be visible and resolved, not silently treated as ordinary optimization penalties.
The reproduction log is `/tmp/phase6_review/warm_guard_review.log`.

### R6-C: efficiency ranges and energy proxy definitions

Replace the approximate '-0.005' fan polytropic-to-isentropic adjustment with the
actual conversion at the relevant FPR; document whether bounds are an envelope
or transformed per draw. Keep the fixed values inside the resulting ranges.
The compressor conversion already uses an explicit equation; document its
constant-gamma/rated-OPR approximation versus the variable-cp production model.

The proposed training-CO/HC-derived eta_b may be retained if its assumptions are
explicit: it is an energy-efficiency proxy used in a temperature-rise scaling,
not a newly identified inverse-cycle parameter. Provide a reproducible thermo
calculation for the hardcoded CO/fuel heating values, phase convention and the
HC-as-fuel approximation. It must use calibration-group records only. Keep any
held-out NOx correlation fit excluded in worker initialization before reporting
v5 NOx numbers; do not emit a full-data-fit NOx value as held-out evidence.

### R6-D: reproducible cache evidence

Solution reuse is a reasonable necessary performance repair, but its only
committed evidence is currently a commit message about temporary benchmarks.
Add a small meaningful regression comparing interleaved fuel/state evaluations
against fresh Solutions (A/B/A order), covering equilibrium state and relevant
fuel-flow/thermal outputs. Do not rely on the temporary old-worktree being present
for reproducibility. Check all shipped mechanism profiles after this combustor
change. Temporary old/new full-cycle dumps are `/tmp/cyc_{head,new,new2}.json`.

### R6-E: supplied papers / independent P6.5 audit

The supplied Ma PDF was reviewed at pp4-6 (including renderedpp5-6). Audit the
numbered Eqs23-24 (stress divergence and full viscosity-product derivatives) and
Eqs31-33 (weights based on CURRENT LOSSES, unlike repo epoch-based weights).
Fig2 has a simplified Laplacian sketch: distinguish its sketch from numbered
equations, and do not assert the authors' implementation without inspecting it.
Journal header: AST168(2026)111002, available online2025-09-23; distinguish dates.
The paper's in-range CFD holdout does not establish external-geometry experimental
accuracy. No new training is authorized.

Wang explicitly uses tanh; the second-derivative rationale is a mathematical
interpretation unless a passage actually states it. Uy/San Juan's Shell2020
80% statement is context, not proof of the repo factor's historical origin.
Kuzhagaliyeva p3 Table2 compares69 held-out mixtures with linear-by-mole mixing.

Supplied PDFs are locally available in
`/Users/arnavpatil/.t3/userdata/attachments/`; every filename is the common prefix
`cc74e140-b743-42fd-af4e-bbc46720c8c1-`, the following ID, and `-pdf.pdf`:

| Paper | ID |
|---|---|
| Ma | eb4f785e-bdc3-42b3-84bf-5ed5f127eabf |
| Uy/San Juan | 5d6d03a2-b2ab-44dc-bd42-4acea02f1177 |
| Kuzhagaliyeva | fc2c66b8-7793-463a-9a6b-0a21574095a4 |
| Gal | a634d3bf-a082-489f-8ce5-9ff00d1723b0 |
| Nath | 1921bcaa-9559-4007-8bdf-589ff32ed8a6 |
| Sahin | 6e07d1e0-03bd-45eb-a95e-c26f71d74acd |
| Wang | f8df93b7-4bd0-4e23-872e-73d5c1ee6eb2 |

Temporary text extractions are in `/tmp/phase6_review/` (not canonical artifacts).
Do not introduce those absolute local paths as dependencies of production code.

## Commands to Run

Use `.venv/bin/python`; dispatch via `bash scripts/run_claude_from_plan.sh
--plan /Users/arnavpatil/Documents/JetEngineSimulation/docs/plan_phase6_review.md
--force-model opus`. Run the approved phase commands after corrected registration.

## Tests

Run solver/cache regressions, then `.venv/bin/python -m pytest tests/ -v`,
`.venv/bin/python scripts/test_emissions.py`, shipped mechanism validation and the
40-file protected-hash check. Until v5 exists, report the intentional missing-v5
test failure separately; never conceal it or call the suite fully passing.

## Acceptance Criteria

All R6-A through R6-E addressed, active registration complies with the approved
rule and predates pilot/full fitting, then the original phase gates applied.
Unexpected code errors cannot count as scientific identifiability failures.

## Rollback Notes

Preserve commits/evidence and user work. No reset. Archive superseded registration
with an explicit cross-reference so the timing and reason remain auditable.

## Escalation Guidance

High complexity; Claude Opus executor. Use the original §9 gates, after resolving
ordinary implementation defects. Continue independent authorized evidence work
before reporting any real scientific blocker. Do not request blanket reapproval.
