# Phase 5 execution review and continuation

## Objective

Complete the already-authorized Phase 5 plan in `docs/plan.md`, addressing the
review findings below first. This is corrective review of the current execution,
not authorization to bypass any scientific escalation gate.

The first dispatcher was interrupted by the reviewer while its scratch profile
was running. Its wrapper reported exit 0 despite `Execution error`; that is NOT
evidence of completion. Existing edits are retained for review. The scratch
profile process was stopped because it spent several minutes without completing
even the two-parameter pilot. No registered training process was interrupted.

## Constraints

- All original plan constraints and gates remain in force.
- Do not stop or relaunch the three active Sajben jobs (PIDs 93999, 94000, 94001
  at review time). They are detached, CPU, sleep-proof, with their own guards.
- Preserve every existing checkpoint and v1–v4 artifact byte-for-byte.
- No production cycle change or v5 fitting before the structural-gap gate is resolved.
- Do not overwrite the original plan; retain both plans and review evidence.
- Keep commits narrow; do not commit growing training logs while jobs write them.

## Repo Context

P5.0 housekeeping and P5.1 launch guard are committed through `b5a4528`.
The import-safe `calibrate_lto.py` refactor and initial `identifiability_profile.py`
are uncommitted. P5.0 needs its post-change full test count recorded. Baseline
reviewer tests: 104 passed, 1 skipped, 9 warnings; `/tmp/jetengine-phase5-baseline-pytest.log`.

## Relevant Files

| Action | Path |
|---|---|
| MODIFY | `scripts/validation/sajben_report_p43.py` |
| MODIFY | `outputs/parameter_provenance.md` |
| MODIFY | `scripts/validation/heat_loss_provenance.md` |
| MODIFY | `scripts/optimization/calibrate_lto.py` |
| MODIFY | `scripts/validation/identifiability_profile.py` |
| CREATE | `tests/test_sajben_report_guard.py` |
| CREATE | `tests/test_calibration_identifiability.py` |
| CREATE | `scripts/validation/takeoff_thrust_gap.py` |
| CREATE | `outputs/phase5_execution_status.md` |

Other files and outputs remain as authorized in the original plan.

## Implementation Phases

### Phase 1 — Correct review findings

1. The terminal guard checks all six files on disk but `main()` scores only the
   supplied positional paths. Once all six exist, passing one checkpoint can
   therefore publish a partial terminal report. Require the scored selection to
   include exactly one of every registered seed/configuration of attempt 3, or
   expand it deterministically to the complete set. Reject duplicates/mismatched
   identities. Preserve reporting of earlier attempts. Add focused tests that
   mock completion evidence and exercise selection, missing/nonzero/mismatched
   completion evidence, and refusal before either report output is written.
2. In `parameter_provenance.md`, the Phase 1 phi row was incorrectly relabeled
   as jointly unidentified with a fitted airflow exponent. Phase 1 used fixed
   per-mode airflow scales, so F2's k_mdot ridge applies to the part-power v2–v4
   formulation. Scope statements carefully; eta_b and pressure loss remain inert.
   Do not claim the v4 sampler value is the exact optimum just because the
   parameter is structurally identifiable. Remove the unsupported surviving claim
   in `heat_loss_provenance.md` that casing loss is below eta_b's resolution;
   the point of §5 is that fuel-flow data provide no such bound. Label obsolete
   Phase 4 future-tense recommendations as superseded rather than current facts.

### Phase 2 — Complete v4 diagnostic and decomposition

3. Make the v4 profile computationally practical. Its fuel-flow objective has the
   exact equation in F2; profile/re-optimize that objective analytically or using
   fast deterministic bounded optimization. Cross-check against actual cycle
   evaluations at the frozen point, ridge points, and representative parameter
   variations. Report any feasibility/crash constraint distinction. Do not rely
   on a global monkeypatch/pool of mutable Cantera objects validated at only one
   point. Preserve a route for future combined objectives without pretending the
   fuel-only shortcut implements them. Keep a minimum-width/flat-valley check so
   box-clipped ridges cannot count as identified. Do not tune the verdict solely
   to force F2; document the structural basis and numerical checks.
4. Add the real regression tests before changing the objective, run v4 profiling,
   and record the verdict and explicit ridge measurements. Review the existing
   import-safe calibration refactor for preserved behavior and artifact guards.
5. Implement/run the planned take-off decomposition. Reviewer independently
   reproduced core 55.3914545 + bypass 186.2185401 = 241.6099946 kN. The analytic
   nozzle subtracts turbine-exit momentum (82.2182069 kg/s × 200.6901102 m/s =
   16.5003810 kN). Restoring that term alone gives 258.1103756 kN, still
   52.7896244 kN below target. These are diagnostic counterfactuals, not a fix.
   Check nozzle pressure/area/choking treatment and input sensitivities as planned.
   NASA's general thrust equation subtracts freestream inlet momentum:
   https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-force/
   Stop and escalate with the decomposition if structural, as the original plan
   requires. Do not fit around a defect or silently adopt an input value.

### Phase 3 — Validate and record status

6. Run the full test suite. Verify existing protected data/model hashes against
   `/var/folders/_s/vc8xfnw52bj84b3kjrvb62zw0000gn/T/phase5-review-lf535b48/protected_sha256.json`
   (this is a session-only evidence location, not a path to hardcode in repo code).
7. Record completed work, tests, active jobs and logs, any gate, and next required
   action in `outputs/phase5_execution_status.md`. Commit completed artifacts.
   If a gate is reached, end with the evidence; leave independent training alive.
   Otherwise continue the original plan without asking again for ordinary work.

## File-Level Edits

Per-file changes are specified in phases 1–3 above. No changes to training
hyperparameters, original checkpoints, chemical YAML, or production cycle equations.

## Commands to Run

```bash
.venv/bin/python scripts/validation/identifiability_profile.py --calibration outputs/calibration_trent1000_ae3_v4.json
.venv/bin/python scripts/validation/takeoff_thrust_gap.py
.venv/bin/python -m pytest tests/ -v
```

## Tests

Focused reporter tests must prove partial selection cannot publish a terminal
outcome. Identifiability tests must reproduce the v4 ridge and phi_to-only verdict
using computed values. Full suite must meet the 104-pass/1-skip baseline, with new
tests increasing coverage. Do not mark P5.1 complete before all runs finish.

## Acceptance Criteria

- Review findings corrected and tests pass.
- Reproducible v4 profile and take-off decomposition written and reviewed.
- Scientific gates honored; status distinguishes completed work from live jobs.
- Protected artifacts unchanged; no fabricated exit status or result.

## Rollback Notes

Revert specific corrective commits if needed; preserve original artifacts and live
training. Archive superseded diagnostic scratch outputs rather than deleting them.

## Escalation Guidance

High complexity; recommended Claude Opus. Structural thrust defects require user
decision under `docs/plan.md`, P5.2 Step 2. If encountered, return the evidence and
a precise proposed repair scope; do not implement a production physics fix.
