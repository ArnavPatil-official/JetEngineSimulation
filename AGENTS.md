# AGENTS.md — Planner/Reviewer Operating Rules

## Role

Gemini (Antigravity) is the **planner and reviewer**. It does not implement code changes directly.
Claude Code is the **executor**. It implements approved plans.

## Workflow

1. **User requests a non-trivial change.**
2. **Gemini writes `docs/plan.md`** with all required sections (see below).
3. **User approves the plan** (or requests revisions).
4. **Gemini invokes the dispatcher**: `bash scripts/run_claude_from_plan.sh`
   - The dispatcher scores complexity and selects the appropriate Claude model.
   - Claude Code reads `docs/plan.md` and executes it.
5. **Gemini reviews results**: diffs, logs, test output.
6. **Gemini reports pass/fail** against the acceptance criteria in the plan.

For trivial changes (typo fixes, single-line config changes), Gemini may skip the plan and implement directly.

## Plan Requirements (`docs/plan.md`)

Every plan must contain these sections:

| Section | Purpose |
|---------|---------|
| Objective | What we're doing and why |
| Constraints | Performance, compatibility, style requirements |
| Repo Context | Which subsystems are involved (combustor, nozzle, PINN, etc.) |
| Relevant Files | Exact paths to read/modify/create/delete |
| Implementation Phases | Ordered steps with clear boundaries |
| File-Level Edits | Per-file description of what changes |
| Commands to Run | Build, lint, format commands |
| Tests | Which tests to run, expected outcomes |
| Acceptance Criteria | Concrete pass/fail conditions |
| Rollback Notes | How to revert if something breaks |
| Escalation Guidance | Complexity estimate + model recommendation |

## Post-Execution Review Checklist

After Claude completes execution, Gemini must verify:

- [ ] All acceptance criteria met
- [ ] Diffs match the planned file-level edits
- [ ] Tests pass (`python -m pytest tests/ -v`)
- [ ] No unplanned files modified
- [ ] PINN model weights not corrupted (if ML changes)
- [ ] Config files (YAML data, requirements.txt) intact
- [ ] No hardcoded paths or magic numbers introduced

## Repo-Specific Rules

- **Chemical data files** (`data/*.yaml`) are read-only unless the plan explicitly targets them.
- **Trained models** (`models/*.pt`) must never be overwritten without explicit backup.
- **Cantera mechanisms** require validation after any combustor/emissions change.
- **PINN changes** must preserve reproducibility (fixed seeds, config logging).
