# Phase 6 execution status

Plan: `docs/plan.md` (Phase 6, approved 2026-09-26). Branch `phase4`, start
`6a56dcc`. Test floor at start: 132 passed / 1 skipped
(`.venv/bin/python -m pytest tests/ -v`).

## Log

| Step | Commit | Result |
|---|---|---|
| Prep: Cantera Solution reuse | `ba333a8` | Reusing one Solution per role, reset to its as-constructed state per call: bit-identical to fresh Solutions (24 randomized combustor cases; 5 full cycles vs a clean HEAD worktree); ~5x faster. Without the reset, equilibrate('HP') differs at ~1e-7. Outside the plan's file list; needed to make P6.1 computationally feasible. |
| P6.1 Step 1 — F-A test | `f2dead8` | `tests/test_holdout_informativeness.py`: v4 diagnosis pinned (2 pass); v5 informativeness test fails until the v5 CSV exists (as intended). F-A added to `outputs/parameter_provenance.md`; manifest M2 flagged SUPERSEDED. 177 rows: model 2.46016229 % vs AE3-ratio baseline 3.21571920 %; 171 rows (excl. AE3 models): 2.50278929 % vs 3.24190221 %. |
| P6.1 Step 2 — run_at_thrust | `920420d`, `1f997f8` | Brent solve of φ for a static-thrust target; bracket from the lowest φ at which the cycle closes to min(φ = 1, T4 guard 2111 K). Explicit `ThrustTargetUnreachable`. 8 tests pass (round trip within 1e-6, monotone thrust and T4, warm start gives the same root, clean failures). |
| Feasibility probe (provisional) | — | AE3 inputs only (no fuel flow read); provisional p_loss 0.045, β 0.8, η_b 0.999. At the hand-set 79.9 kg/s core flow the take-off target (310.9 kN) is unreachable inside the T4 guard (max 274.6 kN); at 96 kg/s it is reached at φ 0.545, T4 1903 K. Part-power feasibility is narrow in (W_ref, k_pi, k_mdot). |
| P6.1 Step 3 — registration | (this commit) | `docs/phase6_p61_registration.md`, `outputs/phase6/p61_registration.json`, split `outputs/phase6/split_p61.json`. |

## Outstanding gates

- P6.1 Steps 4–5 (pilot identifiability, full fit, held-out test).
