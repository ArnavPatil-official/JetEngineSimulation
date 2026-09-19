# Nozzle LE-PINN retrain on the Sajben weak-shock case (P4.3) — 2026-09-18

Scored by `sajben_validation.py` (throat-height mapping, split guard, corrected in P4.2). Pre-registered bands on the worse wall: < 0.1 pass / 0.1–0.25 partial / > 0.25 fail. The training data (WIND RANS) itself scores 0.089 / 0.084 on this metric (P4.1 §3), so the pass band is reachable only by a near-perfect surrogate of the CFD.

## Every attempt, as registered

| Checkpoint | Attempt | Gate attempt? | Physics w | Epochs (best val at) | Internal val MSE | Collapsed | Upper L2 | Lower L2 | Worst | Degenerate | Split verified | Band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `le_pinn_sajben_v5.pt` | P4.3-attempt-1 | yes | 0.05 | 5000 (350) | 1.11e-02 | — | 0.258 | 0.187 | 0.258 | no | yes | **fail** |
| `le_pinn_sajben_v5_dataonly_ref.pt` | P4.3-reference-1-data-only | no (ablation) | 0.0 | 5000 (4999) | 5.37e-04 | — | 0.105 | 0.072 | 0.105 | no | yes | **partial (n/a)** |

Velocity-profile L2 (x/H = 1.729 / 2.882 / 4.611 / 6.340):

- `le_pinn_sajben_v5.pt`: 0.106 / 0.170 / 0.336 / 0.336
- `le_pinn_sajben_v5_dataonly_ref.pt`: 0.062 / 0.110 / 0.161 / 0.161

## Outcome applied to the gate attempt(s)

- **P4.3-attempt-1: FAIL (0.258).** The nozzle PINN is retired to an appendix; the title and framing drop the accuracy claim; production stays analytic.

## Ablation reference (not a gate attempt)

- `le_pinn_sajben_v5_dataonly_ref.pt` (physics weight 0.0): worse wall 0.105 (would fall in the *partial* band). Reported so the effect of the physics term is visible: the comparison between this row and the gate attempt is the ablation.

## Observations recorded for any future attempt (no re-tuning was done)

- `le_pinn_sajben_v5.pt`: best internal validation at epoch 350 of 5000; the run made no further progress after that point (see the training log for the learning-rate trajectory). Whether the registered scheduler interacts badly with the physics warm-up is a question for a separately registered attempt, not a reason to adjust this one.

Artifacts: `outputs/sajben_retrain_v5.csv`, checkpoints listed above, `outputs/logs/train_sajben_v5_*.log`.
