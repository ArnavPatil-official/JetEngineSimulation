# Nozzle LE-PINN retrain on the Sajben weak-shock case (P4.3) — 2026-09-26

Scored by `sajben_validation.py` (throat-height mapping, split guard, corrected in P4.2). Pre-registered bands on the worse wall: < 0.1 pass / 0.1–0.25 partial / > 0.25 fail. The training data (WIND RANS) itself scores 0.089 / 0.084 on this metric (P4.1 §3), so the pass band is reachable only by a near-perfect surrogate of the CFD.

## Every attempt, as registered

| Checkpoint | Attempt | Gate attempt? | Physics w | Epochs (best val at) | Internal val MSE | Collapsed | Upper L2 | Lower L2 | Worst | Degenerate | Split verified | Band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `le_pinn_sajben_v5.pt` | P4.3-attempt-1 | yes | 0.05 | 5000 (350) | 1.11e-02 | — | 0.258 | 0.187 | 0.258 | no | yes | **fail** |
| `le_pinn_sajben_v5_dataonly_ref.pt` | P4.3-reference-1-data-only | no (ablation) | 0.0 | 5000 (4999) | 5.37e-04 | — | 0.105 | 0.072 | 0.105 | no | yes | **partial (n/a)** |
| `le_pinn_sajben_v5_a2.pt` | P4.3-attempt-2 | yes | 0.05 | 5000 (300) | 1.21e-02 | — | 0.245 | 0.166 | 0.245 | no | yes | **partial** |
| `le_pinn_sajben_v5_a3_s42.pt` | P4.3-attempt-3 | yes | 0.05 | 5000 (4999) | 2.15e-02 | — | 0.156 | 0.112 | 0.156 | no | yes | **partial** |
| `le_pinn_sajben_v5_a3_dataonly_s42.pt` | P4.3-attempt-3-dataonly | no (ablation) | 0.0 | 5000 (4999) | 1.17e-03 | — | 0.156 | 0.074 | 0.156 | no | yes | **partial (n/a)** |
| `le_pinn_sajben_v5_a3_s43.pt` | P4.3-attempt-3 | yes | 0.05 | 5000 (4999) | 1.53e-02 | — | 0.144 | 0.114 | 0.144 | no | yes | **partial** |
| `le_pinn_sajben_v5_a3_dataonly_s43.pt` | P4.3-attempt-3-dataonly | no (ablation) | 0.0 | 5000 (4999) | 1.37e-03 | — | 0.106 | 0.061 | 0.106 | no | yes | **partial (n/a)** |
| `le_pinn_sajben_v5_a3_s44.pt` | P4.3-attempt-3 | yes | 0.05 | 5000 (4999) | 1.65e-02 | — | 0.175 | 0.140 | 0.175 | no | yes | **partial** |
| `le_pinn_sajben_v5_a3_dataonly_s44.pt` | P4.3-attempt-3-dataonly | no (ablation) | 0.0 | 5000 (4999) | 1.70e-03 | — | 0.082 | 0.067 | 0.082 | no | yes | **pass (n/a)** |

Velocity-profile L2 (x/H = 1.729 / 2.882 / 4.611 / 6.340):

- `le_pinn_sajben_v5.pt`: 0.106 / 0.170 / 0.336 / 0.336
- `le_pinn_sajben_v5_dataonly_ref.pt`: 0.062 / 0.110 / 0.161 / 0.161
- `le_pinn_sajben_v5_a2.pt`: 0.120 / 0.186 / 0.334 / 0.355
- `le_pinn_sajben_v5_a3_s42.pt`: 0.210 / 0.333 / 0.366 / 0.393
- `le_pinn_sajben_v5_a3_dataonly_s42.pt`: 0.071 / 0.130 / 0.158 / 0.147
- `le_pinn_sajben_v5_a3_s43.pt`: 0.162 / 0.252 / 0.325 / 0.338
- `le_pinn_sajben_v5_a3_dataonly_s43.pt`: 0.077 / 0.127 / 0.160 / 0.149
- `le_pinn_sajben_v5_a3_s44.pt`: 0.149 / 0.232 / 0.321 / 0.333
- `le_pinn_sajben_v5_a3_dataonly_s44.pt`: 0.061 / 0.098 / 0.161 / 0.151

## Outcome applied to the gate attempt(s)

- **P4.3-attempt-1: 0.258 on a single seed, within 0.008 of a band boundary — band not claimed.** A single draw this close to the line does not support a band call; see the multi-seed section for the attempt that carries the outcome.
- **P4.3-attempt-2: 0.245 on a single seed, within 0.005 of a band boundary — band not claimed.** A single draw this close to the line does not support a band call; see the multi-seed section for the attempt that carries the outcome.
- **P4.3-attempt-3 (mean of 3 seeds): PARTIAL (0.159).** The PINN stays non-production. The preprint reports a quantified near-miss against the Sajben benchmark: worse wall 0.159 vs the 0.10 gate, with the training data itself at 0.089. This is a publishable negative result.

## Ablation reference (not a gate attempt)

- `le_pinn_sajben_v5_dataonly_ref.pt` (physics weight 0.0): worse wall 0.105 (would fall in the *partial* band). Reported so the effect of the physics term is visible: the comparison between this row and the gate attempt is the ablation.
- `le_pinn_sajben_v5_a3_dataonly_s42.pt` (physics weight 0.0): worse wall 0.156 (would fall in the *partial* band). Reported so the effect of the physics term is visible: the comparison between this row and the gate attempt is the ablation.
- `le_pinn_sajben_v5_a3_dataonly_s43.pt` (physics weight 0.0): worse wall 0.106 (would fall in the *partial* band). Reported so the effect of the physics term is visible: the comparison between this row and the gate attempt is the ablation.
- `le_pinn_sajben_v5_a3_dataonly_s44.pt` (physics weight 0.0): worse wall 0.082 (would fall in the *pass* band). Reported so the effect of the physics term is visible: the comparison between this row and the gate attempt is the ablation.

## Multi-seed attempts — band called on the distribution

| Attempt | Config | Seeds | Worse-wall L2 per seed | Mean ± sd | Range | Ceiling-relative (mean − 0.089) | Band (mean) | Seeds agree? |
|---|---|---|---|---|---|---|---|---|
| P4.3-attempt-3 | tanh, μ=data, physics w 0.05 | 42, 43, 44 | 0.156 / 0.144 / 0.175 | 0.159 ± 0.016 | 0.144–0.175 | +0.070 | **partial** | yes |
| P4.3-attempt-3-dataonly | tanh, μ=data, physics w 0.0 | 42, 43, 44 | 0.156 / 0.106 / 0.082 | 0.115 ± 0.038 | 0.082–0.156 | +0.026 | **partial** (not claimed) | NO — straddles partial/pass |

**P4.3-attempt-3 vs its matched data-only ablation:** physics-on mean 0.159, data-only mean 0.115, difference +0.044 against a seed spread (larger sd) of 0.038 → physics-on is above data-only beyond the seed spread: **a negative result about this residual formulation**, reported as such (pre-registered reading 3).

The pass band (< 0.10) sits 0.011 above the training data's own score; the ceiling-relative column shows how much of each score is the surrogate's error rather than the CFD's. The gate itself is unchanged.

## Completion evidence for the terminal attempt (P5.1 guard)

Every registered run must have completed with exit code 0 before the terminal attempt is reported.

| Run | Seed | Checkpoint | Evidence | Exit | Check |
|---|---|---|---|---|---|
| P4.3-attempt-3 | 42 | `models/le_pinn_sajben_v5_a3_s42.pt` | launch-guard .done marker | 0 | sha256 578277ca5776 matches; finished 2026-09-26T02:33:53-0400 |
| P4.3-attempt-3-dataonly | 42 | `models/le_pinn_sajben_v5_a3_dataonly_s42.pt` | legacy log trailer (exit=N appended by the pre-guard launcher) | 0 | log names this checkpoint as saved; best val 1.171e-03 matches the checkpoint |
| P4.3-attempt-3 | 43 | `models/le_pinn_sajben_v5_a3_s43.pt` | launch-guard .done marker | 0 | sha256 bc3d536fd5a1 matches; finished 2026-09-26T02:33:07-0400 |
| P4.3-attempt-3-dataonly | 43 | `models/le_pinn_sajben_v5_a3_dataonly_s43.pt` | legacy log trailer (exit=N appended by the pre-guard launcher) | 0 | log names this checkpoint as saved; best val 1.371e-03 matches the checkpoint |
| P4.3-attempt-3 | 44 | `models/le_pinn_sajben_v5_a3_s44.pt` | launch-guard .done marker | 0 | sha256 0a48dff75f76 matches; finished 2026-09-26T02:33:54-0400 |
| P4.3-attempt-3-dataonly | 44 | `models/le_pinn_sajben_v5_a3_dataonly_s44.pt` | legacy log trailer (exit=N appended by the pre-guard launcher) | 0 | log names this checkpoint as saved; best val 1.702e-03 matches the checkpoint |

## Observations recorded for any future attempt (no re-tuning was done)

- `le_pinn_sajben_v5.pt`: best internal validation at epoch 350 of 5000; the run made no further progress after that point (see the training log for the learning-rate trajectory). Whether the registered scheduler interacts badly with the physics warm-up is a question for a separately registered attempt, not a reason to adjust this one.
- `le_pinn_sajben_v5_a2.pt`: best internal validation at epoch 300 of 5000; the run made no further progress after that point (see the training log for the learning-rate trajectory). Whether the registered scheduler interacts badly with the physics warm-up is a question for a separately registered attempt, not a reason to adjust this one.

Artifacts: `outputs/sajben_retrain_v5.csv`, checkpoints listed above, `outputs/logs/train_sajben_v5_*.log`.
