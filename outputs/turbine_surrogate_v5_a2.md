# Turbine PINN v5 — surrogate fidelity (P4.4) — 2026-09-18

Checkpoint `models/turbine_pinn_v5_a2.pt` (git `4cfa9af1`, seed 42, 6000 epochs). Claim: **surrogate of run_turbine_analytic (eta_poly 0.9); no independent accuracy**. Gate fixed before the run: p5 and raw T5 within 1 % of the analytic path on held-out engine models (max over conditions), and full-cycle thrust and TSFC within 1 %.

## Verdict: **GATE MISSED — retire the turbine PINN to an appendix; production stays analytic**

## Surrogate fidelity across the LTO envelope

| Split | Conditions | max \|Δp5\|/p5 | mean \|Δp5\|/p5 | max \|ΔT5\|/T5 (raw) | mean \|ΔT5\|/T5 (raw) |
|---|---|---|---|---|---|
| held-out models | 144 | 6.390 % | 1.894 % | 0.262 % | 0.104 % |
| training models | 576 | 6.345 % | 1.751 % | 0.261 % | 0.084 % |

Worst held-out p5 case: Trent 1000-E / APPROACH / Jet-A1 — analytic 1.973 bar, surrogate 2.099 bar (+6.390 %). Held-out τ range 0.259–0.393.

## Full cycle, calibration engine, `turbine_model="pinn"` vs analytic

| Mode | Fuel | Thrust analytic (kN) | Thrust surrogate (kN) | Δ | TSFC analytic | TSFC surrogate | Δ |
|---|---|---|---|---|---|---|---|
| TAKE-OFF | Jet-A1 | 241.61 | 242.01 | +0.165 % | 9.595 | 9.579 | -0.165 % |
| TAKE-OFF | HEFA-SPK | 241.58 | 241.97 | +0.158 % | 9.588 | 9.573 | -0.158 % |
| TAKE-OFF | FT-SPK | 241.58 | 241.95 | +0.154 % | 9.578 | 9.564 | -0.154 % |
| TAKE-OFF | ATJ-SPK | 241.43 | 241.70 | +0.113 % | 9.549 | 9.539 | -0.113 % |
| APPROACH | Jet-A1 | 82.04 | 82.49 | +0.551 % | 7.543 | 7.502 | -0.548 % |
| APPROACH | HEFA-SPK | 82.03 | 82.48 | +0.545 % | 7.538 | 7.497 | -0.542 % |
| APPROACH | FT-SPK | 82.03 | 82.47 | +0.542 % | 7.531 | 7.490 | -0.539 % |
| APPROACH | ATJ-SPK | 81.97 | 82.38 | +0.506 % | 7.509 | 7.471 | -0.504 % |
| IDLE | Jet-A1 | 26.53 | 26.45 | -0.305 % | 9.130 | 9.158 | +0.306 % |
| IDLE | HEFA-SPK | 26.53 | 26.45 | -0.303 % | 9.124 | 9.151 | +0.304 % |
| IDLE | FT-SPK | 26.53 | 26.45 | -0.304 % | 9.114 | 9.142 | +0.305 % |
| IDLE | ATJ-SPK | 26.51 | 26.43 | -0.294 % | 9.087 | 9.114 | +0.294 % |

## What this does and does not show

- The v5 turbine PINN reproduces the analytic work-consistent expansion it was trained on. That is a speed/differentiability claim about a surrogate of the production model.
- It is **not** evidence that either model matches a real turbine: no turbine measurement or CFD exists in this repository, and none was used.
- The legacy checkpoint `models/turbine_pinn.pt` remains as adjudicated in Phase 3 (p5 −41.5 %); it lacked the work-fraction input and could not represent the expansion ratio.

## Artifacts
- `outputs/turbine_surrogate_fidelity_v5_a2.csv`
- `outputs/turbine_surrogate_cycle_check_v5_a2.csv`
- `outputs/turbine_envelope_v5.csv`
