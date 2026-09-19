# Turbine PINN v5 — surrogate fidelity (P4.4) — 2026-09-18

Checkpoint `models/turbine_pinn_v5.pt` (git `0405f744`, seed 42, 3000 epochs). Claim: **surrogate of run_turbine_analytic (eta_poly 0.9); no independent accuracy**. Gate fixed before the run: p5 and raw T5 within 1 % of the analytic path on held-out engine models (max over conditions), and full-cycle thrust and TSFC within 1 %.

## Verdict: **GATE MISSED — retire the turbine PINN to an appendix; production stays analytic**

## Surrogate fidelity across the LTO envelope

| Split | Conditions | max \|Δp5\|/p5 | mean \|Δp5\|/p5 | max \|ΔT5\|/T5 (raw) | mean \|ΔT5\|/T5 (raw) |
|---|---|---|---|---|---|
| held-out models | 144 | 28.250 % | 16.316 % | 0.066 % | 0.027 % |
| training models | 576 | 33.589 % | 18.288 % | 0.073 % | 0.030 % |

Worst held-out p5 case: Trent 1000-D3 / TAKE-OFF / ATJ-SPK — analytic 3.791 bar, surrogate 2.720 bar (-28.250 %). Held-out τ range 0.259–0.393.

## Full cycle, calibration engine, `turbine_model="pinn"` vs analytic

| Mode | Fuel | Thrust analytic (kN) | Thrust surrogate (kN) | Δ | TSFC analytic | TSFC surrogate | Δ |
|---|---|---|---|---|---|---|---|
| TAKE-OFF | Jet-A1 | 241.61 | 229.37 | -5.066 % | 9.595 | 10.107 | +5.337 % |
| TAKE-OFF | HEFA-SPK | 241.58 | 229.30 | -5.083 % | 9.588 | 10.102 | +5.355 % |
| TAKE-OFF | FT-SPK | 241.58 | 229.30 | -5.083 % | 9.578 | 10.091 | +5.355 % |
| TAKE-OFF | ATJ-SPK | 241.43 | 228.92 | -5.183 % | 9.549 | 10.071 | +5.466 % |
| APPROACH | Jet-A1 | 82.04 | 78.11 | -4.796 % | 7.543 | 7.923 | +5.038 % |
| APPROACH | HEFA-SPK | 82.03 | 78.08 | -4.814 % | 7.538 | 7.920 | +5.058 % |
| APPROACH | FT-SPK | 82.03 | 78.08 | -4.815 % | 7.531 | 7.911 | +5.059 % |
| APPROACH | ATJ-SPK | 81.97 | 77.94 | -4.921 % | 7.509 | 7.897 | +5.176 % |
| IDLE | Jet-A1 | 26.53 | 26.93 | +1.485 % | 9.130 | 8.996 | -1.464 % |
| IDLE | HEFA-SPK | 26.53 | 26.92 | +1.487 % | 9.124 | 8.990 | -1.465 % |
| IDLE | FT-SPK | 26.53 | 26.92 | +1.487 % | 9.114 | 8.981 | -1.465 % |
| IDLE | ATJ-SPK | 26.51 | 26.91 | +1.496 % | 9.087 | 8.953 | -1.474 % |

## What this does and does not show

- The v5 turbine PINN reproduces the analytic work-consistent expansion it was trained on. That is a speed/differentiability claim about a surrogate of the production model.
- It is **not** evidence that either model matches a real turbine: no turbine measurement or CFD exists in this repository, and none was used.
- The legacy checkpoint `models/turbine_pinn.pt` remains as adjudicated in Phase 3 (p5 −41.5 %); it lacked the work-fraction input and could not represent the expansion ratio.

## Artifacts
- `outputs/turbine_surrogate_fidelity_v5.csv`
- `outputs/turbine_surrogate_cycle_check_v5.csv`
- `outputs/turbine_envelope_v5.csv`
