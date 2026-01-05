# HyChem Validation Mode

## Purpose
- High-fidelity Jet-A1 benchmark using Stanford HyChem (`data/A1highT.yaml`).
- Keeps CRECK available for blend studies but isolates validation runs so results are not mixed.

## How to Run
- CLI: `python integrated_engine.py --mode validation`
- Programmatic:
```python
from integrated_engine import IntegratedTurbofanEngine
engine = IntegratedTurbofanEngine(mechanism_profile="validation")
result = engine.run_hychem_validation_case(phi=0.5, combustor_efficiency=0.98)
print(f"Thrust: {result['performance']['thrust_kN']:.2f} kN")
print(f"TSFC: {result['performance']['tsfc_mg_per_Ns']:.2f} mg/(N·s)")
```

## Guardrails
- Fuel is fixed to Jet-A1; do not pass SAF blends into HyChem mode.
- Validation outputs are not comparable to CRECK-based blend runs because mechanisms differ.
- Uses the same turbine/nozzle PINN pipeline and fallback logic as blends mode; TSFC is set to infinity if thrust ≤ 0.

## What to Expect
- Prints compressor/combustor/turbine/nozzle summaries plus a validation metadata block:
  - `mechanism`: `HyChem`
  - `mechanism_file`: `data/A1highT.yaml`
  - `fuel`: `Jet-A1`
  - `purpose`: `ICAO validation benchmark`

## Related Files
- `integrated_engine.py`: `run_hychem_validation_case` and CLI mode handling.
- `data/A1highT.yaml`: HyChem mechanism.
- `simulation/fuels.py`: Jet-A1 surrogate used for validation.
