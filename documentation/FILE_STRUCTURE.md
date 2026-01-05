# Jet Engine Simulation - File Structure

```
JetEngineSimulation/
├── integrated_engine.py        # Orchestrates compressor → combustor → turbine → nozzle
├── optimize_blend.py           # Optuna demo for SAF blend search
├── data/                       # Mechanisms and reference data
│   ├── A1highT.yaml            # HyChem Jet-A1 mechanism (validation mode)
│   ├── A2NOx.yaml              # NOx mechanism (reference)
│   ├── creck_c1c16_full.yaml   # CRECK mechanism (blends mode)
│   ├── icao_engine_data.csv    # ICAO reference data
│   ├── processors/             # CTI/YAML conversion utilities
│   └── raw/                    # Source PDFs/mechanism text files
├── simulation/                 # Core physics modules
│   ├── compressor/             # Compressor model
│   ├── combustor/              # Combustor model + tests
│   ├── turbine/                # Turbine PINN + helpers
│   ├── nozzle/                 # Nozzle PINN + helpers/visualization
│   ├── engine_types.py         # Dataclasses for flow/metrics
│   ├── emissions.py            # Emissions estimators
│   ├── fuels.py                # Fuel surrogates and blending utilities
│   └── thermo_utils.py         # Thermo bridges (combustor → PINNs)
├── tests/                      # Regression/physics checks
│   ├── test_choking_detection.py
│   ├── test_nozzle_pinn_fix.py
│   ├── test_nozzle_regression.py
│   ├── verify_physics_corrections.py
│   └── verify_thermo_fix.py
├── evaluation/                 # Exploratory analysis scripts
│   ├── cantera_eda.py
│   ├── icao_eda.py
│   └── visualize_networks.py
├── documentation/              # Project docs (overview, quickstart, guides)
├── nozzle_pinn.pt              # Trained nozzle PINN checkpoint (v3.1+)
├── turbine_pinn.pt             # Trained turbine PINN checkpoint
├── nozzle_validation_dual_thermo.png
├── optuna_landscape.png
└── creck_results.csv           # Example output dataset
```
