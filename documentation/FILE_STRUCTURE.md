# Jet Engine Simulation - File Structure

```
JetEngineSimulation/
├── integrated_engine.py        # Orchestrates compressor → combustor → turbine → nozzle
├── requirements.txt            # Python dependencies
│
├── simulation/                 # Core physics modules
│   ├── __init__.py
│   ├── compressor/             # Compressor model
│   ├── combustor/              # Combustor model + tests
│   ├── turbine/                # Turbine PINN + helpers
│   ├── nozzle/                 # Nozzle PINN + helpers/visualization
│   ├── engine_types.py         # Dataclasses for flow/metrics
│   ├── emissions.py            # Emissions estimators
│   ├── fuels.py                # Fuel surrogates and blending utilities
│   └── thermo_utils.py         # Thermo bridges (combustor → PINNs)
│
├── models/                     # Trained PINN checkpoints
│   ├── turbine_pinn.pt         # Trained turbine PINN checkpoint
│   └── nozzle_pinn.pt          # Trained nozzle PINN checkpoint (v3.1+)
│
├── data/                       # Mechanisms and reference data
│   ├── A1highT.yaml            # HyChem Jet-A1 mechanism (validation mode)
│   ├── A2NOx.yaml              # NOx mechanism (reference)
│   ├── creck_c1c16_full.yaml   # CRECK mechanism (blends mode)
│   ├── icao_engine_data.csv    # ICAO reference data
│   ├── processors/             # CTI/YAML conversion utilities
│   └── raw/                    # Source PDFs/mechanism text files
│
├── scripts/                    # Runnable scripts
│   ├── optimization/
│   │   ├── optimize_blend.py   # Multi-objective SAF blend optimization
│   │   └── calibrate_lto.py    # LTO calibration against ICAO data
│   ├── visualization/
│   │   ├── pareto_visual.py    # Pareto front 2D/3D plots
│   │   ├── marked_visuals.py   # Parallel-coordinates with Pareto highlights
│   │   ├── optimization_plot.py # Parse logs and plot
│   │   ├── plot_validation.py  # ICAO validation bar chart
│   │   ├── visualize_results.py # PINN physics profiles
│   │   └── nozzle_2d_geometry.py # Nozzle wall contour generator
│   ├── generate_report.py      # PDF report generator
│   ├── test_emissions.py       # Emissions estimator test
│   └── verify_requirements.py  # Requirements verification
│
├── outputs/                    # Generated outputs
│   ├── plots/                  # Generated images and reports
│   └── results/                # CSV data and logs
│
├── tests/                      # Regression/physics checks
│   ├── test_choking_detection.py
│   ├── test_nozzle_pinn_fix.py
│   ├── test_nozzle_regression.py
│   ├── verify_physics_corrections.py
│   └── verify_thermo_fix.py
│
├── evaluation/                 # Exploratory analysis scripts
│   ├── cantera_eda.py
│   ├── icao_eda.py
│   └── visualize_networks.py
│
└── documentation/              # Project docs
    ├── COMPREHENSIVE_DOCUMENTATION.md
    ├── FILE_STRUCTURE.md
    ├── HYCHEM_VALIDATION_SUMMARY.md
    ├── NOZZLE_PINN_GUIDE.md
    ├── QUICKSTART_FUEL_DEPENDENT.md
    ├── CHANGELOG_HYCHEM.md
    ├── EMISSIONS_ESTIMATOR_SUMMARY.md
    └── archived/
```

> **Note:** All scripts in `scripts/` should be run from the project root directory.
> Relative paths (e.g. `data/...`, `models/...`, `outputs/...`) resolve from the CWD.
