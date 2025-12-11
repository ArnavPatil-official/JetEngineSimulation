# Jet Engine Simulation - File Structure

## Project Overview
This project contains a comprehensive jet engine simulation with components for compressor, combustor, turbine, and nozzle modeling, along with optimization and evaluation tools.

## Directory Structure

```
JetEngineSimulation/
├── data/                           # Data files and configuration
│   ├── A1highT.yaml                # High temperature jet fuel mechanism
│   ├── A2NOx.yaml                  # NOx formation mechanism
│   ├── creck_c1c16_full.yaml       # CRECK full mechanism
│   ├── icao_engine_data.csv        # ICAO engine performance data
│   ├── isooctane.yaml              # Isooctane fuel mechanism
│   ├── n_dodecane_hychem.yaml      # n-Dodecane HyChem mechanism
│   │
│   ├── processors/                 # Data processing scripts
│   │   ├── cti_to_yaml.py          # Convert CTI to YAML format
│   │   ├── engine_data_conversion.py  # Engine data conversion utilities
│   │   └── mechanism_conversion.py # Chemical mechanism conversion
│   │
│   └── raw/                        # Raw data files
│       ├── ICAO_RR_TRENT_1000/     # Rolls-Royce Trent 1000 ICAO data
│       │   └── [83 PDF files]      # Engine certification documents
│       │
│       └── chemical_mechanisms/    # Chemical kinetics mechanisms
│           ├── jet_fuel/           # Jet fuel mechanisms
│           │   ├── A1highT.cti
│           │   └── A2NOx.cti
│           │
│           ├── kinetics/           # Kinetic mechanisms
│           │   ├── CRECK_2003_TOT_HT_LT.CKI.txt
│           │   ├── NC12H26_Hybrid_2019-10-17_mech.txt
│           │   ├── c7_c12_2methylalkanes_c8_c12_nalkanes_v1.1_mech_CnF_inp.txt
│           │   ├── ic8_ver3_mech.txt
│           │   └── icetane_2009-06-21_mech.txt
│           │
│           └── thermodynamics/     # Thermodynamic data
│               ├── CRECK_2003_TOT_HT_LT.CKT.txt
│               ├── NC12H26_Hybrid_2019-10-17_therm.txt
│               ├── c7_c20_2methylalkanes_c8_c16_nalkanes_v1_therm_CnF_dat.txt
│               ├── isocetane_2009-06-21_therm_dat_v15.txt
│               └── n_heptane_v3.1_therm.dat.txt
│
├── evaluation/                     # Analysis and evaluation scripts
│   ├── cantera_eda.py              # Cantera exploratory data analysis
│   └── icao_eda.py                 # ICAO data exploratory analysis
│
├── optimization/                   # Optimization algorithms
│   └── optimize_blend.py           # Fuel blend optimization
│
├── simulation/                     # Core simulation modules
│   ├── __init__.py                 # Package initialization
│   ├── engine.py                   # Main engine simulation
│   ├── fuels.py                    # Fuel properties and models
│   │
│   ├── combustor/                  # Combustion chamber simulation
│   │   ├── combustor.py            # Combustor physics and chemistry
│   │   └── test_combustor.py       # Combustor unit tests
│   │
│   ├── compressor/                 # Compressor simulation
│   │   └── compressor.py           # Compressor thermodynamics
│   │
│   ├── nozzle/                     # Nozzle simulation
│   │   ├── nozzle.py               # Nozzle flow calculations
│   │   └── visualize_nozzle.py     # Nozzle visualization tools
│   │
│   └── turbine/                    # Turbine simulation
│       ├── __init__.py             # Turbine package initialization
│       ├── turbine.py              # Turbine thermodynamics
│       ├── turbine_boundary.py     # Turbine boundary conditions
│       └── visualize_turbine.py    # Turbine visualization tools
│
├── nozzle_pinn.pt                  # Pre-trained nozzle PINN model
├── turbine_pinn.pt                 # Pre-trained turbine PINN model
│
└── .venv/                          # Python virtual environment
    └── [virtual environment files]
```

## Component Breakdown

### Core Simulation Components

1. **Compressor** ([simulation/compressor/](simulation/compressor/))
   - Handles compression stage thermodynamics
   - Pressure ratio and efficiency calculations

2. **Combustor** ([simulation/combustor/](simulation/combustor/))
   - Combustion chemistry using Cantera
   - NOx and emissions modeling
   - Temperature and pressure evolution

3. **Turbine** ([simulation/turbine/](simulation/turbine/))
   - Turbine expansion calculations
   - Boundary condition handling
   - Visualization capabilities

4. **Nozzle** ([simulation/nozzle/](simulation/nozzle/))
   - Exhaust nozzle flow modeling
   - Thrust calculations
   - Flow visualization

### Supporting Modules

- **Fuels** ([simulation/fuels.py](simulation/fuels.py))
  - Fuel property database
  - Blend composition handling
  - Chemical mechanism integration

- **Engine** ([simulation/engine.py](simulation/engine.py))
  - Complete engine cycle integration
  - Component interconnection
  - Performance calculations

### Machine Learning Models

- **PINN Models** (Physics-Informed Neural Networks)
  - `nozzle_pinn.pt`: Trained model for nozzle flow prediction
  - `turbine_pinn.pt`: Trained model for turbine performance

### Data Processing

- **Chemical Mechanisms**: YAML and CTI format mechanisms for various fuels
- **ICAO Data**: Real-world engine performance and emissions data
- **Conversion Tools**: Scripts to process and convert mechanism formats

### Optimization

- **Fuel Blend Optimization** ([optimization/optimize_blend.py](optimization/optimize_blend.py))
  - Optimizes fuel blend compositions
  - Performance and emissions trade-offs

### Evaluation & Analysis

- **Cantera EDA** ([evaluation/cantera_eda.py](evaluation/cantera_eda.py))
  - Chemical kinetics analysis
  - Mechanism validation

- **ICAO EDA** ([evaluation/icao_eda.py](evaluation/icao_eda.py))
  - Real engine data analysis
  - Model validation against certification data

## File Counts

- **Total Directories**: 16
- **Total Files**: 124
- **Python Scripts**: ~15
- **YAML Mechanisms**: 5
- **ICAO PDFs**: 83
- **Raw Mechanism Files**: 10

## Technologies Used

- Python 3.12
- Cantera (chemical kinetics)
- PyTorch (PINN models)
- NumPy, SciPy (numerical computing)
- Matplotlib (visualization)
- Optuna (optimization)

## Notes

- Virtual environment (`.venv/`) and cache files (`__pycache__/`) are excluded from version control
- PINN models are pre-trained and stored as `.pt` files in the root directory
- ICAO data provides real-world validation benchmarks for the Rolls-Royce Trent 1000 engine family
