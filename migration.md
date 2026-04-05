# Colab Migration Audit

## Scope
This document covers the Sajben transonic diffuser path, the core PINN and engine modeling stack, the optimization routines, and the dependency files needed to run them reproducibly in Google Colab.

Excluded from the main inventory:
- tests and test directories
- EDA-only / visualization-only scripts
- README / docs / .git / __pycache__ / env files
- pure plotting or report-generation files, which are listed in an appendix

## Step 1: File Audit

### Required source and dependency files

| File Path | Purpose (1 line) | Dependencies it imports | Required? |
|---|---|---|---|
| [integrated_engine.py](integrated_engine.py) | Main Brayton-cycle orchestrator that links compressor, combustor, turbine, nozzle, emissions, and fuel handling. | os, sys, torch, numpy, cantera, pandas, pathlib.Path, typing, sklearn.linear_model.LinearRegression, simulation.compressor.compressor, simulation.combustor.combustor, simulation.thermo_utils, simulation.nozzle.nozzle, simulation.turbine.turbine | Yes |
| [simulation/compressor/compressor.py](simulation/compressor/compressor.py) | Cantera-based compressor thermodynamics with isentropic efficiency losses. | cantera | Yes |
| [simulation/combustor/combustor.py](simulation/combustor/combustor.py) | Constant-pressure combustor with chemical equilibrium and fuel-dependent properties. | cantera, numpy, typing | Yes |
| [simulation/emissions.py](simulation/emissions.py) | Emissions-index utilities for CO and NOx estimates. | typing, cantera, numpy | Yes |
| [simulation/fuels.py](simulation/fuels.py) | Fuel surrogate definitions and SAF blending utilities. | __future__, dataclasses.dataclass, typing | Yes |
| [simulation/thermo_utils.py](simulation/thermo_utils.py) | Bridges combustor output to turbine/nozzle condition dictionaries. | typing | Yes |
| [simulation/engine_types.py](simulation/engine_types.py) | Typed flow-state, fuel, and performance result containers. | dataclasses.dataclass, dataclasses.field, typing | Yes |
| [simulation/nozzle/nozzle.py](simulation/nozzle/nozzle.py) | Fuel-dependent nozzle PINN, fallback solver, and nozzle training/inference API. | torch, torch.nn, numpy, matplotlib.pyplot, pathlib.Path, typing, warnings, optional pandas | Yes |
| [simulation/nozzle/le_pinn.py](simulation/nozzle/le_pinn.py) | LE-PINN architecture, Sajben parsing helpers, training, fine-tuning, and validation support. | __future__, sys, warnings, pathlib.Path, dataclasses.dataclass, typing, numpy, torch, torch.nn | Yes |
| [simulation/turbine/turbine.py](simulation/turbine/turbine.py) | Fuel-dependent turbine PINN and runtime API for integrated engine use. | torch, torch.nn, numpy, pandas, time, pathlib.Path, typing | Yes |
| [scripts/parse_sajben_cfd.py](scripts/parse_sajben_cfd.py) | Generates the Sajben CFD training dataset from raw geometry and inlet decks. | __future__, re, sys, pathlib.Path, numpy, torch, simulation.nozzle.le_pinn | Yes |
| [scripts/validation/train_sajben.py](scripts/validation/train_sajben.py) | Trains LE-PINN on Sajben synthetic diffuser conditions. | __future__, argparse, sys, warnings, pathlib.Path, numpy, torch, torch.nn, simulation.nozzle.le_pinn | Yes |
| [scripts/validation/sajben_validation.py](scripts/validation/sajben_validation.py) | Validates LE-PINN against Sajben experimental wall-pressure and velocity data. | __future__, argparse, sys, warnings, pathlib.Path, numpy, torch, simulation.nozzle.le_pinn | Yes |
| [scripts/validation/finetune_sajben.py](scripts/validation/finetune_sajben.py) | Fine-tunes LE-PINN on the processed Sajben CFD dataset. | __future__, argparse, sys, pathlib.Path, simulation.nozzle.le_pinn | Yes |
| [scripts/optimization/optimize_blend.py](scripts/optimization/optimize_blend.py) | Multi-objective fuel-blend optimization with emissions and performance metrics. | sys, pathlib.Path, optuna, numpy, matplotlib.pyplot, pandas, contextlib, io, re, mpl_toolkits.mplot3d.Axes3D, integrated_engine, simulation.fuels, optuna.trial.TrialState, os | Yes |
| [scripts/optimization/calibrate_lto.py](scripts/optimization/calibrate_lto.py) | Optuna-based LTO calibration routine for combustor efficiency and throttle settings. | sys, pathlib.Path, optuna, numpy, logging, integrated_engine, matplotlib.pyplot | Yes |
| [requirements.txt](requirements.txt) | Dependency manifest for Colab/environment reproducibility. | none | Yes |

### Required raw data and model artifacts

| File Path | Purpose (1 line) | Dependencies it imports | Required? |
|---|---|---|---|
| [data/raw/data.Mach46.txt](data/raw/data.Mach46.txt) | Sajben / Hseih experimental wall-pressure and velocity data used by validation. | none | Yes |
| [data/raw/sajben.x.fmt](data/raw/sajben.x.fmt) | Sajben Plot3D geometry file used by parser and validation geometry checks. | none | Yes |
| [data/raw/cfd_datasets/nasa/sajben.dat.{1..20}](data/raw/cfd_datasets/nasa/) | Raw NASA inlet/back-pressure decks used to build the Sajben CFD dataset. | none | Yes |
| [data/creck_c1c16_full.yaml](data/creck_c1c16_full.yaml) | CRECK chemical mechanism used for blend studies. | none | Yes |
| [data/A1highT.yaml](data/A1highT.yaml) | HyChem mechanism used for Jet-A1 validation mode. | none | Yes |
| [data/icao_engine_data.csv](data/icao_engine_data.csv) | ICAO engine emissions data used for emissions calibration. | none | Yes |
| [data/processed/master_shock_dataset.pt](data/processed/master_shock_dataset.pt) | Serialized Sajben CFD dataset used by fine-tuning and CFD validation. | none | Maybe |
| [models/turbine_pinn.pt](models/turbine_pinn.pt) | Trained turbine PINN checkpoint loaded by the integrated engine. | none | Maybe |
| [models/nozzle_pinn.pt](models/nozzle_pinn.pt) | Trained nozzle PINN checkpoint loaded by the integrated engine. | none | Maybe |
| [models/le_pinn.pt](models/le_pinn.pt) | Base LE-PINN checkpoint used by validation and fallback workflows. | none | Maybe |
| [models/le_pinn_cfd.pt](models/le_pinn_cfd.pt) | CFD-finetuned LE-PINN checkpoint used for dataset-driven validation. | none | Maybe |
| [models/le_pinn_sajben.pt](models/le_pinn_sajben.pt) | Sajben-trained LE-PINN checkpoint used by Sajben validation. | none | Maybe |
| [models/le_pinn_sajben_finetuned.pt](models/le_pinn_sajben_finetuned.pt) | Fine-tuned Sajben checkpoint created after CFD fine-tuning. | none | Maybe |

### Notes on Required vs Maybe
- `Yes` means the file is an authoritative input or source file that should be copied into Colab storage.
- `Maybe` means the file is a generated artifact or checkpoint that can be regenerated from the source pipeline, but copying it avoids rerunning training or preprocessing.

## Appendix: Plotting / Reporting-Only Files
These are excluded from the main pipeline inventory because they are visualization, reporting, or EDA only.

- [dashboard.py](dashboard.py)
- [scripts/generate_report.py](scripts/generate_report.py)
- [evaluation/cantera_eda.py](evaluation/cantera_eda.py)
- [evaluation/icao_eda.py](evaluation/icao_eda.py)
- [evaluation/visualize_networks.py](evaluation/visualize_networks.py)
- [scripts/visualization/marked_visuals.py](scripts/visualization/marked_visuals.py)
- [scripts/visualization/nozzle_2d_geometry.py](scripts/visualization/nozzle_2d_geometry.py)
- [scripts/visualization/optimization_plot.py](scripts/visualization/optimization_plot.py)
- [scripts/visualization/pareto_visual.py](scripts/visualization/pareto_visual.py)
- [scripts/visualization/plot_validation.py](scripts/visualization/plot_validation.py)
- [scripts/visualization/visualize_results.py](scripts/visualization/visualize_results.py)
- [simulation/nozzle/visualize_nozzle.py](simulation/nozzle/visualize_nozzle.py)
- [simulation/turbine/visualize_turbine_pinn.py](simulation/turbine/visualize_turbine_pinn.py)
- [outputs/pinn_architecture_diagram.py](outputs/pinn_architecture_diagram.py)

## Step 2: Dependency Graph

### Core import graph

```text
integrated_engine.py
  -> simulation/compressor/compressor.py
  -> simulation/combustor/combustor.py
  -> simulation/thermo_utils.py
  -> simulation/nozzle/nozzle.py
  -> simulation/turbine/turbine.py
  -> simulation/emissions.py
  -> simulation/fuels.py
  -> simulation/engine_types.py

scripts/optimization/optimize_blend.py
  -> integrated_engine.py
  -> simulation/fuels.py

scripts/optimization/calibrate_lto.py
  -> integrated_engine.py

scripts/parse_sajben_cfd.py
  -> simulation/nozzle/le_pinn.py

scripts/validation/train_sajben.py
  -> simulation/nozzle/le_pinn.py

scripts/validation/sajben_validation.py
  -> simulation/nozzle/le_pinn.py

scripts/validation/finetune_sajben.py
  -> simulation/nozzle/le_pinn.py
```

### Imported files not already listed
No additional core-module dependencies were found outside the inventory above. The current pipeline does not import `simulation/nozzle/nozzle_conditions.py`, `simulation/turbine/turbine_boundary.py`, or any `data/processors/*.py` modules.

### Colab-breaking local paths
The following path patterns are the ones that will break or behave inconsistently in Colab if left as hardcoded defaults:
- `data/creck_c1c16_full.yaml`
- `data/A1highT.yaml`
- `data/icao_engine_data.csv`
- `models/turbine_pinn.pt`
- `models/nozzle_pinn.pt`
- `models/le_pinn.pt`
- `models/le_pinn_cfd.pt`
- `models/le_pinn_sajben.pt`
- `models/le_pinn_sajben_finetuned.pt`
- `outputs/results/optimization_results.csv`
- `outputs/plots/*.png`
- `data/raw/data.Mach46.txt`
- `data/raw/sajben.x.fmt`
- `data/raw/cfd_datasets/nasa/sajben.dat.*`

## Step 3: Anchor Block

Place this at the top of the main Colab notebook before any imports from the repo:

```python
# --- ENVIRONMENT ANCHOR ----------------------------------------------------
import os
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/RREngineData/"
else:
    BASE_DIR = "/Users/arnavpatil/Desktop/JetEngineSimulation"

DATA_DIR = os.path.join(BASE_DIR, "data/")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed/")
MODEL_DIR = os.path.join(BASE_DIR, "models/")
CONFIG_DIR = os.path.join(BASE_DIR, "configs/")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results/")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots/")
RAW_CFD_DIR = os.path.join(RAW_DATA_DIR, "cfd_datasets/")
RAW_NASA_DIR = os.path.join(RAW_CFD_DIR, "nasa/")

RAW_MACH46 = os.path.join(RAW_DATA_DIR, "data.Mach46.txt")
RAW_SAJBEN = os.path.join(RAW_DATA_DIR, "sajben.x.fmt")
SAJBEN_DAT_DIR = RAW_NASA_DIR

CRECK_MECH = os.path.join(DATA_DIR, "creck_c1c16_full.yaml")
HYCHEM_MECH = os.path.join(DATA_DIR, "A1highT.yaml")
ICAO_DATA = os.path.join(DATA_DIR, "icao_engine_data.csv")
MASTER_SHOCK_DATASET = os.path.join(PROCESSED_DIR, "master_shock_dataset.pt")

TURBINE_CKPT = os.path.join(MODEL_DIR, "turbine_pinn.pt")
NOZZLE_CKPT = os.path.join(MODEL_DIR, "nozzle_pinn.pt")
LE_PINN_CKPT = os.path.join(MODEL_DIR, "le_pinn.pt")
LE_PINN_CFD_CKPT = os.path.join(MODEL_DIR, "le_pinn_cfd.pt")
LE_PINN_SAJBEN_CKPT = os.path.join(MODEL_DIR, "le_pinn_sajben.pt")
LE_PINN_SAJBEN_FINETUNED_CKPT = os.path.join(MODEL_DIR, "le_pinn_sajben_finetuned.pt")

OPT_RESULTS_CSV = os.path.join(RESULTS_DIR, "optimization_results.csv")
PARETO_PNG = os.path.join(PLOTS_DIR, "pareto_3d.png")
PARALLEL_PNG = os.path.join(PLOTS_DIR, "parallel_coordinates.png")

sys.path.insert(0, BASE_DIR)
# ---------------------------------------------------------------------------
```

## Step 4: Path Surgery

### `integrated_engine.py`
- BEFORE: `creck_mechanism_path: str = "data/creck_c1c16_full.yaml"`
  AFTER:  `creck_mechanism_path: str = str(Path(__file__).resolve().parent / "data" / "creck_c1c16_full.yaml")`

- BEFORE: `hychem_mechanism_path: str = "data/A1highT.yaml"`
  AFTER:  `hychem_mechanism_path: str = str(Path(__file__).resolve().parent / "data" / "A1highT.yaml")`

- BEFORE: `turbine_pinn_path: str = "models/turbine_pinn.pt"`
  AFTER:  `turbine_pinn_path: str = str(Path(__file__).resolve().parent / "models" / "turbine_pinn.pt")`

- BEFORE: `nozzle_pinn_path: str = "models/nozzle_pinn.pt"`
  AFTER:  `nozzle_pinn_path: str = str(Path(__file__).resolve().parent / "models" / "nozzle_pinn.pt")`

- BEFORE: `icao_data_path: str = "data/icao_engine_data.csv"`
  AFTER:  `icao_data_path: str = str(Path(__file__).resolve().parent / "data" / "icao_engine_data.csv")`

- BEFORE: `creck_mechanism_path="data/creck_c1c16_full.yaml"`
  AFTER:  `creck_mechanism_path=str(Path(__file__).resolve().parent / "data" / "creck_c1c16_full.yaml")`

- BEFORE: `hychem_mechanism_path="data/A1highT.yaml"`
  AFTER:  `hychem_mechanism_path=str(Path(__file__).resolve().parent / "data" / "A1highT.yaml")`

- BEFORE: `turbine_pinn_path="turbine_pinn.pt"`
  AFTER:  `turbine_pinn_path=str(Path(__file__).resolve().parent / "models" / "turbine_pinn.pt")`

- BEFORE: `nozzle_pinn_path="nozzle_pinn.pt"`
  AFTER:  `nozzle_pinn_path=str(Path(__file__).resolve().parent / "models" / "nozzle_pinn.pt")`

### `scripts/optimization/optimize_blend.py`
- BEFORE: `creck_mechanism_path="data/creck_c1c16_full.yaml"`
  AFTER:  `creck_mechanism_path=str(Path(__file__).resolve().parent.parent.parent / "data" / "creck_c1c16_full.yaml")`

- BEFORE: `hychem_mechanism_path="data/A1highT.yaml"`
  AFTER:  `hychem_mechanism_path=str(Path(__file__).resolve().parent.parent.parent / "data" / "A1highT.yaml")`

- BEFORE: `turbine_pinn_path="models/turbine_pinn.pt"`
  AFTER:  `turbine_pinn_path=str(Path(__file__).resolve().parent.parent.parent / "models" / "turbine_pinn.pt")`

- BEFORE: `nozzle_pinn_path="models/nozzle_pinn.pt"`
  AFTER:  `nozzle_pinn_path=str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt")`

- BEFORE: `icao_data_path="data/icao_engine_data.csv"`
  AFTER:  `icao_data_path=str(Path(__file__).resolve().parent.parent.parent / "data" / "icao_engine_data.csv")`

- BEFORE: `df_results.to_csv('outputs/results/optimization_results.csv', index=False)`
  AFTER:  `df_results.to_csv(str(Path(__file__).resolve().parent.parent.parent / "outputs" / "results" / "optimization_results.csv"), index=False)`

- BEFORE: `os.makedirs('outputs/plots', exist_ok=True)`
  AFTER:  `os.makedirs(str(Path(__file__).resolve().parent.parent.parent / "outputs" / "plots"), exist_ok=True)`

- BEFORE: `plt.savefig('outputs/plots/pareto_3d.png', dpi=300)`
  AFTER:  `plt.savefig(str(Path(__file__).resolve().parent.parent.parent / "outputs" / "plots" / "pareto_3d.png"), dpi=300)`

- BEFORE: `plt.savefig('outputs/plots/parallel_coordinates.png', dpi=300)`
  AFTER:  `plt.savefig(str(Path(__file__).resolve().parent.parent.parent / "outputs" / "plots" / "parallel_coordinates.png"), dpi=300)`

### `scripts/validation/train_sajben.py`
- BEFORE: `SAVE_PATH = str(_ROOT / "models" / "le_pinn_sajben.pt")`
  AFTER:  `SAVE_PATH = str(_ROOT / "models" / "le_pinn_sajben.pt")`

This one is already Colab-safe because it is derived from `Path(__file__)`; no code change is needed unless you want to parameterize the output root further.

### `scripts/validation/sajben_validation.py`
- BEFORE: `DATA_FILE  = _ROOT / "data" / "raw" / "data.Mach46.txt"`
  AFTER:  `DATA_FILE  = _ROOT / "data" / "raw" / "data.Mach46.txt"`

- BEFORE: `GEOM_FILE  = _ROOT / "data" / "raw" / "sajben.x.fmt"`
  AFTER:  `GEOM_FILE  = _ROOT / "data" / "raw" / "sajben.x.fmt"`

- BEFORE: `DEFAULT_MODEL = _ROOT / "models" / "le_pinn_sajben.pt"`
  AFTER:  `DEFAULT_MODEL = _ROOT / "models" / "le_pinn_sajben.pt"`

- BEFORE: `FALLBACK_MODEL = _ROOT / "models" / "le_pinn.pt"`
  AFTER:  `FALLBACK_MODEL = _ROOT / "models" / "le_pinn.pt"`

These are already derived from the repository root and are fine once the repo is mounted in Drive.

### `scripts/validation/finetune_sajben.py`
- BEFORE: `dataset_path   = str(REPO_ROOT / "data"   / "processed" / "master_shock_dataset.pt")`
  AFTER:  `dataset_path   = str(REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt")`

- BEFORE: `pretrained     = str(REPO_ROOT / "models" / "le_pinn_sajben.pt")`
  AFTER:  `pretrained     = str(REPO_ROOT / "models" / "le_pinn_sajben.pt")`

- BEFORE: `save_path      = str(REPO_ROOT / "models" / "le_pinn_sajben_finetuned.pt")`
  AFTER:  `save_path      = str(REPO_ROOT / "models" / "le_pinn_sajben_finetuned.pt")`

These are already derived from the repository root and are fine.

### `scripts/parse_sajben_cfd.py`
- BEFORE: `geom_path=str(REPO_ROOT / "data" / "raw" / "sajben.x.fmt")`
  AFTER:  `geom_path=str(REPO_ROOT / "data" / "raw" / "sajben.x.fmt")`

- BEFORE: `dat_dir=str(REPO_ROOT / "data" / "raw" / "cfd_datasets" / "nasa")`
  AFTER:  `dat_dir=str(REPO_ROOT / "data" / "raw" / "cfd_datasets" / "nasa")`

- BEFORE: `output_path=str(REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt")`
  AFTER:  `output_path=str(REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt")`

These are already derived from the repository root and are fine.

### `simulation/turbine/turbine.py`
- BEFORE: `}, "turbine_pinn.pt")`
  AFTER:  `}, str(Path(__file__).resolve().parent.parent.parent / "models" / "turbine_pinn.pt"))`

This saves the checkpoint into the `models/` directory instead of the repository root.

### `simulation/nozzle/nozzle.py`
- BEFORE: `def save_model(model, filename="nozzle_pinn.pt", conditions=None, training_info=None):`
  AFTER:  `def save_model(model, filename=None, conditions=None, training_info=None):`
  plus set the default inside the function to `str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt")`.

- BEFORE: `def load_model(filename="nozzle_pinn.pt"):`
  AFTER:  `def load_model(filename=None):`
  plus set the default inside the function to `str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt")`.

- BEFORE: `def train_nozzle(num_epochs=5001, lr=1e-3, save_path="nozzle_pinn.pt", verbose=True):`
  AFTER:  `def train_nozzle(num_epochs=5001, lr=1e-3, save_path=None, verbose=True):`
  plus set the default inside the function to `str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt")`.

- BEFORE: `save_path = REPO_ROOT / 'nozzle_validation_dual_thermo.png'`
  AFTER:  `save_path = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'plots' / 'nozzle_validation_dual_thermo.png'`

- BEFORE: `def test_inlet_consistency(model_path="nozzle_pinn.pt"):`
  AFTER:  `def test_inlet_consistency(model_path=None):`
  plus default to the `models/nozzle_pinn.pt` path inside the function.

- BEFORE: `def test_mass_conservation(model_path="nozzle_pinn.pt"):`
  AFTER:  `def test_mass_conservation(model_path=None):`
  plus default to the `models/nozzle_pinn.pt` path inside the function.

- BEFORE: `def test_integration_case(model_path="nozzle_pinn.pt"):`
  AFTER:  `def test_integration_case(model_path=None):`
  plus default to the `models/nozzle_pinn.pt` path inside the function.

- BEFORE: `def test_thermo_sensitivity(model_path="nozzle_pinn.pt"):`
  AFTER:  `def test_thermo_sensitivity(model_path=None):`
  plus default to the `models/nozzle_pinn.pt` path inside the function.

- BEFORE: `save_path="nozzle_pinn.pt"` in the module's `__main__` block
  AFTER:  `save_path=str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt")`

- BEFORE: `test_inlet_consistency("nozzle_pinn.pt")`
  AFTER:  `test_inlet_consistency(str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt"))`

- BEFORE: `test_mass_conservation("nozzle_pinn.pt")`
  AFTER:  `test_mass_conservation(str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt"))`

- BEFORE: `test_integration_case("nozzle_pinn.pt")`
  AFTER:  `test_integration_case(str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt"))`

- BEFORE: `test_thermo_sensitivity("nozzle_pinn.pt")`
  AFTER:  `test_thermo_sensitivity(str(Path(__file__).resolve().parent.parent.parent / "models" / "nozzle_pinn.pt"))`

### `simulation/nozzle/le_pinn.py`
No hardcoded local file path needs surgery for the core pipeline; its model and dataset paths are already derived from `_REPO_ROOT`.

## Step 5: Transfer Checklist

### 1. Files to upload to `/MyDrive/RREngineData/`

Upload these under the same subdirectory structure:

- `data/raw/data.Mach46.txt`
- `data/raw/sajben.x.fmt`
- `data/raw/cfd_datasets/nasa/sajben.dat.1` through `data/raw/cfd_datasets/nasa/sajben.dat.20`
- `data/creck_c1c16_full.yaml`
- `data/A1highT.yaml`
- `data/icao_engine_data.csv`
- `data/processed/master_shock_dataset.pt` if you want to skip re-running dataset generation
- `models/turbine_pinn.pt` if you want to skip retraining turbine inference weights
- `models/nozzle_pinn.pt` if you want to skip retraining nozzle inference weights
- `models/le_pinn.pt` if you want the fallback / base LE-PINN checkpoint
- `models/le_pinn_cfd.pt` if you want the CFD-finetuned LE-PINN checkpoint
- `models/le_pinn_sajben.pt` if you want the Sajben-trained checkpoint
- `models/le_pinn_sajben_finetuned.pt` if you want the fine-tuned Sajben checkpoint

### 2. Files to push to GitHub (code only, no data)

- `integrated_engine.py`
- `simulation/compressor/compressor.py`
- `simulation/combustor/combustor.py`
- `simulation/emissions.py`
- `simulation/fuels.py`
- `simulation/thermo_utils.py`
- `simulation/engine_types.py`
- `simulation/nozzle/nozzle.py`
- `simulation/nozzle/le_pinn.py`
- `simulation/turbine/turbine.py`
- `scripts/parse_sajben_cfd.py`
- `scripts/validation/train_sajben.py`
- `scripts/validation/sajben_validation.py`
- `scripts/validation/finetune_sajben.py`
- `scripts/optimization/optimize_blend.py`
- `scripts/optimization/calibrate_lto.py`
- `requirements.txt`
- `migration.md` if you want the audit itself in version control

### 3. pip install commands needed in Colab

Preferred reproducible install:
```bash
pip install -r requirements.txt
```

If you want the minimal explicit package list from the imports scan:
```bash
pip install cantera optuna pandas matplotlib scikit-learn
```

Notes:
- `torch` and `numpy` are excluded from the explicit list per your instruction.
- `matplotlib` covers `mpl_toolkits.mplot3d`.
- `scikit-learn` is required for `LinearRegression` in `integrated_engine.py`.

### 4. Colab-specific setup steps

1. Use a GPU runtime if you plan to train or fine-tune PINNs.
2. Mount Google Drive before importing repo modules.
3. Put the repository and data under `/content/drive/MyDrive/RREngineData/`.
4. Run the anchor block first, then import the modules.
5. For Sajben training scripts, override the default Apple MPS device with `--device cuda` or `--device cpu`; Colab does not provide MPS.
6. If you keep any legacy relative path calls, set the notebook working directory to `BASE_DIR` after mounting Drive.
7. Keep generated checkpoints and processed datasets in `models/` and `data/processed/` respectively so the existing module defaults keep resolving cleanly.

## Summary
The pipeline is cleanly split into:
- raw Sajben inputs and engine mechanism data
- source code for the engine, PINNs, optimization, and Sajben processing
- optional regenerated artifacts under `data/processed/` and `models/`

The current codebase is already mostly `Path(__file__)`-driven, so Colab migration is mostly a matter of mounting Drive, installing the missing packages, and normalizing the remaining hardcoded checkpoint/output defaults.
