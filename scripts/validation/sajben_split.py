#!/usr/bin/env python3
"""
P4.3 — Build the Sajben training set from the NASA/WIND RANS solution and
declare the train / evaluation split BEFORE any training run.

Route (a) of the P4.1 audit (``outputs/sajben_data_audit.md``): the training
signal is the converged Spalart-Allmaras solution of the weak-shock case
shipped in the NPARC archive (``sajben.cgd`` + ``sajben.cfl``, decoded by
``simulation.nozzle.wind_cff``).  The evaluation set is the experiment
(``data/raw/data.Mach46.txt``), which never enters training.  Leakage is
therefore impossible by construction, and the split record written into
the dataset — and copied into every checkpoint trained on it — lets
``sajben_validation.py`` verify that rather than trust it.

Dataset layout (matches ``finetune_on_cfd_data``)::

    inputs  (N, 6)  [x, y, A5, A6, P0, T0]    SI; A5/A6/P0/T0 are constants
    targets (N, 9)  [rho, u, v, P, T, 0, 0, 0, mu_l + mu_t]
    sample_weights (N,)  ones
    split   dict    declared split record (see ``SPLIT`` below)

Conventions, recorded here so they are not re-derived later:

* One flow case only (weak shock, exit static 16.055 psi). The inputs
  A5, A6, P0, T0 are single constants and normalise to zero; the network is
  a function of (x, y) for this case. It is a single-condition surrogate.
* A5 = pi (H/2)^2 and A6 = pi (H/2 sqrt(AR))^2 follow ``build_sajben_grid``
  in ``sajben_validation.py`` so the scorer's throat-height reconstruction
  (H = 2 sqrt(A5/pi)) is exact.
* The inflow column i = 0 is excluded: it carries the uniform-inflow
  condition imposed on no-slip walls (corner spike, P4.1 §5).
* Reynolds-stress columns 5-7 are zero, not NaN.
* mu_eff (column 8) is the WIND laminar + eddy viscosity.

Usage::

    python scripts/validation/sajben_split.py            # writes the dataset
    python scripts/validation/sajben_split.py --check    # re-derive and compare hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.wind_cff import load_wind_solution  # noqa: E402

NASA_DIR = REPO_ROOT / "data" / "raw" / "cfd_datasets" / "nasa" / "transdif01"
WIND_CGD = NASA_DIR / "sajben.cgd"
WIND_CFL = NASA_DIR / "sajben.cfl"
EXP_FILE = REPO_ROOT / "data" / "raw" / "data.Mach46.txt"
OUT_PATH = REPO_ROOT / "data" / "processed" / "sajben_wind_dataset.pt"

EXCLUDED_I = [0]            # inflow column, see module docstring
AR_EXIT_NOMINAL = 1.5       # only used for the A6 constant


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset() -> dict:
    sol = load_wind_solution(WIND_CGD, WIND_CFL)
    H = sol.h_throat
    keep_i = np.array([i for i in range(sol.ni) if i not in EXCLUDED_I])

    f = {k: getattr(sol, k)[:, keep_i] for k in ("x", "y", "rho", "u", "v", "p", "T", "mu_l", "mu_t")}
    n = f["x"].size
    A5 = np.pi * (H / 2.0) ** 2
    A6 = np.pi * (H / 2.0 * np.sqrt(AR_EXIT_NOMINAL)) ** 2

    inputs = np.column_stack([
        f["x"].ravel(), f["y"].ravel(),
        np.full(n, A5), np.full(n, A6),
        np.full(n, sol.p0), np.full(n, sol.T0),
    ]).astype(np.float32)
    zeros = np.zeros(n)
    targets = np.column_stack([
        f["rho"].ravel(), f["u"].ravel(), f["v"].ravel(), f["p"].ravel(), f["T"].ravel(),
        zeros, zeros, zeros, (f["mu_l"] + f["mu_t"]).ravel(),
    ]).astype(np.float32)

    split = {
        "attempt": "P4.3-attempt-1",
        "declared": "2026-09-18",
        "train_source": str(WIND_CFL.relative_to(REPO_ROOT)),
        "train_grid": str(WIND_CGD.relative_to(REPO_ROOT)),
        "train_sha256": sha256(WIND_CFL),
        "train_grid_sha256": sha256(WIND_CGD),
        "train_rows": int(n),
        "train_excluded_i": EXCLUDED_I,
        "eval_source": str(EXP_FILE.relative_to(REPO_ROOT)),
        "eval_sha256": sha256(EXP_FILE),
        "eval_rows_in_train": 0,
        "eval_metric": "sajben_validation.py wall-Cp shape-L2 (throat-height mapping), pre-registered bands "
                       "<0.10 pass / 0.10-0.25 partial / >0.25 fail (docs/plan.md P4.3)",
        "internal_val_fraction": 0.2,
        "internal_val_seed": 42,
        "note": "single flow case (weak shock); A5, A6, P0, T0 are constants",
    }
    return {
        "inputs": torch.from_numpy(inputs),
        "targets": torch.from_numpy(targets),
        "sample_weights": torch.ones(n, dtype=torch.float32),
        "split": split,
        "reference": sol.reference,
        "provenance": sol.provenance,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the file on disk")
    args = ap.parse_args()

    ds = build_dataset()
    if args.check:
        on_disk = torch.load(OUT_PATH, map_location="cpu", weights_only=False)
        same = (torch.equal(on_disk["inputs"], ds["inputs"])
                and torch.equal(on_disk["targets"], ds["targets"])
                and on_disk["split"]["train_sha256"] == ds["split"]["train_sha256"])
        print("dataset on disk", "matches" if same else "DIFFERS FROM", "a fresh rebuild")
        sys.exit(0 if same else 1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        old = torch.load(OUT_PATH, map_location="cpu", weights_only=False)
        if torch.equal(old["inputs"], ds["inputs"]) and torch.equal(old["targets"], ds["targets"]):
            print(f"{OUT_PATH.relative_to(REPO_ROOT)} already up to date (identical rebuild).")
            return
    torch.save(ds, OUT_PATH)
    print(f"Saved {OUT_PATH.relative_to(REPO_ROOT)}: inputs {tuple(ds['inputs'].shape)}, "
          f"targets {tuple(ds['targets'].shape)}, sha256 {sha256(OUT_PATH)[:16]}…")
    print("Declared split:")
    print(json.dumps(ds["split"], indent=2))


if __name__ == "__main__":
    main()
