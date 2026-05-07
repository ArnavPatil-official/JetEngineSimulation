"""
Train LE-PINN on Sajben transonic diffuser conditions.

This script now serves as a wrapper around the Sajben fine-tuning workflow
that uses the actual processed Sajben CFD dataset (master_shock_dataset.pt).

For synthetic pre-training, the dataset-backed fine-tune function can be
run with a fresh init (no pretrained checkpoint). This ensures Sajben
training always uses the correct processed dataset with planar geometry.

Saves checkpoint to  models/le_pinn_sajben.pt

Usage::

    python scripts/validation/train_sajben.py [--epochs 5000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from simulation.nozzle.le_pinn import finetune_on_cfd_data

# ---------------------------------------------------------------------------
# Sajben dataset path and training config
# ---------------------------------------------------------------------------
DATASET_PATH = str(_ROOT / "data" / "processed" / "master_shock_dataset.pt")
SAVE_PATH = str(_ROOT / "models" / "le_pinn_sajben.pt")


def train_sajben_le_pinn(
    n_epochs: int = 5000,
    lr: float = 1e-5,
    save_path: str | None = None,
    device: str = "mps",
    verbose: bool = True,
    physics_loss_weight: float = 0.05,
) -> tuple:
    """
    Train LE-PINN on Sajben dataset using dataset-backed fine-tuning path.

    This wrapper ensures Sajben training uses the processed dataset
    (master_shock_dataset.pt) with planar geometry mode, not synthetic-only.
    """
    import torch

    # ---- Phase 1: Lock Correct Sajben Data Path ----
    if verbose:
        print("=" * 70)
        print("LE-PINN SAJBEN TRAINING (dataset-backed, planar geometry)")
        print("=" * 70)

    # Validate dataset existence
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Sajben dataset not found: {DATASET_PATH}\n"
            f"Run 'python3 scripts/parse_sajben_cfd.py' to generate it."
        )

    # Load and validate schema
    try:
        dataset = torch.load(DATASET_PATH, weights_only=True)
    except TypeError:
        dataset = torch.load(DATASET_PATH)

    required_keys = {"inputs", "targets"}
    missing = required_keys - set(dataset.keys())
    if missing:
        raise ValueError(
            f"Dataset schema validation failed. Missing keys: {missing}\n"
            f"Expected keys: {required_keys}"
        )

    if verbose:
        print("Dataset validation passed:")
        print(f"  Resolved path: {Path(DATASET_PATH).resolve()}")
        print(f"  Inputs shape : {dataset['inputs'].shape}")
        print(f"  Targets shape: {dataset['targets'].shape}")
        if "sample_weights" in dataset:
            print(f"  Sample weights: {dataset['sample_weights'].shape}")
        else:
            print(f"  Sample weights: None")
        print(f"  Geometry mode: planar (Sajben 2D diffuser)")
        print()

    # Redirect to dataset-backed fine-tuning function with geometry="planar"
    model, history = finetune_on_cfd_data(
        dataset_path=DATASET_PATH,
        pretrained_path=None,  # Fresh init (no pretrained checkpoint)
        save_path=save_path,
        n_epochs=n_epochs,
        lr=lr,
        physics_loss_weight=physics_loss_weight,
        device=device,
        verbose=verbose,
        geometry="planar",  # Sajben is 2D planar, not axisymmetric
    )

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LE-PINN on Sajben diffuser dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Number of training epochs (default: 5000)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Compute device (default: mps)",
    )
    parser.add_argument(
        "--physics-weight", type=float, default=0.05,
        dest="physics_weight",
        help="Physics loss weight (default: 0.05)",
    )
    args = parser.parse_args()

    model, history = train_sajben_le_pinn(
        n_epochs=args.epochs,
        lr=args.lr,
        save_path=SAVE_PATH,
        device=args.device,
        physics_loss_weight=args.physics_weight,
        verbose=True,
    )


if __name__ == "__main__":
    main()
