#!/usr/bin/env python3
"""
Fine-tune the LE-PINN on the NASA Sajben transonic diffuser dataset.

Prerequisites
-------------
Run ``scripts/validation/sajben_split.py`` to build
``data/processed/sajben_wind_dataset.pt`` (declared train/eval split), and
``scripts/validation/train_sajben.py`` to produce the initialisation
``models/le_pinn_sajben_v5.pt``.

Fine-tuning keeps the initialisation's normalisers, refuses a collapsed or
out-of-domain initialisation, restores the best-validation weights and
records provenance (P4.2 guarantees in ``finetune_on_cfd_data``).

Usage
-----
    python scripts/validation/finetune_sajben.py [--epochs N] [--device auto|cpu|mps|cuda]

Output
------
Fine-tuned checkpoint: ``models/le_pinn_sajben_v5_finetuned.pt`` (never overwritten)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.le_pinn import finetune_on_cfd_data
from scripts.validation.train_sajben import resolve_device

DATASET_PATH = str(REPO_ROOT / "data" / "processed" / "sajben_wind_dataset.pt")
PRETRAINED_PATH = str(REPO_ROOT / "models" / "le_pinn_sajben_v5.pt")
SAVE_PATH = str(REPO_ROOT / "models" / "le_pinn_sajben_v5_finetuned.pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LE-PINN on Sajben transonic diffuser data."
    )
    parser.add_argument("--epochs",  type=int,   default=500,
                        help="Number of fine-tuning epochs (default: 500)")
    parser.add_argument("--lr",      type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--device",  type=str,   default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Compute device (default: auto -> cpu)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="RNG seed for the train/val split (default: 42)")
    parser.add_argument("--pretrained", type=str, default=PRETRAINED_PATH,
                        help=f"Initialisation checkpoint (default: {PRETRAINED_PATH})")
    parser.add_argument("--out", type=str, default=SAVE_PATH,
                        help=f"Output checkpoint (default: {SAVE_PATH}); never overwritten")
    parser.add_argument("--physics-weight", type=float, default=0.05,
                        dest="physics_weight",
                        help="Physics loss weight (default: 0.05)")
    parser.add_argument("--physics-max-points", type=int, default=None,
                        dest="physics_max_points",
                        help="Max training points used for physics loss per epoch "
                             "(default: auto cap on mps)")
    parser.add_argument(
        "--physics-debug",
        action="store_true",
        help="Print per-term normalized physics residual losses during fine-tuning",
    )
    args = parser.parse_args()

    dataset_path   = DATASET_PATH
    pretrained     = args.pretrained
    save_path      = args.out
    device         = resolve_device(args.device)

    # ---- Phase 1: Lock Correct Sajben Data Path ----
    # Verify dataset existence and schema before training starts
    import torch
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Sajben dataset not found: {dataset_path}\n"
            f"Run 'python scripts/validation/sajben_split.py' to generate it."
        )

    # Load and validate schema
    try:
        dataset = torch.load(dataset_path, weights_only=False)
    except TypeError:
        dataset = torch.load(dataset_path)

    required_keys = {"inputs", "targets"}
    missing = required_keys - set(dataset.keys())
    if missing:
        raise ValueError(
            f"Dataset schema validation failed. Missing keys: {missing}\n"
            f"Expected keys: {required_keys}"
        )

    # Print dataset summary for traceability
    print("Dataset validation passed:")
    print(f"  Resolved path: {Path(dataset_path).resolve()}")
    print(f"  Inputs shape : {dataset['inputs'].shape}")
    print(f"  Targets shape: {dataset['targets'].shape}")
    if "sample_weights" in dataset:
        print(f"  Sample weights: {dataset['sample_weights'].shape}")
    else:
        print(f"  Sample weights: None")
    print(f"  Geometry mode: planar (Sajben 2D diffuser)")
    print()

    # Use fresh init if the pretrained checkpoint is missing
    if not Path(pretrained).exists():
        print(f"Warning: pretrained checkpoint not found at {pretrained}. "
              "Fine-tuning from random init.")
        pretrained = None

    print("=" * 60)
    print("LE-PINN Fine-tuning — Sajben Transonic Diffuser")
    print("=" * 60)
    print(f"  dataset  : {dataset_path}")
    print(f"  pretrained: {pretrained or '(fresh init)'}")
    print(f"  save_path: {save_path}")
    print(f"  epochs   : {args.epochs}")
    print(f"  lr       : {args.lr}")
    print(f"  device   : {device}")
    print(f"  seed     : {args.seed}")
    print(f"  phys wt  : {args.physics_weight}")
    print(f"  phys max : {args.physics_max_points if args.physics_max_points is not None else 'auto'}")
    print()

    model, history = finetune_on_cfd_data(
        dataset_path=dataset_path,
        pretrained_path=pretrained,
        save_path=save_path,
        n_epochs=args.epochs,
        lr=args.lr,
        physics_loss_weight=args.physics_weight,
        physics_max_points=args.physics_max_points,
        device=device,
        verbose=True,
        geometry="planar",  # Sajben is a 2D planar diffuser (not axisymmetric)
        physics_debug=args.physics_debug,
        seed=args.seed,
        extra_payload={"split": dataset.get("split")},
    )

    # Summary
    final_train = history["loss_total"][-1]
    final_val   = history["val_loss"][-1]
    print()
    print("=" * 60)
    print("Fine-tuning complete.")
    print(f"  Final train loss : {final_train:.6f}")
    print(f"  Final val   loss : {final_val:.6f}")
    print(f"  Checkpoint saved : {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
