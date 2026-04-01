#!/usr/bin/env python3
"""
Fine-tune the LE-PINN on the NASA Sajben transonic diffuser dataset.

Prerequisites
-------------
Run ``scripts/parse_sajben_cfd.py`` first to generate
``data/processed/master_shock_dataset.pt``.

Usage
-----
    python scripts/validation/finetune_sajben.py [--epochs N] [--device mps|cpu|cuda]

Output
------
Fine-tuned checkpoint: ``models/le_pinn_sajben_finetuned.pt``
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LE-PINN on Sajben transonic diffuser data."
    )
    parser.add_argument("--epochs",  type=int,   default=500,
                        help="Number of fine-tuning epochs (default: 500)")
    parser.add_argument("--lr",      type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--device",  type=str,   default="mps",
                        choices=["mps", "cpu", "cuda"],
                        help="Compute device (default: mps on Apple Silicon)")
    parser.add_argument("--physics-weight", type=float, default=0.05,
                        dest="physics_weight",
                        help="Physics loss weight (default: 0.05)")
    parser.add_argument("--physics-max-points", type=int, default=None,
                        dest="physics_max_points",
                        help="Max training points used for physics loss per epoch "
                             "(default: auto cap on mps)")
    args = parser.parse_args()

    dataset_path   = str(REPO_ROOT / "data"   / "processed" / "master_shock_dataset.pt")
    pretrained     = str(REPO_ROOT / "models" / "le_pinn_sajben.pt")
    save_path      = str(REPO_ROOT / "models" / "le_pinn_sajben_finetuned.pt")

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
    print(f"  device   : {args.device}")
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
        device=args.device,
        verbose=True,
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
