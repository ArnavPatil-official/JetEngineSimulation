"""
Train LE-PINN on Sajben transonic diffuser conditions.

Generates synthetic isentropic data at atmospheric / room-temperature
conditions through a geometry matching the Sajben diffuser (AR=1.5,
pseudo-axisymmetric throat radius = H/2 = 0.022 m).

Uses a cosine-annealing LR schedule with warm restarts so the data
loss does not stall.

Saves checkpoint to  models/le_pinn_sajben.pt

Usage::

    python scripts/validation/train_sajben.py [--epochs 5000]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Project root
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from simulation.nozzle.le_pinn import (
    LE_PINN,
    MinMaxNormalizer,
    AdaptiveLossWeighting,
    compute_rans_residuals,
    compute_wall_distances,
    compute_wall_normals,
    generate_synthetic_training_data,
    generate_wall_geometry,
    compute_wall_bc_loss,
    _safe_physics_loss,
    _estimate_wall_distances,
)

# ---------------------------------------------------------------------------
# Sajben-equivalent training parameters
# ---------------------------------------------------------------------------
SAJBEN_PARAMS = dict(
    NPR=1.22,              # label only (for checkpoint metadata)
    AR=1.50,               # exit-to-throat area ratio (matches Sajben)
    Throat_Radius=0.022,   # half of throat height H=0.044 m
    P_in=101_325.0,        # atmospheric stagnation pressure [Pa]
    T_in=293.0,            # room-temperature stagnation [K]
)

SAVE_PATH = str(_ROOT / "models" / "le_pinn_sajben.pt")
RANDOM_SEED = 42


def train_sajben_le_pinn(
    n_epochs: int = 5000,
    lr: float = 5e-4,
    save_path: str | None = None,
    device: str = "mps",
    verbose: bool = True,
) -> tuple:
    """
    Train LE-PINN on Sajben conditions with cosine-annealing LR schedule.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    dev = torch.device(device)

    NPR = SAJBEN_PARAMS["NPR"]
    AR = SAJBEN_PARAMS["AR"]
    Throat_Radius = SAJBEN_PARAMS["Throat_Radius"]
    P_in = SAJBEN_PARAMS["P_in"]
    T_in = SAJBEN_PARAMS["T_in"]

    # ---- Data ----
    inputs, targets, wall_dists = generate_synthetic_training_data(
        NPR=NPR, AR=AR, Throat_Radius=Throat_Radius, P_in=P_in, T_in=T_in,
    )

    # Normalizers
    input_norm = MinMaxNormalizer().fit(inputs)
    output_norm = MinMaxNormalizer().fit(targets)

    inputs_n = input_norm.transform(inputs).to(dev)
    targets_n = output_norm.transform(targets).to(dev)
    wall_dists = wall_dists.to(dev)

    # ---- Wall BC data ----
    upper_wall, lower_wall, geom = generate_wall_geometry(NPR, AR, Throat_Radius)
    wall_normals_upper = compute_wall_normals(upper_wall[:, 0], upper_wall[:, 1])
    wall_normals_lower = compute_wall_normals(lower_wall[:, 0], lower_wall[:, 1])
    wall_normals_lower[:, 1] = -wall_normals_lower[:, 1]

    def _midpoints(wall: np.ndarray) -> np.ndarray:
        return 0.5 * (wall[:-1] + wall[1:])
    wall_mid_upper = _midpoints(upper_wall)
    wall_mid_lower = _midpoints(lower_wall)
    n_wall = len(wall_mid_upper) + len(wall_mid_lower)
    wall_xy = np.concatenate([wall_mid_upper, wall_mid_lower], axis=0)
    wall_normals_all = np.concatenate([wall_normals_upper, wall_normals_lower], axis=0)

    wall_inputs_raw = np.column_stack([
        wall_xy[:, 0], wall_xy[:, 1],
        np.full(n_wall, geom["A5"]),
        np.full(n_wall, geom["A6"]),
        np.full(n_wall, P_in),
        np.full(n_wall, T_in),
    ]).astype(np.float32)

    wall_inputs_t = input_norm.transform(
        torch.from_numpy(wall_inputs_raw)
    ).to(dev).requires_grad_(True)
    wall_normals_t = torch.from_numpy(
        wall_normals_all.astype(np.float32)
    ).to(dev)

    # ---- Model ----
    model = LE_PINN().to(dev)

    # ---- Optimizer: AdamW with CosineAnnealingWarmRestarts ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-5,
    )
    # Restart every T_0 epochs — starts at lr, decays to eta_min, then restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=1e-6,
    )

    weighting = AdaptiveLossWeighting(max_epochs=n_epochs)

    history = {
        "loss_total": [], "loss_data": [],
        "loss_physics": [], "loss_bc": [], "lr": [],
    }

    if verbose:
        print("=" * 70)
        print("LE-PINN SAJBEN TRAINING  (cosine-annealing LR)")
        print("=" * 70)
        print(f"  Epochs:   {n_epochs}")
        print(f"  AR={AR},  Throat_Radius={Throat_Radius} m")
        print(f"  P_in={P_in:.0f} Pa,  T_in={T_in:.0f} K,  LR={lr:.0e}")
        print(f"  Points:   {len(inputs)} interior,  {n_wall} wall BC")
        print("=" * 70)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # 1. Data loss
        preds = model(inputs_n, wall_dists)
        loss_data = nn.functional.mse_loss(preds, targets_n)

        # 2. Physics loss
        loss_physics = _safe_physics_loss(model, inputs_n, wall_dists)

        # 3. BC loss
        loss_bc = compute_wall_bc_loss(model, wall_inputs_t, wall_normals_t)

        # 4. Adaptive weighting
        lam_d, lam_p, lam_bc = weighting.compute_weights(epoch)
        loss_total = lam_d * loss_data + lam_p * loss_physics + lam_bc * loss_bc

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(epoch + 1)

        history["loss_total"].append(loss_total.item())
        history["loss_data"].append(loss_data.item())
        history["loss_physics"].append(loss_physics.item())
        history["loss_bc"].append(loss_bc.item())
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if verbose and epoch % 200 == 0:
            print(
                f"Ep {epoch:5d} | Total {loss_total.item():.3e} | "
                f"Data {loss_data.item():.3e} | Physics {loss_physics.item():.3e} | "
                f"BC {loss_bc.item():.3e} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

    # ---- Save ----
    if save_path is not None:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_norm_min": input_norm.data_min,
            "input_norm_max": input_norm.data_max,
            "output_norm_min": output_norm.data_min,
            "output_norm_max": output_norm.data_max,
            "config": {
                "NPR": NPR, "AR": AR, "Throat_Radius": Throat_Radius,
                "P_in": P_in, "T_in": T_in, "n_epochs": n_epochs,
            },
            "seed": RANDOM_SEED,
        }
        torch.save(ckpt, save_path)
        if verbose:
            print(f"\n💾 Checkpoint saved: {save_path}")

    if verbose:
        print("=" * 70)
        print("✅ Sajben-condition LE-PINN training complete!")
        print(f"  Final data loss: {history['loss_data'][-1]:.4e}")
        print(f"  Final physics:   {history['loss_physics'][-1]:.4e}")
        print("=" * 70)

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LE-PINN on Sajben diffuser conditions"
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Number of training epochs (default: 5000)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Initial learning rate (default: 5e-4)",
    )
    args = parser.parse_args()

    model, history = train_sajben_le_pinn(
        n_epochs=args.epochs,
        lr=args.lr,
        save_path=SAVE_PATH,
        verbose=True,
    )


if __name__ == "__main__":
    main()
