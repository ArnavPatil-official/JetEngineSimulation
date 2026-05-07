#!/usr/bin/env python3
"""
Train a unified LE-PINN checkpoint that covers both:
1) engine-cycle nozzle operating conditions (high T/P, larger A5/A6), and
2) Sajben/CFD domain from master_shock_dataset.pt (low T/P, planar diffuser data).

Output checkpoint:
    models/le_pinn_unified.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.nozzle.le_pinn import (
    LE_PINN,
    MinMaxNormalizer,
    AdaptiveLossWeighting,
    generate_synthetic_training_data,
    _estimate_wall_distances,
    _safe_physics_loss,
    RANDOM_SEED,
)


def _safe_torch_load(path: str, **kwargs):
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _auto_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_synthetic_multidomain(
    n_engine_cases: int,
    n_cfd_scale_cases: int,
    n_legacy_cases: int,
    n_axial: int,
    n_radial: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build synthetic LE-PINN data across multiple regimes.

    Returns:
        inputs_syn  : (N, 6)
        targets_syn : (N, 9)
        walls_syn   : (N, 1)
    """
    rng = np.random.default_rng(seed)

    inputs_all = []
    targets_all = []
    walls_all = []

    # Engine-cycle regime (areas and thermodynamics close to integrated cycle)
    for _ in range(n_engine_cases):
        # A5 range ~0.15..0.27 m^2 => throat radius ~0.22..0.29 m
        throat_r = float(rng.uniform(0.22, 0.29))
        ar = float(rng.uniform(1.35, 1.85))
        npr = float(rng.uniform(2.0, 8.0))
        p_in = float(rng.uniform(2.0e5, 4.6e5))
        t_in = float(rng.uniform(1400.0, 2400.0))
        inp, tgt, wd = generate_synthetic_training_data(
            n_axial=n_axial,
            n_radial=n_radial,
            NPR=npr,
            AR=ar,
            Throat_Radius=throat_r,
            P_in=p_in,
            T_in=t_in,
        )
        inputs_all.append(inp)
        targets_all.append(tgt)
        walls_all.append(wd)

    # CFD/Sajben-scale geometry + low T/P regime
    for _ in range(n_cfd_scale_cases):
        # A5 around 0.044 m^2 => throat radius ~0.118 m
        throat_r = float(rng.uniform(0.10, 0.14))
        ar = float(rng.uniform(1.35, 1.75))
        npr = float(rng.uniform(1.05, 1.6))
        p_in = float(rng.uniform(1.0e5, 1.6e5))
        t_in = float(rng.uniform(260.0, 340.0))
        inp, tgt, wd = generate_synthetic_training_data(
            n_axial=n_axial,
            n_radial=n_radial,
            NPR=npr,
            AR=ar,
            Throat_Radius=throat_r,
            P_in=p_in,
            T_in=t_in,
        )
        inputs_all.append(inp)
        targets_all.append(tgt)
        walls_all.append(wd)

    # Legacy LE-PINN axisymmetric regime
    for _ in range(n_legacy_cases):
        throat_r = float(rng.uniform(0.045, 0.065))
        ar = float(rng.uniform(1.45, 1.70))
        npr = float(rng.uniform(5.0, 8.0))
        p_in = float(rng.uniform(5.2e5, 7.2e5))
        t_in = float(rng.uniform(1500.0, 1900.0))
        inp, tgt, wd = generate_synthetic_training_data(
            n_axial=n_axial,
            n_radial=n_radial,
            NPR=npr,
            AR=ar,
            Throat_Radius=throat_r,
            P_in=p_in,
            T_in=t_in,
        )
        inputs_all.append(inp)
        targets_all.append(tgt)
        walls_all.append(wd)

    return (
        torch.cat(inputs_all, dim=0).float(),
        torch.cat(targets_all, dim=0).float(),
        torch.cat(walls_all, dim=0).float(),
    )


def train_unified_le_pinn(
    dataset_path: str,
    save_path: str,
    pretrained_path: str | None,
    n_epochs: int,
    lr: float,
    device: str,
    physics_loss_weight: float,
    physics_points_per_geom: int,
    cfd_weight_boost: float,
    n_engine_cases: int,
    n_cfd_scale_cases: int,
    n_legacy_cases: int,
    n_axial: int,
    n_radial: int,
    max_synth_points: int,
    max_cfd_points: int,
    val_fraction: float,
    verbose: bool,
) -> Tuple[LE_PINN, Dict[str, list]]:
    dev = torch.device(device)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"CFD dataset not found: {dataset_path}")

    # 1) Synthetic multidomain data
    inputs_syn, targets_syn, walls_syn = _build_synthetic_multidomain(
        n_engine_cases=n_engine_cases,
        n_cfd_scale_cases=n_cfd_scale_cases,
        n_legacy_cases=n_legacy_cases,
        n_axial=n_axial,
        n_radial=n_radial,
        seed=RANDOM_SEED,
    )
    if inputs_syn.shape[0] > max_synth_points:
        idx = torch.randperm(inputs_syn.shape[0])[:max_synth_points]
        inputs_syn = inputs_syn[idx]
        targets_syn = targets_syn[idx]
        walls_syn = walls_syn[idx]

    # 2) CFD dataset
    cfd = _safe_torch_load(dataset_path)
    inputs_cfd = cfd["inputs"].float()
    targets_cfd = cfd["targets"].float()
    weights_cfd = cfd["sample_weights"].float() if "sample_weights" in cfd else None

    valid = (
        torch.isfinite(inputs_cfd).all(dim=1)
        & torch.isfinite(targets_cfd[:, :5]).all(dim=1)
    )
    inputs_cfd = inputs_cfd[valid]
    targets_cfd = targets_cfd[valid]
    if weights_cfd is not None:
        weights_cfd = weights_cfd[valid]

    targets_cfd[:, 5:] = torch.nan_to_num(targets_cfd[:, 5:], nan=0.0, posinf=0.0, neginf=0.0)

    if inputs_cfd.shape[0] > max_cfd_points:
        idx = torch.randperm(inputs_cfd.shape[0])[:max_cfd_points]
        inputs_cfd = inputs_cfd[idx]
        targets_cfd = targets_cfd[idx]
        if weights_cfd is not None:
            weights_cfd = weights_cfd[idx]

    cfd_ref_x = inputs_cfd[:, 0]
    cfd_ref_y_abs = inputs_cfd[:, 1].abs()
    walls_cfd = _estimate_wall_distances(inputs_cfd[:, :2], cfd_ref_x, cfd_ref_y_abs).float()

    # 3) Merge datasets
    inputs_all = torch.cat([inputs_syn, inputs_cfd], dim=0)
    targets_all = torch.cat([targets_syn, targets_cfd], dim=0)
    walls_all = torch.cat([walls_syn, walls_cfd], dim=0)

    # geometry flag: 1 = axisymmetric synthetic, 0 = planar CFD
    geom_flag = torch.cat(
        [
            torch.ones(inputs_syn.shape[0], dtype=torch.bool),
            torch.zeros(inputs_cfd.shape[0], dtype=torch.bool),
        ],
        dim=0,
    )

    w_syn = torch.ones(inputs_syn.shape[0], dtype=torch.float32)
    if weights_cfd is None:
        w_cfd = torch.ones(inputs_cfd.shape[0], dtype=torch.float32) * float(cfd_weight_boost)
    else:
        w_cfd = weights_cfd.clone().float()
        w_cfd = w_cfd / (w_cfd.mean() + 1e-12)
        w_cfd = w_cfd * float(cfd_weight_boost)
    data_weights = torch.cat([w_syn, w_cfd], dim=0)

    # 4) Train/val split
    N = inputs_all.shape[0]
    perm = torch.randperm(N)
    n_val = max(1, int(val_fraction * N))
    train_idx = perm[:-n_val]
    val_idx = perm[-n_val:]

    x_train = inputs_all[train_idx]
    y_train = targets_all[train_idx]
    wd_train = walls_all[train_idx]
    g_train = geom_flag[train_idx]
    w_train = data_weights[train_idx]

    x_val = inputs_all[val_idx]
    y_val = targets_all[val_idx]
    wd_val = walls_all[val_idx]

    input_norm = MinMaxNormalizer().fit(x_train)
    output_norm_5 = MinMaxNormalizer().fit(y_train[:, :5])
    # Physics-loss path operates on device tensors, so keep device-local copies.
    input_norm_dev = MinMaxNormalizer()
    input_norm_dev.data_min = input_norm.data_min.to(dev)
    input_norm_dev.data_max = input_norm.data_max.to(dev)
    output_norm_5_dev = MinMaxNormalizer()
    output_norm_5_dev.data_min = output_norm_5.data_min.to(dev)
    output_norm_5_dev.data_max = output_norm_5.data_max.to(dev)

    x_train_n = input_norm.transform(x_train).to(dev)
    y_train_5n = output_norm_5.transform(y_train[:, :5]).to(dev)
    wd_train = wd_train.to(dev)
    g_train = g_train.to(dev)
    w_train = w_train.to(dev)
    w_train = w_train / (w_train.mean() + 1e-12)

    x_val_n = input_norm.transform(x_val).to(dev)
    y_val_5n = output_norm_5.transform(y_val[:, :5]).to(dev)
    wd_val = wd_val.to(dev)

    model = LE_PINN().to(dev)
    if pretrained_path is not None and Path(pretrained_path).exists():
        try:
            ckpt = _safe_torch_load(pretrained_path, map_location=dev)
            model.load_state_dict(ckpt["model_state_dict"])
            if verbose:
                print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as exc:
            print(f"Warning: pretrained load failed ({exc}), training from random init.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-9
    )
    weighting = AdaptiveLossWeighting(max_epochs=n_epochs)

    history: Dict[str, list] = {
        "loss_total": [],
        "loss_data": [],
        "loss_physics": [],
        "val_loss": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state = None

    if verbose:
        print("=" * 74)
        print("LE-PINN UNIFIED TRAINING")
        print("=" * 74)
        print(f"Device: {device}")
        print(f"Synthetic points: {inputs_syn.shape[0]}")
        print(f"CFD points:       {inputs_cfd.shape[0]}")
        print(f"Train/Val:        {x_train.shape[0]} / {x_val.shape[0]}")
        print(f"Physics weight:   {physics_loss_weight}")
        print("=" * 74)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(x_train_n, wd_train)
        sq = (pred[:, :5] - y_train_5n) ** 2
        loss_data = (sq.mean(dim=1) * w_train).mean()

        loss_physics = torch.tensor(0.0, device=dev)
        if physics_loss_weight > 0:
            # Axisymmetric physics sample
            idx_axis = torch.where(g_train)[0]
            if idx_axis.numel() > 0:
                if idx_axis.numel() > physics_points_per_geom:
                    p = torch.randperm(idx_axis.numel(), device=dev)[:physics_points_per_geom]
                    idx_axis = idx_axis[p]
                loss_axis = _safe_physics_loss(
                    model,
                    x_train_n[idx_axis],
                    wd_train[idx_axis],
                    input_norm=input_norm_dev,
                    output_norm=output_norm_5_dev,
                    geometry="axisymmetric",
                )
                if not torch.isfinite(loss_axis):
                    warnings.warn(
                        "Non-finite axisymmetric physics loss detected; "
                        "dropping this term for current epoch.",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                    loss_axis = torch.tensor(0.0, device=dev)
                loss_physics = loss_physics + 0.5 * loss_axis

            # Planar physics sample
            idx_planar = torch.where(~g_train)[0]
            if idx_planar.numel() > 0:
                if idx_planar.numel() > physics_points_per_geom:
                    p = torch.randperm(idx_planar.numel(), device=dev)[:physics_points_per_geom]
                    idx_planar = idx_planar[p]
                loss_planar = _safe_physics_loss(
                    model,
                    x_train_n[idx_planar],
                    wd_train[idx_planar],
                    input_norm=input_norm_dev,
                    output_norm=output_norm_5_dev,
                    geometry="planar",
                )
                if not torch.isfinite(loss_planar):
                    warnings.warn(
                        "Non-finite planar physics loss detected; "
                        "dropping this term for current epoch.",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                    loss_planar = torch.tensor(0.0, device=dev)
                loss_physics = loss_physics + 0.5 * loss_planar

        if not torch.isfinite(loss_data):
            raise RuntimeError("Encountered non-finite data loss; aborting training.")
        if not torch.isfinite(loss_physics):
            warnings.warn(
                "Non-finite total physics loss detected; replacing with zero.",
                RuntimeWarning,
                stacklevel=1,
            )
            loss_physics = torch.tensor(0.0, device=dev)

        lam_d, lam_p, _ = weighting.compute_weights(epoch)
        loss_total = lam_d * loss_data + physics_loss_weight * lam_p * loss_physics
        if not torch.isfinite(loss_total):
            warnings.warn(
                "Non-finite total loss detected; using data-loss-only step.",
                RuntimeWarning,
                stacklevel=1,
            )
            loss_total = loss_data

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_data.item())

        history["loss_total"].append(float(loss_total.item()))
        history["loss_data"].append(float(loss_data.item()))
        history["loss_physics"].append(float(loss_physics.item()))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if epoch % 25 == 0 or epoch == (n_epochs - 1):
            model.eval()
            with torch.no_grad():
                pv = model(x_val_n, wd_val)
                vloss = nn.functional.mse_loss(pv[:, :5], y_val_5n).item()
            history["val_loss"].append(float(vloss))
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(
                    f"Ep {epoch:4d} | Total {loss_total.item():.3e} | "
                    f"Data {loss_data.item():.3e} | Physics {loss_physics.item():.3e} | "
                    f"Val {vloss:.3e} | lr={optimizer.param_groups[0]['lr']:.1e}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_norm_min": input_norm.data_min,
            "input_norm_max": input_norm.data_max,
            "output_norm_min": output_norm_5.data_min,
            "output_norm_max": output_norm_5.data_max,
            "config": {
                "mode": "unified_multidomain",
                "dataset": dataset_path,
                "pretrained": pretrained_path,
                "epochs": n_epochs,
                "lr": lr,
                "physics_loss_weight": physics_loss_weight,
                "physics_points_per_geom": physics_points_per_geom,
                "cfd_weight_boost": cfd_weight_boost,
                "n_engine_cases": n_engine_cases,
                "n_cfd_scale_cases": n_cfd_scale_cases,
                "n_legacy_cases": n_legacy_cases,
                "n_axial": n_axial,
                "n_radial": n_radial,
                "max_synth_points": max_synth_points,
                "max_cfd_points": max_cfd_points,
                "val_fraction": val_fraction,
                "best_val_loss": best_val,
            },
            "seed": RANDOM_SEED,
        },
        save_path,
    )
    if verbose:
        print(f"\nSaved unified checkpoint: {save_path}")
        print("=" * 74)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unified multidomain LE-PINN.")
    parser.add_argument("--dataset", type=str, default=str(REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt"))
    parser.add_argument("--save-path", type=str, default=str(REPO_ROOT / "models" / "le_pinn_unified.pt"))
    parser.add_argument("--pretrained", type=str, default=str(REPO_ROOT / "models" / "le_pinn.pt"))
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--physics-weight", type=float, default=0.03, dest="physics_weight")
    parser.add_argument("--physics-points", type=int, default=4096, dest="physics_points")
    parser.add_argument("--cfd-weight-boost", type=float, default=2.5, dest="cfd_weight_boost")
    parser.add_argument("--engine-cases", type=int, default=20, dest="engine_cases")
    parser.add_argument("--cfd-scale-cases", type=int, default=20, dest="cfd_scale_cases")
    parser.add_argument("--legacy-cases", type=int, default=12, dest="legacy_cases")
    parser.add_argument("--n-axial", type=int, default=64, dest="n_axial")
    parser.add_argument("--n-radial", type=int, default=20, dest="n_radial")
    parser.add_argument("--max-synth-points", type=int, default=110000, dest="max_synth_points")
    parser.add_argument("--max-cfd-points", type=int, default=90000, dest="max_cfd_points")
    parser.add_argument("--val-fraction", type=float, default=0.15, dest="val_fraction")
    args = parser.parse_args()

    device = _auto_device(args.device)
    pretrained = args.pretrained if args.pretrained and Path(args.pretrained).exists() else None

    print("=" * 74)
    print("UNIFIED LE-PINN RETRAIN")
    print("=" * 74)
    print(f"Dataset:     {args.dataset}")
    print(f"Pretrained:  {pretrained or '(none)'}")
    print(f"Save path:   {args.save_path}")
    print(f"Device:      {device}")
    print(f"Epochs/LR:   {args.epochs} / {args.lr}")
    print(f"Phys weight: {args.physics_weight}")
    print("=" * 74)

    train_unified_le_pinn(
        dataset_path=args.dataset,
        save_path=args.save_path,
        pretrained_path=pretrained,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        physics_loss_weight=args.physics_weight,
        physics_points_per_geom=args.physics_points,
        cfd_weight_boost=args.cfd_weight_boost,
        n_engine_cases=args.engine_cases,
        n_cfd_scale_cases=args.cfd_scale_cases,
        n_legacy_cases=args.legacy_cases,
        n_axial=args.n_axial,
        n_radial=args.n_radial,
        max_synth_points=args.max_synth_points,
        max_cfd_points=args.max_cfd_points,
        val_fraction=args.val_fraction,
        verbose=True,
    )


if __name__ == "__main__":
    main()
