"""
P4.2 regression guards for ``finetune_on_cfd_data``.

Background (docs/plan.md F2): fine-tuning ``le_pinn_sajben.pt`` produced a
checkpoint that scored worse than its initialisation. The diagnosis found
(1) the initialisation was a collapsed network (2 of 400 ReLU units alive at
layer 6, outputs constant to 1e-6), so both scores were the score of noise;
(2) the fine-tune path silently re-fitted the normalisers to a different
domain; (3) nothing measured the held-out loss before the first update, so
a worse checkpoint could be saved.

These tests pin the three guarantees that now hold:

* a saved fine-tuned checkpoint never has a worse validation loss than the
  weights it started from (validation split of the fine-tuning data, same
  coordinates);
* a collapsed initialisation is refused;
* a pretrained checkpoint's normalisers are reused, and a dataset outside
  that domain is refused unless the caller opts into re-fitting.

The data used is a synthetic, self-contained dataset (no repo files needed)
so the guards are exercised on a network that is genuinely learning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.nozzle.le_pinn import (  # noqa: E402
    LE_PINN,
    finetune_on_cfd_data,
    network_health,
)


def _make_dataset(path: Path, n: int = 600, seed: int = 0, x_lo: float = 0.0, x_hi: float = 1.0) -> str:
    """Smooth synthetic (N, 6) -> (N, 9) dataset with varying targets."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g) * (x_hi - x_lo) + x_lo
    y = torch.rand(n, generator=g) * 0.05
    inputs = torch.stack([x, y, torch.full_like(x, 0.05), torch.full_like(x, 0.08),
                          torch.full_like(x, 1.0e5), torch.full_like(x, 300.0)], 1)
    rho = 1.0 + 0.3 * torch.sin(3 * x)
    u = 100.0 + 200.0 * x + 50.0 * y
    v = 5.0 * torch.cos(4 * x)
    P = 5.0e4 + 4.0e4 * torch.cos(2 * x)
    T = 250.0 + 40.0 * x
    zeros = torch.zeros_like(x)
    targets = torch.stack([rho, u, v, P, T, zeros, zeros, zeros, 1.7e-5 + 0 * x], 1)
    torch.save({"inputs": inputs.float(), "targets": targets.float()}, path)
    return str(path)


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory) -> tuple[str, str]:
    """A short from-scratch run on the synthetic data -> a healthy checkpoint."""
    d = tmp_path_factory.mktemp("p42")
    data = _make_dataset(d / "data.pt")
    ckpt = str(d / "base.pt")
    finetune_on_cfd_data(dataset_path=data, pretrained_path=None, save_path=ckpt,
                         n_epochs=60, lr=1e-3, physics_loss_weight=0.0,
                         device="cpu", verbose=False)
    return data, ckpt


def test_finetune_never_worse_than_init(pretrained, tmp_path):
    data, ckpt = pretrained
    out = str(tmp_path / "ft.pt")
    _, hist = finetune_on_cfd_data(dataset_path=data, pretrained_path=ckpt, save_path=out,
                                   n_epochs=30, lr=1e-3, physics_loss_weight=0.0,
                                   device="cpu", verbose=False)
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert hist["val_epoch"][0] == -1, "validation must be measured before the first update"
    assert payload["val_loss_final"] <= payload["val_loss_init"] + 1e-12
    assert payload["val_loss_best"] == pytest.approx(payload["val_loss_final"])
    assert payload["val_loss_final"] <= min(hist["val_loss"]) + 1e-12


def test_finetune_with_useless_lr_returns_init_weights(pretrained, tmp_path):
    """If no epoch improves on the init, the saved weights ARE the init weights."""
    data, ckpt = pretrained
    out = str(tmp_path / "ft_nolearn.pt")
    finetune_on_cfd_data(dataset_path=data, pretrained_path=ckpt, save_path=out,
                         n_epochs=5, lr=0.5, physics_loss_weight=0.0,   # lr large enough to diverge
                         device="cpu", verbose=False)
    base = torch.load(ckpt, map_location="cpu", weights_only=False)
    ft = torch.load(out, map_location="cpu", weights_only=False)
    assert ft["val_loss_final"] <= ft["val_loss_init"] + 1e-12
    if ft["best_epoch"] < 0:
        for k, v in base["model_state_dict"].items():
            assert torch.equal(v, ft["model_state_dict"][k])


def test_checkpoint_provenance_fields(pretrained, tmp_path):
    data, ckpt = pretrained
    out = str(tmp_path / "ft_prov.pt")
    finetune_on_cfd_data(dataset_path=data, pretrained_path=ckpt, save_path=out,
                         n_epochs=2, lr=1e-4, physics_loss_weight=0.0,
                         device="cpu", verbose=False)
    p = torch.load(out, map_location="cpu", weights_only=False)
    for key in ("seed", "device", "dataset_sha256", "pretrained_sha256", "git_sha",
                "val_loss_init", "val_loss_best", "val_loss_final", "best_epoch",
                "health_init", "health_final"):
        assert key in p, key
    assert len(p["dataset_sha256"]) == 64 and len(p["pretrained_sha256"]) == 64
    for key in ("n_epochs", "lr", "physics_loss_weight", "normalizers"):
        assert key in p["config"], key
    assert p["config"]["normalizers"].startswith("reused from")


def test_pretrained_normalizers_are_reused(pretrained, tmp_path):
    data, ckpt = pretrained
    out = str(tmp_path / "ft_norm.pt")
    finetune_on_cfd_data(dataset_path=data, pretrained_path=ckpt, save_path=out,
                         n_epochs=2, lr=1e-4, physics_loss_weight=0.0,
                         device="cpu", verbose=False)
    base = torch.load(ckpt, map_location="cpu", weights_only=False)
    ft = torch.load(out, map_location="cpu", weights_only=False)
    for k in ("input_norm_min", "input_norm_max", "output_norm_min", "output_norm_max"):
        assert torch.allclose(base[k].float(), ft[k].float()), k


def test_dataset_outside_pretrained_domain_is_refused(pretrained, tmp_path):
    data, ckpt = pretrained
    far = _make_dataset(tmp_path / "far.pt", x_lo=3.0, x_hi=4.0)   # x outside [0, 1]
    with pytest.raises(ValueError, match="outside the pretrained"):
        finetune_on_cfd_data(dataset_path=far, pretrained_path=ckpt,
                             save_path=str(tmp_path / "x.pt"), n_epochs=1,
                             physics_loss_weight=0.0, device="cpu", verbose=False)
    # explicit opt-in re-fits (and warns)
    with pytest.warns(RuntimeWarning, match="refit_normalizers=True"):
        finetune_on_cfd_data(dataset_path=far, pretrained_path=ckpt,
                             save_path=str(tmp_path / "x.pt"), n_epochs=1,
                             physics_loss_weight=0.0, device="cpu", verbose=False,
                             refit_normalizers=True)


def test_collapsed_init_is_refused(pretrained, tmp_path):
    data, ckpt = pretrained
    base = torch.load(ckpt, map_location="cpu", weights_only=False)
    dead = dict(base)
    sd = {k: v.clone() for k, v in base["model_state_dict"].items()}
    # Kill the last hidden layer of the global net: every ReLU there goes dark,
    # so the output is the final bias everywhere (a constant field).
    sd["global_net.net.12.bias"] = torch.full_like(sd["global_net.net.12.bias"], -1e3)
    dead["model_state_dict"] = sd
    dead_path = str(tmp_path / "dead.pt")
    torch.save(dead, dead_path)

    model = LE_PINN()
    model.load_state_dict(sd)
    inp = torch.rand(200, 6)
    h = network_health(model, inp)
    assert h["collapsed"] and h["alive_units"][-1] == 0

    with pytest.raises(ValueError, match="collapsed"):
        finetune_on_cfd_data(dataset_path=data, pretrained_path=dead_path,
                             save_path=str(tmp_path / "y.pt"), n_epochs=1,
                             physics_loss_weight=0.0, device="cpu", verbose=False)
    # override exists, and the guarantee still holds
    _, hist = finetune_on_cfd_data(dataset_path=data, pretrained_path=dead_path,
                                   save_path=str(tmp_path / "y.pt"), n_epochs=3,
                                   physics_loss_weight=0.0, device="cpu", verbose=False,
                                   allow_collapsed_init=True)
    assert hist["val_loss"][-1] <= hist["val_loss"][0] + 1e-12


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "models" / "le_pinn_sajben.pt").exists(),
    reason="le_pinn_sajben.pt not present",
)
def test_shipped_sajben_checkpoint_is_collapsed():
    """Documents the F2 root cause on the actual artifact."""
    ck = torch.load(Path(__file__).parent.parent / "models" / "le_pinn_sajben.pt",
                    map_location="cpu", weights_only=False)
    model = LE_PINN()
    model.load_state_dict(ck["model_state_dict"])
    inp = torch.rand(2000, 6)
    inp[:, 2:] = 0.0
    h = network_health(model, inp)
    assert h["collapsed"]
    assert min(h["alive_units"]) <= 2
