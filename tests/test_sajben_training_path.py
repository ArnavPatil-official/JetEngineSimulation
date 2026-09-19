"""
Regression tests for the Sajben dataset-backed training entrypoints.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validation import finetune_sajben as finetune_sajben_script
from scripts.validation import train_sajben as train_sajben_script


# P4.3: the training data is the declared-split WIND dataset (route (a) of
# the P4.1 audit), no longer the quasi-1D master_shock_dataset.pt.
MASTER_DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "sajben_wind_dataset.pt"
)


def test_dataset_path_resolves() -> None:
    expected = str(MASTER_DATASET_PATH)
    assert train_sajben_script.DATASET_PATH == expected
    assert finetune_sajben_script.DATASET_PATH == expected


def test_dataset_carries_declared_split() -> None:
    if not MASTER_DATASET_PATH.exists():
        pytest.skip("dataset absent")
    ds = torch.load(MASTER_DATASET_PATH, map_location="cpu", weights_only=False)
    split = ds["split"]
    assert split["eval_rows_in_train"] == 0
    assert split["eval_source"].endswith("data.Mach46.txt")
    assert split["train_source"].endswith("sajben.cfl")
    assert len(split["train_sha256"]) == 64 and len(split["eval_sha256"]) == 64
    assert ds["inputs"].shape[0] == split["train_rows"]


def test_train_script_registers_attempt_and_constant_weights() -> None:
    a = train_sajben_script.ATTEMPT
    assert a["id"] == "P4.3-attempt-1"
    assert a["loss_weighting"]["schedule"] == "PhysicsWarmupWeighting"
    assert a["loss_weighting"]["data"] == 1.0
    assert a["pretrained"] is None
    assert a["seed"] == 42


def test_resolve_device_auto_is_cpu() -> None:
    assert train_sajben_script.resolve_device("auto") == "cpu"


def test_physics_warmup_weighting_never_decays_data() -> None:
    from simulation.nozzle.le_pinn import PhysicsWarmupWeighting
    w = PhysicsWarmupWeighting(max_epochs=100, warmup_fraction=0.5)
    for e in (0, 25, 50, 99):
        lam_d, lam_p, lam_bc = w.compute_weights(e)
        assert lam_d == 1.0 and lam_bc == 1.0
    assert w.compute_weights(0)[1] == 0.0
    assert w.compute_weights(25)[1] == pytest.approx(0.5)
    assert w.compute_weights(50)[1] == 1.0 and w.compute_weights(99)[1] == 1.0


def test_dataset_existence_check_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_dataset = tmp_path / "does_not_exist.pt"
    monkeypatch.setattr(train_sajben_script, "DATASET_PATH", str(missing_dataset))

    with pytest.raises(FileNotFoundError, match="Sajben dataset not found"):
        train_sajben_script.train_sajben_le_pinn(
            n_epochs=1,
            device="cpu",
            verbose=False,
        )


def test_dataset_schema_validation_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid_dataset = tmp_path / "invalid_dataset.pt"
    torch.save({"inputs": torch.randn(8, 6)}, invalid_dataset)
    monkeypatch.setattr(train_sajben_script, "DATASET_PATH", str(invalid_dataset))

    with pytest.raises(ValueError, match="Missing keys"):
        train_sajben_script.train_sajben_le_pinn(
            n_epochs=1,
            device="cpu",
            verbose=False,
        )


@pytest.mark.skipif(not MASTER_DATASET_PATH.exists(), reason="dataset absent")
def test_geometry_mode_is_planar(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_finetune_on_cfd_data(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object(), {"loss_total": [0.0], "loss_data": [0.0], "val_loss": [0.0]}

    monkeypatch.setattr(train_sajben_script, "finetune_on_cfd_data", fake_finetune_on_cfd_data)

    train_sajben_script.train_sajben_le_pinn(
        n_epochs=1,
        device="cpu",
        verbose=False,
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["geometry"] == "planar"


@pytest.mark.skipif(not MASTER_DATASET_PATH.exists(), reason="dataset absent")
def test_finetune_sajben_script_uses_planar_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_finetune_on_cfd_data(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object(), {"loss_total": [0.0], "loss_data": [0.0], "val_loss": [0.0]}

    monkeypatch.setattr(finetune_sajben_script, "finetune_on_cfd_data", fake_finetune_on_cfd_data)
    monkeypatch.setattr(
        sys,
        "argv",
        ["finetune_sajben.py", "--epochs", "1", "--device", "cpu", "--physics-debug"],
    )

    finetune_sajben_script.main()

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["geometry"] == "planar"
    assert kwargs["physics_debug"] is True
