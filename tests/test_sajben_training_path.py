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


MASTER_DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "master_shock_dataset.pt"
)


def test_dataset_path_resolves() -> None:
    expected = str(MASTER_DATASET_PATH)
    assert train_sajben_script.DATASET_PATH == expected
    assert str(finetune_sajben_script.REPO_ROOT / "data" / "processed" / "master_shock_dataset.pt") == expected


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
        ["finetune_sajben.py", "--epochs", "1", "--device", "cpu"],
    )

    finetune_sajben_script.main()

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["geometry"] == "planar"

