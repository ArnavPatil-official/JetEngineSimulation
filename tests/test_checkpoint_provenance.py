"""
P6.8: every Phase-4/5 era checkpoint (models/*_v5*.pt) records its provenance:
seed, device, git SHA, a dataset hash that matches the dataset file, and the
hidden-layer activation.

Checkpoints are protected and never rewritten. Five of them predate the
``activation`` field (LE-PINN attempts 1-2 and the data-only reference; both
turbine surrogates). For those, the activation is derived from the source file
at the checkpoint's own recorded git SHA, which must hard-code exactly one
activation type and offer no activation option.
"""

import hashlib
import re
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = sorted((ROOT / "models").glob("*_v5*.pt"))
SOURCE = {"le_pinn": "simulation/nozzle/le_pinn.py", "turbine_pinn": "simulation/turbine/turbine.py"}
ACTIVATIONS = {"nn.ReLU()": "relu", "nn.Tanh()": "tanh"}


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _dataset_file(ck: dict) -> tuple[Path, str]:
    if "dataset_sha256" in ck:                       # LE-PINN: config['dataset'] (absolute at training time)
        p = Path(ck["config"]["dataset"])
        return (ROOT / "data" / "processed" / p.name if p.is_absolute() else ROOT / p), ck["dataset_sha256"]
    return ROOT / ck["envelope_csv"], ck["envelope_sha256"]   # turbine surrogate: training envelope


def _activation_at_sha(ckpt: Path, sha: str) -> str:
    module = SOURCE["le_pinn" if ckpt.name.startswith("le_pinn") else "turbine_pinn"]
    src = _git("show", f"{sha}:{module}")
    assert "_make_activation" not in src, f"{ckpt.name}: activation was configurable at {sha}"
    found = {name for pat, name in ACTIVATIONS.items() if pat in src}
    assert len(found) == 1, f"{ckpt.name}: ambiguous activation at {sha}: {found}"
    return found.pop()


def test_v5_checkpoints_exist():
    assert len(CHECKPOINTS) >= 10


@pytest.mark.parametrize("ckpt", CHECKPOINTS, ids=lambda p: p.name)
def test_checkpoint_records_provenance(ckpt):
    ck = _load(ckpt)
    assert isinstance(ck["seed"], int)
    assert ck["device"] in ("cpu", "cuda", "mps")
    sha = ck["git_sha"]
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert _git("cat-file", "-t", sha).strip() == "commit", f"{sha} not in this repository"

    path, recorded = _dataset_file(ck)
    assert re.fullmatch(r"[0-9a-f]{64}", recorded)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded, f"{path} changed since training"

    act = ck.get("activation") or ck.get("config", {}).get("activation")
    derived = None if act else _activation_at_sha(ckpt, sha)
    assert (act or derived) in ("relu", "tanh")
