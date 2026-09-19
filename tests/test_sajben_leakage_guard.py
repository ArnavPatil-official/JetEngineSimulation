"""
P4.3 split-leakage guard: ``sajben_validation.py`` must refuse a checkpoint
whose declared training split overlaps the evaluation set, accept one whose
record is clean, and warn (not refuse) on a legacy checkpoint without one.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validation.sajben_validation import check_split_record  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "data" / "raw" / "data.Mach46.txt"
pytestmark = pytest.mark.skipif(not EVAL.exists(), reason="experimental data file absent")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _clean_split() -> dict:
    return {
        "attempt": "test", "train_source": "wind/sajben.cfl", "train_grid": "wind/sajben.cgd",
        "train_sha256": "a" * 64, "train_grid_sha256": "b" * 64, "train_rows": 10,
        "eval_source": str(EVAL.name), "eval_sha256": _sha(EVAL), "eval_rows_in_train": 0,
    }


def test_clean_split_is_accepted(capsys):
    out = check_split_record({"split": _clean_split()}, EVAL)
    assert out["eval_rows_in_train"] == 0
    assert "OK" in capsys.readouterr().out


def test_eval_hash_in_training_sources_is_refused():
    s = _clean_split(); s["train_sha256"] = _sha(EVAL)
    with pytest.raises(ValueError, match="overlaps the evaluation set"):
        check_split_record({"split": s}, EVAL)


def test_eval_file_named_as_training_source_is_refused():
    s = _clean_split(); s["train_source"] = f"data/raw/{EVAL.name}"
    with pytest.raises(ValueError, match="overlaps the evaluation set"):
        check_split_record({"split": s}, EVAL)


def test_nonzero_eval_rows_in_train_is_refused():
    s = _clean_split(); s["eval_rows_in_train"] = 3
    with pytest.raises(ValueError, match="overlaps the evaluation set"):
        check_split_record({"split": s}, EVAL)


def test_legacy_checkpoint_warns_but_scores():
    with pytest.warns(RuntimeWarning, match="no declared train/eval split"):
        assert check_split_record({"config": {"dataset": "x/master_shock_dataset.pt"}}, EVAL) is None


def test_legacy_checkpoint_trained_on_eval_file_is_refused():
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="names the evaluation file"):
            check_split_record({"config": {"dataset": f"data/raw/{EVAL.name}"}}, EVAL)


def test_changed_eval_file_warns():
    s = _clean_split(); s["eval_sha256"] = "0" * 64
    with pytest.warns(RuntimeWarning, match="has changed since the split was declared"):
        check_split_record({"split": s}, EVAL)
