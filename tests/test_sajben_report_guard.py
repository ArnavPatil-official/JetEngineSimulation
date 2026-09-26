"""
P5.1 report guard: ``sajben_report_p43.py`` must never publish a terminal
outcome from a partial selection or from runs without exit-0 completion
evidence, and must refuse before writing either report file.

Runs in a temporary repo root with small stand-in checkpoints and ``.done``
markers; the real checkpoints, logs and reports are never touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validation import sajben_report_p43 as rp  # noqa: E402
from scripts.validation.train_sajben import ATTEMPTS, done_marker_path  # noqa: E402

A3 = ATTEMPTS[3]
KINDS = ("out", "out_dataonly")


def _log(root: Path, seed: int, kind: str) -> Path:
    tag = f"a3_dataonly_s{seed}" if kind == "out_dataonly" else f"a3_s{seed}"
    return root / "outputs" / "logs" / f"train_sajben_v5_{tag}.log"


def _write_run(root: Path, seed: int, kind: str, *, exit_code=0, attempt_id=None, ck_seed=None,
               marker=True, marker_ck=None) -> Path:
    ck = root / A3[kind].format(seed=seed)
    ck.parent.mkdir(parents=True, exist_ok=True)
    aid = attempt_id or (A3["id"] + ("-dataonly" if kind == "out_dataonly" else ""))
    torch.save({"attempt": {"id": aid, "terminal": True}, "seed": seed if ck_seed is None else ck_seed}, ck)
    log = _log(root, seed, kind)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("training output\n")
    if marker:
        done_marker_path(log).write_text(json.dumps({
            "exit_code": exit_code,
            "checkpoint": marker_ck or str(ck.relative_to(root)),
            "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest() if exit_code == 0 else None,
            "finished": "2026-09-26T00:00:00",
        }))
    return ck


def _write_all(root: Path, override: dict | None = None) -> None:
    for s in A3["seeds"]:
        for k in KINDS:
            _write_run(root, s, k, **(override or {}).get((s, k), {}))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(rp, "REPO_ROOT", root)
    monkeypatch.setattr(rp, "OUT_MD", root / "outputs" / "sajben_retrain_v5.md")
    monkeypatch.setattr(rp, "OUT_CSV", root / "outputs" / "sajben_retrain_v5.csv")
    monkeypatch.setattr(rp, "default_log_path", lambda n, s, d: _log(root, s, "out_dataonly" if d else "out"))
    return root


def _all_six(root: Path) -> list[Path]:
    return [root / A3[k].format(seed=s) for s in A3["seeds"] for k in KINDS]


def _run_main(monkeypatch, argv: list[str], stop_after: int = 1):
    scored = []

    def fake_score(p: Path) -> dict:
        scored.append(p.name)
        if len(scored) >= stop_after:
            raise RuntimeError("stop after selection")   # outputs are never reached in these tests
        return {}

    monkeypatch.setattr(rp, "score", fake_score)
    monkeypatch.setattr(sys, "argv", ["sajben_report_p43.py", *argv])
    with pytest.raises((SystemExit, RuntimeError)) as exc:
        rp.main()
    return scored, exc


# ---- selection -----------------------------------------------------------

def test_one_terminal_checkpoint_expands_to_the_full_registered_set(repo):
    _write_all(repo)
    one = repo / A3["out"].format(seed=43)
    assert rp.resolve_selection([one]) == _all_six(repo)


def test_earlier_attempts_kept_in_place_and_terminal_expanded_once(repo, tmp_path):
    _write_all(repo)
    earlier = repo / "models" / "le_pinn_sajben_v5_a2.pt"
    torch.save({"attempt": {"id": "P4.3-attempt-2"}, "seed": 42}, earlier)
    sel = [earlier, repo / A3["out_dataonly"].format(seed=44), repo / A3["out"].format(seed=42)]
    assert rp.resolve_selection(sel) == [earlier.resolve(), *_all_six(repo)]


def test_duplicate_selection_is_refused(repo):
    _write_all(repo)
    p = repo / A3["out"].format(seed=42)
    with pytest.raises(SystemExit, match="more than once"):
        rp.resolve_selection([p, p])


def test_terminal_checkpoint_outside_registered_paths_is_refused(repo):
    stray = repo / "elsewhere" / Path(A3["out"].format(seed=42)).name
    stray.parent.mkdir(parents=True)
    torch.save({"attempt": {"id": A3["id"], "terminal": True}, "seed": 42}, stray)
    with pytest.raises(SystemExit, match="mismatched identity"):
        rp.resolve_selection([stray])


# ---- completion evidence -------------------------------------------------

def test_complete_evidence_passes_guard(repo):
    _write_all(repo)
    ev = rp.terminal_guard(rp.resolve_selection([repo / A3["out"].format(seed=42)]))
    assert len(ev) == 6 and all(e["ok"] for e in ev)


@pytest.mark.parametrize("case, override, needle", [
    ("missing checkpoint", None, "checkpoint missing"),
    ("missing marker (legacy log, no trailer)", {"marker": False}, "no exit=N trailer"),
    ("nonzero exit", {"exit_code": 137}, "exit code 137"),
    ("wrong seed recorded", {"ck_seed": 44}, "mismatched identity"),
    ("wrong attempt recorded", {"attempt_id": "P4.3-attempt-2"}, "mismatched identity"),
    ("marker names another checkpoint", {"marker_ck": "models/other.pt"}, "mismatched identity"),
])
def test_bad_evidence_refuses_before_any_output(repo, monkeypatch, case, override, needle):
    _write_all(repo, {(43, "out"): override} if override is not None else None)
    if override is None:
        (repo / A3["out"].format(seed=43)).unlink()
    scored, exc = _run_main(monkeypatch, [A3["out"].format(seed=42)])
    assert exc.type is SystemExit and needle in str(exc.value), case
    assert "seed 43" in str(exc.value)
    assert scored == []
    assert not rp.OUT_MD.exists() and not rp.OUT_CSV.exists()


def test_partial_selection_scores_all_six_not_the_subset(repo, monkeypatch):
    _write_all(repo)
    scored, exc = _run_main(monkeypatch, [A3["out"].format(seed=42)], stop_after=6)
    assert exc.type is RuntimeError          # reached scoring: selection and guard passed
    assert scored == [p.name for p in _all_six(repo)]
    assert not rp.OUT_MD.exists() and not rp.OUT_CSV.exists()
