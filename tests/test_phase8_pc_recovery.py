"""Manufactured fixtures for the stage-f serialization fix and the one-shot f-i recovery; no real score is run."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.phase8 import pc_recover_f_to_i as recovery
from scripts.phase8.saf_surrogate import registration, run as saf_run, score

SHA = lambda data: hashlib.sha256(data).hexdigest()
FIX = recovery.FIX_PATH


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())
    return SHA(path.read_bytes())


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- serialization regression

def manufactured_concordance():
    properties = {"lifecycle": {"baseline_fossil_gCO2e_MJ": 89.0, "pathways": {
        name: {"triangular": {"min": 10.0, "mode": 20.0, "max": 30.0}} for name in ("HEFA", "FT", "ATJ")}}}
    draws = ("central", *[f"draw_{i:02d}" for i in range(64)])
    queries = [{"fuel": fuel, "op": "IDLE", "draw_id": draw} for fuel in ("HEFA-10", "JetA") for draw in draws]
    ff = np.asarray([2.0 + .01 * i for i in range(65)] + [1.0 + .01 * i for i in range(65)])
    claims = [{"quantity": "ff", "a": "HEFA-10", "b": "JetA", "op": "IDLE", "spread_S_dooley2012_vs_2010": "0.1",
               "in_domain": "True", "claimed": "True", "delta_central": "1.0"}]
    return claims, queries, (ff, ff.copy(), ff.copy()), properties


def test_unconverted_numpy_bool_raises_old_error_and_fixed_writer_serializes_it():
    rows = score.claim_concordance(*manufactured_concordance())
    value = rows[0]["central_sign_concordant"]
    assert isinstance(value, np.bool_) and bool(value)
    with pytest.raises(TypeError, match="Object of type bool is not JSON serializable"):
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False)  # the pre-fix json_bytes call
    text = registration.json_bytes(rows).decode()
    assert json.loads(text)[0]["central_sign_concordant"] is True
    assert '"central_sign_concordant": true' in text


def test_fixed_writer_keeps_ordinary_bytes_and_rejections():
    ordinary = {"b": [1, 2.5, None, True, "x"], "a": {"n": 1e-12}}
    expected = (json.dumps(ordinary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    assert registration.json_bytes(ordinary) == expected
    for bad in (float("nan"), np.float64("nan"), np.float32("inf")):
        with pytest.raises(ValueError):
            registration.json_bytes({"x": bad})
    for bad in (object(), {1, 2}, b"raw"):
        with pytest.raises(TypeError, match="not JSON serializable"):
            registration.json_bytes({"x": bad})
    assert registration.json_bytes({"i": np.int64(3), "f": np.float32(.5)}) == b'{\n  "f": 0.5,\n  "i": 3\n}\n'


def test_current_fix_file_is_exactly_the_authorized_edit_of_the_source_commit():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "show", f"{recovery.SOURCE_COMMIT}:{FIX}"], cwd=root, capture_output=True)
    if result.returncode:
        pytest.skip("source commit is not available in this checkout")
    expected = result.stdout
    for before, after in recovery.AUTHORIZED_EDITS[FIX]:
        assert expected.count(before.encode()) == 1
        expected = expected.replace(before.encode(), after.encode())
    assert SHA(expected) == SHA((root / FIX).read_bytes())


# --------------------------------------------------------------------------- manufactured consumed attempt

class Fixture:
    def __init__(self, tmp_path):
        self.root = root = (tmp_path / "repo").resolve()
        self.saf_rel = "outputs/phase8/saf_surrogate/attempt_001"
        self.saf, self.old = root / self.saf_rel, root / recovery.OLD_METADATA
        named = ["outputs/p73/central.csv", "outputs/p73/draws.csv", "outputs/p73/claims.csv"]
        self.registration = {"artifact_root": self.saf_rel, "sole_score": {"named_paths": named}}
        reg_sha = write(root / recovery.SAF_REGISTRATION, self.registration)
        write(root / "docs/phase7_p73_a1_registration.json", {"outputs": {"directory": "outputs/p73"}})
        write(root / "docs/phase8_nozzle_ode_registration.json", {"outputs": {"root": "outputs/nozzle"}})
        targets = {}
        for name in recovery.SEALED_TARGETS:
            write(self.saf / name, f"sealed {name}".encode())
            targets[f"{self.saf_rel}/{name}"] = SHA((self.saf / name).read_bytes())
        for name in named:
            targets[name] = write(root / name, f"target {name}".encode())
        before, after = recovery.AUTHORIZED_EDITS[FIX][0]
        self.fix_old, self.fix_new = before.encode(), after.encode()
        sources = {"scripts/a.py": "1" * 64, FIX: SHA(self.fix_old)}
        simulator = {"name": "python-v6", "source_hashes": sources, "amendment_sha256": "3" * 64}
        self.identity = {"schema": "pc-python-v6-v1", "git_head": recovery.SOURCE_COMMIT, "registration_sha256": reg_sha,
                         "simulator": simulator, "simulator_identity_sha256": recovery.fingerprint(simulator), "source_hashes": sources}
        write(self.saf / "generation_terminal.json", {"simulator_identity_sha256": self.identity["simulator_identity_sha256"]})
        write(self.saf / "product.json", {"simulator_identity_sha256": self.identity["simulator_identity_sha256"]})
        write(self.saf / "validation.json", {"members": []})
        selection = write(self.saf / "selection.json", {"selection": {"Mphys": {"N": 4096}}})
        predictions = {f"sealed_predictions/p{i}.npz": write(self.saf / f"sealed_predictions/p{i}.npz", f"p{i}".encode()) for i in range(72)}
        weights = {f"models/w{i}.npz": write(self.saf / f"models/w{i}.npz", f"w{i}".encode()) for i in range(24)}
        inputs = {"splits/test.json": write(self.saf / "splits/test.json", [1])}
        freeze = write(self.saf / "predictions_freeze.json", {"registration_sha256": reg_sha, "selection_sha256": selection,
            "validation_sha256": SHA((self.saf / "validation.json").read_bytes()),
            "predictions": predictions, "weights": weights, "inputs": inputs})
        reservation = write(self.saf / "score_reservation.json", {"schema_version": 1, "registration_sha256": reg_sha,
            "identity": self.identity, "binary_sha256": None, "predictions_freeze_sha256": freeze,
            "selection_sha256": selection, "targets": targets})
        self.pins = {"score_reservation.json": reservation, "predictions_freeze.json": freeze, "selection.json": selection,
                     "test_metrics.json": write(self.saf / "test_metrics.json", b'{"a": 1}\n'),
                     "ranking_metrics.json": write(self.saf / "ranking_metrics.json", b'{"b": 2}\n')}
        config = {"backend": "torch", "device": "cpu", "workers": 10}
        write(self.old / "config.json", config)
        self.config_sha = recovery.fingerprint(config)
        write(self.old / "scientific_sources.json", sources)
        write(self.old / "parity20.json", {"status": "PASS"})
        artifact = {f"{self.saf_rel}/selection.json": selection}
        for letter, seconds in zip("abcde", (1.0, 2.0, 3.0, 4.0, 5.0)):
            write(self.old / f"checkpoints/{letter}.json", {"status": "COMPLETE", "config_sha256": self.config_sha,
                  "artifacts": artifact, "elapsed_seconds": seconds, "scientific_verdict": "PASS"})
            write(self.old / f"checkpoints/{letter}.started.json", {"stage": letter})
        write(self.old / "checkpoints/f.started.json", {"stage": "f", "started_utc": "2026-10-05T17:58:02+00:00"})
        for letter in "de":
            write(self.old / f"runs/{letter}/terminal.json", {"identity": self.identity})
            write(self.old / f"runs/{letter}/reservation.json", {"identity": self.identity})
            write(self.old / f"runs/{letter}/released_lease.json", {"identity": self.identity})
        write(self.old / "runs/f/reservation.json", {"identity": self.identity})
        write(self.old / "runs/f/released_lease.json", {"identity": self.identity})
        write(self.old / "runs/f/terminal.json", {"status": "ERROR", "ended_utc": "2026-10-05T17:59:39+00:00",
              "errors": ["TypeError: Object of type bool is not JSON serializable"]})
        write(self.old / "terminals/summary.json", {"status": "ERROR"})
        write(root / "outputs/p73/marker.txt", b"p73")

    def layout(self):
        return recovery.Layout(self.root)

    def new_sources(self):
        return recovery.read_json(self.old / "scientific_sources.json") | {FIX: SHA(self.fix_new), recovery.SELF_PATH: "c" * 64}

    def old_bytes(self, name):
        return self.fix_old

    def verify(self):
        return recovery.verify_original(self.layout(), pins=self.pins)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "active_owner_processes", lambda *a, **k: [])
    return Fixture(tmp_path)


def test_fixture_verifies_without_writing_anything(fx):
    before = snapshot(fx.root)
    original = fx.verify()
    assert snapshot(fx.root) == before and not fx.layout().new.exists()
    assert original["frozen_predictions"] == 72 and original["frozen_weights"] == 24
    assert original["original_f_attempt_seconds"] == pytest.approx(97.0)


@pytest.mark.parametrize("mutate,message", [
    (lambda f: write(f.root / "outputs/p73/draws.csv", b"changed"), "Reservation target changed"),
    (lambda f: write(f.saf / "sealed/named_central_species.npz", b"changed"), "Reservation target changed"),
    (lambda f: write(f.saf / "test_metrics.json", b'{"a": 9}\n'), "Pinned artifact changed"),
    (lambda f: write(f.saf / "sealed_predictions/p3.npz", b"x"), "Artifact drift"),
    (lambda f: write(f.saf / "named_metrics.json", b"{}"), "Unexpected scientific output"),
    (lambda f: write(f.old / "owner.json", {}), "owner record"),
    (lambda f: (f.layout().new / "x").parent.mkdir(parents=True), "recovery is one-shot"),
    (lambda f: write(f.old / "checkpoints/f.json", {}), "crashed state"),
    (lambda f: write(f.old / "checkpoints/g.started.json", {}), "already started"),
    (lambda f: write(f.old / "runs/e/terminal.json", {"identity": {**f.identity, "git_head": "0" * 40}}), "producer identity differs"),
    (lambda f: write(f.saf / "product.json", {"simulator_identity_sha256": "0" * 64}), "another simulator identity"),
    (lambda f: write(f.old / "scientific_sources.json", {"scripts/a.py": "9" * 64}), "source snapshot"),
])
def test_integrity_drift_is_rejected(fx, mutate, message):
    mutate(fx)
    with pytest.raises((recovery.RecoveryError, ValueError, RuntimeError), match=message):
        fx.verify()


def test_changed_old_checkpoint_artifact_is_rejected(fx):
    write(fx.saf / "selection.json", {"selection": {"Mphys": {"N": 1}}})
    with pytest.raises(RuntimeError):
        fx.verify()


def test_symlinked_consumed_file_is_rejected(fx):
    target = fx.saf / "splits/test.json"
    link = fx.saf / "splits/link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(recovery.RecoveryError, match="Symlink"):
        fx.verify()


def test_active_owner_detection_is_argv_based(tmp_path):
    proc = tmp_path / "proc"

    def process(pid, *argv):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")

    process(11, "/usr/bin/python3", "scripts/pc_pipeline.py", "--run")
    process(12, "bash", "-lc", "python scripts/pc_pipeline.py --run")
    process(13, "python", "-u", "scripts/phase8/pc_recover_f_to_i.py", "--run")
    process(14, "python", "unrelated.py")
    process(os.getpid(), "python", "scripts/pc_pipeline.py")
    assert sorted(recovery.active_owner_processes(proc)) == [11, 13]


# --------------------------------------------------------------------------- source allowlist

def source_case():
    old = {"scripts/a.py": "a" * 64, FIX: SHA(b"old fix\n")}
    edits = {FIX: ((" fix\n", " fixed\n"),)}
    new = {**old, FIX: SHA(b"old fixed\n"), recovery.SELF_PATH: "c" * 64}
    return old, new, edits, (lambda name: b"old fix\n")


def test_exact_authorized_source_change_is_accepted_and_mapped():
    old, new, edits, blob = source_case()
    bridge = recovery.verify_sources(old, new, old_bytes=blob, edits=edits)
    assert bridge["changed"] == [FIX] and bridge["added"] == [recovery.SELF_PATH] and bridge["removed"] == []
    assert bridge["mapping"][FIX] == {"original_sha256": old[FIX], "current_sha256": new[FIX]}


@pytest.mark.parametrize("mutate,message", [
    (lambda old, new: new.update({"scripts/a.py": "b" * 64}), "Unauthorized source drift"),
    (lambda old, new: new.update({"scripts/extra.py": "d" * 64}), "Source additions"),
    (lambda old, new: new.pop(recovery.SELF_PATH), "Source additions"),
    (lambda old, new: new.pop("scripts/a.py"), "removed"),
    (lambda old, new: new.update({FIX: SHA(b"old fixed and more\n")}), "exactly the authorized"),
    (lambda old, new: old.update({FIX: SHA(b"other\n")}) or new.update({FIX: SHA(b"x")}), "recorded snapshot"),
])
def test_source_drift_beyond_the_reviewed_fix_is_rejected(mutate, message):
    old, new, edits, blob = source_case()
    mutate(old, new)
    with pytest.raises(recovery.RecoveryError, match=message):
        recovery.verify_sources(old, new, old_bytes=blob, edits=edits)


def test_git_scope_rejects_unauthorized_changes_and_reports_run_blockers():
    def git_for(changed, untracked="", branch=recovery.BRANCH, dirty=""):
        def git(root, *args):
            if args[:2] == ("cat-file", "-t"):
                return "commit"
            return {"rev-parse": "h" * 40, "merge-base": "", "diff": "\n".join(changed) if "--cached" not in args else "",
                    "ls-files": untracked, "branch": branch, "status": dirty}[args[0]]
        return git
    ok = recovery.verify_git(".", git=git_for([FIX, "docs/plan.md"], untracked=recovery.SELF_PATH + "\noutputs/x"))
    assert ok["run_blockers"] == []
    blocked = recovery.verify_git(".", git=git_for([FIX], branch="main", dirty=" M x"))
    assert len(blocked["run_blockers"]) == 2
    for changed in (["simulation/runtime.py"], [FIX, "data/creck_c1c16_full.yaml"]):
        with pytest.raises(recovery.RecoveryError, match="outside the authorized"):
            recovery.verify_git(".", git=git_for(changed))
    with pytest.raises(recovery.RecoveryError, match="outside the authorized"):
        recovery.verify_git(".", git=git_for([FIX], untracked="models/new.pt"))


# --------------------------------------------------------------------------- exact score resume

def test_existing_bytes_are_compared_new_files_wait_for_all_comparisons(tmp_path):
    saf = (tmp_path / "saf").resolve()
    existing = saf / "test_metrics.json"
    write(existing, {"x": 1})
    original = existing.read_bytes()
    resume = recovery.ExactResume(saf)
    with resume.patched():
        assert score.write_once(existing, {"x": 1}) == SHA(original)  # identical regeneration
        assert score.write_once(saf / "named_metrics.json", {"y": np.bool_(True)})
        saf_run.csv_once(saf / "per_seed_metrics.csv", [{"a": 1}])
    assert not (saf / "named_metrics.json").exists() and not (saf / "per_seed_metrics.csv").exists()
    outcome = resume.commit()
    assert outcome == {"byte_identical": ["test_metrics.json"], "created": ["named_metrics.json", "per_seed_metrics.csv"]}
    assert existing.read_bytes() == original and b"true" in (saf / "named_metrics.json").read_bytes()


def test_regenerated_byte_mismatch_is_rejected_before_any_mutation(tmp_path):
    saf = (tmp_path / "saf").resolve()
    write(saf / "test_metrics.json", {"x": 1})
    write(saf / "score_reservation.json", {"r": 1})
    before = snapshot(saf)
    resume = recovery.ExactResume(saf)
    with resume.patched():
        score.write_once(saf / "named_metrics.json", {"y": 1})
        with pytest.raises(recovery.RecoveryError, match="differ"):
            score.write_once(saf / "test_metrics.json", {"x": 2})
        with pytest.raises(recovery.RecoveryError, match="differ"):  # reservation reuse: any new identity is refused
            score.write_once(saf / "score_reservation.json", {"r": 2})
        with pytest.raises(recovery.RecoveryError, match="written twice"):
            score.write_once(saf / "named_metrics.json", {"y": 1})
        with pytest.raises(recovery.RecoveryError, match="escapes"):
            score.write_once(tmp_path / "elsewhere.json", {})
    assert snapshot(saf) == before  # nothing was created or replaced


def test_writers_are_restored_and_commit_never_overwrites(tmp_path):
    saf = (tmp_path / "saf").resolve()
    originals = (score.write_once, saf_run.write_once)
    resume = recovery.ExactResume(saf)
    with resume.patched():
        score.write_once(saf / "late.json", {"a": 1})
    assert (score.write_once, saf_run.write_once) == originals
    write(saf / "late.json", b"appeared meanwhile")
    with pytest.raises(FileExistsError):
        resume.commit()
    assert (saf / "late.json").read_bytes() == b"appeared meanwhile"


def test_csv_resume_matches_and_rejects(tmp_path):
    saf = (tmp_path / "saf").resolve()
    with recovery.ExactResume(saf).patched() as first:
        saf_run.csv_once(saf / "p.csv", [{"a": 1, "b": 2}])
    first.commit()
    before = snapshot(saf)
    with recovery.ExactResume(saf).patched() as same:
        saf_run.csv_once(saf / "p.csv", [{"a": 1, "b": 2}])
    assert same.compared
    with recovery.ExactResume(saf).patched():
        with pytest.raises(recovery.RecoveryError, match="differ"):
            saf_run.csv_once(saf / "p.csv", [{"a": 1, "b": 3}])
    assert snapshot(saf) == before


def test_score_identity_shim_reproduces_the_original_reservation_bytes(fx):
    original = fx.verify()
    bound = recovery.OriginalIdentity(original["identity"], original["reservation"]["binary_sha256"])
    regenerated = registration.json_bytes({**{k: v for k, v in original["reservation"].items() if k != "identity"},
                                           "identity": bound.identity, "binary_sha256": bound.binary_sha256})
    assert regenerated == (fx.saf / "score_reservation.json").read_bytes()
    assert SHA(regenerated) == fx.pins["score_reservation.json"]


# --------------------------------------------------------------------------- source bridge keeps the checks

def test_command_spec_bridge_maps_only_the_authorized_file_and_restores():
    bridge = {"changed": [FIX], "original_sources_sha256": "o", "current_sources_sha256": "c"}
    original = saf_run.command_spec
    saf_run.command_spec = lambda *a, **k: {"source_hashes": {FIX: "old", "scripts/other.py": "recorded"}}
    wrapped_base = saf_run.command_spec
    try:
        with recovery.bridged_command_specs(bridge, {FIX: "current", "scripts/other.py": "drifted"}):
            spec = saf_run.command_spec()
        assert spec["source_hashes"] == {FIX: "current", "scripts/other.py": "recorded"}  # others still checked by children
        assert spec["operational_source_bridge"]["changed"] == [FIX]
        assert saf_run.command_spec is wrapped_base
    finally:
        saf_run.command_spec = original


def stage_producer(folder, identity):
    write(folder / "reservation.json", {"identity": identity})
    terminal = write(folder / "terminal.json", {"identity": identity, "status": "COMPLETE", "exit_code": 0,
        "reservation_sha256": SHA((folder / "reservation.json").read_bytes()), "artifact_hashes": {}})
    write(folder / "released_lease.json", {"identity": identity, "state": "RELEASED", "terminal_sha256": terminal})


def test_original_data_view_consumes_unchanged_proofs_but_not_tampered_or_foreign_ones(tmp_path):
    from scripts.phase8 import python_pc
    simulator = {"name": "python-v6", "source_hashes": {"a": "1"}, "amendment_sha256": "x"}
    identity = {"schema": "pc-python-v6-v1", "simulator": simulator, "simulator_identity_sha256": recovery.fingerprint(simulator)}
    stage_producer(tmp_path / "d", identity)
    live = SimpleNamespace(root=tmp_path, simulator_identity_sha256="live-new-sha", pc_lease_path=tmp_path / "x.json")
    view = recovery.OriginalDataView(live, simulator, identity["simulator_identity_sha256"])
    assert view.root == tmp_path and view.simulator_identity == simulator
    assert python_pc.verify_stage_producer(tmp_path, tmp_path / "d", expected_simulator=view.simulator_identity_sha256)["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="another Python simulator identity"):
        python_pc.verify_stage_producer(tmp_path, tmp_path / "d", expected_simulator=live.simulator_identity_sha256)
    write(tmp_path / "d/terminal.json", {"identity": identity, "status": "COMPLETE", "exit_code": 0,
          "reservation_sha256": "0" * 64, "artifact_hashes": {}})
    with pytest.raises(ValueError, match="release bytes changed"):
        python_pc.verify_stage_producer(tmp_path, tmp_path / "d", expected_simulator=view.simulator_identity_sha256)


# --------------------------------------------------------------------------- dry-run and one-attempt behavior

class FakeWorkflow:
    seen = []
    fail_at = None

    def __init__(self, root, metadata, *, original, bridge, current_sources, workers, backend, device):
        self.metadata, self.workers = Path(metadata), workers

    def _stage(self, letter):
        assert (self.metadata / "recovery/consumed.json").is_file()  # marker precedes all work
        FakeWorkflow.seen.append(letter)
        if letter == FakeWorkflow.fail_at:
            raise RuntimeError("boom " + letter)
        return {"status": "COMPLETE", "execution_complete": True, "scientific_verdict": "PASS", "artifacts": {}}

    a = lambda self: self._stage("a")
    f = lambda self: self._stage("f")
    g = lambda self: self._stage("g")
    h = lambda self: self._stage("h")


@contextlib.contextmanager
def fake_inhibitor():
    yield SimpleNamespace(metadata={"fake": True})


def clean_git(root, *args):
    return {"cat-file": "commit", "rev-parse": "h" * 40, "merge-base": "", "diff": "", "ls-files": "",
            "branch": recovery.BRANCH, "status": ""}[args[0]]


def recover(fx, monkeypatch, **kwargs):
    for key in recovery.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CATJET_SIMULATOR_BACKEND", "python")
    return recovery.run_recovery(fx.root, workflow_class=FakeWorkflow, pins=fx.pins, git=clean_git, sleep_inhibitor=fake_inhibitor,
        publish=lambda: {"status": "COMPLETE", "execution_complete": True, "branch": "pc-run-x", "artifacts": {}},
        new_sources=fx.new_sources(), old_bytes=fx.old_bytes, **kwargs)


def test_dry_run_is_the_default_writes_nothing_and_never_runs_science(fx, monkeypatch, capsys):
    before = snapshot(fx.root)
    layout, original = fx.layout(), fx.verify()
    bridge = recovery.verify_sources(original["old_sources"], fx.new_sources(), old_bytes=fx.old_bytes)
    info = {"head": "h", "changed_since_source": [], "run_blockers": ["not on branch"]}
    monkeypatch.setattr(recovery, "verify_all", lambda root, **k: (layout, original, bridge, info, {}))
    monkeypatch.setattr(recovery, "run_recovery", lambda *a, **k: pytest.fail("dry-run must not execute"))
    assert recovery.main([]) == 0 and recovery.main(["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert '"status": "VERIFIED"' in printed and '"run_ready": false' in printed
    assert "ff_MAE" not in printed and "kendall" not in printed  # no score content is read or printed
    assert snapshot(fx.root) == before and not layout.new.exists()
    with pytest.raises(SystemExit):
        recovery.main(["--dry-run", "--run"])


def test_run_arms_marker_preserves_originals_and_records_the_bridge(fx, monkeypatch):
    FakeWorkflow.seen, FakeWorkflow.fail_at = [], None
    old_before, saf_before = snapshot(fx.old), snapshot(fx.saf)
    summary = recover(fx, monkeypatch)
    assert summary["status"] == "COMPLETE" and FakeWorkflow.seen == list("afgh")
    new = fx.layout().new
    assert snapshot(fx.old) == old_before and all(snapshot(fx.saf)[k] == v for k, v in saf_before.items())
    for letter in "bcde":  # carried a-e receipts keep their original bytes, hence the original config SHA
        assert (new / f"checkpoints/{letter}.json").read_bytes() == (fx.old / f"checkpoints/{letter}.json").read_bytes()
    assert (new / "runs/d/terminal.json").read_bytes() == (fx.old / "runs/d/terminal.json").read_bytes()
    record = recovery.read_json(new / "recovery/recovery_record.json")
    assert record["original_config_sha256"] == fx.config_sha != record["new_config_sha256"]
    assert record["source_mapping"]["added"] == [recovery.SELF_PATH] and record["source_mapping"]["changed"] == [FIX]
    f_receipt = recovery.read_json(new / "checkpoints/f.json")  # f is charged the consumed attempt plus the resume
    assert f_receipt["original_attempt_elapsed_seconds"] == pytest.approx(97.0)
    assert f_receipt["elapsed_seconds"] == pytest.approx(f_receipt["recovery_elapsed_seconds"] + 97.0)
    assert not (new / "owner.json").exists() and list((new / "terminals").glob("*.json"))


def test_second_run_refuses_after_success_failure_or_interruption(fx, monkeypatch):
    FakeWorkflow.seen, FakeWorkflow.fail_at = [], "f"
    failed = recover(fx, monkeypatch)
    assert failed["status"] == "ERROR" and failed["stopped_at"] == "f" and "boom f" in failed["traceback"]
    new = fx.layout().new
    assert (new / "recovery/consumed.json").is_file() and (new / "checkpoints/a.json").is_file()
    kept = snapshot(new)
    FakeWorkflow.seen, FakeWorkflow.fail_at = [], None
    with pytest.raises(recovery.RecoveryError, match="one-shot"):
        recover(fx, monkeypatch)
    assert FakeWorkflow.seen == [] and set(snapshot(new)) == set(kept)  # evidence kept, nothing re-run


def test_second_run_refuses_even_if_only_the_marker_exists(fx, monkeypatch):
    write(fx.layout().new / "recovery/consumed.json", {"pid": 1})
    FakeWorkflow.seen = []
    with pytest.raises(recovery.RecoveryError, match="one-shot"):
        recover(fx, monkeypatch)
    assert FakeWorkflow.seen == []


def test_run_refuses_unverified_or_uncommitted_state_before_creating_anything(fx, monkeypatch):
    for key in recovery.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    dirty = lambda root, *args: " M x" if args[0] == "status" else clean_git(root, *args)
    with pytest.raises(recovery.RecoveryError, match="Run blocked"):
        recovery.run_recovery(fx.root, workflow_class=FakeWorkflow, pins=fx.pins, git=dirty,
                              new_sources=fx.new_sources(), old_bytes=fx.old_bytes)
    assert not fx.layout().new.exists()
    write(fx.root / "outputs/p73/draws.csv", b"drifted target")
    with pytest.raises(recovery.RecoveryError, match="Reservation target changed"):
        recover(fx, monkeypatch)
    assert not fx.layout().new.exists()
