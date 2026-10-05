"""Real Python-v6 identity and portable owned stage runs; no compiled-core claim."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .pc_runtime import PCGateError, read_json, sha256_file, utc, write_once
from simulation.runtime import host_identity, liveness, process_birth, require_ac

AMENDMENT = "docs/phase8_backend_pc_amendment.json"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hashes(root):
    root = Path(root).resolve()
    files = {root / "integrated_engine.py", root / "requirements.txt", root / AMENDMENT}
    for folder in ("simulation", "scripts"):
        files.update((root / folder).rglob("*.py"))
    files.update((root / "docs").glob("*registration*.json"))
    mapping_path = root / "outputs/archive/pre_phase6/MAPPING.json"
    mapping = read_json(mapping_path) if mapping_path.is_file() else {}
    pinned = {}
    for name in ("outputs/logs/static_thrust_accounting_protected_sha256.json",
                 "outputs/phase7/protected_sha256_phase7.json", "outputs/phase8/protected_sha256_phase8.json"):
        for old, expected in read_json(root/name).items():
            actual = mapping.get(old,old)
            if sha256_file(root/actual) != expected:
                raise PCGateError("Frozen scientific dependency changed: "+old)
            pinned[actual] = expected
    for name in ("data/creck_c1c16_full.yaml", "data/fuel_properties_v7.yaml", "data/corsia_lca_values.yaml",
                 "data/icao_engine_data.csv", "outputs/phase7/calibration_v6.json",
                 "outputs/phase7/calibration_v6_rows.csv", "outputs/phase6/split_p61.json",
                 "outputs/phase7/identifiability_profile_v6.json", "outputs/phase7/holdout_icao_validation_v6.json"):
        files.add(root / name)
    if any(not p.is_file() for p in files):
        raise PCGateError("Python-v6 scientific source/dependency is missing")
    return {**pinned, **{p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}}


class PythonContext:
    execution_profile = "pc"
    simulator_backend = "python"
    binary_path = None
    binary_sha256 = None

    def __init__(self, root, registration, *, sources, metadata_dir, parity=None, workers=10,
                 expected_consumer_identity=None, require_g0=True, **_):
        self.root = Path(root).resolve()
        self.registration = str(registration)
        if require_g0 and (not parity or parity.get("status") != "PASS"):
            raise PCGateError("The 20-row frozen Python-v6 parity PASS is required")
        self.workers = int(workers)
        self.pc_lease_path = Path(metadata_dir) / "stage_owner.json"
        self.simulator_identity = {"name": "python-v6", "source_hashes": dict(sources),
            "amendment_sha256": sha256_file(self.root / AMENDMENT)}
        self.simulator_identity_sha256 = fingerprint(self.simulator_identity)
        self.identity = {"schema": "pc-python-v6-v1", "execution_profile": "pc",
            "registration_sha256": sha256_file(self.root / self.registration),
            "simulator": self.simulator_identity, "simulator_identity_sha256": self.simulator_identity_sha256,
            "source_hashes": dict(sources), "host": host_identity(), "platform": platform.platform(),
            "python": sys.version, "workers": self.workers,
            "parity": copy.deepcopy(parity), "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()}
        self._captured_identity = copy.deepcopy(self.identity)
        self.original_context = {"schema": "pc-python-v6-parity", "parity": copy.deepcopy(parity)}
        self._expected_files = copy.deepcopy((expected_consumer_identity or {}).get("files", {}))
        self._owned = None
        self.assert_current()

    def assert_light_current(self):
        if self.identity != self._captured_identity:
            raise PCGateError("Captured Python-v6 identity changed")
        if sha256_file(self.root / self.registration) != self.identity["registration_sha256"]:
            raise PCGateError("Scientific registration drift")
        if sha256_file(self.root / AMENDMENT) != self.simulator_identity["amendment_sha256"]:
            raise PCGateError("PC amendment drift")

    def assert_current(self):
        self.assert_light_current()
        for name, expected in self.identity["source_hashes"].items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or sha256_file(path) != expected:
                raise PCGateError(f"Python-v6 source/dependency drift: {name}")
        for name, entry in self._expected_files.items():
            if sha256_file(self.root / name) != entry["sha256"]:
                raise PCGateError(f"Consumer source drift: {name}")
        return self.identity

    def require_idle_ac(self):
        self.assert_current()
        return require_ac()

    def acquire_run(self, output_dir, registration_sha256, identity=None):
        return PythonRun(self, output_dir, registration_sha256, identity)


class PythonRun:
    def __init__(self, context, output_dir, registration_sha256, identity=None):
        context.require_idle_ac()
        self.context, self.identity = context, copy.deepcopy(context.identity)
        if registration_sha256 != self.identity["registration_sha256"] or (identity is not None and identity != self.identity):
            raise PCGateError("Reservation identity differs from Python context")
        self.out = Path(output_dir)
        if not self.out.is_absolute():
            self.out = context.root / self.out
        self.out = self.out.resolve()
        if not self.out.is_relative_to(context.root / "outputs") or self.out.exists():
            raise PCGateError("Stage reservation must use a fresh output directory")
        self.lease_path = context.pc_lease_path
        self.owner_birth, self._released = process_birth(os.getpid()), False
        if not self.owner_birth or self.owner_birth == "<dead>":
            raise PCGateError("Owner process birth is unreadable")
        lease = {"owner_pid": os.getpid(), "owner_birth": self.owner_birth, "host": host_identity(),
                 "identity": self.identity, "argv": list(sys.argv), "state": "STARTING", "children": [],
                 "output_dir": str(self.out.relative_to(context.root)), "created_utc": utc()}
        write_once(self.lease_path, lease)
        try:
            self.out.mkdir(parents=True, exist_ok=False)
            write_once(self.out / "reservation.json", lease)
            self.reservation_sha256 = sha256_file(self.out / "reservation.json")
            lease["reservation_sha256"] = self.reservation_sha256
            self._write_lease(lease)
            context._owned = self.reservation_sha256
            self.assert_current()
        except BaseException:
            self.lease_path.unlink(missing_ok=True)
            raise

    def _write_lease(self, lease):
        temporary = self.lease_path.with_suffix(".tmp")
        with temporary.open("x") as stream:
            json.dump(lease, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, self.lease_path)

    def _assert_owned(self):
        if self._released:
            raise PCGateError("Released stage cannot mutate another reservation")
        lease = read_json(self.lease_path)
        if (lease.get("owner_pid") != os.getpid() or lease.get("owner_birth") != self.owner_birth
                or lease.get("host") != host_identity() or lease.get("reservation_sha256") != self.reservation_sha256):
            raise PCGateError("Portable host/PID ownership was lost")
        if sha256_file(self.out / "reservation.json") != self.reservation_sha256:
            raise PCGateError("Stage reservation bytes changed")
        frozen = read_json(self.out / "reservation.json")
        mutable = {"children", "state", "reservation_sha256", "terminal_sha256"}
        if any(lease.get(k) != v for k, v in frozen.items() if k not in mutable):
            raise PCGateError("Stage lease immutable fields changed")

    def assert_current(self):
        self._assert_owned()
        if self.identity != self.context._captured_identity:
            raise PCGateError("Run identity differs from captured Python-v6 context")
        self.context.require_idle_ac()

    def light_training_check(self):
        self._assert_owned()
        if self.identity != self.context._captured_identity:
            raise PCGateError("Run identity differs from captured Python-v6 context")
        self.context.assert_light_current(); require_ac()

    def record_children(self, children):
        self._assert_owned()
        if any(not isinstance(c.get("pid"), int) or not c.get("birth") for c in children):
            raise PCGateError("Child PID/birth is required")
        lease = read_json(self.lease_path)
        lease.update(children=list(children), state="RUNNING")
        self._write_lease(lease)

    def release(self, terminal):
        self._assert_owned()
        terminal = dict(terminal)
        try:
            self.assert_current()
        except (PCGateError, OSError, ValueError) as exc:
            terminal.update(status="ERROR", state="ERROR", exit_code=1, outputs_complete=False)
            terminal.setdefault("errors", []).append(str(exc))
        lease = read_json(self.lease_path)
        if terminal.get("ambiguous_child") or any(liveness(c["pid"], c["birth"], host=lease["host"])
                not in {"dead", "reused"} for c in lease["children"]):
            lease["state"] = "AMBIGUOUS_CHILD"; self._write_lease(lease)
            raise PCGateError("Child exit unproven; portable lease retained")
        terminal.update(identity=copy.deepcopy(self.context._captured_identity), reservation_sha256=self.reservation_sha256, ended_utc=utc())
        write_once(self.out / "terminal.json", terminal)
        lease.update(state="RELEASED", terminal_sha256=sha256_file(self.out / "terminal.json"))
        write_once(self.out / "released_lease.json", lease)
        self.lease_path.unlink(); self.context._owned = None; self._released = True
        return terminal
