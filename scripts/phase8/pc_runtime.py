"""Local (non-Mac) PC execution context/run for the portable screening pipeline.

``scripts.phase8.scientific_workflow_gate.Context`` hard-requires either the
registered OPERATIONS registration or a fresh Mac G0 receipt tied to a Mac
compiled core (pmset/sysctl/sw_vers, six recorded C++ workers). A rebuilt PC
core cannot satisfy that identity, and that module is read-only here: it is
never modified, monkeypatched or imitated with manufactured Mac receipts.

This module mirrors gate.Context/gate.Run's essential, observable safety
properties (one fresh lease per run, PID+birth ownership proof, idle-AC
requirement, children-liveness-before-release) with real local provenance:
the actual local core/source hashes and a freshly computed local G0 parity
result, never a claimed or inferred completed Mac main chain. It does not
read, write or depend on any historical Mac registration, lease or evidence
file, and the historical Mac gate/chain stay fully independent of it.
"""
from __future__ import annotations

import hashlib
import copy
import importlib.machinery
import importlib.util
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEAD = "<dead>"


class PCGateError(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)  # Atomic complete publication, still exclusive-create.
    finally:
        tmp.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def process_birth(pid):
    """Portable (Linux/macOS) start-time identity; same contract as ac_workflow.process_birth."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return DEAD
    except PermissionError:
        pass
    except (OSError, OverflowError, TypeError):
        return None
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True).stdout
    except OSError:
        return None
    if not out.strip():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return DEAD
        except OSError:
            pass
        return None
    return " ".join(out.split())


def liveness(pid, birth):
    if not isinstance(pid, int) or pid <= 0 or not birth or birth == DEAD:
        return "unknown"
    now = process_birth(pid)
    if now is None:
        return "unknown"
    if now == DEAD:
        return "dead"
    return "alive" if now == birth else "reused"


def read_power_linux(power_supply_dir="/sys/class/power_supply"):
    """Linux/WSL2 power read: no battery (desktop/VM) or AC-online (laptop).

    Never calls pmset/sysctl/sw_vers; those are macOS-only. Most desktops and
    every WSL2 guest expose no /sys/class/power_supply battery device - that
    absence is recorded as actual metadata, not treated as a refusal.
    """
    base = Path(power_supply_dir)
    if not base.is_dir():
        return {"battery_present": False, "on_ac": True,
                "note": "no /sys/class/power_supply; treated as mains (e.g. WSL2 guest)"}
    kinds = {}
    for entry in base.iterdir():
        type_path = entry / "type"
        if type_path.exists():
            kinds.setdefault(type_path.read_text().strip(), []).append(entry)
    batteries = kinds.get("Battery", [])
    if not batteries:
        return {"battery_present": False, "on_ac": True, "note": "no battery device; desktop/VM treated as mains"}
    # Linux distinguishes USB_C/USB_PD/USB_PD_DRP and other USB chargers
    # from plain USB. Their online property is still the external-power proof.
    mains = [entry for kind, entries in kinds.items() if kind == "Mains" or kind.startswith("USB")
             for entry in entries]
    online = any((p / "online").exists() and (p / "online").read_text().strip() == "1" for p in mains)
    return {"battery_present": True, "on_ac": online, "note": f"{len(batteries)} battery device(s) present"}


def require_idle_ac_linux():
    power = read_power_linux()
    if power["battery_present"] and not power["on_ac"]:
        raise PCGateError("Scientific work requires AC power (battery present, not on mains)")
    return power


def select_core(build_dir, module_name="catjet_core"):
    """Explicitly locate, hash and import the PC-built core before any use.

    Mirrors teacher.load_selected_core's explicit-path contract: the module is
    imported from one verified file path, never discovered via a sys.path
    search. Once cached in sys.modules it is what every later consumer
    (SAF/P7.3 helpers, and fork-inherited G0/benchmark worker processes on
    Linux) resolves to, since plain ``import <module_name>`` checks
    sys.modules first.
    """
    build_dir = Path(build_dir).resolve()
    candidates = [build_dir / f"{module_name}{s}" for s in importlib.machinery.EXTENSION_SUFFIXES
                  if (build_dir / f"{module_name}{s}").is_file()]
    if not candidates:
        raise PCGateError(f"No compiled {module_name}<ext> in {build_dir}; run cpp/build_pc.sh first")
    if len(candidates) != 1:
        raise PCGateError(f"Ambiguous compiled {module_name} selection in {build_dir}")
    path = candidates[0]
    binary_sha256 = sha256_file(path)
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise PCGateError(f"Foreign {module_name} was already imported")
        if getattr(existing, "_pc_binary_sha256", None) != binary_sha256:
            raise PCGateError(f"Loaded {module_name} does not match the selected binary bytes")
        return existing, path, binary_sha256
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PCGateError(f"Selected {module_name} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if Path(module.__file__).resolve() != path or sha256_file(path) != binary_sha256:
            raise PCGateError(f"Selected {module_name} changed during import")
        module._pc_binary_sha256 = binary_sha256
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, path, binary_sha256


def local_g0_parity(root, build_dir, out_dir, *, workers=6):
    """Fresh local G0 parity on this PC's own compiled core (never the Mac's).

    Reuses scripts.phase8.g0_parity.compare/regenerate/regenerate_ae3_design_point
    unmodified; only the core import path differs (explicit PC build directory,
    injected into sys.path before any catjet_core/simulation import - never by
    editing simulation/catjet_backend.py or scripts/phase8/v6_backend.py).
    Writes fresh tables and a verdict into a new ``out_dir`` (refuses to reuse
    or overwrite an existing one, like g0_parity.main's own --out-dir).
    """
    root, build_dir, out_dir = Path(root).resolve(), Path(build_dir).resolve(), Path(out_dir)
    if out_dir.exists():
        raise PCGateError("G0 output directory already exists; no overwrite or retry")
    _, binary_path, binary_sha256 = select_core(build_dir, "catjet_core")
    for extra in (str(build_dir), str(root), str(root / "scripts" / "optimization"), str(root / "scripts" / "phase8")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    import g0_parity as g0
    out_dir.mkdir(parents=True)
    runs = {}
    for backend in ("cpp", "python"):
        generated = g0.regenerate(backend, workers, backend, out_dir)
        runs[backend] = {"wall_s": generated["wall_s"],
                         "checks": {name: g0.compare(g0.FROZEN[name], path) for name, path in generated["paths"].items()}}
    ae3_path = g0.regenerate_ae3_design_point("cpp", out_dir)
    runs["cpp"]["checks"]["ae3"] = g0.compare(g0.FROZEN["ae3"], ae3_path)
    verdict = "PASS" if all(check["match"] for check in runs["cpp"]["checks"].values()) else "FAIL"
    if sha256_file(binary_path) != binary_sha256:
        raise PCGateError("Selected local core changed during G0 parity")
    doc = {"gate": "G0 (local PC; not the historical Mac chain)", "verdict": verdict,
           "core": {"path": str(binary_path.relative_to(root)), "sha256": binary_sha256},
           "machine": platform.platform(), "python": platform.python_version(),
           "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                       text=True, check=True).stdout.strip(),
           "backends": runs,
           "note": "local diagnostic evidence; does not extend or replace historical Mac G0 evidence"}
    write_once(out_dir / "g0_parity.json", doc)
    return doc


class PCContext:
    """Local PC provenance context: real local binary/source hashes, local G0.

    Never requires or manufactures a completed Mac main chain; deliberately
    does not reuse scientific_workflow_gate.Context's Mac-build-bound _g0().
    """

    def __init__(self, root, registration, *, binary_path, binary_sha256, source_hashes,
                 g0_record, identity_extra=None, expected_consumer_identity=None, require_g0=True):
        self.root = Path(root).resolve()
        self.execution_profile = "pc"
        self.pc_lease_path = self.root / "outputs/phase8/pc_runtime/owner_lease.json"
        self.registration = str(registration) if registration is not None else None
        self.binary_path = Path(binary_path).resolve()
        self.binary_sha256 = binary_sha256
        if sha256_file(self.binary_path) != binary_sha256:
            raise PCGateError("Selected local core hash drift")
        self.require_g0 = require_g0
        if require_g0 and not (isinstance(g0_record, dict) and g0_record.get("verdict") == "PASS"):
            raise PCGateError("Local fresh G0 parity PASS is required")
        if isinstance(g0_record, dict) and g0_record.get("core") is not None and g0_record["core"] != {
                "path": str(self.binary_path.relative_to(self.root)), "sha256": binary_sha256}:
            raise PCGateError("Local G0 used a different core")
        self._g0_record = copy.deepcopy(g0_record)
        self.original_context = {"schema": "pc-local-g0", "g0": copy.deepcopy(g0_record), "created_utc": utc()}
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, capture_output=True,
                              text=True, check=True).stdout.strip()
        self.identity = {"schema": "pc-local-identity-v1", "root": str(self.root),
            "registration_sha256": sha256_file(self.root / self.registration) if self.registration else None,
            "git_head": head, "binary_path": str(self.binary_path.relative_to(self.root)),
            "binary_sha256": binary_sha256, "source_hashes": copy.deepcopy(dict(source_hashes)),
            "platform": platform.platform()}
        extra = copy.deepcopy(identity_extra or {})
        if any(name in self.identity and value != self.identity[name] for name, value in extra.items()):
            raise PCGateError("Extra identity metadata cannot replace captured local provenance")
        self.identity.update(extra)
        self._captured_identity = copy.deepcopy(self.identity)
        self._expected_files = copy.deepcopy((expected_consumer_identity or {}).get("files", {}))
        self._owned = None
        self.assert_current()

    def assert_current(self):
        """Check the bytes captured at creation, rather than trusting stored hashes."""
        self.assert_light_current()
        try:
            if sha256_file(self.binary_path) != self.binary_sha256:
                raise PCGateError("Selected local core hash drift")
            for name, expected in self.identity["source_hashes"].items():
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root) or sha256_file(path) != expected:
                    raise PCGateError(f"Captured local source hash drift: {name}")
            for name, entry in self._expected_files.items():
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root) or sha256_file(path) != entry.get("sha256"):
                    raise PCGateError(f"Consumer source differs from frozen expectation: {name}")
        except OSError as exc:
            raise PCGateError(f"Captured local provenance is unreadable: {exc}") from exc
        return self.identity

    def assert_light_current(self):
        """Small epoch guard; full source/core validation remains a boundary check."""
        if self.identity != self._captured_identity:
            raise PCGateError("Captured local identity was changed")
        try:
            if self.registration and sha256_file(self.root / self.registration) != self.identity["registration_sha256"]:
                raise PCGateError("Local registration hash drift")
            g0_name, g0_hash = self.identity.get("g0_record_path"), self.identity.get("g0_record_sha256")
            if g0_name is not None or g0_hash is not None:
                if not g0_name or not g0_hash:
                    raise PCGateError("Local G0 evidence needs both its path and hash")
                path = (self.root / g0_name).resolve()
                if not path.is_relative_to(self.root) or sha256_file(path) != g0_hash:
                    raise PCGateError("Local G0 evidence hash drift")
                record = read_json(path)
                if record != self._g0_record or record.get("verdict") != "PASS" or record.get("core") != {
                        "path": self.identity["binary_path"], "sha256": self.binary_sha256}:
                    raise PCGateError("Local G0 evidence differs from the selected PASS/core")
        except OSError as exc:
            raise PCGateError(f"Captured local provenance is unreadable: {exc}") from exc
        return self.identity

    def require_idle_ac(self):
        self.assert_current()
        return require_idle_ac_linux()

    def acquire_run(self, output_dir, registration_sha256, identity=None):
        return PCRun(self, output_dir, registration_sha256, identity)


class PCRun:
    def __init__(self, context, output_dir, registration_sha256, identity=None):
        context.require_idle_ac()
        self.context = context
        self._released = False
        self.identity = copy.deepcopy(context.identity)
        if registration_sha256 is not None and registration_sha256 != self.identity["registration_sha256"]:
            raise PCGateError("Reservation identity differs from current context")
        if identity is not None and identity != self.identity:
            raise PCGateError("Reservation identity differs from current context")
        self.out = Path(output_dir)
        if not self.out.is_absolute():
            self.out = context.root / self.out
        self.out = self.out.resolve()
        if not self.out.is_relative_to(context.root / "outputs") or self.out.exists():
            raise PCGateError("Run output must be a fresh directory under outputs")
        birth = process_birth(os.getpid())
        if not birth or birth == DEAD:
            raise PCGateError("Owner birth unreadable")
        self.owner_birth = birth
        self.lease_path = context.pc_lease_path
        lease = {"owner_pid": os.getpid(), "owner_birth": birth, "argv": list(sys.argv),
                 "identity": self.identity, "output_dir": str(self.out.relative_to(context.root)),
                 "state": "STARTING", "children": [], "created_utc": utc()}
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
            current = read_json(self.lease_path) if self.lease_path.exists() else {}
            if current.get("owner_pid") == lease["owner_pid"] and current.get("owner_birth") == lease["owner_birth"] \
                    and current.get("output_dir") == lease["output_dir"]:
                self.lease_path.unlink(missing_ok=True)
            context._owned = None
            raise

    def _write_lease(self, doc):
        tmp = self.lease_path.with_name(f".{self.lease_path.name}.{os.getpid()}.tmp")
        with tmp.open("x") as stream:
            json.dump(doc, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.lease_path)

    def assert_current(self):
        self._assert_owned()
        self.context.require_idle_ac()
        if self.context.identity != self.identity:
            raise PCGateError("Scientific identity drift during owned PC run")

    def light_training_check(self):
        self._assert_owned()
        self.context.assert_light_current()
        if self.context.identity != self.identity:
            raise PCGateError("Scientific identity drift during owned PC run")
        require_idle_ac_linux()

    def _assert_owned(self):
        if self._released:
            raise PCGateError("Released run object cannot acquire or mutate another lease")
        lease = read_json(self.lease_path)
        if lease.get("reservation_sha256") != self.reservation_sha256 or lease.get("owner_pid") != os.getpid() \
                or lease.get("owner_birth") != self.owner_birth \
                or lease.get("output_dir") != str(self.out.relative_to(self.context.root)):
            raise PCGateError("This run no longer owns the exact reservation")
        if sha256_file(self.out / "reservation.json") != self.reservation_sha256:
            raise PCGateError("Owned reservation bytes changed")
        reservation = read_json(self.out / "reservation.json")
        mutable = {"state", "children", "reservation_sha256", "terminal_sha256"}
        if any(lease.get(k) != value for k, value in reservation.items() if k not in mutable) \
                or set(lease) - set(reservation) - mutable:
            raise PCGateError("Lease immutable identity differs from owned reservation")

    def record_children(self, children):
        self._assert_owned()
        lease = read_json(self.lease_path)
        for child in children:
            if not isinstance(child, dict) or not isinstance(child.get("pid"), int) or not child.get("birth"):
                raise PCGateError("Child ownership requires PID and birth")
        lease["children"] = list(children)
        lease["state"] = "RUNNING"
        self._write_lease(lease)

    def release(self, terminal):
        terminal = dict(terminal)
        self._assert_owned()
        try:
            self.assert_current()
        except (PCGateError, OSError, ValueError) as exc:
            terminal.update(status="ERROR", state="ERROR", exit_code=1, outputs_complete=False)
            terminal.setdefault("errors", []).append(str(exc))
            terminal.setdefault("identity_problems", []).append(str(exc))
        lease = read_json(self.lease_path)
        alive = [child for child in lease.get("children", []) if liveness(child["pid"], child["birth"]) not in {"dead", "reused"}]
        if alive or terminal.get("ambiguous_child"):
            lease["state"] = "AMBIGUOUS_CHILD"
            self._write_lease(lease)
            raise PCGateError("Child exit unproven; lease retained")
        terminal.update(identity=self.identity, reservation_sha256=self.reservation_sha256, ended_utc=utc())
        write_once(self.out / "terminal.json", terminal)
        lease["state"] = "RELEASED"
        lease["terminal_sha256"] = sha256_file(self.out / "terminal.json")
        write_once(self.out / "released_lease.json", lease)
        self.lease_path.unlink()
        self.context._owned = None
        self._released = True
        return terminal
