#!/usr/bin/env python3
"""Run the Phase 8 screening workflow on Linux/WSL2; A2 is deferred.

Help and dry-run use only the Python standard library and never create outputs.
Scientific procedures live in the existing stage implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ("preflight", "build", "focused_checks", "g0", "p73_a1", "saf",
          "nozzle", "product", "freeze")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS")
REGISTRATIONS = {
    "p73_a1": "docs/phase7_p73_a1_registration.json",
    "saf": "docs/phase8_saf_surrogate_registration.json",
    "nozzle": "docs/phase8_nozzle_ode_registration.json",
    "freeze": "docs/phase8_screening_tool_registration.json",
}
FOCUSED_TESTS = (
    "tests/test_phase7_p73_a1_cpp.py", "tests/test_phase8_saf_surrogate.py",
    "tests/test_phase8_saf_surrogate_torch.py", "tests/test_phase8_screening_product.py",
    "tests/test_phase8_freeze_screening.py", "tests/test_phase8_pc_pipeline.py",
    "tests/test_phase8_pc_runtime.py", "tests/test_phase8_pc_saf.py",
    "tests/test_phase8_pc_finish.py",
)


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    device: str = "auto"
    cpp_prefix: Path | None = None
    build_dir: Path | None = None
    metadata_dir: Path | None = None

    def paths(self):
        root = self.root.resolve()
        return (root, (self.cpp_prefix or Path(os.environ.get("CATJET_CPP_PREFIX",
                str(Path.home() / "miniforge3/envs/catjet-cpp-pc")))).resolve(),
                (self.build_dir or root / "cpp/build_pc").resolve(),
                (self.metadata_dir or root / "outputs/phase8/pc_pipeline").resolve())


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_outputs(root):
    p73, saf, nozzle = (read_json(root / REGISTRATIONS[k]) for k in ("p73_a1", "saf", "nozzle"))
    return {"p73_a1": root / p73["outputs"]["directory"],
            "saf": root / saf["artifact_root"],
            "nozzle": root / nozzle["outputs"]["root"],
            "freeze": root / read_json(root / REGISTRATIONS["freeze"])["outputs"]["root"]}


def validate_paths(config):
    root, prefix, build, metadata = config.paths()
    if (not build.is_relative_to(root / "cpp") or build == root / "cpp"
            or any(build.is_relative_to(old) for old in (root / "cpp/build", root / "cpp/build_next"))):
        raise ValueError("--build-dir must be a separate directory under cpp/, such as cpp/build_pc")
    if not metadata.is_relative_to(root / "outputs/phase8") or metadata == root / "outputs/phase8":
        raise ValueError("--metadata-dir must be a new directory under outputs/phase8/")
    for stage, path in scientific_outputs(root).items():
        if metadata == path or metadata.is_relative_to(path) or path.is_relative_to(metadata):
            raise ValueError(f"--metadata-dir overlaps the fixed {stage} scientific output")
    return root, prefix, build, metadata


def describe(config):
    root, prefix, build, metadata = validate_paths(config)
    return {"execution_profile": "pc", "platform": "Linux or Windows WSL2",
            "backend": "torch", "requested_device": config.device, "a2": "DEFERRED",
            "stages": list(STAGES), "cpp_prefix": str(prefix), "build_dir": str(build),
            "metadata_dir": str(metadata),
            "scientific_outputs": {k: str(v) for k, v in scientific_outputs(root).items()},
            "budgets": "Existing registrations; no reduced fits or score retries",
            "run_command": [sys.executable, str(root / "pc_pipeline.py"), "--run",
                            "--device", config.device, "--cpp-prefix", str(prefix),
                            "--build-dir", str(build), "--metadata-dir", str(metadata)]}


def protected_files(root):
    mapping_path = root / "outputs/archive/pre_phase6/MAPPING.json"
    mapping = read_json(mapping_path) if mapping_path.exists() else {}
    result = {}
    for name in ("outputs/logs/static_thrust_accounting_protected_sha256.json",
                 "outputs/phase7/protected_sha256_phase7.json",
                 "outputs/phase8/protected_sha256_phase8.json"):
        for old, expected in read_json(root / name).items():
            actual = mapping.get(old, old)
            if not (root / actual).is_file() or sha256(root / actual) != expected:
                raise RuntimeError(f"Protected file missing or changed: {old}")
            result[actual] = expected
    return result


def capture_sources(root):
    files = {root / "pc_pipeline.py", root / "requirements.txt", root / "cpp/build_pc.sh"}
    for folder in ("scripts", "simulation", "pinn"):
        files.update((root / folder).rglob("*.py"))
    for suffix in ("*.cpp", "*.h", "*.hpp", "CMakeLists.txt"):
        files.update(p for p in (root / "cpp").rglob(suffix)
                     if not any(part.startswith("build") for part in p.relative_to(root / "cpp").parts))
    files.update((root / "docs").glob("*registration*.json"))
    files.update((root / "docs").glob("*registration*.md"))
    return {**protected_files(root),
            **{p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}}


def preflight(config):
    root, prefix, build, metadata = validate_paths(config)
    if platform.system() != "Linux":
        raise RuntimeError("PC execution requires Linux or WSL2; use --dry-run on other hosts")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("PC execution requires Python 3.12 (see PC_SETUP.md)")
    for key in THREAD_ENV:
        if os.environ.get(key, "1") != "1":
            raise RuntimeError(f"Set {key}=1 for the registered one-thread-per-worker procedure")
        os.environ[key] = "1"
    versions = {}
    for package in ("numpy", "scipy", "pandas", "torch", "cantera", "PyYAML", "pybind11", "pytest", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    if versions["torch"].split("+")[0] != "2.9.1" or versions["cantera"] != "3.2.0":
        raise RuntimeError("Install pinned torch 2.9.1 and Cantera 3.2.0 from PC_SETUP.md")
    for executable in ("cmake", "ninja"):
        if not (prefix / "bin" / executable).is_file():
            raise RuntimeError(f"Missing {prefix}/bin/{executable}; see PC_SETUP.md")
    if not (root / ".venv/bin/python").is_file():
        raise RuntimeError("Create the repository .venv using Python 3.12")
    if metadata.exists():
        raise FileExistsError(f"Metadata attempt already exists: {metadata}")
    for stage, path in scientific_outputs(root).items():
        if path.exists():
            raise FileExistsError(f"Fixed {stage} output already exists: {path}; preserve this consumed attempt")
    if (root / "outputs/phase8/pc_runtime/owner_lease.json").exists():
        raise RuntimeError("A PC run already holds or retains its owner lease")
    from scripts.phase8.saf_surrogate.train_torch import resolve_device
    from scripts.phase8.pc_runtime import require_idle_ac_linux
    import torch
    torch.set_num_threads(1)
    device = resolve_device(config.device)
    return {"state": "PASS", "versions": versions, "device": device,
            "power": require_idle_ac_linux(), "protected_file_count": len(protected_files(root)),
            "platform": platform.platform(), "python": sys.version,
            "cpp_prefix": str(prefix), "build_dir": str(build)}


def checked_command(argv, log, *, cwd, env=None):
    with Path(log).open("xb") as stream:
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log}")
    return {"state": "PASS", "argv": [str(x) for x in argv], "exit_code": result.returncode,
            "log": str(log), "log_sha256": sha256(log)}


def stage_result(value):
    if not isinstance(value, dict):
        raise TypeError("A pipeline stage must return its measured result dictionary")
    state = str(value.get("state", value.get("status", value.get("verdict", "ERROR")))).upper()
    if state == "COMPLETE":
        state = "PASS"
    if state not in {"PASS", "FAIL", "INCOMPLETE", "ERROR", "BLOCKED"}:
        raise ValueError(f"Unknown stage result {state}")
    return {**value, "state": state}


class Workflow:
    def __init__(self, config):
        self.config = config
        self.root, self.prefix, self.build, self.metadata = config.paths()
        self.device = config.device
        self.core_path = self.core_sha = self.sources = self.g0_record = None

    def context_factory(self, root, registration, **kwargs):
        from scripts.phase8.pc_runtime import PCContext
        if Path(root).resolve() != self.root:
            raise ValueError("Consumer root differs from the selected PC checkout")
        return PCContext(root, registration, binary_path=self.core_path, binary_sha256=self.core_sha,
            source_hashes=self.sources, g0_record=self.g0_record,
            identity_extra={"execution_profile": "pc", "g0_record_path":
                            str((self.metadata / "g0/g0_parity.json").relative_to(self.root)),
                            "g0_record_sha256": sha256(self.metadata / "g0/g0_parity.json")}, **kwargs)

    def preflight(self):
        result = preflight(self.config)
        self.device = result["device"]
        return result

    def build_core(self):
        env = dict(os.environ, CATJET_CPP_PREFIX=str(self.prefix))
        result = checked_command(["bash", str(self.root / "cpp/build_pc.sh"),
            "--build-dir", str(self.build)], self.metadata / "build.log", cwd=self.root, env=env)
        checked_command([str(self.prefix / "bin/ctest"), "--test-dir", str(self.build),
            "--output-on-failure"], self.metadata / "cpp_tests.log", cwd=self.root, env=env)
        from scripts.phase8.pc_runtime import select_core
        _, self.core_path, self.core_sha = select_core(self.build)
        self.sources = capture_sources(self.root)
        return {**result, "binary_path": str(self.core_path), "binary_sha256": self.core_sha}

    def focused_checks(self):
        return checked_command([sys.executable, "-m", "pytest", *FOCUSED_TESTS, "-v"],
                               self.metadata / "focused_checks.log", cwd=self.root)

    def g0(self):
        from scripts.phase8.pc_runtime import PCContext, local_g0_parity, sha256_file
        context = PCContext(self.root, None, binary_path=self.core_path, binary_sha256=self.core_sha,
                            source_hashes=self.sources, g0_record=None, require_g0=False)
        owned = context.acquire_run(self.metadata / "g0_owner", None)
        owned.record_children([])
        try:
            result = local_g0_parity(self.root, self.build, self.metadata / "g0", workers=6)
            owned.assert_current()
            owned.release({"state": result["verdict"], "status": result["verdict"],
                "execution_complete": True, "artifact_hashes": {
                    str(p.relative_to(self.root)): sha256_file(p)
                    for p in (self.metadata / "g0").rglob("*") if p.is_file()}})
        except BaseException as error:
            if not owned._released:
                owned.release({"state": "ERROR", "status": "ERROR", "reason": str(error)})
            raise
        self.g0_record = result
        return {**result, "state": result["verdict"], "evidence_path":
                str((self.metadata / "g0/g0_parity.json").relative_to(self.root))}

    def p73_a1(self):
        from scripts.phase8.p73_a1_cpp import execute
        return execute(self.root, self.root / REGISTRATIONS["p73_a1"],
                       consumer_root=self.root, gate_factory=self.context_factory)

    def saf(self):
        from scripts.phase8.pc_saf import execute
        return execute(self.root, self.context_factory, device=self.device, backend="torch")

    def nozzle(self):
        from scripts.phase8.pc_finish import run_nozzle
        return run_nozzle(self.root, self.context_factory)

    def product(self):
        from scripts.phase8.pc_finish import check_product
        return check_product(self.root, self.context_factory)

    def freeze(self):
        from scripts.phase8.pc_finish import freeze
        return freeze(self.root, self.context_factory)

    def handlers(self):
        return {name: getattr(self, "build_core" if name == "build" else name) for name in STAGES}


def execute(config, *, handlers=None):
    """Synchronous stages; retain scientific FAIL, stop execution ERROR.

    Injected handlers allow manufactured tests without scientific imports.
    A failed G0 prevents later scientific computation using an invalid core.
    """
    from scripts.phase8.pc_runtime import utc, write_once
    root, _, _, metadata = validate_paths(config)
    workflow = Workflow(config) if handlers is None else None
    handlers = workflow.handlers() if workflow is not None else handlers
    summary = {"execution_profile": "pc", "a2": "DEFERRED", "started_utc": utc(), "stages": []}
    created = False
    active_stage = "preflight"
    try:
        for name in STAGES:
            active_stage = name
            if name != "preflight" and not created:
                metadata.mkdir(parents=True, exist_ok=False)
                created = True
                write_once(metadata / "plan.json", describe(config))
            value = stage_result(handlers[name]())
            summary["stages"].append({"stage": name, **value})
            if created:
                write_once(metadata / f"{name}.json", value)
            if value["state"] in {"ERROR", "BLOCKED"}:
                raise RuntimeError(f"{name} could not complete: {value}")
            if name in {"g0", "p73_a1"} and value["state"] != "PASS":
                summary["state"] = "FAIL"
                summary["stopped_after"] = name
                break
        else:
            summary["state"] = "PASS" if all(r["state"] == "PASS" for r in summary["stages"]) else "INCOMPLETE"
    except BaseException as error:
        summary.update(state="ERROR", error=f"{type(error).__name__}: {error}",
                       stopped_at=active_stage)
    summary["finished_utc"] = utc()
    if created:
        write_once(metadata / "terminal.json", summary)
    return summary


def legacy_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print stage/path plan without imports or writes")
    mode.add_argument("--preflight", action="store_true", help="Check PC dependencies, power and fresh outputs")
    mode.add_argument("--run", action="store_true", help="Build and run the full fixed screening workflow")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpp-prefix", type=Path, help="Conda C++ dependency prefix")
    parser.add_argument("--build-dir", type=Path, help="Separate build directory under cpp/")
    parser.add_argument("--metadata-dir", type=Path, help="Fresh metadata directory under outputs/phase8/")
    args = parser.parse_args(argv)
    config = Config(device=args.device, cpp_prefix=args.cpp_prefix, build_dir=args.build_dir,
                    metadata_dir=args.metadata_dir)
    try:
        result = describe(config) if args.dry_run else preflight(config) if args.preflight else execute(config)
    except Exception as error:
        print(json.dumps({"state": "ERROR", "error": f"{type(error).__name__}: {error}"}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    state = result.get("state", "PASS")
    return 0 if state == "PASS" else 2 if state == "ERROR" else 1


def main(argv=None):
    """Compatibility command: use the resumable Python-v6 PC entry point."""
    from scripts.pc_pipeline import main as python_pc_main
    return python_pc_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
