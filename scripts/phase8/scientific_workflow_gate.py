"""Strict post-chain evidence and exclusive ownership for screening studies.

Registration: docs/phase8_screening_operations_registration.json. This module
does not mutate the original workflow, infer completion from a PID, or recover
an existing lease. Export original evidence before integrating new sources.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

OPERATIONS = "docs/phase8_screening_operations_registration.json"
AMENDMENT = "docs/phase8_screening_operations_amendment_a1.json"


class GateError(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise GateError(f"Unreadable JSON evidence: {path}") from exc
    if not isinstance(doc, dict):
        raise GateError(f"Expected JSON object: {path}")
    return doc


def write_once(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x") as f:
            json.dump(doc, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # Atomic complete publication, still exclusive-create.
    finally:
        tmp.unlink(missing_ok=True)
    return doc


def relative(root, name):
    name = str(name)
    if Path(name).is_absolute() or ".." in Path(name).parts or name in ("", "."):
        raise GateError(f"Unsafe relative evidence path: {name}")
    return Path(root) / name


def git(root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateError("Git evidence unreadable") from exc


def file_identity(path):
    path = Path(path)
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        data = os.readlink(path).encode()
        mode = "120000"
    elif stat.S_ISREG(st.st_mode):
        data = path.read_bytes()
        mode = "100755" if st.st_mode & 0o111 else "100644"
    else:
        raise GateError(f"Not a tracked regular file or symlink: {path}")
    result = {"mode": mode, "sha256": hashlib.sha256(data).hexdigest(),
              "blob": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
    if mode == "120000":
        result["target_sha256"] = digest(path)
    return result


def require_committed(root, paths):
    paths = sorted(set(map(str, paths)))
    if git(root, "status", "--porcelain", "--", *paths).strip():
        raise GateError("Required evidence or registration is uncommitted/dirty")
    _, entries = _tree(root, "HEAD", paths)
    if set(entries) != set(paths):
        raise GateError("Required evidence is absent from committed tree")
    for name, expected in entries.items():
        actual = file_identity(relative(root, name))
        if actual["mode"] != expected["mode"] or actual["blob"] != expected["blob"]:
            raise GateError(f"Required committed evidence differs: {name}")


def _ac(root):
    path = Path(root) / "scripts/phase8/ac_workflow.py"
    op = operations(root)
    if digest(path) != op["original_pins"]["scripts/phase8/ac_workflow.py"]:
        raise GateError("Original workflow validator changed")
    spec = importlib.util.spec_from_file_location("_screening_original_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operations(root):
    doc = read(Path(root) / OPERATIONS)
    if doc.get("id") != "P8-SCREENING-OPERATIONS-20261004":
        raise GateError("Foreign screening operations registration")
    return doc


def _original_validate(root, identity):
    """Use verified original validator semantics with a private proven identity.

    Only the Workflow factory differs in a private globals mapping. Neither
    the old file nor its module globals are patched. Callers must prove every
    original byte/mode/module and the original tree before reaching this call.
    """
    ac = _ac(root)
    original = ac.validate_terminal_context

    def factory(*args, **kwargs):
        kwargs["identity"] = lambda: identity
        return ac.Workflow(*args, **kwargs)

    fn = types.FunctionType(original.__code__, {**original.__globals__, "Workflow": factory},
                            original.__name__, original.__defaults__, original.__closure__)
    fn.__kwdefaults__ = original.__kwdefaults__.copy()
    try:
        return fn(Path(root), expected_identity=identity, require_idle=True)
    except ac.Blocked as exc:
        raise GateError(f"Original terminal proof refused: {exc}") from exc


def _tree(root, head, paths):
    raw = git(root, "ls-tree", "-r", head, "--", *paths)
    entries = {}
    for line in raw.splitlines():
        header, name = line.split("\t", 1)
        mode, kind, blob = header.split()
        if kind != "blob" or mode not in ("100644", "100755", "120000"):
            raise GateError("Unsupported original scientific tree entry")
        entries[name] = {"mode": mode, "blob": blob}
    return raw, entries


def export_main_context(root):
    root = Path(root).resolve()
    op = operations(root)
    ac = _ac(root)
    # This check rejects active ownership before any large source hashing.
    context = ac.validate_terminal_context(root, require_idle=True)
    identity = context["chain"]["identity"]
    if not ac.same_sources(identity, context["identity"]):
        raise GateError("Terminal and current scientific identity differ")
    main = read(root / op["paths"]["main_registration"])
    raw, tree = _tree(root, identity["git_head"], main["identity"]["tracked_paths"])
    if hashlib.sha256(raw.encode()).hexdigest() != identity["tracked_tree_sha256"]:
        raise GateError("Original launch tree hash mismatch")
    files = {}
    for name, entry in tree.items():
        actual = file_identity(relative(root, name))
        if any(actual[k] != entry[k] for k in ("mode", "blob")):
            raise GateError(f"Original source differs at export: {name}")
        files[name] = actual
    wf = ac.Workflow(root=root)
    evidence = {str(p.relative_to(root)): digest(p) for p in sorted(wf.out.rglob("*"))
                if p.is_file() and p != wf.lease_path}
    if not evidence:
        raise GateError("No raw main workflow evidence")
    normalized = _original_validate(root, identity)
    doc = {"schema": 1, "id": "P8-SCREENING-MAIN-DEPENDENCY", "created_utc": utc(),
           "operations_registration_sha256": digest(root / OPERATIONS),
           "main_registration_sha256": digest(root / op["paths"]["main_registration"]),
           "identity": identity, "original_files": files, "raw_evidence": evidence,
           "context": normalized}
    # No export based on a context that drifted while it was hashed.
    again = ac.validate_terminal_context(root, require_idle=True)
    if again != context:
        raise GateError("Original context changed during export")
    return write_once(root / op["paths"]["main_dependency"], doc)


def _declared_new_paths(root, op):
    paths = set(op["new_paths"])
    for name in _registrations(root, op):
        reg = read(root / name)
        paths.update(reg.get("new_source_paths", []))
        paths.update(reg.get("relevant_new_files", []))
        paths.update(reg.get("relevant_files", {}).get("implementation_create", []))
        contract = reg.get("implementation_contract", {})
        if contract.get("new_consumer"):
            paths.add(contract["new_consumer"])
        paths.update(reg.get("new_paths", []))
    for name in paths:
        relative(root, name)
    return paths


def _registrations(root, op):
    amendment = read(root / AMENDMENT)
    if amendment.get("id") != "P8-SCREENING-OPERATIONS-20261004-A1" \
            or amendment.get("parent_sha256") != digest(root / OPERATIONS) \
            or amendment.get("additional_registrations") != ["docs/phase8_screening_tool_registration.json"]:
        raise GateError("Unrecognized operations extension registration")
    return op["sibling_registrations"] + amendment["additional_registrations"]


def freeze_extensions(root):
    root = Path(root).resolve()
    op = operations(root)
    dep = read(root / op["paths"]["main_dependency"])
    _prove_original(root, op, dep)
    declared = _declared_new_paths(root, op)
    main = read(root / op["paths"]["main_registration"])
    roots = main["identity"]["tracked_paths"]
    science = {p for p in declared if any(p == r or p.startswith(r.rstrip("/")+"/") for r in roots)}
    _, current = _tree(root, "HEAD", main["identity"]["tracked_paths"])
    added = set(current) - set(dep["original_files"])
    if not added <= declared:
        raise GateError(f"Undeclared new scientific paths: {sorted(added-declared)}")
    missing = science - set(current)
    if missing:
        raise GateError(f"Registered new sources not integrated: {sorted(missing)}")
    if git(root, "status", "--porcelain", "--", *main["identity"]["tracked_paths"]).strip():
        raise GateError("Dirty or untracked scientific source at extension freeze")
    doc = {"schema": 1, "id": "P8-SCREENING-SOURCE-EXTENSION", "created_utc": utc(),
           "main_dependency_sha256": digest(root / op["paths"]["main_dependency"]),
           "operations_registration_sha256": digest(root / OPERATIONS),
           "operations_amendment_sha256": digest(root / AMENDMENT),
           "git_head": git(root, "rev-parse", "HEAD").strip(),
           "files": {p: file_identity(root / p) for p in sorted(added)},
           "other_declared_files": {p: file_identity(root / p) for p in sorted(declared-science)},
           "registrations": {p: digest(root / p) for p in _registrations(root, op)}}
    return write_once(root / op["paths"]["source_extension_manifest"], doc)


def _prove_original(root, op, dep):
    if dep.get("id") != "P8-SCREENING-MAIN-DEPENDENCY" \
            or dep.get("operations_registration_sha256") != digest(root / OPERATIONS) \
            or dep.get("main_registration_sha256") != digest(root / op["paths"]["main_registration"]):
        raise GateError("Foreign or changed main dependency registration")
    main = read(root / op["paths"]["main_registration"])
    if (root / main["lease_path"]).exists():
        raise GateError("Original workflow lease remains present")
    identity = dep["identity"]
    raw, tree = _tree(root, identity["git_head"], main["identity"]["tracked_paths"])
    if hashlib.sha256(raw.encode()).hexdigest() != identity["tracked_tree_sha256"] \
            or set(tree) != set(dep["original_files"]):
        raise GateError("Attested original tree differs from launch tree")
    for name, entry in tree.items():
        actual = file_identity(relative(root, name))
        if actual != dep["original_files"][name] or any(actual[k] != entry[k] for k in ("mode", "blob")):
            raise GateError(f"Original scientific source changed: {name}")
    modules = {str(p.relative_to(root)): digest(p) for p in sorted(root.glob(main["identity"]["built_modules_glob"]))}
    if modules != identity["built_modules"]:
        raise GateError("Original built module set/hash changed")
    for name, expected in dep["raw_evidence"].items():
        if digest(relative(root, name)) != expected:
            raise GateError(f"Raw main evidence changed: {name}")
    actual = _original_validate(root, identity)
    if actual != dep["context"]:
        raise GateError("Original raw terminal semantics differ from attestation")
    return actual


def _check_extensions(root, op, dep, allow_new_paths=None):
    path = root / op["paths"]["source_extension_manifest"]
    extension = read(path)
    if extension.get("id") != "P8-SCREENING-SOURCE-EXTENSION" \
            or extension.get("main_dependency_sha256") != digest(root / op["paths"]["main_dependency"]) \
            or extension.get("operations_registration_sha256") != digest(root / OPERATIONS):
        raise GateError("Foreign or changed source-extension manifest")
    allowed = _declared_new_paths(root, op)
    main = read(root / op["paths"]["main_registration"])
    _, current = _tree(root, "HEAD", main["identity"]["tracked_paths"])
    added = set(current) - set(dep["original_files"])
    if added != set(extension["files"]) or not added <= allowed:
        raise GateError("Unregistered or missing scientific source extension")
    if allow_new_paths is not None and not set(allow_new_paths) <= allowed:
        raise GateError("Caller requested unregistered new paths")
    for name, expected in extension["files"].items():
        if file_identity(relative(root, name)) != expected:
            raise GateError(f"New scientific source changed: {name}")
    for name, expected in extension.get("other_declared_files", {}).items():
        if file_identity(relative(root, name)) != expected:
            raise GateError(f"Other registered file changed: {name}")
    if extension.get("operations_amendment_sha256") != digest(root / AMENDMENT) \
            or extension["registrations"] != {p: digest(root / p) for p in _registrations(root, op)}:
        raise GateError("Sibling study registration changed")
    if git(root, "status", "--porcelain", "--", *main["identity"]["tracked_paths"]).strip():
        raise GateError("Dirty or untracked scientific sources")
    return extension


def _scientific_expected(root, registration, expected):
    if expected is None:
        return
    if not isinstance(expected, dict) or not isinstance(expected.get("files"), dict):
        raise GateError("Malformed expected consumer identity")
    if expected.get("registration_sha256") != digest(relative(root, registration)):
        raise GateError("Consumer registration differs from frozen expectation")
    for name, entry in expected["files"].items():
        if not isinstance(entry, dict) or entry.get("mode") not in {"100644", "100755", "120000"} \
                or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64 \
                or any(c not in "0123456789abcdef" for c in entry["sha256"]):
            raise GateError("Malformed expected consumer file identity")
        actual = file_identity(relative(root, name))
        if any(actual.get(k) != v for k, v in entry.items()):
            raise GateError(f"Consumer source differs from frozen expectation: {name}")


def recompare_g0(root, op, verdict, files):
    """Execute the unchanged original comparison function, without its runners."""
    import math
    import pandas as pd
    source = root / "scripts/phase8/g0_parity.py"
    if digest(source) != op["original_pins"]["scripts/phase8/g0_parity.py"]:
        raise GateError("Original G0 comparison source changed")
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compare")
    scope = {"Path": Path, "pd": pd, "math": math, "ROOT": root, "RTOL": 1e-9, "ATOL": 1e-12}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    frozen = {"cal": "outputs/phase7/calibration_v6_rows.csv",
              "hold": "outputs/phase7/holdout_icao_validation_v6.csv",
              "hold_summary": "outputs/phase7/holdout_icao_validation_summary_v6.csv",
              "ae3": "outputs/design_point_summary_v5.csv"}
    for backend, expected in (("cpp", set(frozen)), ("python", set(frozen)-{"ae3"})):
        checks = verdict.get("backends", {}).get(backend, {}).get("checks", {})
        if set(checks) != expected:
            raise GateError(f"Incomplete raw G0 {backend} check set")
        for name, check in checks.items():
            if not isinstance(check, dict) or check.get("frozen") != frozen[name]:
                raise GateError("Foreign G0 frozen reference")
            stems = {"cal": "calibration_v6_rows", "hold": "holdout_icao_validation_v6",
                     "hold_summary": "holdout_icao_validation_summary_v6", "ae3": "design_point_summary_v5"}
            target = op["paths"]["g0_target"]
            if check.get("new") != f"{target}/{stems[name]}_{backend}.csv":
                raise GateError("Foreign G0 regenerated table path")
            for field in ("frozen", "new"):
                relative(root, check[field])
                if check[field] not in files:
                    raise GateError("G0 table is outside hashed raw evidence")
            actual = scope["compare"](root / check["frozen"], root / check["new"])
            if actual != check or (backend == "cpp" and actual["match"] is not True):
                raise GateError("Recomputed original G0 comparison differs or fails")


def _g0(root, op, context):
    path = root / op["paths"]["g0_evidence"]
    doc = read(path)
    if doc.get("id") != "P8-SCREENING-G0-EVIDENCE" or doc.get("verdict") != "PASS" \
            or doc.get("main_dependency_sha256") != digest(root / op["paths"]["main_dependency"]):
        raise GateError("Fresh verified G0 evidence unavailable")
    for name, expected in doc.get("files", {}).items():
        if digest(relative(root, name)) != expected:
            raise GateError(f"Fresh G0 evidence changed: {name}")
    if not doc.get("files") or not doc.get("worker_proofs"):
        raise GateError("G0 receipt has no raw table/worker evidence")
    require_committed(root, [op["paths"]["g0_evidence"], *doc["files"]])
    terminal_path = doc.get("terminal")
    if terminal_path not in doc["files"] or read(relative(root, terminal_path)).get("status") != "PASS":
        raise GateError("G0 receipt lacks a successful released-run terminal")
    extension_sha = digest(root / op["paths"]["source_extension_manifest"])
    recorded = doc.get("identity", {})
    if recorded.get("extension_sha256") != extension_sha \
            or recorded.get("operations_sha256") != digest(root / OPERATIONS):
        raise GateError("G0 source-extension identity differs")
    extension = read(root / op["paths"]["source_extension_manifest"])
    expected_g0_identity = {"registration_sha256": digest(root / OPERATIONS),
                            "operations_sha256": digest(root / OPERATIONS),
                            "main_dependency_sha256": digest(root / op["paths"]["main_dependency"]),
                            "extension_sha256": extension_sha, "new_files": extension["files"],
                            "core": doc["core"], "g0_sha256": None}
    if recorded != expected_g0_identity:
        raise GateError("G0 receipt differs from complete frozen original G0 context")
    attempt = op["paths"]["g0_attempt"]
    spec_path = f"{attempt}/command_spec.json"
    reservation_path = f"{attempt}/reservation.json"
    released_path = f"{attempt}/released_lease.json"
    if not {spec_path, reservation_path, released_path} <= set(doc["files"]):
        raise GateError("G0 receipt lacks actual reservation/spec/release evidence")
    spec, reservation, released = [read(relative(root, p)) for p in (spec_path, reservation_path, released_path)]
    argv = spec.get("child_argv")
    if spec.get("root") != str(root) or spec.get("attempt") != attempt \
            or not isinstance(argv, list) or len(argv) != 5 \
            or argv[1:] != [str(root / "scripts/phase8/post_chain_g0.py"), "_execute", "--spec", str(root / spec_path)]:
        raise GateError("G0 command differs from the exact registered wrapper")
    terminal = read(relative(root, terminal_path))
    if terminal_path != f"{attempt}/terminal.json" or terminal.get("identity") != recorded \
            or reservation.get("identity") != recorded or released.get("identity") != recorded \
            or spec.get("identity") != recorded or spec.get("target") != op["paths"]["g0_target"] \
            or spec.get("workers") != 6 or spec.get("core") != doc["core"] \
            or spec.get("owner_pid") != reservation.get("owner_pid") \
            or terminal.get("reservation_sha256") != digest(root / reservation_path) \
            or released.get("reservation_sha256") != terminal.get("reservation_sha256") \
            or released.get("terminal_sha256") != digest(root / terminal_path) \
            or released.get("state") != "RELEASED" or released.get("owner_pid") != reservation.get("owner_pid") \
            or released.get("owner_birth") != reservation.get("owner_birth"):
        raise GateError("G0 run, source and release identities differ")
    expected = terminal.get("expected_outputs")
    hashes = terminal.get("artifact_hashes", {})
    if terminal.get("exit_code") != 0 or terminal.get("outputs_complete") is not True \
            or terminal.get("errors") or not isinstance(expected, list) or not expected \
            or not set(expected) <= set(hashes) or any(doc["files"].get(p) != h for p,h in hashes.items()):
        raise GateError("G0 success coverage or terminal hash references differ")
    core_path = relative(root, doc["core"]["path"])
    sha = digest(core_path)
    if sha != doc["core"]["sha256"]:
        raise GateError("G0 core hash changed")
    allowed = context["identity"]["built_modules"].copy()
    build = context["stages"].get("validation_build", {})
    if build.get("state") == "PASS":
        allowed.update(build.get("outputs", {}))
    if allowed.get(doc["core"]["path"]) != sha:
        raise GateError("G0 core is outside authoritative original/build evidence")
    for proof in doc["worker_proofs"]:
        proof_path = proof.get("evidence_path")
        if proof.get("core") != doc["core"] or not proof.get("pid") or not proof.get("birth") \
                or proof_path not in doc["files"] or read(relative(root, proof_path)) != {k:v for k,v in proof.items() if k != "evidence_path"}:
            raise GateError("Foreign or missing G0 worker provenance")
        if proof.get("main_dependency_sha256") != doc["main_dependency_sha256"] \
                or proof.get("extension_sha256") != extension_sha \
                or proof.get("spec_sha256") != doc["files"][spec_path]:
            raise GateError("G0 worker source provenance differs")
    roles = [p.get("role") for p in doc["worker_proofs"]]
    if roles.count("cpp_worker") != 6 or roles.count("ae3_parent") != 1 \
            or len(set((p["pid"], p["birth"]) for p in doc["worker_proofs"])) != 7:
        raise GateError("G0 requires six actual C++ workers and the AE3 parent")
    ac = _ac(root)
    all_workers = doc.get("python_worker_proofs", []) + doc["worker_proofs"]
    python_workers = doc.get("python_worker_proofs", [])
    py_pairs = {(w.get("pid"), w.get("birth")) for w in python_workers}
    cpp_pairs = {(w.get("pid"), w.get("birth")) for w in doc["worker_proofs"]}
    if len(python_workers) != 6 or len(py_pairs) != 6 or py_pairs & cpp_pairs \
            or any(w.get("role") != "python_worker" or not isinstance(w.get("pid"), int) \
                   or w["pid"] <= 0 or not w.get("birth") for w in python_workers):
        raise GateError("G0 lacks actual Python pool ownership evidence")
    for worker in all_workers:
        if worker.get("role") == "python_worker":
            proof_path = worker.get("evidence_path")
            if proof_path not in doc["files"] or read(relative(root, proof_path)) != {k:v for k,v in worker.items() if k != "evidence_path"} \
                    or worker.get("spec_sha256") != doc["files"][spec_path] \
                    or worker.get("main_dependency_sha256") != doc["main_dependency_sha256"] \
                    or worker.get("extension_sha256") != extension_sha:
                raise GateError("G0 Python pool provenance differs")
        if not any(c.get("pid") == worker.get("pid") and c.get("birth") == worker.get("birth")
                   for c in released.get("children", [])) \
                or ac.liveness(worker.get("pid"), worker.get("birth")) not in {"dead", "reused"}:
            raise GateError("G0 worker ending/ownership is unproven")
    command_exit = doc.get("command_exit")
    if command_exit not in doc["files"] or read(relative(root, command_exit)).get("exit_code") != 0 \
            or read(relative(root, command_exit)).get("waited") is not True:
        raise GateError("G0 command has no waited successful exit")
    command = read(relative(root, command_exit))
    parent = next(p for p in doc["worker_proofs"] if p["role"] == "ae3_parent")
    if command_exit != f"{attempt}/command.exit.json" or command.get("pid") != parent["pid"] \
            or command.get("birth") != parent["birth"] or command.get("argv") != spec.get("child_argv") \
            or not any(c.get("pid") == parent["pid"] and c.get("birth") == parent["birth"]
                       and c.get("argv") == command.get("argv") for c in released.get("children", [])):
        raise GateError("G0 waited command does not match actual recorded parent")
    verdict_paths = [p for p in doc["files"] if p.endswith("/g0_parity.json")]
    if len(verdict_paths) != 1:
        raise GateError("G0 receipt must name one raw verdict")
    verdict = read(root / verdict_paths[0])
    checks = verdict.get("backends", {}).get("cpp", {}).get("checks", {})
    if verdict.get("verdict") != "PASS" or set(checks) != {"cal", "hold", "hold_summary", "ae3"} \
            or not all(isinstance(c, dict) and c.get("match") is True for c in checks.values()):
        raise GateError("Raw G0 verdict/check set is not a complete PASS")
    recompare_g0(root, op, verdict, doc["files"])
    return doc, core_path, sha


class Context:
    def __init__(self, root, registration, allow_new_paths=None, expected=None, require_g0=True):
        self.root = Path(root).resolve()
        registration = Path(registration)
        if registration.is_absolute():
            try:
                registration = registration.resolve().relative_to(self.root)
            except ValueError as exc:
                raise GateError("Consumer registration is outside main checkout") from exc
        self.registration = str(registration)
        self.allow_new_paths = allow_new_paths
        self.expected = expected
        self.require_g0 = require_g0
        if not require_g0 and self.registration != OPERATIONS:
            raise GateError("Only the registered G0 wrapper may bypass its own prerequisite")
        self._owned = None
        self._refresh()

    def _refresh(self):
        op = operations(self.root)
        if self.registration not in _registrations(self.root, op)+[OPERATIONS]:
            raise GateError("Consumer registration is outside the registered screening sequence")
        require_committed(self.root, [OPERATIONS, AMENDMENT, self.registration,
                                     *_registrations(self.root, op),
                                     op["paths"]["main_dependency"],
                                     op["paths"]["source_extension_manifest"]])
        dep = read(self.root / op["paths"]["main_dependency"])
        context = _prove_original(self.root, op, dep)
        extension = _check_extensions(self.root, op, dep, self.allow_new_paths)
        _scientific_expected(self.root, self.registration, self.expected)
        g0 = None
        if self.require_g0:
            g0, binary, sha = _g0(self.root, op, context)
        else:
            candidates = [p for p in context["identity"]["built_modules"] if Path(p).name.startswith("catjet_core.")]
            if len(candidates) != 1:
                raise GateError("Expected one retained original G0 core")
            binary = self.root / candidates[0]
            sha = digest(binary)
        self.original_context = context
        self.binary_path, self.binary_sha256 = binary, sha
        self.identity = {"registration_sha256": digest(relative(self.root, self.registration)),
                         "operations_sha256": digest(self.root / OPERATIONS),
                         "main_dependency_sha256": digest(self.root / op["paths"]["main_dependency"]),
                         "extension_sha256": digest(self.root / op["paths"]["source_extension_manifest"]),
                         "new_files": extension["files"], "core": {"path": str(binary.relative_to(self.root)), "sha256": sha},
                         "g0_sha256": digest(self.root / op["paths"]["g0_evidence"]) if g0 else None}
        self.op = op
        return context

    def require_idle_ac(self):
        self._refresh()
        ac = _ac(self.root)
        power = ac.read_power()
        if not ac.on_ac(power):
            raise GateError("AC power is absent or unreadable")
        lease_path = self.root / self.op["paths"]["owner_lease"]
        if lease_path.exists():
            lease = read(lease_path)
            if self._owned is None or lease.get("reservation_sha256") != self._owned \
                    or lease.get("owner_pid") != os.getpid() \
                    or ac.liveness(lease.get("owner_pid"), lease.get("owner_birth")) != "alive":
                raise GateError("A post-chain owner lease already exists")
        return {"power": power, "context": self.original_context}

    def acquire_run(self, output_dir, registration_sha256, identity=None):
        return Run(self, output_dir, registration_sha256, identity)


class Run:
    def __init__(self, context, output_dir, registration_sha256, identity=None):
        context.require_idle_ac()
        self.context = context
        self._released = False
        self.identity = context.identity
        if registration_sha256 != self.identity["registration_sha256"] \
                or (identity is not None and identity != self.identity):
            raise GateError("Reservation identity differs from current context")
        self.out = Path(output_dir)
        if not self.out.is_absolute():
            self.out = relative(context.root, self.out)
        if not self.out.resolve().is_relative_to(context.root / "outputs") or self.out.exists():
            raise GateError("Run output must be a fresh directory under outputs")
        ac = _ac(context.root)
        birth = ac.process_birth(os.getpid())
        if not birth or birth == ac.DEAD:
            raise GateError("Owner birth unreadable")
        self.owner_birth = birth
        self.lease_path = context.root / context.op["paths"]["owner_lease"]
        lease = {"owner_pid": os.getpid(), "owner_birth": birth, "argv": sys.argv,
                 "registration_sha256": registration_sha256, "identity": self.identity,
                 "output_dir": str(self.out.relative_to(context.root)), "state": "STARTING",
                 "children": [], "created_utc": utc()}
        write_once(self.lease_path, lease)
        try:
            self.out.mkdir(parents=True, exist_ok=False)
            write_once(self.out / "reservation.json", lease)
            self.reservation_sha256 = digest(self.out / "reservation.json")
            lease["reservation_sha256"] = self.reservation_sha256
            self._write_lease(lease)
            context._owned = self.reservation_sha256
            self.assert_current()
        except BaseException:
            # No child may start until construction returned successfully.
            current = read(self.lease_path)
            if current.get("owner_pid") == lease["owner_pid"] and current.get("owner_birth") == lease["owner_birth"] \
                    and current.get("output_dir") == lease["output_dir"]:
                self.lease_path.unlink()
            context._owned = None
            raise

    def _write_lease(self, doc):
        tmp = self.lease_path.with_name(f".{self.lease_path.name}.{os.getpid()}.tmp")
        with tmp.open("x") as f:
            json.dump(doc, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.lease_path)

    def assert_current(self):
        self._assert_owned()
        self.context.require_idle_ac()
        if self.context.identity != self.identity:
            raise GateError("Scientific identity drift during owned run")

    def _assert_owned(self):
        if self._released:
            raise GateError("Released run object cannot acquire or mutate another lease")
        lease = read(self.lease_path)
        if lease.get("reservation_sha256") != self.reservation_sha256 \
                or lease.get("owner_pid") != os.getpid() \
                or lease.get("output_dir") != str(self.out.relative_to(self.context.root)):
            raise GateError("This run no longer owns the exact reservation")
        if digest(self.out / "reservation.json") != self.reservation_sha256:
            raise GateError("Owned reservation bytes changed")
        reservation = read(self.out / "reservation.json")
        mutable = {"state", "children", "reservation_sha256", "terminal_sha256"}
        if any(lease.get(k) != v for k,v in reservation.items() if k not in mutable) \
                or set(lease)-set(reservation)-mutable:
            raise GateError("Lease immutable identity differs from owned reservation")

    def record_children(self, children):
        # Ownership updates are metadata; keep child evidence even after drift.
        self._assert_owned()
        lease = read(self.lease_path)
        for child in children:
            if not isinstance(child, dict) or not isinstance(child.get("pid"), int) \
                    or not child.get("birth") or not isinstance(child.get("argv"), list):
                raise GateError("Child ownership requires PID, birth and argv")
        lease["children"] = list(children)
        lease["state"] = "RUNNING"
        self._write_lease(lease)

    def release(self, terminal):
        terminal = dict(terminal)
        if terminal.get("status") not in {"COMPLETE", "PASS", "FAIL", "ERROR", "INCOMPLETE", "BLOCKED"}:
            raise GateError("Terminal status is required")
        try:
            self.assert_current()
        except (GateError, OSError, ValueError) as exc:
            terminal["status"] = "ERROR"
            terminal.setdefault("errors", []).append(str(exc))
        ac = _ac(self.context.root)
        lease = read(self.lease_path)
        if lease.get("owner_pid") != os.getpid() or lease.get("owner_birth") != self.owner_birth \
                or lease.get("reservation_sha256") != self.reservation_sha256:
            raise GateError("Lease ownership lost; retained for review")
        alive = [c for c in lease["children"] if ac.liveness(c["pid"], c["birth"]) not in {"dead", "reused"}]
        if alive or terminal.get("ambiguous_child"):
            lease["state"] = "AMBIGUOUS_CHILD"
            self._write_lease(lease)
            raise GateError("Child exit unproven; lease retained")
        for name, expected in terminal.get("artifact_hashes", {}).items():
            try:
                matches = digest(relative(self.context.root, name)) == expected
            except (OSError, GateError):
                matches = False
            if not matches:
                terminal["status"] = "ERROR"
                terminal.setdefault("errors", []).append(f"Output changed: {name}")
        coverage = terminal.get("expected_outputs")
        complete = terminal.get("outputs_complete") is True and isinstance(coverage, list) and bool(coverage) \
            and set(coverage) <= set(terminal.get("artifact_hashes", {})) and terminal.get("exit_code") == 0
        if terminal["status"] in {"PASS", "COMPLETE"} and (terminal.get("errors") or not complete):
            terminal["status"] = "ERROR"
            terminal.setdefault("errors", []).append("Success lacks complete artifact hashes")
        if terminal["status"] not in {"PASS", "COMPLETE"}:
            terminal["exit_code"] = terminal.get("exit_code") or 1
            terminal["outputs_complete"] = False
        terminal.update(identity=self.identity, reservation_sha256=self.reservation_sha256, ended_utc=utc())
        write_once(self.out / "terminal.json", terminal)
        lease["state"] = "RELEASED"
        lease["terminal_sha256"] = digest(self.out / "terminal.json")
        write_once(self.out / "released_lease.json", lease)
        self.lease_path.unlink()
        self.context._owned = None
        self._released = True
        return terminal


def prepare_context(root, registration, *, allow_new_paths=None, expected_consumer_identity=None, require_g0=True):
    return Context(root, registration, allow_new_paths, expected_consumer_identity, require_g0)


def validate_consumer_terminal(root, registration_path, output_dir, *, expected_binary_sha256=None, artifact_paths=None, allow_scientific_fail=False):
    """Read-only raw producer proof without decoding sealed numerical labels."""
    ctx = prepare_context(root, registration_path)
    root = ctx.root
    out = Path(output_dir)
    if not out.is_absolute():
        out = relative(root, out)
    if not out.resolve().is_relative_to(root / "outputs"):
        raise GateError("Producer output is outside registered outputs")
    reg = read(root / ctx.registration)
    if reg.get("registration_id") == "P7.3-A1" and out != root / reg["outputs"]["directory"]:
        raise GateError("Producer output differs from registration")
    if reg.get("id") == "P8-S-20261004" and out != root / reg["artifact_root"]:
        raise GateError("Surrogate producer output differs from its registered attempt")
    terminal = read(out / "terminal.json")
    reservation = read(out / "reservation.json")
    released = read(out / "released_lease.json")
    for record in (terminal, reservation, released):
        if record.get("identity") != ctx.identity:
            raise GateError("Producer source/core/G0 identity differs")
    if expected_binary_sha256 is not None and expected_binary_sha256 != ctx.binary_sha256:
        raise GateError("Producer and consumer C++ cores differ")
    failed_complete = allow_scientific_fail is True and terminal.get("status") == "FAIL" \
        and terminal.get("execution_complete") is True and terminal.get("scientific_verdict") == "FAIL" \
        and isinstance(terminal.get("exit_code"), int) and terminal["exit_code"] != 0
    complete = terminal.get("status") in {"COMPLETE", "PASS"} and terminal.get("exit_code") == 0 \
        and terminal.get("outputs_complete") is True
    if not (complete or failed_complete) or terminal.get("errors") \
            or terminal.get("reservation_sha256") != digest(out / "reservation.json") \
            or released.get("reservation_sha256") != terminal.get("reservation_sha256") \
            or released.get("terminal_sha256") != digest(out / "terminal.json") \
            or released.get("state") != "RELEASED" \
            or released.get("owner_pid") != reservation.get("owner_pid") \
            or released.get("owner_birth") != reservation.get("owner_birth"):
        raise GateError("Producer terminal/reservation/release proof is incomplete")
    expected = terminal.get("expected_outputs")
    hashes = terminal.get("artifact_hashes", {})
    if not isinstance(expected, list) or not expected or not set(expected) <= set(hashes):
        raise GateError("Producer success lacks expected artifact coverage")
    if reg.get("registration_id") == "P7.3-A1":
        required = set(reg["outputs"]["quantitative"] + reg["outputs"]["provenance"])
        required.discard(str((out / "terminal.json").relative_to(root)))
        required.add(reg["outputs"]["technical_README"])
        required.add(str((out / "parity/parity.json").relative_to(root)))
        if not required <= set(expected):
            raise GateError("Producer coverage differs from its registered artifacts")
        _scientific_expected(root, ctx.registration, terminal.get("consumer_identity"))
        consumer = terminal.get("consumer_identity")
        own = set(reg.get("new_source_paths", [])) | {ctx.registration, ctx.registration.removesuffix(".json")+".md"}
        if not isinstance(consumer, dict) or not consumer.get("files") or set(consumer["files"]) != own:
            raise GateError("Producer lacks the complete declared consumer source identity")
        command = read(out / "command.exit.json")
        if command.get("exit_code") != 0 or command.get("status") != terminal["status"] \
                or command.get("identity") != ctx.identity or command.get("in_process_completed") is not True \
                or command.get("owner_pid") != reservation.get("owner_pid") \
                or command.get("owner_birth") != reservation.get("owner_birth") \
                or command.get("argv") != reservation.get("argv"):
            raise GateError("Producer completed-command proof differs from reservation")
    selected_hashes = hashes
    if artifact_paths is not None:
        if not isinstance(artifact_paths, (list, tuple, set)) or not artifact_paths or not set(artifact_paths) <= set(hashes):
            raise GateError("Producer artifact allowlist is absent or outside frozen terminal hashes")
        selected_hashes = {p: hashes[p] for p in artifact_paths}
    for name, sha in selected_hashes.items():
        if digest(relative(root, name)) != sha:
            raise GateError(f"Producer artifact changed: {name}")
    paths = [str((out / n).relative_to(root)) for n in ("terminal.json", "reservation.json", "released_lease.json")]
    require_committed(root, [*paths, *selected_hashes])
    ac = _ac(root)
    for child in released.get("children", []):
        if ac.liveness(child.get("pid"), child.get("birth")) not in {"dead", "reused"}:
            raise GateError("Producer child ending is unproven")
    # Metadata-only provenance projection; sealed reference values stay unopened.
    return {"status": terminal["status"], "identity": terminal["identity"],
            "terminal_sha256": digest(out / "terminal.json"), "artifact_hashes": hashes,
            "verified_artifact_paths": sorted(selected_hashes),
            "scientific_verdict": terminal.get("scientific_verdict"),
            "output_dir": str(out.relative_to(root)), "core_sha256": ctx.binary_sha256}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("export-main", "freeze-extensions"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(argv)
    doc = export_main_context(a.root) if a.command == "export-main" else freeze_extensions(a.root)
    print(json.dumps({"id": doc["id"], "created_utc": doc["created_utc"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
