"""Reviewable stdlib post-chain orchestration. Default validation never launches.

Run only a committed ARMED config with its explicitly supplied SHA256. Exact
original completion and lease release precede export, integration and scientific
imports. No retries, history rewrite, broad staging, push or stale-lease recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("focused_tests", "g0", "p73", "surrogate_checks", "surrogate", "nozzle_checks", "nozzle", "product_tests", "freeze")
TEST_STAGES = {"focused_tests", "surrogate_checks", "nozzle_checks", "product_tests"}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
ACTIVE_CHILDREN = {}


class Blocked(RuntimeError):
    pass


class AmbiguousChild(Blocked):
    """An exact owned child may remain alive; ownership must be retained."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise Blocked("expected a JSON object")
    return value


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary,path)  # Publish complete bytes without overwriting.
    finally:
        temporary.unlink(missing_ok=True)


def relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or str(path) in ("", "."):
        raise Blocked("unsafe repository-relative path")
    result = Path(root)/path
    # Missing destinations are permitted, symlinked parents are not.
    if result.resolve() != result.absolute():
        raise Blocked("path resolves through a symlink")
    return result


def git(root, *args, binary=False):
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode()


def birth(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True)
    value = " ".join(result.stdout.split())
    if result.returncode == 0 and value:
        return value
    if result.returncode == 1 and not value and not result.stderr.strip():
        return None
    raise Blocked("process birth is unreadable")


def require_nice():
    value = subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "nice="], text=True).strip()
    if not value or int(value) < 15:
        raise Blocked("bootstrap must run under nice -n15 or lower priority")


def prove_committed(root, paths):
    for name in paths:
        path = relative(root, name)
        raw = git(root, "ls-tree", "HEAD", "--", name).strip()
        if not raw:
            raise Blocked("required path is not committed")
        header, actual = raw.split("\t", 1)
        mode, kind, blob = header.split()
        expected_mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        if actual != name or kind != "blob" or mode != expected_mode or git(root, "cat-file", "blob", blob, binary=True) != path.read_bytes():
            raise Blocked("required committed file differs in mode or bytes")
    if git(root, "status", "--porcelain", "--", *paths).strip():
        raise Blocked("required paths are not clean")


def require_ac():
    value = subprocess.check_output(["pmset", "-g", "ps"], text=True)
    if "AC Power" not in value.splitlines()[0]:
        raise Blocked("AC power absent or unreadable")
    return value


def git_preflight(root):
    listing = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    if any(Path(line.split(maxsplit=1)[1]).name == "git" or Path(line.split(maxsplit=1)[1]).name.startswith("git-") for line in listing.splitlines() if len(line.split(maxsplit=1)) == 2):
        raise Blocked("a Git process is running")
    common = Path(git(root, "rev-parse", "--git-common-dir").strip())
    if not common.is_absolute():
        common = Path(root)/common
    index = Path(git(root, "rev-parse", "--git-path", "index.lock").strip())
    if not index.is_absolute():
        index = Path(root)/index
    if index.exists() or (common/"index.lock").exists():
        raise Blocked("Git index lock is present; no automatic removal")
    if git(root, "ls-files", "-u").strip():
        raise Blocked("unmerged index entries")


def committed_entry(root, entry):
    commit, name = entry.get("commit"), entry.get("source_path")
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise Blocked("source commit must be an exact full object id")
    relative(root, name)
    raw = git(root, "ls-tree", commit, "--", name).strip()
    if not raw or "\n" in raw:
        raise Blocked("reviewed source must identify exactly one blob")
    header, actual_name = raw.split("\t", 1)
    mode, kind, blob = header.split()
    if actual_name != name or kind != "blob" or mode not in ("100644", "100755"):
        raise Blocked("only exact regular reviewed source files are permitted")
    data = git(root, "cat-file", "blob", blob, binary=True)
    if entry.get("mode") != mode or entry.get("blob") != blob or entry.get("sha256") != hashlib.sha256(data).hexdigest():
        raise Blocked("reviewed source commit/blob/mode/SHA256 mismatch")
    return data


def index_projection(root, excluded):
    entries = git(root, "ls-files", "--stage", "-z").split("\0")
    return [line for line in entries if line and line.split("\t", 1)[1] not in excluded]


def selective_commit(root, paths, message):
    paths = sorted(set(paths))
    if not paths or not isinstance(message, str) or not message.strip():
        raise Blocked("explicit nonempty selective commit required")
    for name in paths:
        if not relative(root, name).is_file():
            raise Blocked("commit path is missing or not a regular file")
    git_preflight(root)
    retained = index_projection(root, set(paths))
    git(root, "add", "--", *paths)
    git_preflight(root)
    git(root, "commit", "--only", "-m", message, "--", *paths)
    if index_projection(root, set(paths)) != retained:
        raise Blocked("unrelated index entries changed during selective commit")
    return git(root, "rev-parse", "HEAD").strip()


def validate_config(root, config, armed=False):
    if config.get("schema_version") != 1 or config.get("registration_id") != "P8-SCREENING-BOOTSTRAP-20261004":
        raise Blocked("foreign bootstrap config")
    if armed and str(Path(root).resolve()) != config.get("main_root"):
        raise Blocked("armed destination differs from exact main repository")
    if config.get("stage_order") != list(STAGES) or config.get("tag", {}).get("name") != "freeze-2026-10-18":
        raise Blocked("unregistered stage order or local tag")
    original = config.get("original", {})
    for key in ("registration", "chain_path", "lease_path", "operations_registration"):
        relative(root, original[key])
    if original.get("session") != "20261004T022331Z-22543" or not original["chain_path"].endswith("chain."+original["session"]+".json"):
        raise Blocked("foreign original chain session")
    if original.get("owner_pid") != 22543 or original.get("owner_birth") != "Sat Oct 3 22:23:31 2026" or original.get("launch_head") != "8afb0e0f5c225396f2f486b6083bdf09c59a353b":
        raise Blocked("foreign original owner or launch")
    relative(root, config["output_dir"])
    if config["output_dir"] != "outputs/phase8/screening_operations/bootstrap_20261004" or config.get("scientific_lease_path") != "outputs/phase8/screening_operations/owner.lease.json":
        raise Blocked("foreign bootstrap output or scientific lease")
    if type(config.get("wait_poll_seconds")) is not int or not 1 <= config["wait_poll_seconds"] <= 30:
        raise Blocked("wait interval must be 1..30 seconds")
    pending = []
    for key in ("provider", "sources", "stages", "driver_sha256", "governing_files", "tag_guard"):
        if not config.get(key):
            pending.append(key)
    if not armed:
        return {"state": config.get("state"), "pending": pending, "execution_authorized": False}
    if config.get("state") != "ARMED" or pending or set(config["stages"]) != set(STAGES):
        raise Blocked("config is unarmed or incomplete")
    if config["tag"].get("authorized") is not True:
        raise Blocked("armed config lacks root authorization for the final local tag")
    for key in ("export_commit_message","integration_commit_message","extension_commit_message","usage_commit_message","bootstrap_evidence_commit_message"):
        if not isinstance(config.get(key),str) or not config[key].strip():
            raise Blocked("exact selective commit messages are missing")
    if digest(__file__) != config["driver_sha256"]:
        raise Blocked("bootstrap driver differs from reviewed config")
    provider = config["provider"]
    committed_entry(root, provider)
    for name, sha in config["governing_files"].items():
        if not HEX64.fullmatch(sha) or digest(relative(root, name)) != sha:
            raise Blocked("prospective governing registration differs from armed config")
    prove_committed(root, list(config["governing_files"]))
    for key in ("registration", "operations_registration"):
        if config["governing_files"].get(original[key]) != original[key+"_sha256"]:
            raise Blocked("original/export registrations lack governing identity")
    names = set()
    for entry in config["sources"]:
        name = entry["destination"]
        relative(root, name)
        if name in names:
            raise Blocked("duplicate integration destination")
        names.add(name)
        committed_entry(root, entry)
    gate_name = "scripts/phase8/scientific_workflow_gate.py"
    gate_sources = [entry for entry in config["sources"] if entry["destination"] == gate_name]
    fields = ("commit","source_path","blob","mode","sha256")
    if provider.get("source_path") != gate_name or len(gate_sources) != 1 or any(provider.get(key) != gate_sources[0].get(key) for key in fields):
        raise Blocked("export provider and integrated scientific gate must be the same exact reviewed blob")
    for entry in config.get("post_test_docs", []):
        if entry.get("destination") != "README.md" or not HEX40.fullmatch(entry.get("expected_old_blob", "")):
            raise Blocked("post-test documentation may update only the reviewed README")
        committed_entry(root, entry)
    for stage_id in STAGES:
        stage = config["stages"][stage_id]
        if stage.get("kind") not in ("pytest", "consumer") or not isinstance(stage.get("argv"), list) or not stage["argv"]:
            raise Blocked("missing exact stage argv/kind")
        if not all(isinstance(arg, str) for arg in stage["argv"]):
            raise Blocked("stage argv must be literal strings")
        if (stage["kind"] == "pytest") != (stage_id in TEST_STAGES):
            raise Blocked("consumer and test roles may not be interchanged")
        relative(root, stage["output_dir"])
        relative(root, stage["registration"])
        if not isinstance(stage.get("commit_message"),str) or not stage["commit_message"].strip():
            raise Blocked("stage selective commit message is missing")
        if stage["registration"] not in config["governing_files"]:
            raise Blocked("stage registration not bound to governing committed bytes")
        if not isinstance(stage.get("artifact_roots"), list) or not stage["artifact_roots"]:
            raise Blocked("stage artifact commit roots must be explicit")
        for name in stage["artifact_roots"]:
            if not str(relative(root, name).relative_to(root)).startswith("outputs/"):
                raise Blocked("artifact roots must be registered outputs")
        executable = Path(stage["argv"][0])
        if not executable.is_absolute():
            raise Blocked("stage must pin the actual absolute interpreter path")
        if digest(executable) != stage.get("executable_sha256"):
            raise Blocked("stage executable identity mismatch")
        if stage_id == "focused_tests" and (stage["kind"] != "pytest" or stage["registration"] != original["operations_registration"]):
            raise Blocked("pre-G0 controls must use the operations registration")
        if stage_id in ("surrogate_checks", "nozzle_checks") and stage["registration"] != original["operations_registration"]:
            raise Blocked("numerical controls require owned operations context with fresh G0")
        if stage_id == "product_tests" and stage["kind"] != "pytest":
            raise Blocked("product verification requires actual pytest")
        if stage["kind"] == "pytest" and (stage["argv"][1:3] != ["-m", "pytest"] or not stage.get("test_files")):
            raise Blocked("pytest requires its exact registered file list")
        if stage["kind"] == "pytest":
            tests = stage["test_files"]
            if stage["argv"][3:3+len(tests)] != tests or type(stage.get("minimum_passed")) is not int or stage["minimum_passed"] < 1:
                raise Blocked("pytest command does not bind ordered files and positive count")
            for name in tests:
                relative(root, name)
            junit = stage["junit_path"]
            relative(root, junit)
            if not junit.startswith(stage["output_dir"].rstrip("/")+"/") or "--junitxml="+junit not in stage["argv"]:
                raise Blocked("pytest JUnit path must belong to its fresh owned output")
            tails = stage["argv"][3+len(tests):]
            if tails != ["-v", "--junitxml="+junit]:
                raise Blocked("pytest flags may not skip or select cases")
            if stage_id == "focused_tests" and any("numerical" in name or "nozzle_ode" in name for name in tests):
                raise Blocked("G0 must precede numerical source controls")
            if stage_id == "product_tests":
                declared = read(relative(root, stage["registration"]))["quantitative_freeze_2026_10_04"]["verification_receipt"]
                if tests != declared["registered_tests"] or stage.get("verification") != declared:
                    raise Blocked("final verification differs from registered exact seven-file contract")
                if stage["junit_path"] != declared["junit_xml"]:
                    raise Blocked("final JUnit path differs from registered receipt")
                if stage.get("owner_registration") != original["operations_registration"]:
                    raise Blocked("final numerical fixtures require OPS parent ownership")
    guard = config["tag_guard"]
    if guard.get("argv", [])[1:] != ["scripts/phase8/freeze_screening.py", "--main-root", str(root), "--verify-receipt", "outputs/freeze/freeze_receipt.json", "--verification-receipt", config["stages"]["product_tests"]["verification"]["path"]]:
        raise Blocked("tag guard command must use exact registered receipts")
    executable = Path(guard["argv"][0])
    if not executable.is_absolute(): executable = Path(root)/executable
    if digest(executable) != guard.get("executable_sha256"):
        raise Blocked("tag guard interpreter differs")
    if not config.get("post_test_docs") and not config.get("reviewed_usage_identity"):
        raise Blocked("reviewed post-test usage documentation is missing")
    for install in config.get("dependency_installs", []):
        pins = install.get("pins")
        allowed = {"scipy==1.16.3", "PyYAML==6.0.3", "Cantera==3.2.0", "pandas==2.3.3",
            "python-dateutil==2.9.0.post0", "pytz==2025.2", "tzdata==2025.2", "six==1.17.0",
            "ruamel.yaml==0.18.16", "ruamel.yaml.clib==0.2.15"}
        if not isinstance(pins, list) or not pins or len(set(pins)) != len(pins) or not set(pins) <= allowed or install.get("authorized") is not True:
            raise Blocked("dependency installation is not explicitly pinned and authorized")
        if install.get("argv", [])[1:] != ["-m", "pip", "install", "--no-deps", *pins] or not Path(install["argv"][0]).is_absolute() or digest(install["argv"][0]) != install.get("executable_sha256"):
            raise Blocked("dependency installation command differs from exact pins")
    return {"state": "ARMED", "pending": [], "execution_authorized": True}


def wait_original(root, config, out):
    original = config["original"]
    for key in ("registration", "operations_registration"):
        if digest(relative(root, original[key])) != original[key+"_sha256"]:
            raise Blocked("original or operations registration changed before export")
    chain_path, lease = relative(root, original["chain_path"]), relative(root, original["lease_path"])
    while True:
        for name, sha in config["governing_files"].items():
            if digest(relative(root, name)) != sha:
                raise Blocked("governing bytes changed while waiting")
        if chain_path.exists() and not lease.exists():
            chain = read(chain_path)
            if chain.get("session") != original["session"] or chain.get("identity", {}).get("git_head") != original["launch_head"]:
                raise Blocked("exact chain launch identity mismatch")
            if chain.get("status") not in ("COMPLETE", "FINISHED_WITH_FLAGS_OR_FAILURES", "STOPPED_WITH_BLOCKERS"):
                raise Blocked("malformed terminal chain status")
            # Ending is an additional race guard, never completion evidence.
            if birth(original["owner_pid"]) != original["owner_birth"]:
                write_once(out/"original_ready.json", {"chain_path":original["chain_path"], "chain_sha256":digest(chain_path),
                    "session":original["session"], "status":chain["status"], "original_lease_absent":True,
                    "owner_pid":original["owner_pid"], "owner_birth":original["owner_birth"], "owner_ended":True, "observed_utc":utc()})
                return
        time.sleep(config["wait_poll_seconds"])


def load_provider(root, config, out):
    path = out/"provider/scientific_workflow_gate.py"
    write_once(path, committed_entry(root, config["provider"]))
    spec = importlib.util.spec_from_file_location("_reviewed_screening_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def integrate(root, config, attestation):
    protected = set(attestation["original_files"])
    staged = []
    for entry in config["sources"]:
        name = entry["destination"]
        path = relative(root, name)
        if name in protected or path.exists() or path.is_symlink() or git(root, "ls-files", "--", name).strip():
            raise Blocked("integration may add only genuinely NEW non-original paths")
        data = committed_entry(root, entry)
        write_once(path, data)
        path.chmod(0o755 if entry["mode"] == "100755" else 0o644)
        staged.append(name)
    return selective_commit(root, staged, config["integration_commit_message"])


def child_launch(spec_path, expected_sha):
    spec = read(spec_path)
    root = Path(spec["root"])
    handshake = Path(spec["handshake"])
    own_birth = birth(os.getpid())
    if not own_birth or digest(spec_path) != expected_sha:
        raise Blocked("child launch identity unreadable")
    lease = read(spec["bootstrap_owner_lease"])
    if os.getppid() != spec["owner_pid"] or birth(spec["owner_pid"]) != spec["owner_birth"] or any(lease.get(key) != spec.get(key) for key in ("owner_pid","owner_birth")) or lease.get("driver_sha256") != digest(__file__) or lease.get("config_sha256") != spec.get("config_sha256"):
        raise Blocked("private launcher lacks its exact live bootstrap owner")
    # Same-PID exec preserves the independently observed birth and wait handle.
    current_nice = int(subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "nice="], text=True).strip())
    if current_nice < 15: os.nice(15-current_nice)
    require_nice()
    write_once(handshake, {"pid":os.getpid(), "birth":own_birth, "argv":spec["argv"], "spec_sha256":digest(spec_path),
                          "parent_pid":spec["owner_pid"], "parent_birth":spec["owner_birth"],
                          "actual_launcher_argv":sys.orig_argv,
                          "native_launcher_command":subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "command="], text=True).strip()})
    deadline = time.monotonic()+60
    go_path = Path(spec["go"])
    while not go_path.exists():
        if birth(spec["owner_pid"]) != spec["owner_birth"] or time.monotonic() > deadline:
            raise Blocked("bootstrap owner ended or GO never arrived")
        time.sleep(.05)
    go = read(go_path)
    if go != {"action":"GO", "pid":os.getpid(), "birth":own_birth, "spec_sha256":digest(spec_path)}:
        raise Blocked("child was not authorized by exact GO")
    require_ac()
    require_nice()
    executable = Path(spec["argv"][0])
    if not executable.is_absolute():
        executable = root/executable
    if digest(executable) != spec["executable_sha256"]:
        raise Blocked("child executable drift")
    os.chdir(root)
    os.execve(str(executable), spec["argv"], {**os.environ, **THREADS})


def spawn(root, stage, proof_dir, guard, owned=None, on_ready=None, log_path=None):
    proof_dir.mkdir(parents=True, exist_ok=False)
    paths = {name:proof_dir/(name+".json") for name in ("command_spec", "handshake", "go", "command_exit")}
    log = Path(log_path) if log_path is not None else proof_dir/"command.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    guard()
    spec = {"root":str(root), "owner_pid":os.getpid(), "owner_birth":birth(os.getpid()), "argv":stage["argv"],
            "executable_sha256":stage["executable_sha256"], "handshake":str(paths["handshake"]), "go":str(paths["go"]), "started_utc":utc()}
    owner_lease = root/"outputs/phase8/screening_operations/bootstrap_20261004/owner.lease.json"
    lease = read(owner_lease)
    if lease.get("owner_pid") != spec["owner_pid"] or lease.get("owner_birth") != spec["owner_birth"] or lease.get("driver_sha256") != digest(__file__) or not HEX64.fullmatch(lease.get("config_sha256","")):
        raise Blocked("spawn has no exact reviewed bootstrap ownership")
    spec.update(bootstrap_owner_lease=str(owner_lease),config_sha256=lease["config_sha256"])
    write_once(paths["command_spec"], spec)
    # Pass the spec hash through argv, avoiding a self-hash in the document.
    launcher = [sys.executable, str(Path(__file__).resolve()), "_child", "--spec", str(paths["command_spec"]),
                "--spec-sha256", digest(paths["command_spec"])]
    with log.open("xb", buffering=0) as stream:
        child = subprocess.Popen(launcher, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        ACTIVE_CHILDREN[child.pid] = None
        observed_birth = None
        handshake, error = None, None
        try:
            observed_birth = birth(child.pid)
            ACTIVE_CHILDREN[child.pid] = observed_birth
            if not observed_birth:
                raise Blocked("new child birth unavailable")
            if owned is not None:
                owned.record_children([{"pid":child.pid, "birth":observed_birth, "argv":stage["argv"]}])
            deadline = time.monotonic()+60
            while not paths["handshake"].exists():
                guard()
                if child.poll() is not None or time.monotonic() > deadline:
                    raise Blocked("spawn has no exact child handshake; no GO sent")
                time.sleep(.05)
            handshake = read(paths["handshake"])
            if handshake["pid"] != child.pid or observed_birth != handshake["birth"] or birth(child.pid) != handshake["birth"] or handshake["spec_sha256"] != digest(paths["command_spec"]) or handshake.get("actual_launcher_argv") != launcher or not handshake.get("native_launcher_command"):
                raise Blocked("native child birth/actual launcher/spec mismatch")
            if on_ready is not None: on_ready(handshake)
            guard()
            require_nice()
            if child.poll() is not None or birth(child.pid) != observed_birth:
                raise Blocked("recorded child ended before GO")
            write_once(paths["go"], {"action":"GO", "pid":child.pid, "birth":observed_birth, "spec_sha256":digest(paths["command_spec"])})
            while child.poll() is None:
                guard()
                time.sleep(1)
            exit_code = child.wait()
            guard()
        except BaseException as exc:
            error = exc
            # Only the exact new owned child/session may be stopped. Original
            # owner/chain processes are never signaled or recovered.
            if child.poll() is None:
                try:
                    exact_child = bool(observed_birth) and birth(child.pid) == observed_birth and os.getpgid(child.pid) == child.pid
                except (Blocked,OSError):
                    exact_child = False
                if not exact_child:
                    raise AmbiguousChild("owned child ending ambiguous; leases retained") from exc
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    raise AmbiguousChild("owned child ending ambiguous; leases retained") from exc
            exit_code = child.wait()
        ACTIVE_CHILDREN.pop(child.pid, None)
    end = {**spec, "pid":child.pid, "birth":observed_birth, "waited":True, "exit_code":exit_code,
           "ended_utc":utc(), "log_sha256":digest(log), "command_spec_sha256":digest(paths["command_spec"]),
           "errors":[] if error is None else [f"{type(error).__name__}: {error}"]}
    write_once(paths["command_exit"], end)
    if error is not None: raise error
    return end, paths, log


def pytest_counts(path):
    try:
        root = ET.parse(path).getroot()
        suites, cases = list(root.iter("testsuite")), list(root.iter("testcase"))
        total = sum(int(suite.attrib["tests"]) for suite in suites)
        invalid = any(int(suite.attrib[key]) != 0 for suite in suites for key in ("errors","failures","skipped"))
    except (ET.ParseError,KeyError,ValueError) as exc:
        raise Blocked("JUnit is malformed or lacks declared totals") from exc
    if not suites or not cases or len(cases) != total or invalid or any(list(case.iter(tag)) for case in cases for tag in ("failure", "error", "skipped")):
        raise Blocked("focused tests did not complete with every case PASS")
    return len(cases)


def prove_pytest_log(path, count):
    log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", Path(path).read_text())
    passed = re.findall(r"\b(\d+) passed\b", log)
    if passed != [str(count)] or re.search(r"\b[1-9]\d* (?:failed|errors?|skipped|deselected)\b", log):
        raise Blocked("raw pytest log and all-cases JUnit PASS disagree")


def in_artifact_roots(name, roots):
    return any(name == prefix or name.startswith(prefix.rstrip("/")+"/") for prefix in roots)


def completed_paths(root, stage, identity, actual_exit):
    output = relative(root, stage["output_dir"])
    terminal, reservation, released = [read(output/name) for name in ("terminal.json", "reservation.json", "released_lease.json")]
    success = terminal.get("status") in ("PASS", "COMPLETE") and terminal.get("exit_code") == 0 and terminal.get("outputs_complete") is True
    failed = stage.get("allow_scientific_fail") is True and terminal.get("status") == "FAIL" and terminal.get("execution_complete") is True and terminal.get("scientific_verdict") == "FAIL" and type(terminal.get("exit_code")) is int and terminal["exit_code"] != 0
    if not (success or failed) or terminal.get("errors") or actual_exit != terminal.get("exit_code"):
        raise Blocked("waited producer did not complete a valid registered PASS or scientific FAIL")
    if any(record.get("identity") != identity for record in (terminal, reservation, released)) or terminal.get("reservation_sha256") != digest(output/"reservation.json") or released.get("reservation_sha256") != terminal["reservation_sha256"] or released.get("terminal_sha256") != digest(output/"terminal.json") or released.get("state") != "RELEASED" or any(released.get(key) != reservation.get(key) for key in ("owner_pid", "owner_birth")):
        raise Blocked("raw source/reservation/released-lease join is incomplete")
    expected, hashes = terminal.get("expected_outputs"), terminal.get("artifact_hashes")
    if not isinstance(expected, list) or not expected or not isinstance(hashes, dict) or not set(expected) <= set(hashes):
        raise Blocked("producer artifact coverage is incomplete")
    paths = set()
    for name, sha in hashes.items():
        if not HEX64.fullmatch(sha) or digest(relative(root, name)) != sha:
            raise Blocked("producer artifact hash differs before commit")
        if in_artifact_roots(name, stage["artifact_roots"]): paths.add(name)
        else: prove_committed(root, [name])  # Frozen inputs are checked, never restaged.
    paths.update(str((output/name).relative_to(root)) for name in ("terminal.json", "reservation.json", "released_lease.json"))
    for name in stage.get("additional_commit_paths", []):
        if not in_artifact_roots(name, stage["artifact_roots"]):
            raise Blocked("additional receipt is outside registered artifact roots")
        paths.add(name)
    if not all(in_artifact_roots(name, stage["artifact_roots"]) for name in paths):
        raise Blocked("artifact commit escapes registered roots")
    return sorted(paths), terminal


def stage_run(root, config, stage_id, gate, out, bootstrap_guard):
    bootstrap_guard()
    stage = config["stages"][stage_id]
    ctx = gate.prepare_context(root, stage["registration"], require_g0=stage_id != "focused_tests" and stage_id != "g0")
    ctx.require_idle_ac()
    identity = ctx.identity
    def guard():
        bootstrap_guard()
        fresh = gate.prepare_context(root, stage["registration"], require_g0=stage_id not in ("focused_tests", "g0"))
        if fresh.identity != identity:
            raise Blocked("stage scientific identity changed")
        require_ac()
    if stage["kind"] == "pytest":
        owner_registration = stage.get("owner_registration",stage["registration"])
        owner_context = gate.prepare_context(root,owner_registration,require_g0=stage_id != "focused_tests")
        owner_context.require_idle_ac()
        owned = owner_context.acquire_run(stage["output_dir"],owner_context.identity["registration_sha256"])
        terminal_identity = owner_context.identity
        junit = relative(root, stage["junit_path"])
        try:
            verification = stage.get("verification")
            op_paths = gate.operations(root)["paths"]
            extension = read(root/op_paths["source_extension_manifest"])
            tests = {**read(root/op_paths["main_dependency"])["original_files"], **identity["new_files"], **extension.get("other_declared_files",{})}
            test_files = {name:tests[name] for name in stage["test_files"]}
            raw_spec = relative(root, verification["command_spec"]) if verification else owned.out/"test_command_spec.json"
            raw_exit = relative(root, verification["command_exit"]) if verification else owned.out/"test_command.exit.json"
            raw_log = relative(root, verification["command_log"]) if verification else owned.out/"test_command.log"
            receipt_path = relative(root, verification["path"]) if verification else owned.out/"verification.json"
            command = {}
            def ready(handshake):
                executable = Path(stage["argv"][0])
                if not executable.is_absolute(): executable = root/executable
                command.update({"argv":stage["argv"], "workdir":str(root), "owner_pid":os.getpid(), "owner_birth":birth(os.getpid()),
                    "child_pid":handshake["pid"], "child_birth":handshake["birth"],
                    "interpreter":str(executable.absolute()), "interpreter_sha256":digest(executable)})
                write_once(raw_spec,{"command":command, "identity":identity, "test_files":test_files})
            def owned_guard():
                bootstrap_guard()
                owned.assert_current()
            end, paths, log = spawn(root, stage, owned.out/"raw", owned_guard, owned, ready, raw_log)
            count = pytest_counts(junit) if end["exit_code"] == 0 else 0
            if count < stage["minimum_passed"]:
                raise Blocked("focused test PASS count below registered minimum")
            prove_pytest_log(log, count)
            exit_record = {"status":"PASS", "exit_code":0, "waited":True, "command":command, "identity":identity, "test_files":test_files,
                "passed":count, "skipped":0, "ended_utc":end["ended_utc"], "log_sha256":digest(raw_log), "command_spec_sha256":digest(raw_spec),"junit_sha256":digest(junit)}
            write_once(raw_exit,exit_record)
            receipt = {"status":"PASS", "exit_code":0, "waited":True, "identity":identity, "passed":count, "skipped":0,
                "command":command, "test_files":test_files, "ended_utc":end["ended_utc"],
                "junit":{"path":stage["junit_path"],"sha256":digest(junit),"all_cases_pass":True,"passed":count},
                "raw_sha256":{str(p.relative_to(root)):digest(p) for p in (raw_spec,raw_log,raw_exit,junit)}}
            write_once(receipt_path,receipt)
            files = sorted({str(p.relative_to(root)) for p in owned.out.rglob("*") if p.is_file()} | {str(p.relative_to(root)) for p in (raw_spec,raw_log,raw_exit,receipt_path)})
            owned.release({"status":"PASS", "exit_code":0, "outputs_complete":True, "expected_outputs":files,
                           "artifact_hashes":{p:digest(relative(root,p)) for p in files}})
        except AmbiguousChild:
            raise  # Neither ownership layer can release an unproven child.
        except BaseException as exc:
            owned.release({"status":"ERROR", "exit_code":1, "outputs_complete":False, "errors":[str(exc)]})
            raise
    else:
        owner_registration, terminal_identity = stage["registration"], identity
        if relative(root, stage["output_dir"]).exists():
            raise Blocked("occupied study output; no implicit retry or overwrite")
        end, paths, log = spawn(root, stage, out/"commands"/stage_id, guard)
        if (root/gate.operations(root)["paths"]["owner_lease"]).exists():
            raise Blocked("consumer did not release exact scientific ownership")
        terminal = read(relative(root, stage["output_dir"])/"terminal.json")
        if end["exit_code"] != terminal.get("exit_code"):
            raise Blocked("actual waited exit disagrees with producer terminal")
    output = relative(root,stage["output_dir"])
    paths, terminal = completed_paths(root,stage,terminal_identity,end["exit_code"])
    commit = selective_commit(root,paths,stage["commit_message"])
    if stage_id == "focused_tests":
        prove_committed(root,paths)
        proof = {"status":terminal["status"],"identity":identity,"terminal_sha256":digest(output/"terminal.json"),"pre_g0_pure_controls":True}
    elif stage_id == "g0":
        fresh = gate.prepare_context(root,stage["registration"])
        fresh.require_idle_ac()  # Independently validates exact fresh G0 raw evidence.
        proof = {"status":"PASS","identity":fresh.identity,"g0_sha256":fresh.identity["g0_sha256"]}
    else:
        proof = gate.validate_consumer_terminal(root,owner_registration,output,
            expected_binary_sha256=ctx.binary_sha256,allow_scientific_fail=stage.get("allow_scientific_fail") is True)
    write_once(out/(stage_id+".receipt.json"), {"stage":stage_id,"commit":commit,"producer":proof,"completed_utc":utc()})


def dependencies(root, config, out, guard):
    # Only package metadata is inspected. No MLX/core/scientific module import.
    script = "import importlib.metadata as m,json,sys; canonical=lambda n:n.lower().replace('-','_').replace('.','_'); d={canonical(p.metadata.get('Name','')):p.version for p in m.distributions()}; print(json.dumps({n:d.get(canonical(n)) for n in sys.argv[1:]}))"
    for index, item in enumerate(config.get("dependency_installs", [])):
        interpreter, pins = item["argv"][0], item["pins"]
        names = [pin.split("==")[0] for pin in pins]
        query = {"argv":[interpreter,"-c",script,*names],"executable_sha256":item["executable_sha256"]}
        before, _, log = spawn(root,query,out/"commands"/f"dependency_{index}_before",guard)
        if before["exit_code"]:
            raise Blocked("dependency metadata inspection failed")
        versions = json.loads(log.read_text())
        missing = []
        for pin in pins:
            name, version = pin.split("==")
            if versions.get(name) is None: missing.append(pin)
            elif versions[name] != version:
                raise Blocked("existing dependency version differs; no environment overwrite")
        if missing:
            install = {**item,"argv":[interpreter,"-m","pip","install","--no-deps",*missing]}
            result, _, _ = spawn(root,install,out/"commands"/f"dependency_{index}_install",guard)
            if result["exit_code"]:
                raise Blocked("explicit missing pinned dependency installation failed")
        after, _, log = spawn(root,query,out/"commands"/f"dependency_{index}_after",guard)
        final = json.loads(log.read_text())
        if after["exit_code"] or any(final.get(pin.split("==")[0]) != pin.split("==")[1] for pin in pins):
            raise Blocked("pinned dependency versions are unproven")
        write_once(out/f"dependency_{index}.receipt.json",{"before":versions,"installed_pins":missing,"after":final,"interpreter":interpreter,"interpreter_sha256":item["executable_sha256"]})


def update_usage(root, config):
    paths = []
    for entry in config.get("post_test_docs", []):
        name = entry["destination"]
        prove_committed(root,[name])
        raw = git(root,"ls-tree","HEAD","--",name).strip().split("\t",1)[0].split()
        if raw[2] != entry["expected_old_blob"]:
            raise Blocked("README changed since reviewed usage update")
        data = committed_entry(root,entry)
        path = relative(root,name)
        temporary = path.with_name(path.name+".bootstrap-new")
        write_once(temporary,data)
        temporary.chmod(0o755 if entry["mode"] == "100755" else 0o644)
        os.replace(temporary,path)
        paths.append(name)
    if paths: return selective_commit(root,paths,config["usage_commit_message"])
    if not config.get("reviewed_usage_identity"):
        raise Blocked("reviewed conditional README usage identity is missing")
    prove_committed(root,["README.md"])
    if digest(root/"README.md") != config["reviewed_usage_identity"]["sha256"]:
        raise Blocked("existing reviewed usage README differs")
    return git(root,"rev-parse","HEAD").strip()


def execute(root, config_path, expected_sha):
    config = read(config_path)
    if not HEX64.fullmatch(expected_sha or "") or digest(config_path) != expected_sha:
        raise Blocked("exact reviewed config SHA256 is required")
    validate_config(root,config,armed=True)
    source_root = Path(__file__).resolve().parents[2]
    config_rel = str(Path(config_path).resolve().relative_to(source_root))
    driver_rel = str(Path(__file__).resolve().relative_to(source_root))
    prove_committed(source_root,[config_rel,driver_rel])
    require_nice()
    out = relative(root,config["output_dir"])
    out.mkdir(parents=True,exist_ok=False)
    lease = {"owner_pid":os.getpid(),"owner_birth":birth(os.getpid()),"argv":sys.orig_argv,
             "config_sha256":expected_sha,"driver_sha256":digest(__file__),"started_utc":utc()}
    write_once(out/"owner.lease.json",lease)
    write_once(out/"config.json",config)
    errors = []
    try:
        wait_original(root,config,out)
        require_ac()
        gate = load_provider(root,config,out)
        attestation = gate.export_main_context(root)
        # The exporter returns its document, not a path or summary receipt.
        if attestation["identity"]["git_head"] != config["original"]["launch_head"]:
            raise Blocked("strict export differs from configured original launch")
        ready = read(out/"original_ready.json")
        if attestation.get("raw_evidence",{}).get(config["original"]["chain_path"]) != ready["chain_sha256"] or attestation["context"].get("chain",{}).get("session") != config["original"]["session"]:
            raise Blocked("strict export did not bind exact configured completed chain")
        selective_commit(root,[gate.operations(root)["paths"]["main_dependency"]],config["export_commit_message"])
        def original_guard():
            require_ac()
            require_nice()
            if digest(config_path) != expected_sha or digest(__file__) != config["driver_sha256"]:
                raise Blocked("bootstrap driver/config changed")
            if digest(out/"provider/scientific_workflow_gate.py") != config["provider"]["sha256"]:
                raise Blocked("reviewed export/validation provider changed")
            committed_entry(root,config["provider"])
            copied = out/"provider/scientific_workflow_gate.py"
            if ("100755" if copied.stat().st_mode & stat.S_IXUSR else "100644") != config["provider"]["mode"]:
                raise Blocked("reviewed provider mode changed")
            if read(out/"owner.lease.json") != lease:
                raise Blocked("bootstrap owner changed")
            for name,sha in config["governing_files"].items():
                if digest(relative(root,name)) != sha: raise Blocked("governing bytes changed")
            gate._prove_original(root,gate.operations(root),attestation)
        dependencies(root,config,out,original_guard)
        original_guard()
        integrate(root,config,attestation)
        gate.freeze_extensions(root)
        selective_commit(root,[gate.operations(root)["paths"]["source_extension_manifest"]],config["extension_commit_message"])
        for stage_id in STAGES:
            stage_run(root,config,stage_id,gate,out,original_guard)
            if stage_id == "product_tests":
                commit = update_usage(root,config)
                write_once(out/"usage.receipt.json",{"commit":commit,"README_sha256":digest(root/"README.md")})
        tag = config["tag"]
        if tag.get("authorized") is not True:
            raise Blocked("local tag was not explicitly armed")
        guard_stage = config["tag_guard"]
        identity = gate.prepare_context(root,config["stages"]["freeze"]["registration"]).identity
        def final_guard():
            original_guard()
            current = gate.prepare_context(root,config["stages"]["freeze"]["registration"])
            if current.identity != identity: raise Blocked("final source/G0 identity changed")
            current.require_idle_ac()
        end,_,log = spawn(root,guard_stage,out/"commands/tag_guard",final_guard)
        if end["exit_code"]:
            raise Blocked("quantitative tag guard refused actual artifacts")
        review = json.loads(log.read_text())
        if review.get("status") != "READY_FOR_ROOT_REVIEW" or review.get("local_tag") != tag["name"] or review.get("tag_created") is not False or review.get("scientific_verdict") not in ("PASS","FAIL"):
            raise Blocked("tag guard returned a foreign or incomplete review")
        final_guard()
        write_once(out/"tag_review.json",review)
        raw_paths = [str(path.relative_to(root)) for path in out.rglob("*") if path.is_file() and path.name != "owner.lease.json"]
        selective_commit(root,raw_paths,config["bootstrap_evidence_commit_message"])
        final_guard()
        git_preflight(root)
        if git(root,"tag","--list",tag["name"]).strip():
            raise Blocked("local tag exists; never move or overwrite it")
        final = git(root,"rev-parse","HEAD").strip()
        git(root,"tag",tag["name"],final)
        exit_code = 0 if review["scientific_verdict"] == "PASS" else 1
        write_once(out/"terminal.json",{"status":"COMPLETE" if exit_code == 0 else "FAIL","exit_code":exit_code,
            "execution_complete":True,"errors":[],"tag":tag["name"],"tag_created":True,"commit":final,
            "scientific_verdict":review["scientific_verdict"],"config_sha256":expected_sha,"ended_utc":utc()})
        return exit_code
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        if not (out/"terminal.json").exists():
            write_once(out/"terminal.json",{"status":"BLOCKED","exit_code":1,"execution_complete":False,"errors":errors,"ended_utc":utc()})
        raise
    finally:
        # No common scientific lease is ever deleted here.
        scientific_lease = relative(root,config["scientific_lease_path"])
        if ACTIVE_CHILDREN or scientific_lease.exists():
            raise AmbiguousChild("live or unresolved scientific ownership; bootstrap lease retained")
        active = read(out/"owner.lease.json")
        if active != lease:
            raise Blocked("bootstrap ownership changed; retained")
        write_once(out/"released_lease.json",{**lease,"terminal_sha256":digest(out/"terminal.json"),"released_utc":utc()})
        (out/"owner.lease.json").unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command",choices=("validate","run","_child"))
    parser.add_argument("--main-root",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--spec-sha256")
    args = parser.parse_args(argv)
    if args.command == "_child":
        if not args.spec or digest(args.spec) != args.spec_sha256:
            raise Blocked("child spec hash differs")
        return child_launch(args.spec,args.spec_sha256)
    if not args.main_root or not args.config:
        parser.error("exact main root and config are required")
    root = args.main_root.resolve()
    if args.command == "validate":
        config = read(args.config)
        result = validate_config(root,config,armed=config.get("state") == "ARMED")
        print(json.dumps({**result,"execution_authorized":False,"config_identity_complete":config.get("state") == "ARMED","config_sha256":digest(args.config)}))
        return 0
    return execute(root,args.config.resolve(),args.config_sha256)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (Blocked,OSError,ValueError,subprocess.SubprocessError) as exc:
        print(f"BLOCKED: {exc}",file=sys.stderr)
        raise SystemExit(1)
