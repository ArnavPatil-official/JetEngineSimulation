#!/usr/bin/env python3
"""Record-owned, AC-only Phase 8 recovery workflow
(docs/phase8_queue_recovery_registration.json, P8-QUEUE-RECOVERY-20261003
and its amendment_1).

    caffeinate -i .venv/bin/python scripts/phase8/ac_workflow.py chain \
        --registration docs/phase8_queue_recovery_registration.json

Subcommands
  chain          the registered stages in order, one heavy process at a time:
                 benchmark run 2, its two reruns, A2 calibration, Track 4
                 focused pytest, the single diagnostic, full pytest, hashes
  queue          one registered benchmark queue (run_benchmark_queue.sh)
  status         lease and records, read-only; --require-idle exits 1 unless
                 no lease is held and a finished chain record exists
  recover-stale  remove a provably stale lease, with a recorded reason
  _child         internal launcher: handshake, wait for go, exec the command

Completion comes only from registered record files and benchmark result.json
files, never from process names, log text or a dead PID. One owner lease is
created atomically (O_EXCL); the liveness of the owner and of its child is
PID plus process birth time. Every child is a launcher that waits for an
explicit go before exec of the registered command, so the lease records the
child before any heavy work starts. AC power and source identity are
rechecked after every wait, immediately before the spawn and again before the
go; an unreadable power source blocks. Every record and log is write-once.
Standard library only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRATION = "docs/phase8_queue_recovery_registration.json"
REGISTRATION_ID = "P8-QUEUE-RECOVERY-20261003"
DEAD = "<dead>"
ELIGIBLE = ("PASS", "FAIL", "INCOMPLETE", "INVALID_POWER")   # terminal benchmark verdicts
BLOCKING = ("REFUSED", "ERROR_PARTIAL")                       # terminal execution, never queue-complete
QUEUE_DONE = ("TERMINAL_COMPLETE", "TERMINAL_COMPLETE_WITH_FLAGS")
STAGE_STATES = ("PASS", "FAIL", "REFUSED")
RUN_FILES = ("manifest.json", "progress.jsonl", "result.json")
ABORTED_EXIT = 125                                            # launcher told to abort: nothing executed


class Blocked(Exception):
    """A precondition stops this spec/stage without consuming it."""


class Refused(Exception):
    """Ownership or registration refusal; nothing was launched."""


class LeaseLost(RuntimeError):
    """The lease on disk is no longer this session's."""


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError:
        return None


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_once(path: Path, obj) -> str:
    """Create a JSON file that must not exist; return the sha256 of its bytes."""
    data = (json.dumps(obj, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return sha256_bytes(data)


def on_ac(pmset_text: str | None, marker: str = "'AC Power'") -> bool:
    return bool(pmset_text) and marker in pmset_text.splitlines()[0]


def _run(argv: list, cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def read_power() -> str | None:
    return _run(["pmset", "-g", "ps"])


def process_birth(pid: int) -> str | None:
    """Start time of a live process, DEAD if it does not exist, None if unknown."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return DEAD
    except PermissionError:
        pass
    except (OSError, OverflowError, TypeError):
        return None
    out = _run(["ps", "-o", "lstart=", "-p", str(pid)])
    if out is None:                      # ps exits 1 if the process ended in between
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return DEAD
        except OSError:
            pass
        return None
    return " ".join(out.split()) or None


def liveness(pid, birth, birth_fn=process_birth) -> str:
    """'alive', 'dead', 'reused' (same PID, other birth time) or 'unknown'.

    A stopped process is alive: only exit or PID reuse makes an owner stale."""
    if not isinstance(pid, int) or pid <= 0 or not birth or birth == DEAD:
        return "unknown"
    now = birth_fn(pid)
    if now is None:
        return "unknown"
    if now == DEAD:
        return "dead"
    return "alive" if now == birth else "reused"


def git_identity(root: Path, paths: list, modules_glob: str) -> dict:
    """HEAD, a hash of the tracked tree under ``paths`` (must be clean) and built modules."""
    head = _run(["git", "rev-parse", "--verify", "HEAD"], root)
    status = _run(["git", "status", "--porcelain", "--", *paths], root)
    tree = _run(["git", "ls-tree", "-r", "HEAD", "--", *paths], root)
    if head is None or status is None or tree is None:
        raise Blocked("git HEAD, status or tree unreadable (fail closed)")
    if status.strip():
        raise Blocked("uncommitted or untracked changes in identity paths:\n" + status.rstrip())
    modules = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.glob(modules_glob))}
    if any(d is None for d in modules.values()):
        raise Blocked("built module identity unreadable (fail closed)")
    return {"git_head": head.strip(), "tracked_tree_sha256": sha256_bytes(tree.encode()),
            "built_modules": modules}


def same_sources(a, b) -> bool:
    return (isinstance(a, dict) and isinstance(b, dict)
            and isinstance(a.get("tracked_tree_sha256"), str)
            and isinstance(a.get("built_modules"), dict)
            and a.get("tracked_tree_sha256") == b.get("tracked_tree_sha256")
            and a.get("built_modules") == b.get("built_modules"))


def parse_specs(lines) -> list[dict]:
    specs = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (4, 5) or (len(parts) == 5 and parts[4] != "native") \
                or not parts[0].isdigit() or not parts[3].isdigit():
            raise Blocked(f"malformed benchmark spec {line!r}")
        specs.append({"arm": int(parts[0]), "variant": parts[1], "workload": parts[2],
                      "workers": int(parts[3]), "native": len(parts) == 5, "text": " ".join(parts)})
    return specs


def run_name(spec: dict) -> str:
    return (f"arm{spec['arm']}{spec['variant']}{'-native' if spec['native'] else ''}"
            f"_{spec['workload']}_{spec['workers']}w")


def child_main(handshake: str, go_fd: int, command: list) -> int:
    """Launcher: record this process, wait for the driver's go, exec the command (same PID).

    Anything other than an explicit go (abort, EOF because the driver died) exits
    ABORTED_EXIT without executing the command."""
    try:
        write_once(Path(handshake), {"pid": os.getpid(), "birth": process_birth(os.getpid()),
                                     "command": command, "written_utc": utc()})
    except OSError as exc:
        print(f"launcher: handshake not written ({exc}); aborting", flush=True)
        os.close(go_fd)
        return ABORTED_EXIT
    with os.fdopen(go_fd, "rb") as f:
        msg = f.read()
    if msg != b"go" or not command:
        print(f"launcher: no go ({msg!r}); registered command not executed", flush=True)
        return ABORTED_EXIT
    sys.stdout.flush()
    os.execvp(command[0], command)
    return 127                          # not reached


class Workflow:
    def __init__(self, root: Path = ROOT, registration: str = REGISTRATION, *, power=read_power,
                 birth=process_birth, spawn=None, sleep=time.sleep, identity=None):
        self.root = Path(root)
        self.reg_path = self.root / registration
        self.reg_bytes = self.reg_path.read_bytes()
        self.reg = json.loads(self.reg_bytes)
        if self.reg.get("id") != REGISTRATION_ID:
            raise Refused(f"unexpected registration id {self.reg.get('id')!r}")
        self.reg_sha = sha256_bytes(self.reg_bytes)
        self.out = self.root / self.reg["output_root"]
        self.records = self.out / "records"
        self.lease_path = self.root / self.reg["lease_path"]
        self.lock_path = self.lease_path.with_name(self.lease_path.name + ".recovery.lock")
        self.marker = self.reg["power"]["ac_marker"]
        self.poll_s = self.reg["power"]["poll_s"]
        self.power, self.birth, self.sleep = power, birth, sleep
        self.spawn = spawn or self._spawn
        ident = self.reg["identity"]
        self.identity = identity or (lambda: git_identity(self.root, ident["tracked_paths"],
                                                          ident["built_modules_glob"]))
        self.session = self.session_dir = self.lease = self.start_identity = None
        self.child = None           # recorded child that has not been waited for
        self.ambiguous = None       # set before a spawn, cleared only by waited-exit evidence
        self.n_spawn = 0
        self.benchmark_children = []
        self.notes = []

    # ------------------------------------------------------------------ basics
    def rel(self, path: Path) -> str:
        return str(Path(path).relative_to(self.root))

    def log(self, msg: str) -> None:
        line = f"{utc()} {msg}"
        print(line, flush=True)
        if self.session_dir is not None:
            with open(self.session_dir / "driver.log", "a") as f:
                f.write(line + "\n")

    def _spawn(self, argv: list, logf, go_fd: int, handshake: Path):
        argv = [str(self.root / a) if i == 0 and "/" in a and not os.path.isabs(a) else a
                for i, a in enumerate(argv)]
        launcher = [sys.executable, str(Path(__file__).resolve()), "_child", "--handshake", str(handshake),
                    "--go-fd", str(go_fd), "--", *argv]
        return subprocess.Popen(launcher, cwd=self.root, stdin=subprocess.DEVNULL, stdout=logf,
                                stderr=subprocess.STDOUT, pass_fds=(go_fd,))

    # ------------------------------------------------------------------- lease
    def start(self, command: str) -> None:
        """Freeze the start identity, take the lease, create the session records."""
        self.start_identity = self.identity()
        own_birth = self.birth(os.getpid())
        if not own_birth or own_birth == DEAD:
            raise Refused("cannot read this process's birth time (fail closed)")
        session = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        lease = {"registration_id": self.reg["id"], "registration_sha256": self.reg_sha,
                 "session": session, "command": command, "owner_pid": os.getpid(),
                 "owner_birth": own_birth, "host": platform.node(), "acquired_utc": utc(),
                 "identity": self.start_identity, "state": "STARTING", "stage": None,
                 "child": None, "pending_child": None, "updated_utc": utc()}
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            other = read_json(self.lease_path)
            if not isinstance(other, dict):
                raise Refused(f"{self.rel(self.lease_path)} exists but is malformed; inspect it by hand")
            state = liveness(other.get("owner_pid"), other.get("owner_birth"), self.birth)
            if state in ("alive", "unknown"):
                raise Refused(f"lease held by a {state} owner (pid {other.get('owner_pid')}, session "
                              f"{other.get('session')}, state {other.get('state')}); duplicate ownership refused")
            raise Refused(f"stale lease: owner pid {other.get('owner_pid')} is {state}; nothing is inferred "
                          "about its work. Recover explicitly with: ac_workflow.py recover-stale --reason ...")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(lease, indent=2) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.lease, self.session = lease, session
            session_dir = self.out / "sessions" / session
            session_dir.mkdir(parents=True, exist_ok=False)
            (session_dir / "driver.log").open("x").close()
            self.session_dir = session_dir
            parent = os.getppid()
            write_once(session_dir / "launch.json", {
                "registration_id": self.reg["id"], "registration_sha256": self.reg_sha,
                "session": session, "command": command, "argv": sys.argv, "cwd": os.getcwd(),
                "owner_pid": os.getpid(), "owner_birth": own_birth, "parent_pid": parent,
                "parent_args": (_run(["ps", "-o", "args=", "-p", str(parent)]) or "").strip() or None,
                "host": platform.node(), "python": sys.version.split()[0], "started_utc": utc(),
                "identity": self.start_identity, "lease": self.rel(self.lease_path)})
        except BaseException:
            # nothing was spawned: this session's own fresh lease is removed, never left stale
            os.unlink(self.lease_path)
            self.lease = self.session = None
            raise
        self.log(f"session {session} owns {self.rel(self.lease_path)} ({command}); "
                 f"HEAD {self.start_identity['git_head']}")

    def update_lease(self, **changes) -> None:
        current = read_json(self.lease_path)
        if not isinstance(current, dict) or current.get("session") != self.session \
                or current.get("owner_pid") != os.getpid():
            raise LeaseLost(f"{self.rel(self.lease_path)} is no longer owned by session {self.session}")
        self.lease.update(changes, updated_utc=utc())
        tmp = self.lease_path.with_name(f".{self.lease_path.name}.{self.session}.tmp")
        tmp.write_text(json.dumps(self.lease, indent=2) + "\n")
        os.replace(tmp, self.lease_path)

    def release(self, outcome: str) -> bool:
        """Release the lease; False (lease kept) while a child is recorded or ambiguous."""
        if self.child is not None or self.ambiguous is not None:
            try:
                self.update_lease(state="AMBIGUOUS_CHILD", outcome=outcome)
            except (LeaseLost, OSError) as exc:
                self.log(f"could not mark the lease AMBIGUOUS_CHILD: {exc}")
            self.log("lease kept: a child was spawned or being spawned without waited-exit evidence")
            return False
        self.update_lease(state="RELEASED", outcome=outcome, released_utc=utc())
        dest = self.out / "leases" / "released" / f"{self.session}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(self.lease_path, dest)      # never overwrites
        os.unlink(self.lease_path)
        self.log(f"lease released ({outcome}) -> {self.rel(dest)}")
        return True

    def _child_state(self, lease: dict) -> str:
        """'none', 'dead' (provably ended), 'alive', 'unknown' or 'ambiguous' (no identity at all)."""
        child, pending = lease.get("child"), lease.get("pending_child")
        if not child and not pending:
            return "ambiguous" if lease.get("state") in ("STARTING_CHILD", "CHILD_SPAWNED",
                                                         "AMBIGUOUS_CHILD") else "none"
        idents = []
        if isinstance(child, dict) and child.get("birth"):
            idents.append((child.get("pid"), child.get("birth")))
        hs_path = (child or {}).get("handshake") or (pending or {}).get("handshake")
        hs = read_json(self.root / hs_path) if hs_path else None
        if isinstance(hs, dict) and hs.get("birth"):
            idents.append((hs.get("pid"), hs.get("birth")))
        if not idents:
            return "ambiguous"
        states = [liveness(p, b, self.birth) for p, b in idents]
        if all(s in ("dead", "reused") for s in states):
            return "dead"
        return "alive" if "alive" in states else "unknown"

    def recover_stale(self, reason: str, attest_no_child: str | None = None) -> str:
        """Remove a provably stale lease under an exclusive recovery lock, revalidating
        the lease file's identity immediately before removal."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Refused("another stale recovery holds the recovery lock") from None
            try:
                st, data = os.stat(self.lease_path), self.lease_path.read_bytes()
            except FileNotFoundError:
                raise Refused("no lease to recover") from None
            ident = (st.st_dev, st.st_ino, sha256_bytes(data))
            try:
                other = json.loads(data)
            except ValueError:
                other = None
            if not isinstance(other, dict):
                raise Refused("lease is malformed; inspect it by hand")
            owner = liveness(other.get("owner_pid"), other.get("owner_birth"), self.birth)
            if owner not in ("dead", "reused"):
                raise Refused(f"lease is not stale: owner {owner} (a stopped owner is alive)")
            child = self._child_state(other)
            attestation = None
            if child == "ambiguous" and attest_no_child:
                attestation = attest_no_child
            elif child not in ("none", "dead"):
                raise Refused(f"lease is not provably stale: child {child}; inspect by hand"
                              + ("; after verifying no child exists use --attest-no-child TEXT"
                                 if child == "ambiguous" else ""))
            try:
                st2, data2 = os.stat(self.lease_path), self.lease_path.read_bytes()
            except FileNotFoundError:
                raise Refused("lease vanished during recovery; nothing removed") from None
            if (st2.st_dev, st2.st_ino, sha256_bytes(data2)) != ident:
                raise Refused("lease changed during recovery; nothing removed")
            dest = self.out / "leases" / "stale" / f"{other.get('session')}-recovered-{os.getpid()}.json"
            write_once(dest, {"registration_id": self.reg["id"], "recovered_utc": utc(), "reason": reason,
                              "by_pid": os.getpid(), "owner_state": owner, "child_state": child,
                              "operator_attestation_no_child": attestation, "lease_sha256": ident[2],
                              "stale_lease": other, "inferred_completion": None,
                              "note": "no completion is inferred from a dead PID; records decide"})
            os.unlink(self.lease_path)
            return self.rel(dest)
        finally:
            os.close(fd)

    # ------------------------------------------------------- power and launch
    def wait_for_ac(self, what: str) -> None:
        waiting = False
        while True:
            text = self.power()
            if on_ac(text, self.marker):
                if waiting:
                    self.update_lease(state="AC_PRESENT", waiting_for=None)
                    self.log(f"AC power present; continuing to {what}")
                return
            if not waiting:
                state = "WAITING_FOR_AC" if text else "WAITING_FOR_POWER_READING"
                self.update_lease(state=state, waiting_for=what)
                self.log(f"{state} before {what}: "
                         f"{text.splitlines()[0] if text else 'pmset unreadable (blocks)'}")
                waiting = True
            self.sleep(self.poll_s)

    def _check_sources(self, label: str, sha_checks: dict | None) -> dict:
        for path, digest in (sha_checks or {}).items():
            if sha256_file(self.root / path) != digest:
                raise Blocked(f"{path} differs from its registered sha256; {label} not launched")
        ident = self.identity()
        if not same_sources(ident, self.start_identity):
            raise Blocked(f"source identity drift before {label}; not launched")
        if ident["git_head"] != self.start_identity["git_head"]:
            note = (f"HEAD {ident['git_head']} at {label} (start {self.start_identity['git_head']}); "
                    "identity paths unchanged")
            if note not in self.notes:
                self.notes.append(note)
        return ident

    def _preflight(self, label: str, heavy: bool, sha_checks: dict | None, precondition=None) -> dict:
        while True:
            if heavy:
                self.wait_for_ac(label)
            ident = self._check_sources(label, sha_checks)
            if precondition is not None:
                precondition()
            if not heavy or on_ac(self.power(), self.marker):
                return ident
            self.log(f"power left AC between the wait and the launch of {label}; waiting again")

    def _go_problem(self, birth, label: str, heavy: bool, sha_checks: dict | None, precondition=None):
        """(kind, reason) that forbids the go, or None. Checked after the spawn is recorded."""
        if not birth or birth == DEAD:
            return "birth", "child birth time unreadable; its liveness could not be recorded"
        try:
            self._check_sources(label, sha_checks)
            if precondition is not None:
                precondition()
        except Blocked as exc:
            return "identity", str(exc)
        if heavy and not on_ac(self.power(), self.marker):
            return "power", "power left AC before the go"
        return None

    def _spawn_once(self, stage: str, label: str, argv: list, heavy: bool, sha_checks, precondition=None) -> dict:
        self.n_spawn += 1
        safe = label.replace("/", "__")
        log_path = self.session_dir / "logs" / f"{safe}.{self.n_spawn}.log"
        handshake = self.session_dir / "handshakes" / f"{self.n_spawn}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handshake.parent.mkdir(parents=True, exist_ok=True)
        pending = {"label": label, "argv": argv, "log": self.rel(log_path),
                   "handshake": self.rel(handshake), "since_utc": utc()}
        self.ambiguous = pending                       # fail closed from here until a waited exit
        self.update_lease(state="STARTING_CHILD", stage=stage, child=None, pending_child=pending)
        r, w = os.pipe()
        problem = None
        try:
            with open(log_path, "x") as logf:
                try:
                    proc = self.spawn(argv, logf, r, handshake)
                finally:
                    os.close(r)
                birth = self.birth(proc.pid)
                self.child = {"pid": proc.pid, "birth": None if birth == DEAD else birth, "argv": argv,
                              "label": label, "log": self.rel(log_path), "handshake": self.rel(handshake),
                              "spawned_utc": utc()}
                self.update_lease(state="CHILD_SPAWNED", child=self.child)
                problem = self._go_problem(birth, label, heavy, sha_checks, precondition)
                if not problem:
                    self.update_lease(state="RUNNING")  # durable before the child can receive its go
                try:
                    os.write(w, b"abort" if problem else b"go")
                except OSError as exc:              # launcher already gone: the command never ran
                    problem = problem or ("launcher", f"launcher unreachable before the go: {exc}")
                os.close(w)
                w = None
                if problem:
                    self.log(f"abort {label} (pid {proc.pid}): {problem[1]}")
                else:
                    self.log(f"start {label} pid {proc.pid}: {' '.join(argv)}")
                code = proc.wait()
        finally:
            if w is not None:
                os.close(w)                         # EOF: a launcher without its go aborts
        child = {**self.child, "exit_code": code, "waited": True, "went": problem is None,
                 "abort": None if problem is None else {"kind": problem[0], "reason": problem[1]},
                 "ended_utc": utc(), "log_sha256": sha256_file(log_path),
                 "handshake_sha256": sha256_file(handshake)}
        self.child = self.ambiguous = None             # waited-exit evidence exists now
        self.update_lease(state="RECORDING", child=None, pending_child=None, last_child=child)
        self.log(f"end {label} exit={code}{'' if problem is None else ' (aborted before the go)'}")
        return child

    def launch(self, stage: str, label: str, argv: list, heavy: bool = True, sha_checks: dict | None = None,
               precondition=None) -> dict:
        while True:
            self._preflight(label, heavy, sha_checks, precondition)
            child = self._spawn_once(stage, label, argv, heavy, sha_checks, precondition)
            if child["went"]:
                end, problems = self._identity_after(label)
                return {**child, "heavy": heavy, "identity": dict(self.start_identity),
                        "end_identity": end, "identity_problems": problems}
            if child["abort"]["kind"] != "power":
                raise Blocked(f"{label} not launched: {child['abort']['reason']} "
                              f"(launcher exit {child['exit_code']}, waited)")

    def _identity_after(self, label: str) -> tuple[dict | None, list]:
        try:
            end = self.identity()
        except (Blocked, OSError, ValueError, TypeError) as exc:
            problem = f"identity unreadable after {label}: {exc}"
            self.notes.append(problem)
            return None, [problem]
        problems = [] if same_sources(end, self.start_identity) else [f"source identity drift during {label}"]
        self.notes.extend(problems)
        return end, problems

    def _file_problem(self, path, digest) -> str | None:
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64 \
                or any(c not in "0123456789abcdef" for c in digest):
            return f"missing or malformed file hash: {path!r}"
        candidate = self.root / path
        try:
            candidate.resolve().relative_to(self.root.resolve())
        except (ValueError, OSError):
            return f"evidence path outside root: {path!r}"
        if Path(path).is_absolute() or sha256_file(candidate) != digest:
            return f"evidence {path} missing or changed"
        return None

    def _launch_problems(self, launch, command: list, *, allow_failed_identity=False) -> list:
        if not isinstance(launch, dict):
            return ["launch evidence missing"]
        p = []
        if launch.get("waited") is not True or launch.get("went") is not True \
                or type(launch.get("exit_code")) is not int:
            p.append("no waited-exit evidence for the registered command")
        if launch.get("argv") != command:
            p.append("launch command differs from registration")
        for path_key, hash_key in (("log", "log_sha256"), ("handshake", "handshake_sha256")):
            problem = self._file_problem(launch.get(path_key), launch.get(hash_key))
            if problem:
                p.append(problem)
        handshake = (read_json(self.root / launch["handshake"])
                     if self._file_problem(launch.get("handshake"), launch.get("handshake_sha256")) is None else None)
        normalized = [str(self.root / a) if i == 0 and "/" in a and not os.path.isabs(a) else a
                      for i, a in enumerate(command)]
        if not isinstance(handshake, dict) or handshake.get("pid") != launch.get("pid") \
                or handshake.get("birth") != launch.get("birth") or handshake.get("command") != normalized \
                or type(launch.get("pid")) is not int or not launch.get("birth") or launch.get("birth") == DEAD:
            p.append("launcher handshake identity/command missing or mismatched")
        if not same_sources(launch.get("identity"), self.start_identity):
            p.append("launch source identity differs from current sources")
        identity_failed = not same_sources(launch.get("end_identity"), launch.get("identity"))
        recorded_failure = bool(launch.get("identity_problems"))
        if identity_failed != recorded_failure:
            p.append("end identity evidence and recorded failure disagree")
        if (identity_failed or recorded_failure) and not allow_failed_identity:
            p.append("end source identity missing or changed")
        return p

    # ---------------------------------------------------------- benchmark queue
    def validate_result(self, run_dir: Path, spec: dict) -> dict:
        val = self.reg["benchmark"]["validation"]
        v = {"run_dir": self.rel(run_dir), "problems": [], "timing_problems": [],
             "files": {self.rel(run_dir / n): sha256_file(run_dir / n) for n in RUN_FILES}}
        result, manifest = read_json(run_dir / "result.json"), read_json(run_dir / "manifest.json")
        if not isinstance(result, dict):
            v["state"] = "NO_RESULT"
            return v
        verdict = result.get("verdict")
        if verdict not in val["terminal_verdicts"] or verdict not in ELIGIBLE:
            v["state"] = "MALFORMED"
            v["problems"].append(f"result verdict {verdict!r}")
            return v
        if not isinstance(manifest, dict):
            v["problems"].append("manifest.json missing or malformed")
        if any(d is None for d in v["files"].values()):
            v["problems"].append("manifest/result/progress evidence missing or unreadable")
        for key in ("arm", "variant", "workload", "workers"):
            for label, doc in (("manifest", manifest), ("result", result)):
                if isinstance(doc, dict) and doc.get(key) != spec[key]:
                    v["problems"].append(f"{label} {key} {doc.get(key)!r} != spec {spec[key]!r}")
        for label, doc in (("manifest", manifest), ("result", result)):
            if isinstance(doc, dict) and (doc.get("repeats") != val["repeats"]
                                         or doc.get("warmups") != val["warmups"]):
                v["problems"].append(f"{label} repeats/warmups {doc.get('repeats')}/{doc.get('warmups')}")
        powers = {}
        try:
            for line in (run_dir / "progress.jsonl").read_text().splitlines():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("progress entry is not an object")
                if "repeat" in obj:
                    rep = obj["repeat"]
                    if type(rep) is not int or rep in powers or not 0 <= rep < val["repeats"] + val["warmups"]:
                        raise ValueError("invalid/duplicate repeat")
                    powers[rep] = obj.get("power")
        except (OSError, ValueError, TypeError):
            v["problems"].append("progress.jsonl missing or malformed")
        if v["problems"]:
            v["state"] = "MALFORMED"
            return v
        v["state"] = v["verdict"] = verdict
        comps = result.get("comparisons")
        n_runs = val["repeats"] + val["warmups"]
        comps_ok = isinstance(comps, list) and len(comps) == n_runs and all(
            isinstance(c, dict) and c.get("repeat") == rep
            and isinstance(c.get("internal"), dict) and c["internal"].get("match") is True
            and (c.get("reference") is None or (isinstance(c["reference"], dict)
                                                and c["reference"].get("match") is True))
            for rep, c in enumerate(comps))
        timings = result.get("timings_s")
        if not (isinstance(timings, list) and len(timings) == val["timing_samples"]
                and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in timings)):
            v["timing_problems"].append(f"timing samples {timings!r}")
        if result.get("registered_protocol") is not True or manifest.get("registered_protocol") is not True:
            v["timing_problems"].append("registered_protocol is not true")
        if not comps_ok:
            v["timing_problems"].append("comparison evidence missing, malformed or not matching")
        for rep in range(n_runs):
            p = powers.get(rep)
            if not isinstance(p, str):
                v["timing_problems"].append(f"repeat {rep}: no power reading")
            elif not on_ac(p, self.marker):
                v["timing_problems"].append(f"repeat {rep}: {p}")
        if verdict != "PASS":
            v["timing_problems"].append(f"verdict {verdict}")
        v["comparisons_match"] = comps_ok
        v["numerical_reference_valid"] = verdict in ("PASS", "INVALID_POWER") and comps_ok
        v["timing_valid"] = not v["timing_problems"]
        v["median_s"] = result.get("median_s")
        return v

    def _flags_for(self, qname: str, name: str) -> list:
        return [f for f in self.reg.get("flags", []) if f["record"] == f"{qname}/{name}"]

    def _plan_specs(self, qname: str) -> list[dict]:
        q = self.reg["queues"][qname]
        plan = (self.root / q["plan"]).read_bytes()
        if sha256_bytes(plan) != q["plan_sha256"]:
            raise Blocked(f"{q['plan']} differs from its registered hash")
        specs = parse_specs(plan.decode().splitlines())
        if len(specs) != q["n_specs"] or len({run_name(s) for s in specs}) != len(specs):
            raise Blocked(f"{q['plan']} has {len(specs)} specs (registered {q['n_specs']}) or duplicate names")
        return specs

    def _benchmark_sha_checks(self) -> dict:
        b = self.reg["benchmark"]
        if sha256_file(self.root / b["script"]) != b["script_sha256"]:
            raise Blocked(f"{b['script']} differs from the registered script_sha256; no benchmark work")
        return {b["script"]: b["script_sha256"]}

    def _benchmark_command(self, qname: str, spec: dict) -> list:
        b = self.reg["benchmark"]
        fields = {**spec, "out_dir": self.reg["queues"][qname]["out_dir"]}
        return [a.format(**fields) for a in b["command"]] + ([b["native_flag"]] if spec["native"] else [])

    def _historical_evidence(self, qname: str, name: str) -> dict | None:
        hist = self.reg.get("historical_records", {}).get(qname, {}).get(name)
        if hist is None:
            return None
        out = Path(self.reg["queues"][qname]["out_dir"])
        return {str(out / (f if "/" in f else f"{name}/{f}")): d for f, d in hist.items()}

    @staticmethod
    def _unlinked_flag(qname: str, name: str) -> str:
        return (f"{qname}/{name}: terminal result found without a spec record or registered "
                "identity (its launch log is not linked)")

    def spec_record_problems(self, qname: str, spec: dict, rec) -> list:
        """Everything a spec record must satisfy to be trusted now, evidence re-hashed."""
        if not isinstance(rec, dict):
            return ["spec record unreadable or malformed"]
        p = []
        for key, want in (("registration_id", self.reg["id"]), ("registration_sha256", self.reg_sha),
                          ("queue", qname), ("run_name", run_name(spec)), ("spec", spec["text"]),
                          ("execution_terminal", True)):
            if rec.get(key) != want:
                p.append(f"{key} {rec.get(key)!r} != {want!r}")
        state = rec.get("state")
        if state not in ELIGIBLE + BLOCKING:
            p.append(f"state {state!r}")
        if rec.get("completion_eligible") is not (state in ELIGIBLE):
            p.append(f"completion_eligible {rec.get('completion_eligible')!r} inconsistent with {state!r}")
        if not same_sources(rec.get("identity"), self.start_identity):
            p.append("spec record source identity differs from current sources")
        name = run_name(spec)
        run_dir = self.root / self.reg["queues"][qname]["out_dir"] / name
        v = self.validate_result(run_dir, spec)
        required = {path for path, digest in v["files"].items() if digest is not None}
        if state in ELIGIBLE:
            if v["state"] != state:
                p.append(f"result content state {v['state']!r} != record {state!r}")
            for key in ("timing_valid", "numerical_reference_valid", "comparisons_match", "timing_problems",
                        "median_s", "problems"):
                if rec.get(key) != v.get(key):
                    p.append(f"recorded {key} differs from result validation")
        origin = rec.get("origin")
        hist = self._historical_evidence(qname, name)
        expected_prov = []
        if origin == "this_session":
            launch = rec.get("launch")
            p += self._launch_problems(launch, self._benchmark_command(qname, spec),
                                       allow_failed_identity=state in BLOCKING)
            if isinstance(launch, dict):
                required.update(launch.get(k) for k in ("log", "handshake") if isinstance(launch.get(k), str))
                if state in ELIGIBLE and launch.get("exit_code") != (0 if state == "PASS" else 1):
                    p.append("benchmark exit disagrees with terminal verdict")
        elif origin == "pre_existing_registered":
            if hist is None or rec.get("launch") is not None:
                p.append("registered historical origin has no matching registration or has a launch")
            else:
                required.update(hist)
                for path, digest in hist.items():
                    evidence_map = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
                    if evidence_map.get(path) != digest:
                        p.append(f"registered historical evidence missing or mismatched: {path}")
        elif origin == "found_without_record":
            expected_prov = [self._unlinked_flag(qname, name)]
            if hist is not None or rec.get("launch") is not None or state not in ELIGIBLE:
                p.append("unlinked result origin inconsistent with registration/evidence")
        else:
            p.append(f"unknown origin {origin!r}")
        if rec.get("registered_flags") != self._flags_for(qname, name) \
                or rec.get("provenance_flags") != expected_prov:
            p.append("registered/provenance flags missing or changed")
        evidence = rec.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            p.append("no evidence hashes")
        else:
            if set(evidence) != required:
                p.append("evidence file set differs from required run/origin evidence")
            for path, digest in evidence.items():
                problem = self._file_problem(path, digest)
                if problem:
                    p.append(problem)
        if state in ELIGIBLE and isinstance(evidence, dict):
            run_dir = f"{self.reg['queues'][qname]['out_dir']}/{run_name(spec)}"
            missing = [n for n in RUN_FILES if f"{run_dir}/{n}" not in evidence]
            if missing:
                p.append(f"eligible record lacks run evidence {missing}")
        return p

    def check_queue_completion(self, qname: str) -> tuple[dict, str]:
        """The single strict validator of a queue completion record (resume, owner release, A2)."""
        q = self.reg["queues"][qname]
        path = self.records / f"{qname}.completion.json"
        if not path.is_file():
            raise Blocked(f"queue {qname} has no completion record")
        digest, doc = sha256_file(path), read_json(path)
        if not isinstance(doc, dict):
            raise Blocked(f"{self.rel(path)} is malformed")
        p = []
        for key, want in (("registration_id", self.reg["id"]), ("registration_sha256", self.reg_sha),
                          ("queue", qname), ("plan", q["plan"]), ("plan_sha256", q["plan_sha256"]),
                          ("out_dir", q["out_dir"]), ("terminal", True)):
            if doc.get(key) != want:
                p.append(f"{key} {doc.get(key)!r} != {want!r}")
        if doc.get("status") not in QUEUE_DONE:
            p.append(f"status {doc.get('status')!r}")
        if not same_sources(doc.get("identity"), self.start_identity):
            p.append("completion source identity differs from current sources")
        try:
            specs = self._plan_specs(qname)
        except Blocked as exc:
            raise Blocked(f"{self.rel(path)}: {exc}") from None
        expected = {run_name(s): s for s in specs}
        if doc.get("n_specs") != len(expected):
            p.append("n_specs differs from registered plan")
        recs = doc.get("spec_records")
        if not isinstance(recs, dict) or set(recs) != set(expected):
            p.append("spec_records names differ from the registered plan")
            recs = recs if isinstance(recs, dict) else {}
        loaded = {}
        for name, d in recs.items():
            rec_path = self.records / qname / f"{name}.json"
            if name not in expected or sha256_file(rec_path) != d:
                p.append(f"spec record {name} missing or changed")
                continue
            rec = read_json(rec_path)
            p += [f"{name}: {x}" for x in self.spec_record_problems(qname, expected[name], rec)]
            if isinstance(rec, dict):
                loaded[name] = rec
                if rec.get("completion_eligible") is not True:
                    p.append(f"{name}: state {rec.get('state')} is not completion-eligible")
        if not p and len(loaded) == len(expected):
            summary = self._summary(loaded)
            for key in summary:
                if doc.get(key) != summary[key]:
                    p.append(f"{key} {doc.get(key)!r} != recomputed {summary[key]!r}")
        if p:
            raise Blocked(f"{self.rel(path)} invalid: " + "; ".join(p[:12]))
        return doc, digest

    @staticmethod
    def _summary(recs: dict) -> dict:
        states = {}
        for r in recs.values():
            states[r["state"]] = states.get(r["state"], 0) + 1
        all_pass = all(r["state"] == "PASS" for r in recs.values())
        timing = all(r.get("timing_valid") is True for r in recs.values())
        flags = [f for r in recs.values() for f in r.get("registered_flags", [])]
        prov = [f for r in recs.values() for f in r.get("provenance_flags", [])]
        valid = all_pass and timing and not flags and not prov
        return {"states": states, "all_pass": all_pass, "all_timing_valid": timing,
                "registered_flags": flags, "provenance_flags": prov,
                "scientifically_valid_benchmark": valid,
                "status": "TERMINAL_COMPLETE" if valid else "TERMINAL_COMPLETE_WITH_FLAGS",
                "timing_invalid": sorted(n for n, r in recs.items() if r.get("timing_valid") is not True)}

    def _spec(self, qname: str, q: dict, spec: dict, stage: str, sha_checks: dict) -> str:
        name = run_name(spec)
        out_dir = self.root / q["out_dir"]
        run_dir = out_dir / name
        rec_path = self.records / qname / f"{name}.json"
        if rec_path.exists():
            problems = self.spec_record_problems(qname, spec, read_json(rec_path))
            if problems:
                raise Blocked(f"spec record {self.rel(rec_path)} invalid: {problems}")
            return sha256_file(rec_path)          # terminal: never rerun, even if it blocks completion
        launch, prov = None, []
        hist = self._historical_evidence(qname, name)
        if run_dir.exists():
            v = self.validate_result(run_dir, spec)
            if v["state"] not in ELIGIBLE:
                raise Blocked(f"{name}: occupied result directory without a valid terminal result "
                              f"({v['state']}: {v['problems']}); never an implicit rerun")
            evidence = dict(v["files"])
            if hist is not None:
                hist_paths = hist
                bad = [p for p, d in hist_paths.items() if sha256_file(self.root / p) != d]
                if bad:
                    raise Blocked(f"historical record {name} differs from its registered identity: {bad}")
                evidence.update(hist_paths)
                origin = "pre_existing_registered"
            else:
                origin = "found_without_record"
                prov.append(self._unlinked_flag(qname, name))
            self.log(f"{qname}/{name}: existing terminal result {v['state']} ({origin}); skipped, "
                     "benchmark.py not invoked")
        else:
            argv = self._benchmark_command(qname, spec)
            self.update_lease(stage=f"{stage}:{name}")
            launch = self.launch(stage, f"{qname}/{name}", argv, sha_checks=sha_checks,
                                 precondition=lambda: self._check_existing_queue_records(qname))
            self.benchmark_children.append({"pid": launch["pid"], "birth": launch["birth"]})
            origin = "this_session"
            evidence = {launch["log"]: launch["log_sha256"], launch["handshake"]: launch["handshake_sha256"]}
            if run_dir.exists():
                v = self.validate_result(run_dir, spec)
                if v["state"] not in ELIGIBLE:
                    v["problems"].append(f"benchmark.py exit {launch['exit_code']} left {v['state']}")
                    v["state"] = "ERROR_PARTIAL"
                evidence.update({p: d for p, d in v["files"].items() if d is not None})
            else:
                v = {"state": "REFUSED", "problems": [
                    f"benchmark.py exit {launch['exit_code']} without creating {self.rel(run_dir)}"],
                    "timing_problems": ["no result"]}
            v.setdefault("numerical_reference_valid", False)
            v.setdefault("timing_valid", False)
            launch_problems = self._launch_problems(launch, argv)
            try:
                self._check_existing_queue_records(qname)
            except Blocked as exc:
                launch_problems.append(str(exc))
            if v["state"] in ELIGIBLE and launch["exit_code"] != (0 if v["state"] == "PASS" else 1):
                launch_problems.append("benchmark exit disagrees with terminal verdict")
            if launch_problems:
                v["problems"].extend(launch_problems)
                v["state"] = "ERROR_PARTIAL"
                v["timing_valid"] = v["numerical_reference_valid"] = False
        v.pop("files", None)
        rec = {"registration_id": self.reg["id"], "registration_sha256": self.reg_sha, "queue": qname,
               "spec": spec["text"], "run_name": name, "origin": origin, "session": self.session,
               "identity": self.start_identity,
               "execution_terminal": True, **v, "completion_eligible": v["state"] in ELIGIBLE,
               "evidence": evidence, "registered_flags": self._flags_for(qname, name),
               "provenance_flags": prov, "launch": launch, "recorded_utc": utc()}
        write_once(rec_path, rec)
        if not rec["completion_eligible"]:
            self.log(f"{qname}/{name}: {v['state']} recorded; it blocks queue completion and is not retried")
        return sha256_file(rec_path)

    def _check_existing_queue_records(self, qname: str) -> None:
        for spec in self._plan_specs(qname):
            path = self.records / qname / f"{run_name(spec)}.json"
            if path.exists():
                problems = self.spec_record_problems(qname, spec, read_json(path))
                if problems:
                    raise Blocked(f"existing queue evidence changed before launch: {problems}")

    def run_queue(self, qname: str, stage: str) -> dict:
        q = self.reg["queues"][qname]
        sha_checks = self._benchmark_sha_checks()
        specs = self._plan_specs(qname)
        completion = self.records / f"{qname}.completion.json"
        if completion.exists():
            doc, digest = self.check_queue_completion(qname)
            return {"state": doc["status"], "completion": self.rel(completion),
                    "completion_sha256": digest, "origin": "earlier_session"}
        hashes, blockers = {}, []
        for spec in specs:
            try:
                hashes[run_name(spec)] = self._spec(qname, q, spec, stage, sha_checks)
            except Blocked as exc:
                blockers.append({"run_name": run_name(spec), "reason": str(exc)})
                self.log(f"BLOCKED {qname}/{run_name(spec)}: {exc}")
        recs = {n: read_json(self.records / qname / f"{n}.json") for n in hashes}
        ineligible = sorted(n for n, r in recs.items() if not (isinstance(r, dict) and r.get("completion_eligible")))
        if blockers or ineligible:
            return {"state": "BLOCKED", "blockers": blockers,
                    "ineligible_records": {n: (recs[n] or {}).get("state") for n in ineligible},
                    "n_records": len(hashes), "n_specs": len(specs),
                    "note": "no queue completion record: a spec is blocked, refused or partial"}
        summary = self._summary(recs)
        doc = {"registration_id": self.reg["id"], "registration_sha256": self.reg_sha, "queue": qname,
               "plan": q["plan"], "plan_sha256": q["plan_sha256"], "out_dir": q["out_dir"],
               "n_specs": len(specs), "session": self.session, "identity": self.start_identity,
               "terminal": True, **summary,
               "note": "terminal completion: every spec reached a completion-eligible terminal record; "
                       "this is not by itself a scientifically valid benchmark",
               "spec_records": hashes, "completed_utc": utc()}
        write_once(completion, doc)
        doc, digest = self.check_queue_completion(qname)
        self.log(f"{qname}: {doc['status']} {doc['states']}")
        return {"state": doc["status"], "completion": self.rel(completion),
                "completion_sha256": digest, "origin": "this_session"}

    def _owner_release_path(self) -> Path:
        return self.records / "benchmark_owner_released.json"

    def check_owner_release(self) -> dict:
        """Validate the benchmark-owner release record against freshly validated completions."""
        digests = {q: self.check_queue_completion(q)[1] for q in self.reg["queues"]}
        path = self._owner_release_path()
        doc = read_json(path)
        if not isinstance(doc, dict):
            raise Blocked(f"{self.rel(path)} missing or malformed")
        if doc.get("registration_id") != self.reg["id"] or doc.get("registration_sha256") != self.reg_sha \
                or doc.get("completions") != digests or doc.get("benchmark_children_alive") is not False \
                or not same_sources(doc.get("identity"), self.start_identity):
            raise Blocked(f"{self.rel(path)} does not match the validated queue completion records")
        return doc

    def release_benchmark_owner(self) -> dict:
        try:
            digests = {q: self.check_queue_completion(q)[1] for q in self.reg["queues"]}
        except Blocked as exc:
            return {"state": "NOT_RELEASED", "reason": str(exc)}
        path = self._owner_release_path()
        if path.exists():
            self.check_owner_release()
            return {"state": "RELEASED", "record": self.rel(path), "record_sha256": sha256_file(path),
                    "origin": "earlier_session"}
        alive = [c for c in self.benchmark_children
                 if liveness(c["pid"], c["birth"], self.birth) not in ("dead", "reused")]
        if self.child is not None or self.ambiguous is not None or alive:
            return {"state": "NOT_RELEASED", "reason": f"benchmark child not provably ended: {alive}"}
        write_once(path, {"registration_id": self.reg["id"], "registration_sha256": self.reg_sha,
                          "session": self.session, "owner_pid": os.getpid(),
                          "identity": self.start_identity,
                          "owner_birth": self.lease["owner_birth"], "completions": digests,
                          "benchmark_children": self.benchmark_children,
                          "benchmark_children_alive": False, "released_utc": utc()})
        self.check_owner_release()
        return {"state": "RELEASED", "record": self.rel(path), "record_sha256": sha256_file(path),
                "origin": "this_session"}

    # ------------------------------------------------------------ other stages
    def _report_evidence(self, st: dict):
        """Report status/hash and, beside it, hashes.json re-verified entry by entry."""
        if not st.get("report"):
            return None
        rep_path = self.root / st["report"]
        rep = read_json(rep_path)
        hashes_path = rep_path.parent / "hashes.json"
        entries = read_json(hashes_path)
        bad = None
        if isinstance(entries, dict) and entries:
            bad = []
            for n, d in entries.items():
                if not isinstance(n, str) or Path(n).name != n \
                        or self._file_problem(self.rel(rep_path.parent / n), d):
                    bad.append(str(n))
            actual = {p.name for p in rep_path.parent.iterdir() if p.is_file() and p.name != "hashes.json"}
            if set(entries) != actual or "report.json" not in entries:
                bad.append("hash manifest file set incomplete")
            if isinstance(rep, dict) and rep.get("status") == "PASS" and st["id"] == "track4_diagnostic":
                required = {"config.json", "environment.json", "run_log.txt", "report.json", "report.md",
                            "turbine_score.json", "manufactured_scores.json", "nozzle_scores.json"}
                turbine = rep.get("turbine")
                checkpoint = turbine.get("checkpoint") if isinstance(turbine, dict) else None
                if isinstance(checkpoint, str) and Path(checkpoint).name == checkpoint:
                    required.add(checkpoint)
                else:
                    bad.append("turbine checkpoint not identified")
                bad.extend(sorted(required - set(entries)))
        return {"path": st["report"], "sha256": sha256_file(rep_path),
                "status": rep.get("status") if isinstance(rep, dict) else None,
                "hashes": self.rel(hashes_path), "hashes_sha256": sha256_file(hashes_path),
                "hashes_verified": bad == [], "hashes_mismatched": bad}

    def stage_record_problems(self, st: dict, rec) -> list:
        """Everything a cached command-stage record must satisfy to be trusted now."""
        if not isinstance(rec, dict):
            return ["stage record unreadable or malformed"]
        p = []
        for key, want in (("registration_id", self.reg["id"]), ("registration_sha256", self.reg_sha),
                          ("stage", st["id"]), ("command", st["command"])):
            if rec.get(key) != want:
                p.append(f"{key} {rec.get(key)!r} != {want!r}")
        if not same_sources(rec.get("identity"), self.start_identity):
            p.append("recorded source identity differs from the current sources (stale)")
        state = rec.get("state")
        if state not in STAGE_STATES:
            p.append(f"state {state!r}")
        launch = rec.get("launch")
        p += self._launch_problems(launch, st["command"], allow_failed_identity=state != "PASS")
        if not isinstance(launch, dict) or launch.get("exit_code") != rec.get("exit_code"):
            p.append("launch and stage exits disagree")
        for kind in ("outputs", "retained"):
            evidence = rec.get(kind)
            expected = st.get("expected_outputs" if kind == "outputs" else "retain_unchanged", [])
            if not isinstance(evidence, dict) or set(evidence) != set(expected):
                p.append(f"recorded {kind} set differs from registration")
                continue
            for path, digest in evidence.items():
                if digest is None and state != "PASS" and kind == "outputs" and sha256_file(self.root / path) is None:
                    continue                  # a terminal failure may truthfully have absent outputs
                problem = self._file_problem(path, digest)
                if problem:
                    p.append(problem)
        if st.get("report"):
            now, then = self._report_evidence(st), rec.get("report")
            if not isinstance(then, dict) or any(now[k] != then.get(k) for k in
                                                 ("path", "sha256", "status", "hashes_sha256", "hashes_verified")):
                p.append("report or hashes.json evidence missing or changed")
            if state == "PASS" and (now["status"] != "PASS" or now["hashes_verified"] is not True
                                    or now["sha256"] is None or now["hashes_sha256"] is None):
                p.append("PASS without passing complete report/hash evidence")
        if state == "PASS" and st.get("expected_json_keys"):
            outputs = st.get("expected_outputs", [])
            doc = read_json(self.root / outputs[0]) if outputs else None
            if not isinstance(doc, dict) or any(k not in doc for k in st["expected_json_keys"]):
                p.append("PASS output lacks required JSON content")
        if state == "PASS" and (rec.get("exit_code") != 0 or rec.get("problems")):
            p.append("PASS without exit 0 and complete evidence")
        return p

    def _check_stage_dependencies(self, st: dict, trail: frozenset = frozenset()) -> None:
        if st["id"] in trail:
            raise Blocked("cyclic registered stage dependency")
        for qname in st.get("requires_queue_completions", []):
            self.check_queue_completion(qname)
        if st.get("requires_benchmark_owner_released"):
            self.check_owner_release()
        dep = st.get("requires_stage_pass")
        if dep:
            ok, why = self._stage_pass(dep, trail | {st["id"]})
            if not ok:
                raise Blocked(f"{dep} did not pass ({why}); not run")

    def _stage_pass(self, dep_id: str, trail: frozenset = frozenset()) -> tuple[bool, str]:
        dep = next((s for s in self.reg["stages"] if s["id"] == dep_id), None)
        rec = read_json(self.records / "stages" / f"{dep_id}.json")
        if dep is None or rec is None:
            return False, "no record"
        problems = self.stage_record_problems(dep, rec)
        try:
            self._check_stage_dependencies(dep, trail)
        except Blocked as exc:
            problems.append(str(exc))
        if problems:
            return False, f"record not trusted: {problems}"
        return rec["state"] == "PASS", rec["state"]

    def run_command_stage(self, st: dict, results: dict) -> dict:
        sid = st["id"]
        rec_path = self.records / "stages" / f"{sid}.json"
        self._check_stage_dependencies(st)
        if rec_path.exists():
            problems = self.stage_record_problems(st, read_json(rec_path))
            if problems:
                raise Blocked(f"{self.rel(rec_path)} invalid (never rerun): {problems}")
            rec = read_json(rec_path)
            return {"state": rec["state"], "record": self.rel(rec_path), "record_sha256": sha256_file(rec_path),
                    "origin": "earlier_session"}
        present = [p for p in st.get("must_not_exist", []) if (self.root / p).exists()]
        if present:
            raise Blocked(f"write-once outputs already exist: {present}; never rerun")
        retained = {p: sha256_file(self.root / p) for p in st.get("retain_unchanged", [])}
        if any(d is None for d in retained.values()):
            raise Blocked("registered retained file missing or unreadable; not run")
        def precondition():
            self._check_stage_dependencies(st)
            if any((self.root / p).exists() for p in st.get("must_not_exist", [])):
                raise Blocked("write-once outputs appeared before launch")
            if any(self._file_problem(p, d) for p, d in retained.items()):
                raise Blocked("retained file changed before launch")
        launch = self.launch(sid, f"stages/{sid}", st["command"], heavy=st.get("heavy", True),
                             precondition=precondition)
        code = launch["exit_code"]
        outputs = {p: sha256_file(self.root / p) for p in st.get("expected_outputs", [])}
        problems = self._launch_problems(launch, st["command"])
        try:
            self._check_stage_dependencies(st)
        except Blocked as exc:
            problems.append(f"dependency changed during command: {exc}")
        problems += [f"missing output {p}" for p, d in outputs.items() if d is None]
        keys = st.get("expected_json_keys")
        if keys and outputs and next(iter(outputs.values())):
            doc = read_json(self.root / next(iter(outputs)))
            problems += [f"{next(iter(outputs))} lacks {k!r}" for k in keys
                         if not isinstance(doc, dict) or k not in doc]
        problems += [f"retained file changed: {p}" for p, d in retained.items()
                     if sha256_file(self.root / p) != d]
        report = self._report_evidence(st)
        if report is not None:
            if report["status"] != "PASS":
                problems.append(f"report status {report['status']!r}")
            if not report["hashes_verified"]:
                problems.append(f"hashes.json not verified: {report['hashes_mismatched']}")
        created = any((self.root / p).exists() for p in st.get("must_not_exist", []))
        if code != 0 and st.get("must_not_exist") and not created:
            state = "REFUSED"          # nothing written; still never retried automatically
        elif code == 0 and not problems:
            state = "PASS"
        else:
            state = "FAIL"
            if code == 0:
                problems.append("exit 0 but the evidence is incomplete")
        rec = {"registration_id": self.reg["id"], "registration_sha256": self.reg_sha, "stage": sid,
               "session": self.session, "command": st["command"], "identity": launch["identity"],
               "state": state, "exit_code": code, "launch": launch, "outputs": outputs,
               "retained": retained, "report": report, "problems": problems, "recorded_utc": utc()}
        write_once(rec_path, rec)
        validation = self.stage_record_problems(st, rec)
        if validation:
            raise Blocked(f"fresh stage evidence invalid (never rerun): {validation}")
        return {"state": state, "record": self.rel(rec_path), "record_sha256": sha256_file(rec_path),
                "exit_code": code, "problems": problems}

    # ------------------------------------------------------------------ chain
    def run_chain(self) -> dict:
        results = {}
        last_queue = max(i for i, s in enumerate(self.reg["stages"]) if s["kind"] == "benchmark_queue")
        for i, st in enumerate(self.reg["stages"]):
            try:
                if st["kind"] == "benchmark_queue":
                    res = self.run_queue(st["queue"], st["id"])
                else:
                    res = self.run_command_stage(st, results)
            except Blocked as exc:
                res = {"state": "BLOCKED", "reason": str(exc)}
            results[st["id"]] = res
            self.log(f"stage {st['id']}: {res['state']}")
            if i == last_queue:
                try:
                    results["benchmark_owner_release"] = self.release_benchmark_owner()
                except Blocked as exc:
                    results["benchmark_owner_release"] = {"state": "NOT_RELEASED", "reason": str(exc)}
        states = [results[s["id"]]["state"] for s in self.reg["stages"]]
        if all(s in ("PASS", "TERMINAL_COMPLETE") for s in states):
            status = "COMPLETE"
        elif "BLOCKED" in states:
            status = "STOPPED_WITH_BLOCKERS"
        else:
            status = "FINISHED_WITH_FLAGS_OR_FAILURES"
        return {"status": status, "stages": results}

    def write_session_record(self, outcome: dict, kind: str = "chain") -> str:
        path = self.records / f"{kind}.{self.session}.json"
        write_once(path, {"registration_id": self.reg["id"], "registration_sha256": self.reg_sha,
                          "session": self.session, "identity": self.start_identity,
                          **outcome, "notes": self.notes, "finished_utc": utc()})
        return self.rel(path)


def validate_terminal_context(root: Path, registration: str = REGISTRATION, *, expected_identity=None,
                              require_idle: bool = True, allow_active_stage: str | None = None) -> dict:
    """Read-only record gate for consumers of the main recovery workflow.

    An idle gate needs a terminal chain with validated record references. The
    only active exception is the actual recorded full_pytest child; it still
    needs valid benchmark completion, A2 and validation-build evidence. No
    process-name or ancestor matching is used and no lease is recovered here.
    """
    wf = Workflow(root=Path(root), registration=registration)
    wf.start_identity = wf.identity()
    if expected_identity is not None and not same_sources(expected_identity, wf.start_identity):
        raise Blocked("expected source identity differs from the main workflow sources")
    lease = read_json(wf.lease_path) if wf.lease_path.exists() else None
    active = False
    if wf.lease_path.exists():
        if allow_active_stage != "full_pytest" or not isinstance(lease, dict):
            raise Blocked("main workflow lease is present; idle completion is not proven")
        child = lease.get("child")
        stage = next((s for s in wf.reg["stages"] if s["id"] == "full_pytest"), None)
        if lease.get("registration_id") != wf.reg["id"] or lease.get("registration_sha256") != wf.reg_sha \
                or not same_sources(lease.get("identity"), wf.start_identity) \
                or lease.get("stage") != "full_pytest" or lease.get("state") != "RUNNING" \
                or liveness(lease.get("owner_pid"), lease.get("owner_birth"), wf.birth) != "alive" \
                or not isinstance(child, dict) or child.get("pid") != os.getpid() \
                or child.get("pid") == lease.get("owner_pid") \
                or liveness(child.get("pid"), child.get("birth"), wf.birth) != "alive" \
                or stage is None or child.get("argv") != stage["command"]:
            raise Blocked("active exception is not the recorded live full_pytest child")
        handshake = child.get("handshake")
        log = child.get("log")
        prefix = wf.out / "sessions" / str(lease.get("session"))
        for path in (handshake, log):
            if not isinstance(path, str) or Path(path).is_absolute() \
                    or not (wf.root / path).resolve().is_relative_to(prefix.resolve()) \
                    or not (wf.root / path).is_file():
                raise Blocked("active full_pytest child has no matching session handshake/log")
        hs = read_json(wf.root / handshake)
        normalized = [str(wf.root / a) if i == 0 and "/" in a and not os.path.isabs(a) else a
                      for i, a in enumerate(stage["command"])]
        if not isinstance(hs, dict) or hs.get("pid") != child["pid"] or hs.get("birth") != child["birth"] \
                or hs.get("command") != normalized:
            raise Blocked("active full_pytest child handshake mismatched")
        active = True
    elif allow_active_stage is not None and allow_active_stage != "full_pytest":
        raise Blocked("unsupported active stage exception")
    queues = {q: wf.check_queue_completion(q)[0] for q in wf.reg["queues"]}
    release = wf.check_owner_release()
    stages = {}
    for st in wf.reg["stages"]:
        if st["kind"] != "command" or (active and st["id"] == "full_pytest"):
            continue
        path = wf.records / "stages" / f"{st['id']}.json"
        if not path.exists():
            continue
        rec = read_json(path)
        problems = wf.stage_record_problems(st, rec)
        if problems:
            raise Blocked(f"{wf.rel(path)} invalid: {problems}")
        if rec["state"] == "PASS":
            wf._check_stage_dependencies(st)
        stages[st["id"]] = rec
    if active:
        for sid in ("a2_calibration", "validation_build"):
            if sid not in stages:
                raise Blocked(f"active full_pytest requires terminal {sid} evidence")
        if stages["validation_build"]["state"] != "PASS":
            raise Blocked("active full_pytest requires validation_build PASS")
        chain = None
    else:
        candidates = sorted(wf.records.glob("chain.*.json"))
        if not candidates:
            raise Blocked("no terminal main workflow chain record")
        chain = read_json(candidates[-1])
        if not isinstance(chain, dict) or chain.get("registration_id") != wf.reg["id"] \
                or chain.get("registration_sha256") != wf.reg_sha \
                or not same_sources(chain.get("identity"), wf.start_identity) \
                or chain.get("status") not in ("COMPLETE", "FINISHED_WITH_FLAGS_OR_FAILURES", "STOPPED_WITH_BLOCKERS"):
            raise Blocked("terminal chain registration/source/status invalid")
        refs = chain.get("stages")
        expected = {s["id"] for s in wf.reg["stages"]} | {"benchmark_owner_release"}
        if not isinstance(refs, dict) or set(refs) != expected:
            raise Blocked("terminal chain stage set differs from registration")
        states = []
        for st in wf.reg["stages"]:
            ref = refs[st["id"]]
            if not isinstance(ref, dict):
                raise Blocked("terminal chain stage reference malformed")
            state = ref.get("state")
            states.append(state)
            if st["kind"] == "benchmark_queue":
                path = wf.records / f"{st['queue']}.completion.json"
                if state != queues[st["queue"]]["status"] or ref.get("completion") != wf.rel(path) \
                        or wf._file_problem(ref.get("completion"), ref.get("completion_sha256")):
                    raise Blocked("terminal chain queue reference invalid")
            elif state == "BLOCKED":
                if not isinstance(ref.get("reason"), str) or st["id"] in stages:
                    raise Blocked("terminal chain blocked stage evidence inconsistent")
            else:
                path = wf.records / "stages" / f"{st['id']}.json"
                if st["id"] not in stages or state != stages[st["id"]]["state"] \
                        or ref.get("record") != wf.rel(path) \
                        or wf._file_problem(ref.get("record"), ref.get("record_sha256")):
                    raise Blocked("terminal chain command reference invalid")
        owner = refs["benchmark_owner_release"]
        if not isinstance(owner, dict) or owner.get("state") != "RELEASED" \
                or owner.get("record") != wf.rel(wf._owner_release_path()) \
                or wf._file_problem(owner.get("record"), owner.get("record_sha256")):
            raise Blocked("terminal chain owner-release reference invalid")
        status = ("COMPLETE" if all(s in ("PASS", "TERMINAL_COMPLETE") for s in states) else
                  "STOPPED_WITH_BLOCKERS" if "BLOCKED" in states else "FINISHED_WITH_FLAGS_OR_FAILURES")
        if chain["status"] != status:
            raise Blocked("terminal chain status differs from its validated stages")
    return {"registration_id": wf.reg["id"], "registration_sha256": wf.reg_sha,
            "identity": wf.start_identity, "queues": queues, "owner_release": release,
            "stages": stages, "chain": chain, "lease": lease if active else None}


def _status(wf: Workflow, require_idle: bool) -> int:
    lease = read_json(wf.lease_path)
    if wf.lease_path.exists():
        state = liveness(lease.get("owner_pid"), lease.get("owner_birth"), wf.birth) \
            if isinstance(lease, dict) else "malformed"
        print(f"lease: {wf.rel(wf.lease_path)} owner {state}: " + json.dumps(
            {k: lease.get(k) for k in ("session", "owner_pid", "state", "stage", "waiting_for", "child",
                                       "pending_child")}
            if isinstance(lease, dict) else None))
    else:
        print("lease: none held")
    chains = sorted(wf.records.glob("chain.*.json")) if wf.records.exists() else []
    for p in sorted(wf.records.glob("*.json")) if wf.records.exists() else []:
        doc = read_json(p) or {}
        print(f"{wf.rel(p)}: {doc.get('status') or doc.get('state')}")
    if require_idle:
        try:
            validate_terminal_context(wf.root, str(wf.reg_path.relative_to(wf.root)))
        except (Blocked, Refused, OSError, ValueError, KeyError, TypeError) as exc:
            print(f"idle completion not validated: {exc}")
            return 1
        return 0
    return 0


def _raise_exit(signum, _frame):
    sys.exit(128 + signum)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chain", "queue", "status", "recover-stale"):
        p = sub.add_parser(name)
        p.add_argument("--registration", default=REGISTRATION)
        if name == "queue":
            p.add_argument("--out-dir", required=True)
            p.add_argument("specs", nargs="*")
        if name == "status":
            p.add_argument("--require-idle", action="store_true")
        if name == "recover-stale":
            p.add_argument("--reason", required=True)
            p.add_argument("--attest-no-child", default=None,
                           help="recorded operator attestation, only for a lease whose child identity is absent")
    p = sub.add_parser("_child")
    p.add_argument("--handshake", required=True)
    p.add_argument("--go-fd", type=int, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "_child":
        return child_main(a.handshake, a.go_fd, a.command[1:] if a.command[:1] == ["--"] else a.command)
    try:
        wf = Workflow(registration=a.registration)
    except (OSError, ValueError, KeyError, Refused) as exc:
        print(f"REFUSED: registration unusable: {exc}", file=sys.stderr)
        return 2
    if a.cmd == "status":
        return _status(wf, a.require_idle)
    if a.cmd == "recover-stale":
        try:
            print(f"stale lease recorded at {wf.recover_stale(a.reason, a.attest_no_child)}")
            return 0
        except Refused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
    qname = None
    if a.cmd == "queue":
        out = os.path.normpath(os.path.relpath(Path(a.out_dir).resolve(), ROOT))
        try:
            given = parse_specs(a.specs)
        except Blocked as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        for name, q in wf.reg["queues"].items():
            plan = parse_specs((ROOT / q["plan"]).read_text().splitlines())
            if os.path.normpath(q["out_dir"]) == out and [s["text"] for s in plan] == [s["text"] for s in given]:
                qname = name
        if qname is None:
            print("REFUSED: out-dir and specs do not match a registered queue", file=sys.stderr)
            return 2
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_exit)
    try:
        wf.start(a.cmd if qname is None else f"queue {qname}")
    except (Refused, Blocked) as exc:
        print(f"REFUSED (nothing launched): {exc}", file=sys.stderr)
        return 2
    outcome = {"status": "ABORTED"}
    finalized = False
    try:
        if qname is None:
            outcome = wf.run_chain()
        else:
            try:
                res = wf.run_queue(qname, f"queue_{qname}")
            except Blocked as exc:
                res = {"state": "BLOCKED", "reason": str(exc)}
            outcome = {"status": res["state"], "stages": {f"queue_{qname}": res}}
    except BaseException as exc:
        outcome = {"status": "ABORTED", "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                   "traceback": traceback.format_exc()}
        raise
    finally:
        try:
            record = wf.write_session_record(outcome, "chain" if qname is None else "queue_session")
            wf.log(f"outcome {outcome['status']}; record {record}")
            finalized = wf.release(outcome["status"])
        except (LeaseLost, OSError, ValueError, TypeError) as exc:
            print(f"FINALIZATION FAILED (nonzero exit): {exc}", file=sys.stderr)
    if not finalized:
        return 3
    ok = ("COMPLETE",) if qname is None else QUEUE_DONE
    return 0 if outcome["status"] in ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
