#!/usr/bin/env python3
"""
Phase 7 durable job supervisor (P7.5). Shell entry point: ``scripts/run_phase7.sh``.

Every job runs from a FROZEN SOURCE SNAPSHOT: ``git archive <commit>`` extracted
outside the working tree (``SNAPSHOT_ROOT/<commit>``), so an edit to the
working tree can never change code that a running job imports. Jobs write into
the snapshot; on success their registered outputs are PROMOTED into the
repository (write-once: an existing different file is never overwritten).

Per job, under ``outputs/phase7/runs/<queue>/<job>/`` in the repository:
    lock                     O_EXCL lock (pid, start time) while an attempt runs
    attempt_<n>.json         launch record: pid, start, commit, tree, argv, env,
                             config hash, input hashes
    attempt_<n>.log          stdout + stderr of the attempt (never overwritten)
    attempt_<n>.exit.json    exit code and finish time (written by the supervisor
                             when it observes the exit)
    attempt_<n>.interrupted.json   an attempt found without an exit record at
                             restart (supervisor or machine died): evidence kept,
                             the exact registered command is re-run
    COMPLETE.json            written only after exit 0 AND every registered
                             output exists AND the job's completion check passes;
                             records output SHA-256s
    FAILED.json              non-zero exit or failed completion check; dependants
                             do not run; no automatic retry
    GATE_CLOSED.json         a registered scientific gate stopped this job
                             (a result, not an infrastructure failure)

Completion is never inferred from a PID disappearing or from a wrapper's zero
exit alone: downstream jobs and reports read COMPLETE.json and re-verify hashes.

Usage:
    python scripts/phase7_supervisor.py launch --queue calib   # detached, caffeinated
    python scripts/phase7_supervisor.py status
    python scripts/phase7_supervisor.py run --queue calib --commit SHA --repo DIR   # (internal)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent.parent          # repo or snapshot root
RUNS_REL = Path("outputs") / "phase7" / "runs"
PYTHON_REL = Path(".venv") / "bin" / "python"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def write_json_new(path: Path, obj) -> None:
    """Write-once JSON record (refuses to replace evidence)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:
        fh.write(json.dumps(obj, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------
# Job and queue definitions
# --------------------------------------------------------------------------
@dataclass
class Job:
    name: str
    argv: list[str]                          # relative to the snapshot root; argv[0] a script
    outputs: list[str]                       # repo-relative paths promoted on success
    requires: list[tuple[str, str]] = field(default_factory=list)   # (queue, job) COMPLETE
    imports: list[str] = field(default_factory=list)   # repo artifacts copied into the snapshot
    gate: Optional[Callable[[Path], dict]] = None      # evaluated on the repo before launch
    check: Optional[Callable[[Path], dict]] = None     # completion check on the snapshot
    env: dict = field(default_factory=dict)
    threads: int = 1                                   # CPU slots the job occupies

    def config(self) -> dict:
        return {"name": self.name, "argv": self.argv, "outputs": self.outputs,
                "requires": self.requires, "imports": self.imports, "env": self.env,
                "threads": self.threads}

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.config(), sort_keys=True).encode()).hexdigest()


def _p7(*names: str) -> list[str]:
    return [f"outputs/phase7/{n}" for n in names]


def gate_blends(repo: Path) -> dict:
    """P7.2 -> P7.3 dependency: A1 PASS and no B0 escalation (registered)."""
    res = json.loads((repo / "outputs/phase7/holdout_icao_validation_v6.json").read_text())
    return res["blend_gate"]


def check_json_key(rel: str, key: str) -> Callable[[Path], dict]:
    def check(root: Path) -> dict:
        d = json.loads((root / rel).read_text())
        return {"ok": key in d, "detail": f"{rel} has '{key}'"}
    return check


def pinn_jobs() -> list[Job]:
    """30 registered P7.4 runs + the report (definitions from the registration)."""
    reg_path = HERE / "outputs" / "phase7" / "p74_registration.json"
    if not reg_path.exists():
        return []
    reg = json.loads(reg_path.read_text())
    threads = int(reg["compute"]["torch_threads_per_run"])
    jobs = []
    for run in reg["runs"]:
        jobs.append(Job(
            name=run["run_id"],
            argv=["scripts/validation/train_sajben_ma.py", "--run-id", run["run_id"]],
            outputs=[run["checkpoint"], run["record"]],
            env={"P7_TORCH_THREADS": str(threads)},
            threads=threads,
            check=check_json_key(run["record"], "final_checkpoint_sha256"),
        ))
    jobs.append(Job(
        name="p74_report",
        argv=["scripts/validation/report_sajben_ma.py"],
        outputs=reg["report_outputs"],
        requires=[("pinn", r["run_id"]) for r in reg["runs"]],
        imports=[p for r in reg["runs"] for p in (r["checkpoint"], r["record"])],
        threads=1,
    ))
    return jobs


def pinn_max_threads() -> int:
    reg_path = HERE / "outputs" / "phase7" / "p74_registration.json"
    if reg_path.exists():
        return int(json.loads(reg_path.read_text())["compute"]["max_threads"])
    return int(os.environ.get("P7_MAX_THREADS", "10"))


def queues_unfinished(repo: Path, names: list[str]) -> bool:
    """True while any job of the named queues is not in a terminal state
    (a queue whose supervisor has not recorded its jobs yet counts as unfinished)."""
    for q in names:
        sup = repo / RUNS_REL / q / "supervisor.json"
        if not sup.exists():
            return True
        jobs = json.loads(sup.read_text()).get("jobs", {})
        if not jobs or any(job_state(repo, q, j) not in ("COMPLETE", "FAILED", "GATE_CLOSED") for j in jobs):
            return True
    return False


def queues() -> dict:
    calib_out = _p7("calibration_v6.json", "calibration_v6_evaluations.csv", "calibration_v6_rows.csv")
    prof_out = _p7("identifiability_profile_v6.json", "identifiability_profile_v6.csv",
                   "identifiability_profile_v6_progress.log")
    hold_out = _p7("holdout_icao_validation_v6.csv", "holdout_icao_validation_summary_v6.csv",
                   "holdout_icao_validation_v6.json")
    blend_out = _p7("p73_blends_v6_central.csv", "p73_blends_v6_draws.csv",
                    "p73_blends_v6_claims.csv", "p73_blends_v6_nvpm.csv",
                    "p73_blends_v6_lifecycle_corsia.csv", "p73_blends_v6.json", "p73_blends_v6.md")
    return {
        "calib": {"max_threads": 6, "jobs": [
            Job("p72_calibrate", ["scripts/optimization/lto_v6.py", "calibrate"], calib_out,
                threads=6, check=check_json_key("outputs/phase7/calibration_v6.json", "selected")),
            Job("p72_profile", ["scripts/optimization/lto_v6.py", "profile"], prof_out,
                requires=[("calib", "p72_calibrate")], threads=6,
                check=check_json_key("outputs/phase7/identifiability_profile_v6.json", "A1")),
            Job("p72_holdout", ["scripts/optimization/lto_v6.py", "holdout"], hold_out,
                requires=[("calib", "p72_profile")], threads=6,
                check=check_json_key("outputs/phase7/holdout_icao_validation_v6.json", "blend_gate")),
        ]},
        "blends": {"max_threads": 6, "jobs": [
            Job("p73_blends", ["scripts/optimization/blend_matched_thrust_v6.py"], blend_out,
                requires=[("calib", "p72_holdout")],
                imports=calib_out + prof_out + hold_out, gate=gate_blends, threads=6,
                check=check_json_key("outputs/phase7/p73_blends_v6.json", "claims")),
        ]},
        # PINN runs never oversubscribe the Mac: while any calib/blends job is
        # unfinished, `reserve_threads` of the cap stay free for those queues.
        "pinn": {"max_threads": None, "jobs": pinn_jobs(),
                 "reserve_for": ["calib", "blends"], "reserve_threads": 6},
    }


# --------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------
def snapshot_root(repo: Path) -> Path:
    return repo.parent / f"{repo.name}_phase7_snapshots"


def make_snapshot(repo: Path, commit: str) -> Path:
    """Extract ``git archive <commit>`` once; the directory is never modified
    by the supervisor except for job outputs and imported upstream artifacts."""
    dest = snapshot_root(repo) / commit
    marker = dest / ".snapshot.json"
    if marker.exists():
        return dest
    tmp = snapshot_root(repo) / f".{commit}.partial{os.getpid()}"
    tmp.mkdir(parents=True)
    arch = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit],
                          check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=arch, check=True)
    (tmp / ".venv").symlink_to(repo / ".venv")          # interpreter only, not source
    tree = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{commit}^{{tree}}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    (tmp / ".snapshot.json").write_text(json.dumps(
        {"commit": commit, "tree": tree, "created": now(),
         "archive_sha256": hashlib.sha256(arch).hexdigest()}, indent=2) + "\n")
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------
def job_dir(repo: Path, queue: str, job: str) -> Path:
    return repo / RUNS_REL / queue / job


def attempts(d: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("attempt_*.json")
                  if p.stem.count(".") == 0 and p.stem.split("_")[1].isdigit())


def job_state(repo: Path, queue: str, job: str) -> str:
    d = job_dir(repo, queue, job)
    for s in ("COMPLETE", "FAILED", "GATE_CLOSED"):
        if (d / f"{s}.json").exists():
            return s
    if (d / "lock").exists():
        return "RUNNING_OR_INTERRUPTED"
    return "PENDING" if not attempts(d) else "INTERRUPTED"


def verify_complete(repo: Path, queue: str, job: str) -> bool:
    """COMPLETE.json present and every promoted output still has its recorded hash."""
    p = job_dir(repo, queue, job) / "COMPLETE.json"
    if not p.exists():
        return False
    rec = json.loads(p.read_text())
    return all((repo / rel).exists() and sha256(repo / rel) == h for rel, h in rec["outputs_sha256"].items())


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def promote(snap: Path, repo: Path, rel: str) -> str:
    src, dst = snap / rel, repo / rel
    h = sha256(src)
    if dst.exists():
        if sha256(dst) != h:
            raise RuntimeError(f"refusing to overwrite {rel}: a different file exists in the repo")
        return h
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".promote.tmp")
    shutil.copy2(src, tmp)
    if sha256(tmp) != h:
        raise RuntimeError(f"copy of {rel} corrupted")
    os.replace(tmp, dst)
    return h


def import_artifacts(repo: Path, snap: Path, rels: list[str], upstream: list[tuple[str, str]]) -> dict:
    """Copy completed upstream outputs into the snapshot, verified against the
    hashes recorded in the upstream COMPLETE.json files."""
    recorded = {}
    for q, j in upstream:
        recorded.update(json.loads((job_dir(repo, q, j) / "COMPLETE.json").read_text())["outputs_sha256"])
    out = {}
    for rel in rels:
        h = sha256(repo / rel)
        if rel in recorded and recorded[rel] != h:
            raise RuntimeError(f"{rel} differs from its upstream completion record")
        dst = snap / rel
        if dst.exists() and sha256(dst) == h:
            out[rel] = h
            continue
        if dst.exists():
            raise RuntimeError(f"snapshot already holds a different {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dst)
        out[rel] = h
    return out


# --------------------------------------------------------------------------
# The run loop
# --------------------------------------------------------------------------
class Supervisor:
    def __init__(self, repo: Path, queue: str, commit: str, snap: Path):
        self.repo, self.queue, self.commit, self.snap = repo, queue, commit, snap
        self.spec = queues()[queue]
        self.running: dict[str, tuple[subprocess.Popen, int, object]] = {}
        self.stop = False

    # -- records -----------------------------------------------------------
    def status_path(self) -> Path:
        return self.repo / RUNS_REL / self.queue / "supervisor.json"

    def heartbeat(self, note: str = "") -> None:
        write_json_atomic(self.status_path(), {
            "queue": self.queue, "supervisor_pid": os.getpid(), "commit": self.commit,
            "snapshot": str(self.snap), "heartbeat": now(), "note": note,
            "jobs": {j.name: job_state(self.repo, self.queue, j.name) for j in self.spec["jobs"]},
        })
        write_status_md(self.repo)

    def recover(self, job: Job) -> None:
        """At (re)start: an attempt with a launch record but no exit record is
        interrupted infrastructure. Keep its evidence and release the lock."""
        d = job_dir(self.repo, self.queue, job.name)
        lock = d / "lock"
        for n in attempts(d):
            if not (d / f"attempt_{n}.exit.json").exists() and not (d / f"attempt_{n}.interrupted.json").exists():
                rec = json.loads((d / f"attempt_{n}.json").read_text())
                if pid_alive(rec["pid"]) and rec.get("supervisor_pid") != os.getpid():
                    raise SystemExit(f"{job.name} attempt {n} (pid {rec['pid']}) is still running; "
                                     "refusing to start a second supervisor")
                # outputs the interrupted attempt left in the snapshot are kept as
                # evidence (moved, never deleted), so the re-run starts clean
                kept = []
                for rel in job.outputs:
                    src = self.snap / rel
                    if src.exists():
                        dst = d / f"attempt_{n}_partial" / rel.replace("/", "__")
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), dst)
                        kept.append(rel)
                write_json_new(d / f"attempt_{n}.interrupted.json",
                               {"found": now(), "launch_record": rec, "partial_outputs_kept": kept,
                                "note": "no exit record: supervisor or host stopped; the exact "
                                        "registered command is re-run (infrastructure recovery)"})
        if lock.exists():
            lock.unlink()

    def deps_ok(self, job: Job) -> Optional[bool]:
        """True: all requirements verified complete. None: still pending. False: blocked."""
        for q, j in job.requires:
            st = job_state(self.repo, q, j)
            if st == "COMPLETE":
                if not verify_complete(self.repo, q, j):
                    raise RuntimeError(f"{q}/{j} COMPLETE but its outputs changed")
                continue
            if st in ("FAILED", "GATE_CLOSED"):
                return False
            return None
        return True

    def launch(self, job: Job) -> None:
        d = job_dir(self.repo, self.queue, job.name)
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(d / "lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {now()}\n".encode())
        os.close(fd)
        n = (attempts(d)[-1] + 1) if attempts(d) else 1
        imported = import_artifacts(self.repo, self.snap, job.imports, job.requires)
        reg_hashes = {str(p.relative_to(self.snap)): sha256(p)
                      for p in sorted((self.snap / "outputs" / "phase7").glob("p7*_registration.json"))}
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1", **job.env)
        env["OMP_NUM_THREADS"] = str(job.threads)
        argv = [str(self.snap / PYTHON_REL)] + job.argv
        log = open(d / f"attempt_{n}.log", "x")
        proc = subprocess.Popen(argv, cwd=self.snap, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        write_json_new(d / f"attempt_{n}.json", {
            "job": job.name, "queue": self.queue, "attempt": n, "pid": proc.pid,
            "supervisor_pid": os.getpid(), "started": now(), "commit": self.commit,
            "snapshot": str(self.snap),
            "snapshot_record": json.loads((self.snap / ".snapshot.json").read_text()),
            "argv": job.argv, "cwd": str(self.snap), "env": job.env, "threads": job.threads,
            "config": job.config(), "config_hash": job.config_hash(),
            "registration_sha256": reg_hashes, "imported_inputs_sha256": imported,
            "inspect": f"tail -f {d / f'attempt_{n}.log'}",
        })
        self.running[job.name] = (proc, n, log)
        self.heartbeat(f"launched {job.name} attempt {n} pid {proc.pid}")

    def finish(self, job: Job, rc: int) -> None:
        proc, n, log = self.running.pop(job.name)
        log.close()
        d = job_dir(self.repo, self.queue, job.name)
        write_json_new(d / f"attempt_{n}.exit.json", {"exit_code": rc, "finished": now()})
        problem = None
        if rc != 0:
            problem = f"exit code {rc}"
        else:
            missing = [o for o in job.outputs if not (self.snap / o).exists()]
            if missing:
                problem = f"missing outputs {missing}"
            elif job.check is not None:
                c = job.check(self.snap)
                if not c["ok"]:
                    problem = f"completion check failed: {c['detail']}"
        if problem:
            write_json_new(d / "FAILED.json", {"attempt": n, "reason": problem, "recorded": now()})
        else:
            hashes = {o: promote(self.snap, self.repo, o) for o in job.outputs}
            write_json_new(d / "COMPLETE.json", {
                "attempt": n, "exit_code": rc, "completed": now(), "commit": self.commit,
                "config_hash": job.config_hash(), "outputs_sha256": hashes})
        (d / "lock").unlink(missing_ok=True)
        self.heartbeat(f"{job.name}: {'FAILED ' + problem if problem else 'COMPLETE'}")

    def run(self) -> int:
        for j in self.spec["jobs"]:
            self.recover(j)
        self.heartbeat("started")

        def _term(signum, _f):
            self.stop = True
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, _term)
        base_cap = self.spec["max_threads"] or pinn_max_threads()
        last_beat = time.time()
        while True:
            for name, (proc, _n, _l) in list(self.running.items()):
                rc = proc.poll()
                if rc is not None:
                    self.finish(next(j for j in self.spec["jobs"] if j.name == name), rc)
            if self.stop:
                for proc, _n, _l in self.running.values():
                    os.killpg(proc.pid, signal.SIGTERM)
                self.heartbeat("stopped by signal; running attempts will be recovered on restart")
                return 1
            used = sum(j.threads for j in self.spec["jobs"] if j.name in self.running)
            cap = base_cap - (self.spec.get("reserve_threads", 0)
                              if queues_unfinished(self.repo, self.spec.get("reserve_for", [])) else 0)
            waiting = False
            for job in self.spec["jobs"]:
                st = job_state(self.repo, self.queue, job.name)
                if job.name in self.running or st in ("COMPLETE", "FAILED", "GATE_CLOSED"):
                    continue
                ok = self.deps_ok(job)
                if ok is None:
                    waiting = True
                    continue
                if ok is False:
                    write_json_new(job_dir(self.repo, self.queue, job.name) / "GATE_CLOSED.json",
                                   {"reason": "an upstream job failed or was gated", "recorded": now()})
                    self.heartbeat(f"{job.name}: blocked by upstream")
                    continue
                if job.gate is not None:
                    g = job.gate(self.repo)
                    if not g["open"]:
                        write_json_new(job_dir(self.repo, self.queue, job.name) / "GATE_CLOSED.json",
                                       {"gate": g, "recorded": now()})
                        self.heartbeat(f"{job.name}: gate closed ({g['reason']})")
                        continue
                if used + job.threads > cap and self.running:
                    waiting = True
                    continue
                self.launch(job)
                used += job.threads
            if not self.running and not waiting:
                self.heartbeat("queue finished")
                return 0
            if time.time() - last_beat > 60:
                self.heartbeat("running")
                last_beat = time.time()
            time.sleep(5)


# --------------------------------------------------------------------------
# Status file (regenerated on every state change)
# --------------------------------------------------------------------------
def write_status_md(repo: Path) -> None:
    lines = ["# Phase 7 job status (auto-generated by scripts/phase7_supervisor.py)", "",
             f"Updated {now()}. States come from run records, not from process lists: "
             "COMPLETE requires exit 0, every registered output and its hash.", "",
             "| Queue | Job | State | Attempts | Detail |", "|---|---|---|---|---|"]
    base = repo / RUNS_REL
    for qd in sorted(p for p in base.glob("*") if p.is_dir()):
        sup = qd / "supervisor.json"
        beat = json.loads(sup.read_text()) if sup.exists() else {}
        names = sorted({p.name for p in qd.glob("*") if p.is_dir()} | set(beat.get("jobs", {})))
        for name in names:
            jd = qd / name
            st = job_state(repo, qd.name, jd.name)
            det = ""
            for s in ("COMPLETE", "FAILED", "GATE_CLOSED"):
                if (jd / f"{s}.json").exists():
                    rec = json.loads((jd / f"{s}.json").read_text())
                    det = rec.get("completed") or rec.get("reason") or json.dumps(rec.get("gate"))
            if st == "RUNNING_OR_INTERRUPTED":
                a = attempts(jd)[-1]
                rec = json.loads((jd / f"attempt_{a}.json").read_text())
                alive = pid_alive(rec["pid"])
                st = "RUNNING" if alive else "INTERRUPTED (no exit record)"
                det = f"pid {rec['pid']} since {rec['started']}"
            lines.append(f"| {qd.name} | {jd.name} | {st} | {len(attempts(jd)) if jd.exists() else 0} | {det} |")
        if beat:
            lines.append(f"| {qd.name} | (supervisor) | pid {beat.get('supervisor_pid')} "
                         f"{'alive' if pid_alive(beat.get('supervisor_pid', -1)) else 'not running'} | | "
                         f"heartbeat {beat.get('heartbeat')}: {beat.get('note')} |")
    write_json_atomic_text(repo / "outputs" / "phase7" / "runs" / "STATUS.md", "\n".join(lines) + "\n")


def write_json_atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_launch(repo: Path, queue: str, commit: Optional[str]) -> None:
    commit = commit or subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                      capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.splitlines()
    dirty = [ln for ln in dirty if not ln.endswith(".DS_Store")]
    if dirty:
        print("NOTE: tracked files differ from HEAD; the snapshot is the COMMIT, not the working tree:")
        print("\n".join(dirty))
    snap = make_snapshot(repo, commit)
    sup = repo / RUNS_REL / queue / "supervisor.json"
    if sup.exists():
        rec = json.loads(sup.read_text())
        if pid_alive(rec.get("supervisor_pid", -1)):
            raise SystemExit(f"queue {queue} already has a live supervisor (pid {rec['supervisor_pid']})")
    logdir = repo / "outputs" / "logs" / "phase7"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    log = open(logdir / f"supervisor_{queue}_{stamp}.log", "x")
    argv = ["/usr/bin/caffeinate", "-ims", str(snap / PYTHON_REL), str(snap / "scripts" / "phase7_supervisor.py"),
            "run", "--queue", queue, "--commit", commit, "--repo", str(repo)]
    p = subprocess.Popen(argv, cwd=snap, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    print(json.dumps({"queue": queue, "commit": commit, "snapshot": str(snap), "caffeinate_pid": p.pid,
                      "log": str(log.name), "started": now()}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("launch")
    a.add_argument("--queue", required=True, choices=["calib", "blends", "pinn"])
    a.add_argument("--commit", default=None)
    r = sub.add_parser("run")
    r.add_argument("--queue", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--repo", required=True)
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "launch":
        cmd_launch(HERE, args.queue, args.commit)
    elif args.cmd == "run":
        repo = Path(args.repo).resolve()
        snap = HERE
        rec = json.loads((snap / ".snapshot.json").read_text())
        if rec["commit"] != args.commit:
            raise SystemExit("supervisor must run from the snapshot of its commit")
        sys.exit(Supervisor(repo, args.queue, args.commit, snap).run())
    else:
        write_status_md(HERE)
        print((HERE / RUNS_REL / "STATUS.md").read_text())


if __name__ == "__main__":
    main()
