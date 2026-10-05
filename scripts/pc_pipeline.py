#!/usr/bin/env python3
"""Resumable a–i Python-v6 PC study; help/dry-run perform no scientific imports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = tuple("abcdefghi")
STAGE_NAMES = ("environment_hardware", "parity20", "p73_a1", "saf_generate", "learning_curve",
               "sole_score", "nozzle_ode", "pc_timing_break_even", "commit_push_pc_run")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
DIRECT_FILE_LIMIT = 75*1024*1024
TRANSPORT_CHUNK_LIMIT = 50*1024*1024


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    workers: int = 10
    backend: str = "torch"
    device: str = "cpu"
    metadata_dir: Path | None = None

    def paths(self):
        root = self.root.resolve()
        metadata = self.metadata_dir or root / "outputs/phase8/pc_python_pipeline"
        if not metadata.is_absolute():
            metadata = root / metadata
        metadata = metadata.resolve()
        if not metadata.is_relative_to(root / "outputs/phase8") or metadata == root / "outputs/phase8":
            raise ValueError("Metadata must be a dedicated directory under outputs/phase8")
        return root, metadata

    def record(self):
        root, metadata = self.paths()
        return {"root":str(root), "metadata_dir":str(metadata), "workers":self.workers,
                "backend":self.backend, "device":self.device, "simulator":"python-v6"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def scientific_outputs(root):
    saf = read_json(root / "docs/phase8_saf_surrogate_registration.json")
    p73 = read_json(root / "docs/phase7_p73_a1_registration.json")
    nozzle = read_json(root / "docs/phase8_nozzle_ode_registration.json")
    return {"p73":root/p73["outputs"]["directory"], "saf":root/saf["artifact_root"],
            "nozzle":root/nozzle["outputs"]["root"]}


def describe(config):
    root, metadata = config.paths()
    outputs = scientific_outputs(root)
    for path in outputs.values():
        if metadata == path or metadata.is_relative_to(path) or path.is_relative_to(metadata):
            raise ValueError("Pipeline metadata overlaps a fixed scientific output")
    if config.workers <= 0 or config.backend not in {"torch","mlx"}:
        raise ValueError("Positive workers and a supported backend are required")
    return {**config.record(), "stages":[{"id":letter,"name":name} for letter,name in zip(STAGES,STAGE_NAMES)],
            "scientific_outputs":{k:str(v) for k,v in outputs.items()}, "a2":"DEFERRED", "cpp_required":False,
            "schedule":{"train_sizes":[64,256,1024,4096],"arms":["Mdata","Mphys"],"seeds":[42,43,44],"fits":24},
            "resume":"Byte-verified completed stages only; consumed partial stages and scoring never reopen"}


def preflight(config):
    plan = describe(config)
    from simulation.runtime import require_ac
    if sys.version_info[:2] != (3,12):
        raise RuntimeError("Use Python3.12 for the pinned study environment")
    for key in THREAD_ENV:
        if os.environ.get(key,"1") != "1":
            raise RuntimeError(f"{key}=1 is required")
        os.environ[key] = "1"
    versions = {name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","cantera",
        "PyYAML","torch" if config.backend=="torch" else "mlx")}
    if versions["cantera"] != "3.2.0":
        raise RuntimeError("Install pinned Cantera3.2.0")
    from simulation.ml_backend import get_backend
    selected = get_backend(config.backend,device=config.device)
    return {"status":"PASS", "plan":plan, "versions":versions, "power":require_ac(),
            "logical_cores":os.cpu_count(), "platform":platform.platform(), "backend":selected.name}


def checked_git(root, *args):
    result = subprocess.run(["git",*args],cwd=root,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def publication_transport(config, paths):
    """Delegate lossless large-file transport; scientific originals stay unchanged."""
    from scripts.phase8.artifact_transport import prepare
    root,metadata=config.paths()
    published,manifest=prepare(root,metadata,paths,threshold_bytes=DIRECT_FILE_LIMIT,chunk_bytes=TRANSPORT_CHUNK_LIMIT)
    return published,{name:entry["sha256"] for name,entry in manifest.get("files",{}).items()}


def restore_artifacts(config):
    """Restore byte-verified large originals from a cloned run branch."""
    from scripts.phase8.artifact_transport import restore
    root,metadata=config.paths()
    describe(config)
    restored=restore(root,metadata,allowed_roots=[metadata,*scientific_outputs(root).values()])
    return {"status":"PASS","restored":restored,"science_executed":False}


def publish_outputs(config, *, git=checked_git):
    """Create a run branch and commit/push only actual owned scientific artifacts."""
    root, metadata = config.paths()
    intent_path = metadata/"publication_intent.json"
    if intent_path.exists():
        intent = validate_publication_intent(config, git=git)
        branch, commit, paths = intent["branch"], intent["commit"], list(intent["artifact_hashes"])
        remote = git(root,"ls-remote","--heads","origin",branch).split()
        if remote and remote[0] != commit:
            raise RuntimeError("Remote run branch differs from the owned publication commit")
        git(root,"push","--set-upstream","origin",branch)
        remote = git(root,"ls-remote","--heads","origin",branch).split()
        if not remote or remote[0] != commit:
            raise RuntimeError("Remote run branch does not match the owned publication commit")
        return {"status":"COMPLETE", "execution_complete":True, "branch":branch,"commit":commit,
                "published_paths":paths,"force":False,"retried_owned_commit":True,"artifacts":{}}
    branch = "pc-run-"+datetime.now(timezone.utc).strftime("%Y%m%d")
    if git(root,"diff","--cached","--name-only"):
        raise RuntimeError("Publication refuses unrelated pre-staged files")
    if git(root,"branch","--list",branch) or git(root,"ls-remote","--heads","origin",branch):
        raise RuntimeError("Run branch already exists; publication never overwrites or force-pushes")
    own = [metadata,*scientific_outputs(root).values()]
    paths = []
    for folder in own:
        if not folder.resolve().is_relative_to(root/"outputs"):
            raise ValueError("Publication scope escaped outputs")
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError("Owned artifact cannot be a symlink")
            name = path.name
            # Active ownership/checkpoint i is not a completed scientific artifact.
            if name in {"owner.json","stage_owner.json","latest_terminal.json","i.started.json","publication_intent.json"} or ".tmp" in name:
                continue
            paths.append(path.relative_to(root).as_posix())
    if not paths:
        raise RuntimeError("No owned artifacts to publish")
    paths,large_originals=publication_transport(config,paths)
    intent = {"schema":"pc-publication-v1", "config_sha256":fingerprint(config.record()), "branch":branch,
              "base_commit":git(root,"rev-parse","HEAD"), "state":"PREPARING",
              "artifact_hashes":{name:sha256(root/name) for name in paths}, "large_original_hashes":large_originals}
    intent_path.write_text(json.dumps(intent,indent=2,allow_nan=False)+"\n")
    git(root,"switch","-c",branch)
    # Batches avoid OS argument length limits; no wildcard or git add-all.
    for start in range(0,len(paths),100):
        git(root,"-c","core.autocrlf=false","add","-f","--",*paths[start:start+100])
    staged = set(git(root,"diff","--cached","--name-only").splitlines())
    if staged != set(paths):
        raise RuntimeError("Staged publication differs from the exact owned artifact set")
    for name in paths:
        if git(root,"hash-object","--no-filters",str(root/name))!=git(root,"rev-parse",":"+name):
            raise RuntimeError("Git filter changed sealed publication bytes: "+name)
    git(root,"commit","-m",f"Record Python-v6 PC run {branch.removeprefix('pc-run-')}")
    commit = git(root,"rev-parse","HEAD")
    intent.update(state="COMMITTED",commit=commit)
    temporary = intent_path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(intent,stream,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,intent_path)
    git(root,"push","--set-upstream","origin",branch)
    remote = git(root,"ls-remote","--heads","origin",branch).split()
    if not remote or remote[0] != commit:
        raise RuntimeError("Remote run branch does not match the pushed commit")
    return {"status":"COMPLETE", "execution_complete":True, "branch":branch,"commit":commit,
            "published_paths":paths,"force":False,"artifacts":{}}


def validate_publication_intent(config, *, git=checked_git):
    """Authenticate the exact local commit for a reversible network push retry."""
    root, metadata = config.paths()
    intent = read_json(metadata/"publication_intent.json")
    if (intent.get("schema")!="pc-publication-v1" or intent.get("state")!="COMMITTED"
            or intent.get("config_sha256")!=fingerprint(config.record())):
        raise RuntimeError("Publication was interrupted before an authenticated commit existed")
    branch, commit = intent["branch"],intent["commit"]
    if (not branch.startswith("pc-run-") or git(root,"branch","--show-current")!=branch
            or git(root,"rev-parse","HEAD")!=commit or git(root,"rev-parse",commit+"^")!=intent["base_commit"]):
        raise RuntimeError("Owned publication branch or commit changed")
    if git(root,"diff","--cached","--name-only"):
        raise RuntimeError("Publication retry refuses staged changes")
    paths=intent["artifact_hashes"]
    changed=set(git(root,"diff-tree","--no-commit-id","--name-only","-r",commit).splitlines())
    if changed!=set(paths):
        raise RuntimeError("Publication commit changed files outside its exact owned intent")
    own=[metadata,*scientific_outputs(root).values()]
    for name,expected in intent.get("large_original_hashes",{}).items():
        path=root/name
        if (path.is_symlink() or not path.is_file() or not any(path.resolve().is_relative_to(p) for p in own)
                or sha256(path)!=expected):
            raise RuntimeError("Original large scientific artifact changed: "+name)
    for name,expected in paths.items():
        path=root/name
        if (path.is_symlink() or not path.resolve().is_file() or not any(path.resolve().is_relative_to(p) for p in own)
                or sha256(path)!=expected or git(root,"hash-object","--no-filters",str(path))!=git(root,"rev-parse",commit+":"+name)):
            raise RuntimeError("Owned publication artifact or committed bytes changed: "+name)
    return intent


def validate_checkpoint(root, checkpoint, config_sha):
    if checkpoint.get("status") != "COMPLETE" or checkpoint.get("config_sha256") != config_sha:
        raise RuntimeError("Resume checkpoint is incomplete or uses another configuration")
    for name, expected in checkpoint.get("artifacts",{}).items():
        original = root/name
        path = original.resolve()
        if not path.is_relative_to(root) or not path.is_file() or original.is_symlink() or sha256(path) != expected:
            raise RuntimeError("Completed stage artifact changed: "+name)
    return checkpoint


def execute(config, *, resume=False, stop_after=None, handlers=None):
    """Skip immutable completed stages; an interrupted consumed stage fails closed."""
    from scripts.phase8.pc_runtime import utc, write_once
    from simulation.runtime import host_identity, liveness, process_birth, require_ac, SleepInhibitor
    plan = describe(config)
    root, metadata = config.paths()
    if stop_after is not None and stop_after not in STAGES:
        raise ValueError("Unknown stop-after stage")
    config_sha = fingerprint(config.record())
    if resume:
        if not metadata.is_dir() or read_json(metadata/"config.json") != config.record():
            raise RuntimeError("Resume requires the existing attempt and its identical configuration")
    else:
        if metadata.exists():
            raise FileExistsError("Attempt exists; use --resume to validate completed checkpoints")
        if handlers is None and any(path.exists() for path in scientific_outputs(root).values()):
            raise FileExistsError("Fixed scientific output is already consumed; no overwrite or replay")
        metadata.mkdir(parents=True,exist_ok=False)
        write_once(metadata/"config.json",config.record())
        write_once(metadata/"plan.json",plan)
    checkpoints = metadata/"checkpoints"
    checkpoints.mkdir(exist_ok=True)
    completed = {}
    incomplete_seen = False
    for letter in STAGES:
        receipt, started = checkpoints/f"{letter}.json", checkpoints/f"{letter}.started.json"
        if receipt.exists():
            if incomplete_seen:
                raise RuntimeError("Completed stages are not a contiguous prefix")
            completed[letter] = validate_checkpoint(root,read_json(receipt),config_sha)
        else:
            incomplete_seen = True
            if started.exists() and not (letter=="i" and (metadata/"publication_intent.json").is_file()):
                raise RuntimeError(f"Stage {letter} was consumed but has no completed checkpoint; preserve partial artifacts")
    lease = metadata/"owner.json"
    if lease.exists():
        previous = read_json(lease)
        state = liveness(previous.get("owner_pid"),previous.get("owner_birth"),host=previous.get("host"))
        if state not in {"dead","reused"} or (metadata/"stage_owner.json").exists():
            raise RuntimeError("An active, foreign-host or ambiguous PC owner lease remains")
        archive = metadata/"leases"/(uuid.uuid4().hex+".json")
        write_once(archive,previous)
        lease.unlink()
    owner = {"owner_pid":os.getpid(),"owner_birth":process_birth(os.getpid()),"host":host_identity(),
             "config_sha256":config_sha,"started_utc":utc(),"argv":list(sys.argv)}
    if not owner["owner_birth"] or owner["owner_birth"]=="<dead>":
        raise RuntimeError("Owner process identity is unreadable")
    write_once(lease,owner)
    os.environ["CATJET_SIMULATOR_BACKEND"]="python"
    for key in THREAD_ENV:
        if os.environ.get(key,"1") != "1":
            lease.unlink()
            raise RuntimeError(f"{key}=1 is required")
        os.environ[key]="1"
    summary = {"status":"RUNNING","started_utc":utc(),"resumed":resume,"stages":[],"a2":"DEFERRED"}
    active = None
    try:
        if handlers is None:
            from scripts.phase8.python_pc import ScientificWorkflow
            workflow = ScientificWorkflow(root,metadata,workers=config.workers,backend=config.backend,device=config.device)
            handlers = {letter:getattr(workflow,letter) for letter in "abcdefgh"}
            handlers["i"] = lambda:publish_outputs(config)
        with SleepInhibitor() as inhibitor:
            summary["sleep_inhibition"] = inhibitor.metadata
            for letter in STAGES:
                active=letter
                require_ac()
                if read_json(lease)!=owner:
                    raise RuntimeError("PC pipeline ownership changed")
                if letter in completed:
                    receipt=completed[letter]
                    summary["stages"].append({"stage":letter,"skipped_verified_complete":True,
                                               "scientific_verdict":receipt.get("scientific_verdict")})
                else:
                    # All previous snapshots remain valid before dispatching any next stage.
                    for prior in completed.values():
                        validate_checkpoint(root,prior,config_sha)
                    started=utc(); before=__import__("time").perf_counter()
                    marker=checkpoints/f"{letter}.started.json"
                    if not marker.exists():
                        write_once(marker,{"stage":letter,"started_utc":started,"owner":owner})
                    result=handlers[letter]()
                    if not isinstance(result,dict) or result.get("status")!="COMPLETE" or result.get("execution_complete") is not True:
                        raise RuntimeError(f"Stage {letter} did not complete: {result}")
                    receipt={**result,"stage":letter,"status":"COMPLETE","config_sha256":config_sha,
                             "started_utc":started,"finished_utc":utc(),"elapsed_seconds":__import__("time").perf_counter()-before}
                    validate_checkpoint(root,receipt,config_sha)
                    write_once(checkpoints/f"{letter}.json",receipt)
                    completed[letter]=receipt
                    summary["stages"].append({"stage":letter,"skipped_verified_complete":False,
                                               "scientific_verdict":receipt.get("scientific_verdict")})
                if letter=="b" and receipt.get("scientific_verdict")!="PASS":
                    summary.update(status="FAIL",stopped_after="b",reason="Frozen Python-v6 parity failed")
                    break
                if letter==stop_after:
                    summary.update(status="PAUSED",stopped_after=letter)
                    break
            else:
                verdicts={r.get("scientific_verdict") for r in completed.values()}
                summary.update(status="COMPLETE",scientific_verdict="FAIL" if "FAIL" in verdicts else
                               "INCOMPLETE" if verdicts & {"BLOCKED","INCOMPLETE"} else "PASS")
    except BaseException as exc:
        summary.update(status="ERROR",stopped_at=active,error=f"{type(exc).__name__}: {exc}")
    finally:
        summary["finished_utc"]=utc()
        write_once(metadata/"terminals"/(uuid.uuid4().hex+".json"),summary)
        if lease.exists() and read_json(lease)==owner:
            lease.unlink()
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--preflight",action="store_true")
    mode.add_argument("--restore-artifacts",action="store_true",help="Restore SHA-verified large originals from published gzip chunks")
    parser.add_argument("--resume",action="store_true",help="Validate and skip completed checkpoints; never replay consumed stages")
    parser.add_argument("--stop-after",choices=STAGES,help="Stop safely after a complete stage")
    parser.add_argument("--workers",type=int,default=10)
    parser.add_argument("--backend",choices=("torch","mlx"),default=None)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default=None)
    parser.add_argument("--metadata-dir",type=Path)
    args=parser.parse_args(argv)
    backend=args.backend or os.environ.get("CATJET_ML_BACKEND","torch").strip().lower()
    device=args.device or ("cpu" if backend=="torch" else "auto")
    config=Config(workers=args.workers,backend=backend,device=device,metadata_dir=args.metadata_dir)
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    try:
        result=describe(config) if args.dry_run else preflight(config) if args.preflight else restore_artifacts(config) if args.restore_artifacts else execute(
            config,resume=args.resume,stop_after=args.stop_after)
    except Exception as exc:
        print(json.dumps({"status":"ERROR","error":f"{type(exc).__name__}: {exc}"},indent=2),file=sys.stderr)
        return 2
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result.get("status","PASS") in {"PASS","COMPLETE","PAUSED"} else 1


if __name__=="__main__":
    raise SystemExit(main())
