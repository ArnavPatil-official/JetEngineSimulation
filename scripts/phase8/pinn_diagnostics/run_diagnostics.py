"""Track 4 runner: one synchronous, write-once execution of the registered diagnostics.

    nice -n 15 .venv/bin/python -m scripts.phase8.pinn_diagnostics.run_diagnostics \
        --registration docs/phase8_track4_registration.json

Before any computation it refuses (exit 3, nothing written) when the Mac is
not on mains power, when a benchmark/calibration Python process is actively
running (parked shells do not count), when the process is niced below the
registered value, when the diagnostic sources or registration differ from
the committed HEAD, or when ``pmset``, ``ps``, ``git status`` or HEAD cannot
be read (fail closed). It refuses (exit 2) when the output directory exists.

The start identity (HEAD, exact registration bytes and hash, source hashes,
envelope hash) is frozen before the output directory is created and is the
only identity written to the config, checkpoint and report. It is rechecked
at the end; any drift is reported beside it and makes the run an ERROR.
Exit 0 only when every rung PASSes; any FAIL, ERROR or BLOCKED rung exits 1.
Outputs: config, checkpoint, scores, report (JSON + Markdown), environment,
start/exit log and final hashes, all under the registered output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = Path(__file__).resolve().parent
SOURCES = ["__init__.py", "turbine_map.py", "nozzle_verification.py", "run_diagnostics.py"]
# heavy Track 1 jobs, as script paths (module form is the same path with dots)
HEAVY_SCRIPTS = ("scripts/phase8/benchmark.py", "scripts/phase8/ablation_ladder.py",
                 "scripts/optimization/lto_v6.py", "scripts/optimization/calibrate_lto.py")
PYTHON_EXE = re.compile(r"python(\d+(\.\d+)*)?w?", re.IGNORECASE)
REPRO = ("nice -n 15 .venv/bin/python -m scripts.phase8.pinn_diagnostics.run_diagnostics "
         "--registration docs/phase8_track4_registration.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Resource checks (pure functions on command output, so they are testable)
# ---------------------------------------------------------------------------

def on_mains(pmset_output: str | None) -> bool:
    return bool(pmset_output) and "'AC Power'" in pmset_output.splitlines()[0]


def python_target(args: str) -> str | None:
    """Script or module a Python command line runs, as a '/' path; None if not Python.

    Parses ``ps -Ao pid=,args=`` text (the ``comm`` column is truncated on macOS,
    e.g. the framework interpreter shows as ``/Library/Framewo``). Handles absolute
    interpreter paths, ``python3.12``/``Python``, interpreter options, ``-m module``
    and ``-c code`` (returned as the code text).
    """
    try:
        tokens = shlex.split(args)
    except ValueError:                   # ps does not quote argv; fall back to whitespace
        tokens = args.split()
    if not tokens or not PYTHON_EXE.fullmatch(os.path.basename(tokens[0])):
        return None
    it = iter(tokens[1:])
    for tok in it:
        if tok in ("-W", "-X"):
            next(it, None)
        elif tok.startswith("-m"):
            mod = tok[2:] or next(it, "")
            return mod.replace(".", "/") + ".py"
        elif tok.startswith("-c"):
            return " ".join([tok[2:] or next(it, "")] + list(it))
        elif tok == "-" or not tok.startswith("-"):
            return tok.replace(os.sep, "/")
    return ""                            # bare interpreter (REPL)


def is_heavy_target(target: str) -> bool:
    """True for a heavy script/module path (any directory prefix), a calibration
    script, or ``-c`` code naming one. Errs towards refusing."""
    if not target:
        return False
    if any(s in target or s[:-3].replace("/", ".") in target for s in HEAVY_SCRIPTS):
        return True
    name = os.path.basename(target.split()[0])
    return name in {os.path.basename(s) for s in HEAVY_SCRIPTS} or "calibrat" in name


def heavy_python_processes(ps_output: str, own_pid: int) -> list:
    """Python interpreters running a benchmark or calibration job (``ps -Ao pid=,args=``).

    Shells (bash/zsh/caffeinate) whose command text merely names such a job
    are parked waiters, not computations, and are ignored.
    """
    hits = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == own_pid:
            continue
        target = python_target(parts[1])
        if target is not None and is_heavy_target(target):
            hits.append(line.strip())
    return hits


def resource_blockers(pmset_output: str | None, ps_output: str | None, own_pid: int, niceness: int,
                      required_nice: int) -> list:
    out = []
    if not on_mains(pmset_output):
        out.append("not on mains (AC) power, or power source unknown")
    if ps_output is None or not ps_output.strip():
        out.append("process list unreadable (ps failed or returned nothing); refusing (fail closed)")
    else:
        heavy = heavy_python_processes(ps_output, own_pid)
        if heavy:
            out.append("active benchmark/calibration Python process: " + " | ".join(heavy))
    if niceness < required_nice:
        out.append(f"niceness {niceness} below the registered {required_nice}")
    return out


def _cmd(args: list) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def live_blockers(required_nice: int) -> list:
    return resource_blockers(_cmd(["pmset", "-g", "batt"]), _cmd(["ps", "-Ao", "pid=,args="]),
                             os.getpid(), os.nice(0), required_nice)


def git_head() -> str | None:
    head = (_cmd(["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"]) or "").strip()
    return head if re.fullmatch(r"[0-9a-f]{40}", head) else None


def dirty_sources(reg_path: Path) -> str | None:
    """``git status --porcelain`` of the sources and registration; None if unreadable."""
    rels = [str((PKG / s).relative_to(ROOT)) for s in SOURCES] + [str(reg_path.resolve().relative_to(ROOT))]
    return _cmd(["git", "-C", str(ROOT), "status", "--porcelain", "--", *rels])


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _envelope_bytes(reg: dict) -> tuple:
    try:
        return (ROOT / reg["inputs"]["synthetic_envelope"]).read_bytes(), None
    except OSError as exc:
        return None, repr(exc)


def identity_snapshot(reg_path: Path, reg_bytes: bytes, head: str | None, reg: dict,
                      env_bytes: bytes | None, env_error: str | None) -> dict:
    """Small identity dictionary: HEAD, registration, sources, envelope."""
    return {"git_head": head,
            "registration_path": str(reg_path.relative_to(ROOT)) if reg_path.is_relative_to(ROOT) else str(reg_path),
            "registration_sha256": hashlib.sha256(reg_bytes).hexdigest(),
            "source_sha256": {s: sha256(PKG / s) for s in SOURCES},
            "synthetic_envelope": reg["inputs"]["synthetic_envelope"],
            "synthetic_envelope_sha256": hashlib.sha256(env_bytes).hexdigest() if env_bytes is not None else None,
            "synthetic_envelope_error": env_error}


def end_identity(reg_path: Path, reg: dict) -> dict:
    """Re-read everything at the end; unreadable items are recorded, never raised."""
    try:
        reg_bytes = reg_path.read_bytes()
    except OSError as exc:
        return {"error": f"registration unreadable at end: {exc!r}"}
    try:
        env_bytes, env_error = _envelope_bytes(reg)
        return identity_snapshot(reg_path, reg_bytes, git_head(), reg, env_bytes, env_error)
    except OSError as exc:
        return {"error": f"identity unreadable at end: {exc!r}"}


def identity_drift(start: dict, end: dict) -> list:
    if "error" in end:
        return [end["error"]]
    out = [k for k in ("git_head", "registration_sha256", "synthetic_envelope_sha256") if start.get(k) != end.get(k)]
    out += [f"source_sha256.{s}" for s in SOURCES if start["source_sha256"].get(s) != end["source_sha256"].get(s)]
    return out


# ---------------------------------------------------------------------------
# Write-once output helpers
# ---------------------------------------------------------------------------

def create_output_dir(path: Path) -> Path:
    """Create the attempt directory; refuse if anything already exists there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def write_once(path: Path, text: str) -> None:
    with open(path, "x") as fh:
        fh.write(text)


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{utc()} {msg}"
        print(line, flush=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")


def environment() -> dict:
    import numpy
    import scipy
    import torch
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_default_dtype": str(torch.get_default_dtype()), "niceness": os.nice(0),
            "argv": sys.argv}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def check_inputs(reg: dict, env_bytes: bytes | None) -> dict:
    """Envelope identity (the bytes read at start) and that the registered box equals its extent."""
    import pandas as pd
    if env_bytes is None:
        raise ValueError("synthetic envelope was unreadable at start")
    got = hashlib.sha256(env_bytes).hexdigest()
    if got != reg["inputs"]["synthetic_envelope_sha256"]:
        raise ValueError(f"envelope sha256 {got} differs from the registration")
    env = pd.read_csv(io.BytesIO(env_bytes), float_precision="round_trip")
    t = reg["turbine"]
    box_ok = (float(env["tau"].min()) == t["tau"][0] and float(env["tau"].max()) == t["tau"][1]
              and float(env["gamma"].min()) == t["gamma"][0] and float(env["gamma"].max()) == t["gamma"][1])
    if not box_ok:
        raise ValueError("registered (tau, gamma) box differs from the envelope extent")
    return {"synthetic_envelope": reg["inputs"]["synthetic_envelope"], "synthetic_envelope_sha256": got,
            "envelope_rows": int(len(env)), "box_equals_envelope_extent": True,
            "ma_pdf_sha256": reg["inputs"]["ma_pdf_sha256"],
            "ma_pdf_note": "recorded in the registration; the PDF is outside the repository and not re-hashed here"}


def error_record(exc: BaseException) -> dict:
    return {"status": "ERROR", "reason": repr(exc), "traceback": traceback.format_exc()}


def aggregate_status(statuses: list) -> str:
    """ERROR > FAIL > BLOCKED > PASS; PASS only if every rung passed."""
    if not statuses:
        return "ERROR"
    for s in ("ERROR", "FAIL", "BLOCKED"):
        if s in statuses:
            return s
    return "PASS" if all(s == "PASS" for s in statuses) else "ERROR"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.3e}" if isinstance(v, float) else str(v)


def markdown_report(rep: dict) -> str:
    t, m, nz, ident = rep["turbine"], rep["manufactured"], rep["nozzle"], rep["start_identity"]
    L = [f"# Track 4 diagnostics — {rep['registration_id']}", "",
         f"**Aggregate status: {rep['status']}.**", "",
         f"Run {rep['started_utc']} → {rep['finished_utc']}, git `{ident['git_head'][:12]}` (start identity), "
         f"seed {rep['seed']}, CPU float64, {rep['environment']['torch_threads']} thread, nice "
         f"{rep['environment']['niceness']}. **Diagnostic only**: no engine, turbine or PINN validation; "
         "no empirical data, holdout or Sajben/WIND data opened; G2/G3 not reopened.", ""]
    if rep["identity_drift"]:
        L += ["**Identity drift between start and end:** " + ", ".join(rep["identity_drift"]) +
              ". The start identity below is what generated these results.", ""]
    L += ["## Summary", "", "| Item | Status | Key raw error | Tolerance |", "|---|---|---|---|"]
    if t.get("status") in ("PASS", "FAIL"):
        L.append(f"| Turbine map (65×65 grid, scored once) | **{t['status']}** | max rel p error "
                 f"{t['max_relative_pressure_error']:.3e} | < {t['threshold_strictly_below']} |")
    else:
        L.append(f"| Turbine map | **{t.get('status', 'ERROR')}** | {t.get('reason', '')} | — |")
    for act, r in m.items():
        if r.get("status") in ("PASS", "FAIL"):
            L.append(f"| MMS Ma Eqs. 22–26, {act} | **{r['status']}** | abs {r['max_abs_forced_residual']:.3e}, "
                     f"rel {r['max_relative_forcing_disagreement']:.3e}, weakest control "
                     f"{min(r['negative_control_discrepancy'].values()):.3e} | ≤ 1e-10 / ≤ 1e-9 / > 1e-7 |")
        else:
            L.append(f"| MMS Ma Eqs. 22–26, {act} | **{r.get('status', 'ERROR')}** | {r.get('reason', '')} | — |")
    for name, r in nz.items():
        if r.get("status") not in ("PASS", "FAIL"):
            L.append(f"| Nozzle {name} | **{r.get('status', 'ERROR')}** | {r.get('reason', '')} | — |")
        elif name == "4_back_pressure_shock":
            worst_pos = max(c["position_abs_error"] for c in r["cases"].values())
            L.append(f"| Nozzle {name} | **{r['status']}** | position {worst_pos:.3e} | ≤ 1e-9 (position), "
                     "≤ 1e-10 (relative) |")
        else:
            errs = r.get("relative_errors") or {k: v for k, v in r["errors"].items()
                                                 if k.endswith(("_rel", "_abs"))}
            worst = max(errs.values())
            L.append(f"| Nozzle {name} | **{r['status']}** | worst invariant {worst:.3e} | ≤ 1e-10 |")
    L += ["", "## Turbine map (Track 4a)", ""]
    if t.get("status") in ("PASS", "FAIL"):
        w = t["worst"]
        L += [f"- Inputs [tau, gamma, eta_p, cp/R]; eta_p fixed and cp/R = gamma/(gamma − 1), so the "
              f"envelope has **{t['independent_varying_dimensions']} varying independent dimensions**.",
              f"- Training: {rep['registration']['turbine']['training_samples']} scrambled Sobol points; Adam "
              f"{rep['registration']['turbine']['adam_steps']} steps (train MSE {t['train']['train_mse_after_adam']:.3e}), "
              f"L-BFGS {t['train']['lbfgs_n_iter']} iterations / {t['train']['lbfgs_func_evals']} evaluations; final "
              f"train MSE {t['train']['train_mse_final']:.3e} (log units).",
              f"- Score (once, {t['grid_points']} points, {t['grid_training_overlap']} shared with training): max "
              f"relative pressure error **{t['max_relative_pressure_error']:.4e}** (mean "
              f"{t['mean_relative_pressure_error']:.3e}); worst at tau {w['tau']:.6f}, gamma {w['gamma']:.6f}."]
    else:
        L += [f"- {t.get('status', 'ERROR')}: {t.get('reason', '')}. No score was produced."]
    L += [f"- Historical retired turbine PINN (P4.4 attempt 2): **{rep['registration']['turbine']['historic_retired_max_relative_pct']} %** "
          "held-out max |Δp5|/p5. Not like-for-like: that network took [x, cp, R, gamma, tau] on 720 cycle "
          "conditions in float32 with a path loss; this diagnostic is a 2D analytic map in float64."]
    L += ["", "## Manufactured solutions (Track 4b, literal Ma Eqs. 22–26)", "",
          "Dimensionless algebra verifier. Eq. 25's printed coefficient (conductivity + mu_t/Pr) is used "
          "literally; its dimensional meaning is ambiguous and is not repaired here.", "",
          "| Activation | Equation | max abs forced residual | max abs forcing | relative disagreement |",
          "|---|---|---|---|---|"]
    for act, r in m.items():
        for eq, v in r.get("per_equation", {}).items():
            L.append(f"| {act} | {eq} | {v['max_abs_forced_residual']:.3e} | {v['max_abs_forcing']:.3e} | "
                     f"{_fmt(v['relative_forcing_disagreement'])} |")
    L += ["", "| Activation | Omitted term (negative control) | max discrepancy |", "|---|---|---|"]
    for act, r in m.items():
        for om, v in r.get("negative_control_discrepancy", {}).items():
            L.append(f"| {act} | {om} | {v:.3e} |")
    L += ["", "## Exact quasi-1D nozzle ladder (no network)", ""]
    for name, r in nz.items():
        L.append(f"### {name}: {r.get('status', 'ERROR')}")
        L.append("")
        L.append("```json")
        L.append(json.dumps({k: v for k, v in r.items() if k not in ("status", "traceback")}, indent=1))
        L.append("```")
        L.append("")
    L += ["## Identity and reproduction", "",
          f"- Registration `{ident['registration_path']}` sha256 `{ident['registration_sha256']}`.",
          f"- Envelope `{ident['synthetic_envelope']}` sha256 `{ident['synthetic_envelope_sha256']}`.",
          "- Sources: " + ", ".join(f"`{k}` `{v[:16]}…`" for k, v in ident["source_sha256"].items()),
          "- Output hashes: `hashes.json` (written last).",
          f"- Reproduce at git `{ident['git_head']}`: `{REPRO} --output-dir <new empty dir>` "
          "(the registered directory is write-once).", "",
          "## Scope limits and blockers", ""] + [f"- {b}" for b in rep["blocked_or_deferred"]] + [""]
    return "\n".join(L)


# ---------------------------------------------------------------------------

def preflight(reg_path: Path) -> tuple:
    """(reg, reg_bytes, start_identity, env_bytes, blockers). Nothing is written."""
    try:
        reg_bytes = reg_path.read_bytes()
        reg = json.loads(reg_bytes)
        required_nice = int(reg["nice"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, None, None, None, [f"registration unreadable: {exc!r}"]
    blockers = live_blockers(required_nice)
    head = git_head()
    if head is None:
        blockers.append("git HEAD unreadable; refusing (fail closed)")
    dirty = dirty_sources(reg_path)
    if dirty is None:
        blockers.append("git status of the diagnostic sources unreadable; refusing (fail closed)")
    elif dirty.strip():
        blockers.append("diagnostic sources/registration differ from HEAD:\n" + dirty)
    if blockers:
        return reg, reg_bytes, None, None, blockers
    try:
        env_bytes, env_error = _envelope_bytes(reg)
        ident = identity_snapshot(reg_path, reg_bytes, head, reg, env_bytes, env_error)
    except (OSError, KeyError, TypeError) as exc:
        return reg, reg_bytes, None, None, [f"start identity unreadable: {exc!r}"]
    return reg, reg_bytes, ident, env_bytes, []


def execute(reg: dict, reg_path: Path, ident: dict, env_bytes: bytes | None, out: Path) -> int:
    """Everything after the output directory exists. Any failure is logged and hashed."""
    log = RunLog(out / "run_log.txt")
    t0, started, exit_code = time.time(), utc(), 1
    try:
        log(f"START {reg['id']} pid {os.getpid()} out {out} head {ident['git_head']}")
        write_once(out / "config.json", json.dumps({"registration": reg,
                                                    "registration_sha256": ident["registration_sha256"],
                                                    "start_identity": ident}, indent=1))
        import numpy as np
        import torch
        torch.set_num_threads(int(reg["threads"]))
        try:
            torch.set_num_interop_threads(int(reg["threads"]))
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        np.random.seed(int(reg["seed"]))
        torch.manual_seed(int(reg["seed"]))
        from . import nozzle_verification, turbine_map
        env = environment()
        write_once(out / "environment.json", json.dumps(env, indent=1))

        # Track 4a: an envelope or turbine error is confined here
        inputs = None
        try:
            inputs = check_inputs(reg, env_bytes)
            log("inputs verified: envelope sha256 and box extent (round-trip floats)")
            turbine = turbine_map.run(reg, out, log, identity=ident)
        except Exception as exc:
            turbine = error_record(exc)
            log(f"turbine ERROR {exc!r}")
        write_once(out / "turbine_score.json", json.dumps(turbine, indent=1))

        # Track 4b: each activation independently
        manufactured = {}
        for act in reg["manufactured"]["activations"]:
            try:
                manufactured[act] = nozzle_verification.verify_manufactured(reg["manufactured"], act)
                log(f"MMS {act}: {manufactured[act]['status']} abs "
                    f"{manufactured[act]['max_abs_forced_residual']:.3e}")
            except Exception as exc:
                manufactured[act] = error_record(exc)
                log(f"MMS {act} ERROR {exc!r}")
        write_once(out / "manufactured_scores.json", json.dumps(manufactured, indent=1))

        # exact nozzle ladder: downstream of both MMS rungs
        failed = [a for a, r in manufactured.items() if r.get("status") != "PASS"]
        if failed or not manufactured:
            why = "manufactured-solution rung did not pass: " + (", ".join(failed) or "none run")
            nozzle = {k: {"status": "BLOCKED", "reason": why} for k in nozzle_verification.RUNGS}
        else:
            try:
                nozzle = nozzle_verification.run_ladder(reg)
            except Exception as exc:
                nozzle = {"ladder": error_record(exc)}
        for k, v in nozzle.items():
            log(f"nozzle {k}: {v['status']}")
        write_once(out / "nozzle_scores.json", json.dumps(nozzle, indent=1))

        end = end_identity(reg_path, reg)
        drift = identity_drift(ident, end)
        statuses = [turbine.get("status")] + [r.get("status") for r in manufactured.values()] \
            + [r.get("status") for r in nozzle.values()]
        status = "ERROR" if drift else aggregate_status(statuses)
        if drift:
            log(f"identity drift: {drift}")
        rep = {"registration_id": reg["id"], "status": status, "registration": reg, "seed": reg["seed"],
               "started_utc": started, "finished_utc": utc(), "wall_s": time.time() - t0,
               "environment": env, "inputs": inputs, "start_identity": ident, "end_identity": end,
               "identity_drift": drift, "turbine": turbine, "manufactured": manufactured, "nozzle": nozzle,
               "reproduction": [f"git checkout {ident['git_head']}", REPRO + " --output-dir <new empty dir>"],
               "blocked_or_deferred": reg["blocked_or_deferred"]}
        md = markdown_report(rep)
        write_once(out / "report.json", json.dumps(rep, indent=1))
        write_once(out / "report.md", md)
        log(f"aggregate status {status}")
        exit_code = 0 if status == "PASS" else 1
    except BaseException as exc:
        exit_code = 1
        try:
            log(f"ERROR {exc!r}\n{traceback.format_exc()}")
        except Exception:
            print(f"ERROR {exc!r} (run log unwritable)", file=sys.stderr)
        if not isinstance(exc, Exception):
            raise
    finally:
        try:
            log(f"EXIT code {exit_code} wall {time.time() - t0:.1f} s")
        except Exception:
            print(f"EXIT code {exit_code} (run log unwritable)", file=sys.stderr)
        try:
            hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "hashes.json"}
            write_once(out / "hashes.json", json.dumps(hashes, indent=1))
        except Exception as exc:
            print(f"hashes.json not written: {exc!r}", file=sys.stderr)
            exit_code = 1
    return exit_code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--registration", required=True)
    ap.add_argument("--output-dir", default=None, help="override (reproduction only); must not exist")
    args = ap.parse_args(argv)
    reg_path = (ROOT / args.registration).resolve() if not Path(args.registration).is_absolute() \
        else Path(args.registration)

    reg, _reg_bytes, ident, env_bytes, blockers = preflight(reg_path)
    if blockers:
        print("BLOCKED (nothing written):\n- " + "\n- ".join(blockers), file=sys.stderr)
        return 3
    out = Path(args.output_dir) if args.output_dir else ROOT / reg["output_dir"]
    try:
        create_output_dir(out)
    except FileExistsError:
        print(f"REFUSED: {out} exists; results are write-once", file=sys.stderr)
        return 2
    return execute(reg, reg_path, ident, env_bytes, out)


if __name__ == "__main__":
    sys.exit(main())
