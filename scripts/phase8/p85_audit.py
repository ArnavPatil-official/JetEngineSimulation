#!/usr/bin/env python3
"""P8.5-A5 stoichiometry audit and pure G1 gates (docs/phase8_p85_amendment_a5.md).

    nice -n 15 .venv/bin/python scripts/phase8/p85_audit.py

Source audit: each mechanism is loaded as a fresh cantera.Solution from the
unchanged file (never a network phase or a clone). For every reaction j and
element e, in exact rational arithmetic on the loaded float64 values:
delta = sum_k a_ek (nu''_kj - nu'_kj), denominator = max(reactant-side atoms,
product-side atoms), pair defect = |delta| / denominator. B = max over A2NOx.
No threshold discards any defect. CRECK is the balanced control.

Pure scoring (no mechanism, no network): closure against 10*B and the CRECK
control (every case converged), sigma0 with the mixed trace rule, temperature
convergence toward HP with its shortfall labels, the reported-only composition
criterion, and the run guards. Tolerances are the user's mechanism-conditioned
allowance, never derived from observed residuals. Writes
outputs/phase8/p85_a5_audit.json once (exit 2 if it exists; exit 3, nothing
written, if the registration or this file is not committed, the Mac is not on
AC, or the main workflow of docs/phase8_queue_recovery_registration.json holds
its lease or has no terminal chain record).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRATION = ROOT / "docs" / "phase8_p85_a5_registration.json"
AMENDMENT = ROOT / "docs" / "phase8_p85_amendment_a5.md"
SOURCE = ROOT / "scripts" / "phase8" / "p85_audit.py"
VALIDATOR = ROOT / "scripts" / "phase8" / "reactor_validation.py"
if str(ROOT / "scripts" / "phase8") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
from pinn_diagnostics.run_diagnostics import on_mains  # noqa: E402

CLOSURE_MULTIPLIER = 10
CONTROL_BOUND = Fraction(1, 2 ** 52)          # registered CRECK equality rule
CONTROL_CLOSURE = 1e-10
CLOSURE_METRICS = ("energy_relative", "element_relative",
                   "max_mixer_energy_relative", "max_mixer_element_relative")
SHORTFALL_MODES = ("APPROACH", "IDLE")
QUEUE_REGISTRATION = ROOT / "docs" / "phase8_queue_recovery_registration.json"
QUEUE_ID = "P8-QUEUE-RECOVERY-20261003"
TERMINAL_CHAIN = ("COMPLETE", "STOPPED_WITH_BLOCKERS", "FINISHED_WITH_FLAGS_OR_FAILURES")
SCOPE_NOTE = ("B bounds the local stoichiometric defect per reaction event; 10*B is the user's "
              "mechanism-conditioned allowance, not a proof bounding accumulated network element or "
              "enthalpy error. B is not derived from observed residuals or outlets. No claim that "
              "species renormalization preserves a stoichiometric proof.")


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


# ---------------------------------------------------------------------------
# Source audit (pure on plain Python data)
# ---------------------------------------------------------------------------

def audit_reactions(elements: list, atoms: list, weights: list, reactions: list) -> dict:
    """Exact per-reaction, per-element stoichiometric defects.

    atoms[k][e]: atom count of element e in species k (float as loaded);
    weights[k]: molecular weight (kg/kmol); reactions: [{"equation",
    "reactants": {k: nu'}, "products": {k: nu''}}] with the loaded float64
    coefficients. Every value is converted with Fraction(float), which is exact.
    """
    a = [[Fraction(x) for x in row] for row in atoms]
    W = [Fraction(w) for w in weights]
    errors, records = [], []
    B, argmax = Fraction(0), None
    worst = {e: (Fraction(0), None) for e in elements}
    for j, rxn in enumerate(reactions):
        r_f = {int(k): float(v) for k, v in rxn["reactants"].items()}
        p_f = {int(k): float(v) for k, v in rxn["products"].items()}
        nu_r = {k: Fraction(v) for k, v in r_f.items()}
        nu_p = {k: Fraction(v) for k, v in p_f.items()}
        species = sorted(set(nu_r) | set(nu_p))
        dnu = {k: nu_p.get(k, Fraction(0)) - nu_r.get(k, Fraction(0)) for k in species}
        dnu_float = {k: p_f.get(k, 0.0) - r_f.get(k, 0.0) for k in species}
        delta, R, P, d, delta_f = {}, {}, {}, {}, {}
        for i, e in enumerate(elements):
            R[e] = sum((a[k][i] * v for k, v in nu_r.items()), Fraction(0))
            P[e] = sum((a[k][i] * v for k, v in nu_p.items()), Fraction(0))
            delta[e] = P[e] - R[e]
            delta_f[e] = sum(atoms[k][i] * dnu_float[k] for k in species)   # float64 evaluation
            den = max(R[e], P[e])
            if den == 0:
                if delta[e] != 0:
                    errors.append({"index": j, "equation": rxn["equation"], "element": e,
                                   "delta": str(delta[e]), "error": "zero turnover with nonzero defect"})
                d[e] = Fraction(0)
            else:
                d[e] = abs(delta[e]) / den
            if d[e] > B:
                B, argmax = d[e], {"index": j, "equation": rxn["equation"], "element": e}
            if d[e] > worst[e][0]:
                worst[e] = (d[e], j)
        if any(delta[e] != 0 for e in elements):
            mass = sum((W[k] * v for k, v in dnu.items()), Fraction(0))
            records.append({
                "index": j, "equation": rxn["equation"],
                "signed_atoms_exact": {e: str(delta[e]) for e in elements if delta[e] != 0},
                "signed_atoms_float64": {e: delta_f[e] for e in elements if delta[e] != 0},
                "reactant_turnover": {e: str(R[e]) for e in elements if delta[e] != 0},
                "product_turnover": {e: str(P[e]) for e in elements if delta[e] != 0},
                "pair_defect_exact": {e: str(d[e]) for e in elements if delta[e] != 0},
                "pair_defect_float": {e: float(d[e]) for e in elements if delta[e] != 0},
                "mass_defect_kg_kmol_exact": str(mass),
                "mass_defect_kg_kmol_float64": sum(weights[k] * dnu_float[k] for k in species)})
    return {"n_reactions": len(reactions), "elements": list(elements),
            "n_pairs": len(reactions) * len(elements),
            "n_reactions_with_nonzero_defect": len(records), "errors": errors,
            "B_exact": str(B), "B_float": float(B), "argmax": argmax,
            "max_by_element": {e: {"exact": str(v), "float": float(v), "index": j}
                               for e, (v, j) in worst.items()},
            "threshold": None, "reactions_with_nonzero_defect": records}


def audit_B(audit: dict) -> Fraction:
    return Fraction(audit["B_exact"])


def allowance_gate(audit: dict) -> dict:
    """A2NOx: B finite and strictly positive, no audit error."""
    B = audit_B(audit)
    reasons = []
    if audit["errors"]:
        reasons.append(f"{len(audit['errors'])} audit errors")
    if not math.isfinite(float(B)) or B <= 0:
        reasons.append("B is not finite and strictly positive")
    return {"pass": not reasons, "reasons": reasons}


def control_gate(audit: dict, bound: Fraction = CONTROL_BOUND) -> dict:
    """CRECK: balanced to floating representation accuracy (max pair defect <= 2^-52)."""
    B = audit_B(audit)
    reasons = []
    if audit["errors"]:
        reasons.append(f"{len(audit['errors'])} audit errors")
    if B > bound:
        reasons.append(f"max pair defect {float(B):.3e} exceeds {float(bound):.3e}")
    return {"pass": not reasons, "reasons": reasons, "rule": "max pair defect <= 2**-52, exact",
            "max_pair_defect_float": float(B)}


def closure_tolerance(B: Fraction, multiplier: int = CLOSURE_MULTIPLIER) -> Fraction:
    if not isinstance(B, Fraction) or not math.isfinite(float(B)) or B <= 0:
        raise ValueError("B must be a finite, strictly positive Fraction")
    return multiplier * B


def strictly_below(value, tol) -> bool:
    """Finite value strictly below tol; exact rational comparison."""
    if tol is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not math.isfinite(value):
        return False
    return Fraction(value) < Fraction(tol)


def closure_gate(cases: dict, tol, metrics=CLOSURE_METRICS) -> dict:
    """Every case converged (all_converged is True) and every metric finite and strictly
    below tol (None: no tolerance, fail). Small closures of a failed solve never pass."""
    out = {}
    for name, vals in cases.items():
        out[name] = {m: {"value": vals.get(m), "pass": strictly_below(vals.get(m), tol)} for m in metrics}
        out[name]["all_converged"] = {"value": vals.get("all_converged"),
                                      "pass": vals.get("all_converged") is True}
    ok = tol is not None and bool(out) and all(v["pass"] for c in out.values() for v in c.values())
    return {"tolerance_float": None if tol is None else float(tol),
            "tolerance_exact": None if tol is None else str(Fraction(tol)), "cases": out, "pass": ok}


def control_closure_gate(runs: dict, control_audit_pass: bool, tol: float = CONTROL_CLOSURE) -> dict:
    """CRECK base-value runs: all converged and every closure strictly below 1e-10."""
    closure = closure_gate(runs, tol)
    return {"audit_control_pass": bool(control_audit_pass), "closure": closure,
            "pass": bool(control_audit_pass) and closure["pass"]}


# ---------------------------------------------------------------------------
# sigma0 and long-residence gates
# ---------------------------------------------------------------------------

def sigma0_gate(T: list, Y: list, converged: list, names: list, t_tol: float = 1e-12,
                reference_cut: float = 1e-8, rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> dict:
    """Primary PSRs identical at sigma = 0. Y[i][k]: PSR i, species k.

    Species with reference max|Y| > reference_cut use the relative spread;
    the others (reference equal to the cut, or zero) the absolute spread.
    """
    finite = all(math.isfinite(t) for t in T) and all(math.isfinite(y) for row in Y for y in row)
    all_conv = bool(converged) and all(bool(c) for c in converged)
    t_rel = (max(T) - min(T)) / max(T) if finite and T and max(T) > 0 else math.inf
    species = []
    for k, name in enumerate(names):
        col = [row[k] for row in Y]
        ref = max(abs(y) for y in col)
        spread = max(col) - min(col)
        if ref > reference_cut:
            rule, metric, ok = "relative", spread / ref, spread / ref <= rel_tol
        else:
            rule, metric, ok = "absolute", spread, spread <= abs_tol
        species.append({"name": name, "reference": ref, "spread": spread, "rule": rule,
                        "metric": metric, "pass": bool(ok and math.isfinite(metric))})
    failing = [s["name"] for s in species if not s["pass"]]
    t_ok = t_rel <= t_tol
    return {"primary_converged": all_conv, "finite": finite, "T_relative_spread": t_rel, "T_pass": bool(t_ok),
            "n_species_fail": len(failing), "failing_species": failing, "species": species,
            "pass": bool(all_conv and finite and t_ok and not failing)}


def shortfall_label(mode: str, converged: bool, abs_dT: float, errors: list, tol: float = 0.1) -> str:
    """Only a finite, converged APPROACH/IDLE shortfall may be called a kinetics/extinction
    limit; nonconvergence and integrator (CVODES) errors stay numerical failures."""
    if not converged or errors or not math.isfinite(abs_dT):
        return "numerical failure (nonconvergence or integrator error); not physics unless independently diagnosed"
    if abs_dT < tol:
        return "none"
    if mode in SHORTFALL_MODES:
        return "kinetics/extinction limit at this residence time (finite, converged shortfall)"
    return "finite, converged shortfall; not labelled physics at this mode"


def temperature_convergence_gate(runs: list, T_eq: float, tol: float = 0.1, mode: str = "") -> dict:
    """runs: [{"label", "scale", "converged", "T", "errors"}] in increasing scale order
    ("errors": integrator messages of the run's PSRs, optional).

    Pass iff every solve converged, |T - T_eq| is finite and non-increasing
    exactly, and the final error is strictly below tol.
    """
    errs = [abs(r["T"] - T_eq) for r in runs]
    finite = math.isfinite(T_eq) and all(math.isfinite(e) for e in errs)
    conv = {r["label"]: bool(r["converged"]) for r in runs}
    monotone = finite and all(errs[i + 1] <= errs[i] for i in range(len(errs) - 1))
    final_ok = finite and bool(errs) and errs[-1] < tol
    return {"T_equilibrium": T_eq, "abs_dT_K": {r["label"]: e for r, e in zip(runs, errs)},
            "scales": {r["label"]: r["scale"] for r in runs}, "converged": conv,
            "monotone_non_increasing": bool(monotone), "final_abs_dT_K": errs[-1] if errs else None,
            "final_strictly_below_K": tol, "final_pass": bool(final_ok),
            "shortfall": {r["label"]: shortfall_label(mode, r["converged"], e, r.get("errors", []), tol)
                          for r, e in zip(runs, errs)},
            "pass": bool(runs and all(conv.values()) and monotone and final_ok)}


def composition_report(Y_net: list, Y_eq: list, names: list, floor: float = 1e-6, tol: float = 1e-5) -> dict:
    """Old G1.1 composition criterion: REPORTED ONLY, never gating."""
    worst, name = -1.0, None
    for y, ye, n in zip(Y_net, Y_eq, names):
        if ye > floor and abs(y - ye) > worst:
            worst, name = abs(y - ye), n
    met = name is not None and worst <= tol
    return {"reported_only": True, "Y_floor": floor, "criterion_max_dY": tol,
            "max_dY_where_Yeq_gt_floor": worst if name is not None else None, "argmax_species": name,
            "criterion_met": bool(met),
            "claim": ("composition also within the old criterion" if met else
                      "no full-state equilibrium claimed; temperature convergence only")}


def a5_verdict(allowance: dict, g11: dict, g12_a2: dict, g12_control: dict, g13: dict) -> tuple:
    checks = {"A5_audit_B_finite_positive": bool(allowance["pass"]),
              "G1.1_temperature_convergence_toward_HP": all(v["pass"] for v in g11.values()) and bool(g11),
              "G1.2_A2NOx_closure_below_10B": all(v["pass"] for v in g12_a2.values()) and bool(g12_a2),
              "G1.2_CRECK_control": bool(g12_control["pass"]),
              "G1.3_sigma0_identical": all(v["pass"] for v in g13.values()) and bool(g13)}
    return ("PASS" if all(checks.values()) else "FAIL"), checks


# ---------------------------------------------------------------------------
# Run guards (pure on command output)
# ---------------------------------------------------------------------------

def workflow_blockers(queue_reg: dict | None, queue_reg_sha256: str | None, lease_exists: bool,
                      chain_records: list) -> list:
    """Main workflow state from its registered records, never from process names:
    no owner lease on disk (live or stale) and a terminal chain record of the same
    registration (id and sha256) with a status in TERMINAL_CHAIN."""
    if queue_reg is None:
        return [f"main workflow registration {rel(QUEUE_REGISTRATION)} absent or unreadable "
                "(integrate this worktree into main first)"]
    out = []
    if queue_reg.get("id") != QUEUE_ID:
        out.append(f"main workflow registration id is {queue_reg.get('id')!r}, expected {QUEUE_ID}")
    if lease_exists:
        out.append(f"main workflow lease {queue_reg.get('lease_path')} is held (live or stale)")
    terminal = [r for r in chain_records if isinstance(r, dict) and r.get("registration_id") == QUEUE_ID
                and r.get("registration_sha256") == queue_reg_sha256 and r.get("status") in TERMINAL_CHAIN]
    if not terminal:
        out.append("no terminal chain record of the main workflow")
    return out


def run_blockers(power_output: str | None, queue_reg: dict | None, queue_reg_sha256: str | None,
                 lease_exists: bool, chain_records: list) -> list:
    out = [] if on_mains(power_output) else ["not on mains (AC) power, or power source unknown"]
    return out + workflow_blockers(queue_reg, queue_reg_sha256, lease_exists, chain_records)


def commit_blockers(paths: list, ls_files_output: str | None, status_output: str | None) -> list:
    """Each path tracked (``git ls-files``) and unmodified (``git status --porcelain``)."""
    if ls_files_output is None or status_output is None:
        return ["git ls-files/status unreadable; refusing (fail closed)"]
    tracked = set(ls_files_output.split())
    out = [f"{p} is not committed" for p in paths if p not in tracked]
    if status_output.strip():
        out.append("uncommitted changes: " + " | ".join(status_output.strip().splitlines()))
    return out


def audit_match_blockers(audit_doc: dict, current: dict) -> list:
    """The committed audit must match the current registration, audit source and mechanisms."""
    out = []
    for key in ("registration_sha256", "audit_source_sha256", "a2nox_sha256", "creck_sha256"):
        if audit_doc.get("identity", {}).get(key) != current.get(key):
            out.append(f"audit {key} does not match the current file")
    if "B_exact" not in audit_doc.get("a2nox", {}):
        out.append("audit has no A2NOx B")
    return out


def cmd(args: list) -> str | None:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit_blockers(paths: list) -> list:
    rels = [rel(p) for p in paths]
    return commit_blockers(rels, cmd(["git", "ls-files", "--", *rels]),
                           cmd(["git", "status", "--porcelain", "--", *rels]))


def git_head() -> str | None:
    head = (cmd(["git", "rev-parse", "--verify", "HEAD"]) or "").strip()
    return head if len(head) == 40 else None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def live_run_blockers() -> list:
    """AC (``pmset -g ps``, as the main workflow reads it) and the main workflow records."""
    power = cmd(["pmset", "-g", "ps"])
    try:
        qbytes = QUEUE_REGISTRATION.read_bytes()
        q = json.loads(qbytes)
        lease = ROOT / q["lease_path"]
        pattern = Path(q["records"]["chain_record"])
    except (OSError, ValueError, KeyError, TypeError):
        return run_blockers(power, None, None, False, [])
    records = [_read_json(p) for p in sorted((ROOT / pattern.parent).glob(pattern.name.replace("<session>", "*")))]
    return run_blockers(power, q, hashlib.sha256(qbytes).hexdigest(), os.path.lexists(lease), records)


# ---------------------------------------------------------------------------
# Mechanism loading (the actual audit) and CLI
# ---------------------------------------------------------------------------

def load_mechanism(path: Path, expected_sha256: str) -> dict:
    """Fresh Solution from the unchanged file; plain data for audit_reactions."""
    import cantera as ct
    import numpy as np

    if sha256(path) != expected_sha256:
        raise AuditError(f"{rel(path)} sha256 differs from the registration")
    g = ct.Solution(str(path))
    if sha256(path) != expected_sha256:
        raise AuditError(f"{rel(path)} changed while loading")

    def dense(m):
        return m.toarray() if hasattr(m, "toarray") else np.asarray(m)

    nu_r, nu_p = dense(g.reactant_stoich_coeffs), dense(g.product_stoich_coeffs)
    reactions = []
    for j in range(g.n_reactions):
        reactions.append({"equation": g.reaction(j).equation,
                          "reactants": {int(k): float(nu_r[k, j]) for k in np.nonzero(nu_r[:, j])[0]},
                          "products": {int(k): float(nu_p[k, j]) for k in np.nonzero(nu_p[:, j])[0]}})
    return {"elements": list(g.element_names), "species": list(g.species_names),
            "atoms": [[float(g.n_atoms(k, e)) for e in g.element_names] for k in range(g.n_species)],
            "weights": [float(w) for w in g.molecular_weights], "reactions": reactions,
            "cantera": ct.__version__}


def audit_mechanisms(reg: dict) -> dict:
    """Both registered mechanisms, audited from their files."""
    out = {}
    for key in ("a2nox", "creck"):
        m = reg["mechanisms"][key]
        data = load_mechanism(ROOT / m["path"], m["sha256"])
        out[key] = {"path": m["path"], "sha256": m["sha256"], "n_species": len(data["species"]),
                    "cantera": data["cantera"],
                    **audit_reactions(data["elements"], data["atoms"], data["weights"], data["reactions"])}
    return out


def current_identity(reg_bytes: bytes, reg: dict) -> dict:
    return {"registration_sha256": hashlib.sha256(reg_bytes).hexdigest(),
            "amendment_sha256": sha256(AMENDMENT), "audit_source_sha256": sha256(SOURCE),
            "a2nox_sha256": sha256(ROOT / reg["mechanisms"]["a2nox"]["path"]),
            "creck_sha256": sha256(ROOT / reg["mechanisms"]["creck"]["path"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    reg_bytes = REGISTRATION.read_bytes()
    reg = json.loads(reg_bytes)
    blockers = git_commit_blockers([REGISTRATION, AMENDMENT, SOURCE]) + live_run_blockers()
    head = git_head()
    if head is None:
        blockers.append("git HEAD unreadable")
    if blockers:
        print("BLOCKED (nothing written):\n- " + "\n- ".join(blockers), file=sys.stderr)
        return 3
    out = ROOT / reg["outputs"]["audit"]
    if out.exists():
        print(f"REFUSED: {out} exists; the audit is write-once", file=sys.stderr)
        return 2
    from benchmark import protected_check

    protected = protected_check()
    if protected["mismatches"]:
        print(f"BLOCKED: protected hashes changed: {protected['mismatches'][:5]}", file=sys.stderr)
        return 3
    ident = {"git_head": head, **current_identity(reg_bytes, reg)}
    audits = audit_mechanisms(reg)
    allowance, control = allowance_gate(audits["a2nox"]), control_gate(audits["creck"])
    tol = closure_tolerance(audit_B(audits["a2nox"])) if allowance["pass"] else None
    doc = {"registration": reg["id"], "a2nox": audits["a2nox"], "creck": audits["creck"],
           "allowance_gate": allowance, "creck_control_gate": control,
           "closure_tolerance_10B_exact": None if tol is None else str(tol),
           "closure_tolerance_10B_float": None if tol is None else float(tol),
           "creck_closure_tolerance": CONTROL_CLOSURE, "scope_note": SCOPE_NOTE,
           "identity": ident,
           "end_identity": {"git_head": git_head(), **current_identity(REGISTRATION.read_bytes(), reg)},
           "machine": platform.platform(), "protected": protected}
    doc["identity_drift"] = [k for k in ident if doc["end_identity"].get(k) != ident[k]]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        f.write(json.dumps(doc, indent=2) + "\n")
    print(f"P8.5-A5 audit: B = {float(audit_B(audits['a2nox'])):.6e} "
          f"allowance {allowance['pass']} CRECK control {control['pass']} drift {doc['identity_drift']}")
    return 0 if allowance["pass"] and control["pass"] and not doc["identity_drift"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
