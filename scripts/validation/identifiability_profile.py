#!/usr/bin/env python3
"""
P5.2 Step 1 — Identifiability profile of the LTO calibration.

For every fitted parameter of a frozen calibration, fix it at points across its
search box, re-optimise every other fitted parameter within its box, and record
the best objective at each point — the profile of the objective. Then run the
explicit ridge check of Phase 5 finding F2 (docs/plan.md): move k_mdot while
re-solving phi_idle / phi_app so idle and approach fuel flow are unchanged, and
see whether the objective moves.

STRUCTURAL BASIS (v2-v4 objective ``fuel_flow_v4``). The cycle sets fuel flow
before any thermodynamics that depends on eta_b, pressure_loss or k_pi:

    FAR = phi * f_st                     (Cantera set_equivalence_ratio; exact)
    m_fuel(mode) = phi_mode * f_st * beta * m_rated * x_mode^k_mdot

so the objective (mean |relative fuel-flow error| over idle/approach/take-off)
is a closed-form function of (k_mdot, phi_idle, phi_app, phi_to), and eta_b,
pressure_loss and k_pi enter it only through whether the cycle *runs* (a
crashed evaluation scores CRASH_PENALTY in calibrate_lto). The profile is
therefore computed exactly from the closed form: with k_mdot fixed each free
phi is its own target's solution clipped to its box; with k_mdot free the
inner problem is one-dimensional in k_mdot (dense grid + bounded Brent refine).
The closed form is cross-checked against real cycle evaluations — at the frozen
point, along the F2 ridge, at every fitted parameter's box ends, and at each
profile's minimiser and box-edge re-optimisations — and every cycle crash
(the feasibility constraint the closed form cannot see) is reported.

Other objectives (e.g. a future fuel-flow + thrust v5 objective) have no closed
form: ``PROFILERS`` has no entry for them and the script refuses, rather than
silently profiling the fuel-only shortcut. A cycle-based inner optimiser plugs
into ``profile_param`` through the same ``inner_min`` callable.

VERDICT RULE (unchanged from the first draft, fixed before any profile was
run; objective units are the calibration's own — 0.001 = 0.1 pp):

A parameter is IDENTIFIED iff all three hold:
  1. the profile rises by at least MARGIN above its minimum at BOTH box edges;
  2. the DELTA-sublevel interval {theta : profile(theta) <= min + DELTA} is
     interior (touches neither box edge) — a minimum on a search-box edge is
     set by the prior box, not by the data;
  3. that interval is narrower than WIDTH_FRAC of the box — a flat valley
     (a ridge of equal objective, F2) rises at the box edges only because the
     other parameters run into their own boxes; its width exposes it.

Outputs (tag from the calibration file name, e.g. ``_v4``):
    outputs/identifiability_profile<tag>.json   verdicts, criteria, ridge, cycle checks
    outputs/identifiability_profile<tag>.csv    every profile point

Usage::

    python scripts/validation/identifiability_profile.py \\
        --calibration outputs/calibration_trent1000_ae3_v4.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "optimization"))

import calibrate_lto as cal  # noqa: E402

logging.getLogger("cantera").setLevel(logging.ERROR)

# ---- verdict rule (see module docstring; fixed before any run) ----
MARGIN = 0.005       # rise at both box edges: 0.5 percentage points
DELTA = 1e-4         # sublevel threshold: 0.01 percentage points
WIDTH_FRAC = 0.02    # identified interval narrower than 2 % of the box
# ---- profile resolution ----
N_GRID = 401         # profile grid across each box (plus the calibrated value)
N_K = 4001           # inner k_mdot grid when k_mdot is free (then Brent-refined)
N_BISECT = 40        # bisection steps locating each end of the DELTA interval
RIDGE_K = (0.58, 0.60)   # F2 ridge points (the calibrated k_mdot is added)
CYCLE_RTOL = 1e-9    # closed form vs cycle: relative fuel-flow agreement required


# --------------------------------------------------------------------------
# Calibration record
# --------------------------------------------------------------------------
def load_calibration(path: Path) -> dict:
    rec = json.loads(path.read_text())
    fixed = rec.get("fixed_parameters", {})
    best = dict(rec["best_params"])
    # v5+ records list what was fitted and what was fixed; v2-v4 sampled all seven
    fitted = list(rec.get("fitted_parameters") or best.keys())
    full = {k: best.get(k, fixed.get(k)) for k in cal.PARAM_BOUNDS}
    missing = [k for k, v in full.items() if v is None]
    if missing:
        raise ValueError(f"{path.name}: no value for {missing}")
    bounds = {k: tuple(v) for k, v in (rec.get("search_bounds") or cal.PARAM_BOUNDS).items()}
    return {
        "path": path, "record": rec, "fitted": fitted, "values": full, "bounds": bounds,
        "beta": fixed.get("combustor_air_fraction", 1.0),
        "xi": fixed.get("combustor_heat_loss_fraction", 0.0),
        "objective": (rec.get("objective") or {}).get("name", "fuel_flow_v4"),
    }


def new_engine():
    with contextlib.redirect_stdout(io.StringIO()):
        return cal.IntegratedTurbofanEngine()


# --------------------------------------------------------------------------
# Closed-form fuel-flow objective (fuel_flow_v4 only)
# --------------------------------------------------------------------------
class FuelFlowProfile:
    """Exact profile of the v2-v4 fuel-flow objective (see module docstring)."""

    name = "fuel_flow_v4"
    INERT = ("eta_combustor", "pressure_loss", "k_pi")   # absent from m_fuel by construction

    def __init__(self, c: dict, engine=None):
        engine = engine or new_engine()
        with contextlib.redirect_stdout(io.StringIO()):
            self.f_st = float(engine._calculate_fuel_air_ratio(cal.FUEL_LIBRARY["Jet-A1"], 1.0))
        self.c = c
        self.modes = list(cal.ICAO_TARGETS)
        self.x = np.array([cal.ICAO_TARGETS[m]["power_fraction"] for m in self.modes])
        self.t = np.array([cal.ICAO_TARGETS[m]["fuel_flow_kg_s"] for m in self.modes])
        self.phi_keys = [cal.PHI_KEYS[m] for m in self.modes]
        self.scale = self.f_st * c["beta"] * cal.BASE_AIRFLOW   # m_fuel = phi * scale * x^k

    def fuel_flow(self, p: dict) -> dict:
        return {m: p[k] * self.scale * xm ** p["k_mdot"]
                for m, k, xm in zip(self.modes, self.phi_keys, self.x)}

    def __call__(self, p: dict) -> float:
        ff = self.fuel_flow(p)
        return float(np.mean([abs(ff[m] - t) / t for m, t in zip(self.modes, self.t)]))

    def _phis(self, k, p: dict, free: set) -> np.ndarray:
        """phi per mode (rows) for each k (columns): free phis solve their own target, clipped."""
        k = np.atleast_1d(np.asarray(k, dtype=float))
        out = np.empty((len(self.modes), k.size))
        for i, key in enumerate(self.phi_keys):
            if key in free:
                lo, hi = self.c["bounds"][key]
                out[i] = np.clip(self.t[i] / (self.scale * self.x[i] ** k), lo, hi)
            else:
                out[i] = p[key]
        return out

    def _g(self, k, p: dict, free: set) -> np.ndarray:
        k = np.atleast_1d(np.asarray(k, dtype=float))
        ff = self._phis(k, p, free) * self.scale * self.x[:, None] ** k[None, :]
        return np.mean(np.abs(ff - self.t[:, None]) / self.t[:, None], axis=0)

    def inner_min(self, name: str, value: float) -> tuple[float, dict]:
        """min over the other fitted parameters, within their boxes, with ``name`` = ``value``."""
        p = dict(self.c["values"])
        p[name] = value
        free = set(self.c["fitted"]) - {name}
        if "k_mdot" in free:
            lo, hi = self.c["bounds"]["k_mdot"]
            ks = np.linspace(lo, hi, N_K)
            g = self._g(ks, p, free)
            i = int(np.argmin(g))
            a, b = ks[max(i - 1, 0)], ks[min(i + 1, N_K - 1)]
            r = minimize_scalar(lambda k: float(self._g(k, p, free)[0]), bounds=(a, b),
                                method="bounded", options={"xatol": 1e-12})
            k_best = float(r.x) if r.fun <= g[i] else float(ks[i])
            p["k_mdot"] = k_best
        phis = self._phis(p["k_mdot"], p, free)[:, 0]
        for key, v in zip(self.phi_keys, phis):
            p[key] = float(v)
        # inert parameters do not enter m_fuel; they stay at the frozen values
        return self(p), p


PROFILERS = {FuelFlowProfile.name: FuelFlowProfile}


# --------------------------------------------------------------------------
# Profile and verdict (objective-agnostic: takes any inner_min)
# --------------------------------------------------------------------------
def profile_param(inner_min, c: dict, name: str) -> dict:
    lo, hi = c["bounds"][name]
    cal_val = float(np.clip(c["values"][name], lo, hi))
    grid = np.unique(np.append(np.linspace(lo, hi, N_GRID), cal_val))
    res = [inner_min(name, float(g)) for g in grid]
    f = np.array([r[0] for r in res])
    points = [(float(g), float(v)) for g, v in zip(grid, f)]
    i_min = int(np.argmin(f))
    fmin = float(f[i_min])
    inside = np.where(f <= fmin + DELTA)[0]
    a, b = int(inside.min()), int(inside.max())

    def crossing(i_in: int, i_out: int) -> float:
        """Bisect between an in-set and an out-of-set grid point for profile = fmin + DELTA."""
        x_in, x_out = float(grid[i_in]), float(grid[i_out])
        for _ in range(N_BISECT):
            mid = 0.5 * (x_in + x_out)
            fm = inner_min(name, mid)[0]
            points.append((mid, float(fm)))
            if fm <= fmin + DELTA:
                x_in = mid
            else:
                x_out = mid
        return 0.5 * (x_in + x_out)

    touches_lo, touches_hi = a == 0, b == len(grid) - 1
    iv_lo = float(grid[0]) if touches_lo else crossing(a, a - 1)
    iv_hi = float(grid[-1]) if touches_hi else crossing(b, b + 1)
    rise_lo, rise_hi = float(f[0] - fmin), float(f[-1] - fmin)
    width_frac = (iv_hi - iv_lo) / (hi - lo)
    identified = (rise_lo >= MARGIN and rise_hi >= MARGIN
                  and not touches_lo and not touches_hi and width_frac < WIDTH_FRAC)
    reasons = []
    if rise_lo < MARGIN or rise_hi < MARGIN:
        reasons.append(f"edge rise {rise_lo:.2e} / {rise_hi:.2e} < margin {MARGIN}")
    if touches_lo or touches_hi:
        reasons.append("minimum interval touches the "
                       + " and ".join(s for s, t in (("lower", touches_lo), ("upper", touches_hi)) if t)
                       + " box edge")
    if width_frac >= WIDTH_FRAC:
        reasons.append(f"flat valley: interval {width_frac:.1%} of the box >= {WIDTH_FRAC:.0%}")
    return {
        "parameter": name, "bounds": [lo, hi], "calibrated": cal_val,
        "objective_at_calibrated_value": float(f[int(np.argmin(np.abs(grid - cal_val)))]),
        "profile_min": fmin, "argmin": float(grid[i_min]),
        "edge_rise_lower": rise_lo, "edge_rise_upper": rise_hi,
        "interval": [iv_lo, iv_hi], "interval_width_frac": width_frac,
        "touches_lower": touches_lo, "touches_upper": touches_hi,
        "identified": bool(identified),
        "why_not": "; ".join(reasons) if not identified else "",
        "points": sorted(points),
        "argmin_params": res[i_min][1],
        "edge_params": {"lower": res[0][1], "upper": res[-1][1]},
    }


def ridge_points(c: dict) -> list[dict]:
    """F2: phi_mode(k) = phi_mode(k0) * x_mode^(k0 - k) holds idle and approach fuel flow fixed.

    Box limits are deliberately ignored: the question is what the data can see.
    """
    v = c["values"]
    k0 = v["k_mdot"]
    out = []
    for k in sorted(set(RIDGE_K) | {k0}):
        p = dict(v)
        p["k_mdot"] = k
        for mode, key in (("Idle", "phi_idle"), ("Approach", "phi_app")):
            p[key] = v[key] * cal.ICAO_TARGETS[mode]["power_fraction"] ** (k0 - k)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# Cycle cross-check (real Cantera cycle, no patching)
# --------------------------------------------------------------------------
class CycleRunner:
    """Real ``calibrate_lto.run_lto_modes`` per mode, cached on the inputs that mode depends on."""

    def __init__(self, c: dict, engine=None):
        self.engine = engine or new_engine()
        self.c = c
        self.cache: dict = {}
        self.n_cycles = 0

    def perf(self, p: dict) -> dict:
        """{mode: performance dict} or {mode: {'crash': message}}."""
        out = {}
        for mode in cal.ICAO_TARGETS:
            key = (mode, p["eta_combustor"], p["pressure_loss"], p["k_pi"], p["k_mdot"],
                   p[cal.PHI_KEYS[mode]])
            if key not in self.cache:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.cache[key] = cal.run_lto_modes(self.engine, p, self.c["beta"],
                                                            self.c["xi"], modes=(mode,))[mode]
                except Exception as exc:  # noqa: BLE001 — recorded as a feasibility finding
                    self.cache[key] = {"crash": f"{type(exc).__name__}: {exc}"}
                self.n_cycles += 1
            out[mode] = self.cache[key]
        return out


def cross_check(cyc: CycleRunner, model: FuelFlowProfile, p: dict, label: str) -> dict:
    perf = cyc.perf(p)
    crashed = {m: v["crash"] for m, v in perf.items() if "crash" in v}
    ff_model = model.fuel_flow(p)
    row = {"label": label, "params": p, "crashed_modes": crashed,
           "objective_closed_form": model(p)}
    if crashed:
        row.update(objective_cycle=cal.CRASH_PENALTY, max_rel_fuel_flow_diff=None, agrees=None)
        return row
    diffs = [abs(perf[m]["fuel_mass_flow"] - ff_model[m]) / ff_model[m] for m in perf]
    row.update(
        objective_cycle=float(cal.fuel_flow_error(perf)),
        max_rel_fuel_flow_diff=float(max(diffs)),
        agrees=bool(max(diffs) <= CYCLE_RTOL),
        **{f"thrust_{m}_kN": float(perf[m]["thrust_N"]) / 1e3 for m in perf},
    )
    return row


def representative_points(c: dict) -> list[tuple[str, dict]]:
    out = [("frozen calibration", dict(c["values"]))]
    for k in c["fitted"]:
        for side, v in zip(("lower", "upper"), c["bounds"][k]):
            p = dict(c["values"])
            p[k] = v
            out.append((f"{k} at {side} box edge ({v})", p))
    return out


# --------------------------------------------------------------------------
def run(calibration: Path, params: list[str] | None = None, cycle_checks: bool = True) -> dict:
    c = load_calibration(calibration.resolve())
    if c["objective"] not in PROFILERS:
        raise SystemExit(
            f"No closed-form profile for objective {c['objective']!r}. The fuel-flow shortcut applies "
            "only to 'fuel_flow_v4'; profiling this objective needs a cycle-based inner optimiser "
            "passed to profile_param() and validated like the fuel-flow one.")
    t0 = time.time()
    engine = new_engine()
    model = PROFILERS[c["objective"]](c, engine)
    profiles = [profile_param(model.inner_min, c, n) for n in (params or c["fitted"])]
    ridge = [{"k_mdot": p["k_mdot"], "phi_idle": p["phi_idle"], "phi_app": p["phi_app"],
              "objective_closed_form": model(p),
              "in_box": all(c["bounds"][q][0] <= p[q] <= c["bounds"][q][1] for q in ("phi_idle", "phi_app")),
              "params": p}
             for p in ridge_points(c)]
    out = {
        "calibration": str(c["path"].relative_to(REPO_ROOT)),
        "objective": c["objective"],
        "method": "closed-form fuel-flow objective (exact inner minimisation); cycle cross-checked",
        "f_st": model.f_st, "beta": c["beta"], "m_rated_kg_s": cal.BASE_AIRFLOW,
        "objective_at_calibration": model(c["values"]),
        "fitted": c["fitted"],
        "inert_in_objective": [k for k in FuelFlowProfile.INERT if k in c["fitted"]],
        "criteria": {"MARGIN": MARGIN, "DELTA": DELTA, "WIDTH_FRAC": WIDTH_FRAC,
                     "N_GRID": N_GRID, "N_K": N_K, "N_BISECT": N_BISECT, "CYCLE_RTOL": CYCLE_RTOL},
        "identified": [p["parameter"] for p in profiles if p["identified"]],
        "not_identified": [p["parameter"] for p in profiles if not p["identified"]],
        "profiles": profiles,
    }
    objs = [r["objective_closed_form"] for r in ridge]
    out["ridge_check"] = {"rows": ridge, "objective_spread": float(max(objs) - min(objs)),
                          "ridge_present": bool(max(objs) - min(objs) <= DELTA)}
    if cycle_checks:
        cyc = CycleRunner(c, engine)
        checks = [cross_check(cyc, model, p, lbl) for lbl, p in representative_points(c)]
        rrows = []
        for r in ridge:
            chk = cross_check(cyc, model, r["params"], f"ridge k_mdot={r['k_mdot']:.4f}")
            r.update({k: v for k, v in chk.items() if k.startswith(("thrust_", "objective_cycle"))})
            rrows.append(chk)
        for pr in profiles:
            for lbl, p in (("profile minimiser", pr["argmin_params"]),
                           ("re-optimised at lower edge", pr["edge_params"]["lower"]),
                           ("re-optimised at upper edge", pr["edge_params"]["upper"])):
                checks.append(cross_check(cyc, model, p, f"{pr['parameter']}: {lbl}"))
        checks += rrows
        thr = [r["thrust_Idle_kN"] for r in ridge if "thrust_Idle_kN" in r]
        if thr:
            out["ridge_check"]["idle_thrust_spread_pct"] = float(100 * (max(thr) - min(thr)) / np.mean(thr))
        ok = [x for x in checks if x["agrees"] is not None]
        out["cycle_cross_check"] = {
            "n_points": len(checks), "n_cycle_mode_runs": cyc.n_cycles,
            "all_feasible_points_agree": all(x["agrees"] for x in ok),
            "max_rel_fuel_flow_diff": max(x["max_rel_fuel_flow_diff"] for x in ok) if ok else None,
            "crashed_points": [{"label": x["label"], "crashed_modes": x["crashed_modes"]}
                               for x in checks if x["crashed_modes"]],
            "points": checks,
        }
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibration", required=True, type=Path)
    ap.add_argument("--params", nargs="*", help="profile only these (default: every fitted parameter)")
    ap.add_argument("--out-tag", default=None, help="output suffix (default from the calibration file name)")
    ap.add_argument("--no-cycle-checks", action="store_true", help="skip the Cantera cross-check (tests only)")
    args = ap.parse_args()
    tag = args.out_tag if args.out_tag is not None else "_" + args.calibration.stem.rsplit("_", 1)[-1]

    out = run(args.calibration, args.params, cycle_checks=not args.no_cycle_checks)
    print(f"Calibration {out['calibration']}  objective {out['objective']}  fitted {out['fitted']}")
    print(f"f_st {out['f_st']:.10f}; objective at the frozen point {out['objective_at_calibration']:.6f}")
    for pr in out["profiles"]:
        print(f"  {pr['parameter']:<14} min {pr['profile_min']:.6f} at {pr['argmin']:.4f}  "
              f"edge rise {pr['edge_rise_lower']:.4f}/{pr['edge_rise_upper']:.4f}  "
              f"interval [{pr['interval'][0]:.4f}, {pr['interval'][1]:.4f}] "
              f"({pr['interval_width_frac']:.1%})  -> "
              f"{'IDENTIFIED' if pr['identified'] else 'not identified: ' + pr['why_not']}")
    rc = out["ridge_check"]
    print(f"Ridge check (F2): objective spread {rc['objective_spread']:.2e} -> "
          f"ridge {'PRESENT' if rc['ridge_present'] else 'broken'}")
    for r in rc["rows"]:
        print(f"  k_mdot {r['k_mdot']:.4f}  phi_idle {r['phi_idle']:.4f}  phi_app {r['phi_app']:.4f}  "
              f"obj {r['objective_closed_form']:.6f}"
              + (f" (cycle {r['objective_cycle']:.6f})  idle {r['thrust_Idle_kN']:.2f} kN  "
                 f"approach {r['thrust_Approach_kN']:.2f} kN" if "thrust_Idle_kN" in r else "")
              + ("" if r["in_box"] else "  [outside phi box]"))
    if "cycle_cross_check" in out:
        cc = out["cycle_cross_check"]
        print(f"Cycle cross-check: {cc['n_points']} points, {cc['n_cycle_mode_runs']} mode runs; "
              f"closed form agrees at every feasible point: {cc['all_feasible_points_agree']} "
              f"(max rel fuel-flow diff {cc['max_rel_fuel_flow_diff']:.1e}); "
              f"crashed points: {len(cc['crashed_points'])}")
        for x in cc["crashed_points"]:
            print(f"  CRASH {x['label']}: {x['crashed_modes']}")

    out_json = REPO_ROOT / "outputs" / f"identifiability_profile{tag}.json"
    out_csv = REPO_ROOT / "outputs" / f"identifiability_profile{tag}.csv"
    pd.DataFrame([{"parameter": p["parameter"], "value": x, "profile_objective": y}
                  for p in out["profiles"] for x, y in p["points"]]).to_csv(out_csv, index=False)
    for p in out["profiles"]:
        p.pop("points")
    out_json.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nIdentified: {out['identified']}\nNot identified: {out['not_identified']}")
    print(f"Runtime {out['runtime_s']} s. Written: {out_json.relative_to(REPO_ROOT)}, "
          f"{out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
