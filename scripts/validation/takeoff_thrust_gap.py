#!/usr/bin/env python3
"""
P5.2 Step 2 — Decompose the take-off thrust gap before fitting to thrust.

Before the accounting repair, the frozen v4 take-off cycle (analytic turbine,
analytic nozzle, xi = 0) gave 241.6 kN against the ICAO rated
thrust of the Trent 1000-AE3 (``Rated Thrust (kN)`` in data/icao_engine_data.csv).
This script decides whether that gap is a mis-set input or a missing/incorrect
model term (docs/plan.md P5.2 Step 2 gate). Everything below is a DIAGNOSTIC
COUNTERFACTUAL computed from the production cycle's own station states; none is
a fix, and no input value is adopted.

1. Stream decomposition: core nozzle vs bypass, momentum vs pressure terms.
2. Thrust-equation audit. The engine-level (NASA general) thrust equation is
       F = m_e u_e - m_0 u_0 + (p_e - p_0) A_e
   with u_0 = 0 on a static stand (freestream inlet momentum). The analytic
   core nozzle instead computes m (u_exit - u_in), subtracting the momentum of
   its own inlet (the turbine exit) — an internal station. The counterfactual
   restores that term.
3. Nozzle treatment: the analytic nozzle always expands fully to ambient
   (p_exit = p_amb by construction) and its design exit area never constrains
   the flow. The core pressure ratio is checked against the critical ratio; a
   convergent (choked) nozzle counterfactual is computed from the same
   stagnation state; the exit area implied by continuity is compared with the
   design area. The bypass stream is checked the same way.
4. Input sensitivities (production thrust and the restored-term counterfactual):
   total airflow (core and bypass scaled together), BPR, FPR. For each, the
   single-input value that would alone close the gap is reported as a
   diagnostic, not as a candidate value.

Outputs: outputs/takeoff_thrust_gap[_<tag>].json and .md. The untagged pair is the
pre-repair evidence (commit f1bd920). Existing outputs are never overwritten. A tagged
run compares itself with the preserved pre-repair JSON (``--compare-to``).

After the static-thrust accounting repair (docs/plan_phase5_nozzle_repair.md), the
term in item 2 is detected from the nozzle's own output (m·u_exit − thrust_momentum),
so a corrected cycle reports zero subtracted momentum. It does not double-count.

Usage::

    python scripts/validation/takeoff_thrust_gap.py \\
        [--calibration outputs/calibration_trent1000_ae3_v4.json] \\
        [--tag after_accounting] [--compare-to outputs/takeoff_thrust_gap.json]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "optimization"))

import calibrate_lto as cal  # noqa: E402

ICAO_CSV = REPO_ROOT / "data" / "icao_engine_data.csv"
NASA_THRUST = "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/thrust-force/"
AIRFLOW_SCALES = (0.9, 1.1, 1.2, 1.3)
BPRS = (8.0, 10.0, 11.0)
FPRS = (1.35, 1.55, 1.65)


def rated_thrust_kN(uid: str) -> float:
    df = pd.read_csv(ICAO_CSV)
    vals = df.loc[df["Unique ID"] == uid, "Rated Thrust (kN)"].unique()
    if len(vals) != 1:
        raise ValueError(f"{ICAO_CSV.name}: expected one rated thrust for {uid}, got {vals}")
    return float(vals[0])


class TakeoffCycle:
    """The production take-off cycle of a frozen calibration, with optional input overrides."""

    def __init__(self, calibration: Path):
        self.rec = json.loads(calibration.read_text())
        fixed = self.rec["fixed_parameters"]
        self.params = dict(self.rec["best_params"])
        self.beta = fixed.get("combustor_air_fraction", 1.0)
        self.xi = fixed.get("combustor_heat_loss_fraction", 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.engine = cal.IntegratedTurbofanEngine()
        self.base_bpr = self.engine.design_point["bypass_ratio"]

    def run(self, airflow_scale: float = 1.0, bpr: float | None = None, fpr: float | None = None) -> dict:
        cal.set_mode_state(self.engine, self.params, self.beta, self.xi, "Takeoff")
        dp = self.engine.design_point
        dp["mass_flow_core"] *= airflow_scale
        dp["bypass_ratio"] = self.base_bpr if bpr is None else bpr
        if fpr is not None:
            dp["fpr"] = fpr
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = self.engine.run_full_cycle(fuel_blend=cal.FUEL_LIBRARY["Jet-A1"],
                                                 phi=self.params["phi_to"],
                                                 combustor_efficiency=self.params["eta_combustor"])
        finally:
            dp["bypass_ratio"] = self.base_bpr
        return decompose(self.engine, res)


def decompose(engine, res: dict) -> dict:
    perf, turb, nozz, fan = res["performance"], res["turbine"], res["nozzle"], res["fan"]
    p_amb = engine.design_point["P_ambient"]
    m_core = perf["total_mass_flow"]
    g, cp, R = turb["gamma"], turb["cp"], turb["R"]
    # convergent-nozzle counterfactual from the same stagnation state (turbine exit)
    T0, p0 = turb["T"], turb["p"]
    pr_crit = ((g + 1) / 2) ** (g / (g - 1))
    if p0 / p_amb > pr_crit:
        T_star = T0 * 2 / (g + 1)
        p_star = p0 / pr_crit
        u_star = float(np.sqrt(g * R * T_star))
        A_star = m_core / (p_star / (R * T_star) * u_star)
        F_conv = m_core * u_star + (p_star - p_amb) * A_star
    else:
        u_star = A_star = p_star = None
        F_conv = m_core * nozz["u"]
    A_full = nozz.get("A_exit_effective", m_core / (nozz["rho"] * nozz["u"]))
    fpr = engine.design_point["fpr"]
    # momentum the core nozzle actually subtracts: m·u_e − its momentum term (0 once repaired)
    subtracted = m_core * nozz["u"] - nozz["thrust_momentum"]
    return {
        "thrust_kN": perf["thrust_kN"],
        "core_kN": perf["thrust_core_kN"],
        "bypass_kN": perf["thrust_bypass_kN"],
        "core_mass_flow_kg_s": m_core,
        "bypass_mass_flow_kg_s": perf["bypass_mass_flow"],
        "fuel_flow_kg_s": perf["fuel_mass_flow"],
        "core_jet_u_m_s": nozz["u"],
        "core_nozzle_inlet_u_m_s": turb["u"],
        "core_momentum_kN": nozz["thrust_momentum"] / 1e3,
        "core_pressure_kN": nozz["thrust_pressure"] / 1e3,
        "nozzle_inlet_momentum_kN": m_core * turb["u"] / 1e3,
        "subtracted_inlet_momentum_kN": subtracted / 1e3,
        "bypass_jet_u_m_s": fan["u_bypass_exit"],
        "turbine_exit_T_K": T0, "turbine_exit_p_Pa": p0, "turbine_exit_gamma": g,
        "core_nozzle_pressure_ratio": p0 / p_amb, "core_critical_pressure_ratio": pr_crit,
        "core_choked": bool(p0 / p_amb > pr_crit),
        "core_A_exit_design_m2": engine.design_point["A_nozzle_exit"],
        "core_A_exit_fully_expanded_m2": A_full,
        "core_A_throat_convergent_m2": A_star,
        "core_convergent_kN": F_conv / 1e3,
        "bypass_pressure_ratio": fpr, "bypass_critical_pressure_ratio": (1.2) ** 3.5,
        "bypass_choked": bool(fpr > 1.2 ** 3.5),
        "fan_work_MW": perf["fan_work_W"] / 1e6,
        "T4_K": res["combustor"]["T_out"],
    }


def restored(d: dict) -> float:
    """Thrust with the nozzle-inlet momentum term restored (engine-level equation, u_0 = 0)."""
    return d["thrust_kN"] + d["subtracted_inlet_momentum_kN"]


PRE_REPAIR_JSON = REPO_ROOT / "outputs" / "takeoff_thrust_gap.json"
DEFECT_TOL_KN = 1e-6


def output_paths(tag: str) -> tuple[Path, Path]:
    stem = "takeoff_thrust_gap" + (f"_{tag}" if tag else "")
    return REPO_ROOT / "outputs" / f"{stem}.json", REPO_ROOT / "outputs" / f"{stem}.md"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibration", type=Path,
                    default=REPO_ROOT / "outputs" / "calibration_trent1000_ae3_v4.json")
    ap.add_argument("--tag", default="",
                    help="output suffix, e.g. 'after_accounting' -> outputs/takeoff_thrust_gap_after_accounting.*")
    ap.add_argument("--compare-to", type=Path, default=PRE_REPAIR_JSON,
                    help="preserved earlier diagnostic to compare against (tagged runs only)")
    args = ap.parse_args()
    json_path, md_path = output_paths(args.tag)
    existing = [p for p in (json_path, md_path) if p.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite preserved diagnostic {[str(p.relative_to(REPO_ROOT)) for p in existing]}; "
                         "pass a new --tag")
    compare = None
    if args.tag:
        cmp_path = args.compare_to.resolve()
        if not cmp_path.exists():
            raise SystemExit(f"--compare-to {cmp_path} does not exist")
        compare = (cmp_path, json.loads(cmp_path.read_text()))
    cyc = TakeoffCycle(args.calibration.resolve())
    target = rated_thrust_kN(cyc.rec["icao_uid"])
    base = cyc.run()
    gap = target - base["thrust_kN"]
    F_restored = restored(base)
    conv_total = base["core_convergent_kN"] + base["bypass_kN"]   # engine-level equation, convergent core

    sens = {"airflow_scale": [], "bpr": [], "fpr": []}
    for s in AIRFLOW_SCALES:
        d = cyc.run(airflow_scale=s)
        sens["airflow_scale"].append({"value": s, "thrust_kN": d["thrust_kN"], "restored_kN": restored(d),
                                      "T4_K": d["T4_K"], "fuel_flow_kg_s": d["fuel_flow_kg_s"]})
    for b in BPRS:
        d = cyc.run(bpr=b)
        sens["bpr"].append({"value": b, "thrust_kN": d["thrust_kN"], "restored_kN": restored(d),
                            "T4_K": d["T4_K"], "core_kN": d["core_kN"], "bypass_kN": d["bypass_kN"]})
    for f in FPRS:
        d = cyc.run(fpr=f)
        sens["fpr"].append({"value": f, "thrust_kN": d["thrust_kN"], "restored_kN": restored(d),
                            "T4_K": d["T4_K"], "core_kN": d["core_kN"], "bypass_kN": d["bypass_kN"]})

    def closing(key: str, lo: float, hi: float, fn) -> dict:
        out = {}
        for label, f in (("production", lambda d: d["thrust_kN"]), ("restored_term", restored)):
            h = lambda v: f(fn(v)) - target  # noqa: E731
            try:
                a, b = h(lo), h(hi)
                out[label] = float(brentq(h, lo, hi, xtol=1e-4)) if a * b < 0 else None
            except Exception as exc:  # noqa: BLE001 — cycle infeasible in the bracket
                out[label] = f"no solution in [{lo}, {hi}]: {type(exc).__name__}: {exc}"
        return {"input": key, "bracket": [lo, hi], **out}

    closes = [
        closing("airflow_scale", 1.0, 1.6, lambda v: cyc.run(airflow_scale=v)),
        closing("bpr", cyc.base_bpr, 16.0, lambda v: cyc.run(bpr=v)),
        closing("fpr", base["bypass_pressure_ratio"], 2.0, lambda v: cyc.run(fpr=v)),
    ]

    # ---- verdict (the gate) ----
    defect = bool(base["subtracted_inlet_momentum_kN"] > DEFECT_TOL_KN)
    structural = [
        {"term": "core nozzle subtracts its own inlet (turbine-exit) momentum",
         "evidence": (f"F_core = m (u_exit - u_in) with u_in = {base['core_nozzle_inlet_u_m_s']:.4f} m/s; "
                      if defect else
                      f"RESOLVED: F_core = m u_exit (subtracted {base['subtracted_inlet_momentum_kN']:.3e} kN); "
                      f"u_in = {base['core_nozzle_inlet_u_m_s']:.4f} m/s no longer enters thrust; ")
                     + f"engine-level static thrust has u_0 = 0 (NASA general thrust equation, {NASA_THRUST})",
         "effect_kN": base["subtracted_inlet_momentum_kN"],
         "note": "u_in is set by continuity through A_combustor_exit x 1.82 (an area choice), so the "
                 "subtracted term is also an artifact of that area"},
        {"term": "nozzle treatment: always fully expanded; design exit area never constrains the flow",
         "evidence": f"core p05/p_amb = {base['core_nozzle_pressure_ratio']:.3f} > critical "
                     f"{base['core_critical_pressure_ratio']:.3f} (choked for a convergent nozzle); "
                     f"fully-expanded exit area {base['core_A_exit_fully_expanded_m2']:.4f} m2 vs design "
                     f"{base['core_A_exit_design_m2']:.3f} m2 (unused)",
         "effect_kN": base["core_convergent_kN"] - (base["core_kN"] + base["subtracted_inlet_momentum_kN"]),
         "note": "sign: a convergent choked nozzle gives LESS gross thrust than ideal full expansion; "
                 "this treatment does not explain the shortfall, it bounds the core term from above"},
    ]
    residual_after_restore = target - F_restored
    if defect:
        verdict = "STRUCTURAL"
        basis = (
            "The production thrust equation contains an incorrect term (nozzle-inlet momentum subtracted "
            "at an internal station), worth {:.4f} kN; per docs/plan.md P5.2 Step 2 this is a missing/"
            "incorrect model term, not a mis-set input, so fitting thrust now would hide it in a parameter. "
            "Restoring it alone leaves {:.4f} kN ({:.1f} %) unexplained; no single input in the tested "
            "ranges is sourced, so none is adopted.".format(
                base["subtracted_inlet_momentum_kN"], residual_after_restore,
                100 * residual_after_restore / target))
    else:
        verdict = "ACCOUNTING RESOLVED; RESIDUAL UNEXPLAINED (ESCALATE)"
        c_air = next(c for c in closes if c["input"] == "airflow_scale")["production"]
        c_bpr = next(c for c in closes if c["input"] == "bpr")["production"]
        basis = (
            "Core and bypass now both use the static engine-level momentum balance (u_0 = 0); no internal-"
            "station momentum is subtracted. The remaining {:.4f} kN ({:.1f} %) shortfall is NOT explained by "
            "any term audited here: ideal full expansion bounds the core from above (the convergent-nozzle "
            "counterfactual is {:.4f} kN lower), so nozzle treatment cannot close it. Illustrative single-input "
            "closers exist (total airflow x{}, BPR {}), but none is sourced: core airflow {} kg/s is a "
            "hand-set design-point value and BPR {} is the ICAO-sourced value. Under docs/plan.md "
            "('no sourced value, no fixed value'), none is adopted, and P5.2 Step 3 does not resume. The "
            "residual needs either a sourced airflow/cycle input or a separately justified model term.".format(
                gap, 100 * gap / target,
                (base["core_kN"] - base["core_convergent_kN"]),
                f"{c_air:.4f}" if isinstance(c_air, float) else c_air,
                f"{c_bpr:.4f}" if isinstance(c_bpr, float) else c_bpr,
                cyc.engine.design_point["mass_flow_core"], cyc.base_bpr))
    comparison = None
    if compare is not None:
        cmp_path, pre = compare
        pb = pre["base"]
        unchanged_keys = ("fuel_flow_kg_s", "T4_K", "turbine_exit_T_K", "turbine_exit_p_Pa",
                          "core_jet_u_m_s", "core_nozzle_inlet_u_m_s", "bypass_kN", "core_mass_flow_kg_s")
        comparison = {
            "compared_to": str(cmp_path.relative_to(REPO_ROOT)),
            "pre_total_kN": pb["thrust_kN"], "post_total_kN": base["thrust_kN"],
            "pre_core_kN": pb["core_kN"], "post_core_kN": base["core_kN"],
            "delta_total_kN": base["thrust_kN"] - pb["thrust_kN"],
            "pre_subtracted_inlet_momentum_kN": pb["subtracted_inlet_momentum_kN"],
            "delta_minus_pre_subtracted_kN": base["thrust_kN"] - pb["thrust_kN"] - pb["subtracted_inlet_momentum_kN"],
            "pre_gap_kN": pre["gap_kN"], "post_gap_kN": gap,
            "post_matches_pre_restored_counterfactual_kN": base["thrust_kN"] - pre["restored_term_kN"],
            "unchanged_state_max_rel_diff": max(abs(base[k] - pb[k]) / abs(pb[k]) for k in unchanged_keys),
            "unchanged_state_keys": list(unchanged_keys),
        }
    out = {
        "calibration": str(args.calibration.resolve().relative_to(REPO_ROOT)),
        "configuration": "turbine analytic, nozzle analytic, xi from record, Jet-A1, v4 phi_to/eta_b/p_loss",
        "target_kN": target, "target_source": f"{ICAO_CSV.relative_to(REPO_ROOT)} 'Rated Thrust (kN)', "
                                              f"UID {cyc.rec['icao_uid']}, Power 100 %",
        "base": base, "gap_kN": gap, "gap_pct": 100 * gap / target,
        "restored_term_kN": F_restored, "residual_after_restore_kN": residual_after_restore,
        "convergent_core_engine_level_total_kN": conv_total,
        "sensitivities": sens, "single_input_to_close_gap": closes,
        "structural_findings": structural,
        "accounting_defect_present": defect,
        "verdict": verdict,
        "verdict_basis": basis,
        "tag": args.tag,
        "comparison_to_pre_repair": comparison,
    }
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    write_md(out, md_path)
    print(md_path.read_text())


def write_md(o: dict, md_path: Path) -> None:
    b = o["base"]
    defect = o.get("accounting_defect_present", True)
    title = ("Take-off thrust gap decomposition (P5.2 Step 2)" if defect else
             "Take-off thrust gap after the static-thrust accounting repair (P5.2 Step 2, repeated)")
    L = [f"# {title}\n",
         f"Calibration `{o['calibration']}`; {o['configuration']}. Target {o['target_kN']:.1f} kN "
         f"({o['target_source']}). Generated by `scripts/validation/takeoff_thrust_gap.py`"
         + (f" `--tag {o['tag']}`" if o.get("tag") else "") + ". "
         "**Every counterfactual below is a diagnostic, not a fix; no input value is adopted.**\n",
         "## 1. Where the thrust comes from\n",
         "| Term | kN |", "|---|---|",
         (f"| Core nozzle, momentum m(u_e − u_in) | {b['core_momentum_kN']:.7f} |" if defect else
          f"| Core nozzle, momentum m·u_e (u_e = {b['core_jet_u_m_s']:.2f} m/s) | {b['core_momentum_kN']:.7f} |"),
         f"| Core nozzle, pressure (p_e − p_amb)A_e | {b['core_pressure_kN']:.7f} |",
         f"| Bypass m_bp·u_bp (u_bp = {b['bypass_jet_u_m_s']:.2f} m/s) | {b['bypass_kN']:.7f} |",
         f"| **Model total** | **{b['thrust_kN']:.7f}** |",
         f"| Target (ICAO rated) | {o['target_kN']:.1f} |",
         f"| **Gap** | **{o['gap_kN']:.4f} ({o['gap_pct']:.1f} %)** |\n",
         f"Core stream {b['core_mass_flow_kg_s']:.7f} kg/s (air + fuel), bypass {b['bypass_mass_flow_kg_s']:.2f} kg/s; "
         f"T4 {b['T4_K']:.1f} K; fan work {b['fan_work_MW']:.2f} MW.\n",
         "## 2. Thrust-equation audit\n",
         f"The engine-level static thrust is F = m_e·u_e − m_0·u_0 + (p_e − p_0)A_e with u_0 = 0 "
         f"([NASA general thrust equation]({NASA_THRUST}) subtracts *freestream* inlet momentum). "]
    if defect:
        L += [f"The analytic core nozzle subtracts the momentum at its own inlet, the turbine exit: "
              f"{b['core_mass_flow_kg_s']:.7f} kg/s × {b['core_nozzle_inlet_u_m_s']:.7f} m/s = "
              f"**{b['subtracted_inlet_momentum_kN']:.7f} kN**. That velocity is itself set by continuity through "
              "`A_combustor_exit × 1.82`, an area choice.\n",
              f"- Restoring that term alone: **{o['restored_term_kN']:.7f} kN**, still "
              f"**{o['residual_after_restore_kN']:.7f} kN** ({100 * o['residual_after_restore_kN'] / o['target_kN']:.1f} %) below target.\n"]
    else:
        L += ["**Corrected.** The analytic core nozzle now uses F = ṁ·u_e + (p_e − p_amb)A_e. The momentum it "
              f"subtracts is {b['subtracted_inlet_momentum_kN']:.3e} kN. The turbine-exit momentum "
              f"({b['core_mass_flow_kg_s']:.7f} kg/s × {b['core_nozzle_inlet_u_m_s']:.7f} m/s = "
              f"{b['nozzle_inlet_momentum_kN']:.7f} kN) remains a diagnostic only. The core and the bypass "
              "(m_bp·u_bp) now use the same freestream reference.\n",
              f"- Model total **{b['thrust_kN']:.7f} kN**; **{o['gap_kN']:.7f} kN** "
              f"({o['gap_pct']:.1f} %) below target. This residual is **unexplained**; see the verdict.\n"]
    c = o.get("comparison_to_pre_repair")
    if c:
        L += ["### Comparison with the preserved pre-repair diagnostic\n",
              f"Pre-repair evidence: `{c['compared_to']}` (kept unchanged).\n",
              "| | Pre-repair | This run | Δ |", "|---|---|---|---|",
              f"| Core kN | {c['pre_core_kN']:.7f} | {c['post_core_kN']:.7f} | {c['post_core_kN'] - c['pre_core_kN']:+.7f} |",
              f"| Total kN | {c['pre_total_kN']:.7f} | {c['post_total_kN']:.7f} | {c['delta_total_kN']:+.7f} |",
              f"| Gap to target kN | {c['pre_gap_kN']:.7f} | {c['post_gap_kN']:.7f} | {c['post_gap_kN'] - c['pre_gap_kN']:+.7f} |\n",
              f"- Thrust change minus the pre-repair subtracted term ({c['pre_subtracted_inlet_momentum_kN']:.7f} kN): "
              f"**{c['delta_minus_pre_subtracted_kN']:.3e} kN**. The change is exactly the removed internal momentum.",
              f"- This run minus the pre-repair 'restored term' counterfactual: {c['post_matches_pre_restored_counterfactual_kN']:.3e} kN.",
              f"- Fuel flow, T4, turbine-exit T/p/u, core jet velocity, core mass flow and bypass thrust: max relative "
              f"difference **{c['unchanged_state_max_rel_diff']:.2e}**, so they are unchanged.\n"]
    L += ["## 3. Nozzle pressure / area / choking\n",
         "| Stream | p0/p_amb | critical | choked (convergent)? |", "|---|---|---|---|",
         f"| Core | {b['core_nozzle_pressure_ratio']:.3f} | {b['core_critical_pressure_ratio']:.3f} | {'yes' if b['core_choked'] else 'no'} |",
         f"| Bypass | {b['bypass_pressure_ratio']:.3f} | {b['bypass_critical_pressure_ratio']:.3f} | {'yes' if b['bypass_choked'] else 'no'} |\n",
         "The analytic nozzle always expands to p_amb (the pressure term is zero by construction) and never "
         f"uses its design exit area ({b['core_A_exit_design_m2']:.3f} m²); full expansion needs "
         f"{b['core_A_exit_fully_expanded_m2']:.4f} m². A convergent core nozzle from the same stagnation state "
         f"(throat {b['core_A_throat_convergent_m2']:.4f} m²) gives {b['core_convergent_kN']:.4f} kN gross "
         f"vs {b['core_kN'] + b['subtracted_inlet_momentum_kN']:.4f} kN fully expanded — lower, so nozzle "
         "treatment does not explain the shortfall. The bypass stream is unchoked; its full expansion is exact "
         "for a convergent nozzle.\n",
         "Assumptions of the analytic nozzle and of the convergent counterfactual: turbine-exit T and p are "
         "used as the nozzle total state; the expansion is isentropic with the turbine-exit γ, cp and R; the "
         "ideal nozzle's exit area and the convergent throat area both follow from continuity. **Neither area "
         "is a measured or sourced engine dimension**, and no nozzle geometry is imposed. The design area "
         "is the configured PINN geometry and does not constrain the analytic flow.\n",
         ("## 4. Input sensitivities (production thrust | with restored term)\n" if defect else
          "## 4. Input sensitivities (illustrative only; the 'Restored' column equals production after the repair)\n"),
         "| Input | Value | Thrust kN | Restored kN | T4 K |", "|---|---|---|---|---|",
         f"| baseline | — | {b['thrust_kN']:.2f} | {o['restored_term_kN']:.2f} | {b['T4_K']:.1f} |"]
    for key, rows in o["sensitivities"].items():
        for r in rows:
            L.append(f"| {key} | {r['value']} | {r['thrust_kN']:.2f} | {r['restored_kN']:.2f} | {r['T4_K']:.1f} |")
    L += ["", "Single input that would alone reach the target (diagnostic only — none is sourced, none is adopted):\n",
          "| Input | Bracket | Production | With restored term |", "|---|---|---|---|"]
    for c in o["single_input_to_close_gap"]:
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else ("none in bracket" if v is None else v)  # noqa: E731
        L.append(f"| {c['input']} | {c['bracket']} | {fmt(c['production'])} | {fmt(c['restored_term'])} |")
    L += ["", f"## Verdict: **{o['verdict']}**\n", o["verdict_basis"], ""]
    if o["verdict"] == "STRUCTURAL":
        L.append("Per docs/plan.md P5.2 Step 2 this is an escalation gate: P5.2 Steps 3–4 (v5 objective and fit) "
                 "and everything downstream stop until the user decides the repair.\n")
    elif not defect:
        L.append("Gate status: the approved accounting repair is complete. The original gate still applies, "
                 "because the take-off gap is not explained and cannot be closed with a sourced input. "
                 "P5.2 Steps 3–4 and everything downstream stay stopped until the user decides how to treat the "
                 "residual. The η_b / pressure_loss citation gate is also still open.\n")
    md_path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
