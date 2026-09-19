#!/usr/bin/env python3
"""
P4.4 — Turbine PINN: honest target, or honest retirement.

DECISION, WRITTEN BEFORE THE RUN (docs/plan.md P4.4 step 1)
----------------------------------------------------------
**Surrogate claim.** There is no turbine experimental or CFD dataset in the
repo, so the "physics claim" route (conservation laws from residuals alone,
validated externally) is closed for Phase 4. The v5 turbine PINN is trained
as a neural surrogate of the analytic work-consistent polytropic expansion
that Phase 3 adopted as production (``IntegratedTurbofanEngine.run_turbine_analytic``,
eta_poly = 0.9). It approximates that model fast and differentiably and
carries **no independent accuracy**; the preprint must say so in those words.

Why the shipped checkpoint could not be a surrogate: its inputs are
[x, cp*, R*, gamma*] with inlet-anchored outputs, so its pressure ratio
p5/p4 cannot depend on the work fraction tau = W / (m_dot cp T4) — the
quantity that sets p5/p4 = (1 - tau)^(gamma / (eta (gamma - 1))). It was
trained at one fixed W (57.4 MW) and one m_dot, i.e. one tau, and applies
that expansion ratio everywhere (p5 = 2.77 bar vs 4.73 analytic at
take-off, -41.5 %, outputs/turbine_p5_adjudication.csv). The v5 net adds
tau as an input (``NormalizedTurbinePINN(use_work_fraction=True)``); legacy
checkpoints still load with the default.

PRE-REGISTERED GATE (fixed here before the run)
-----------------------------------------------
* Surrogate fidelity on HELD-OUT engine models across the calibrated LTO
  envelope: |p5_pinn - p5_analytic| / p5_analytic < 1 % and the same for the
  raw network T5 (before the work adjustment in ``run_turbine_pinn``), for
  every held-out condition (max, not mean).
* Full cycle at the calibration engine's three LTO modes x four fuels with
  ``turbine_model="pinn"`` and the v5 checkpoint: total thrust and TSFC
  within 1 % of the analytic path, every case.
* Otherwise the turbine PINN is retired to an appendix and production stays
  analytic. No re-tuning after seeing the result; a changed configuration is
  a new attempt with its own record.

TRAINING CONFIGURATION (attempt 1)
----------------------------------
* Envelope: the v4 calibration (``outputs/calibration_trent1000_ae3_v4.json``)
  evaluated over every ICAO Trent 1000 certification record x 3 LTO modes
  (exactly as ``holdout_icao_validation.py`` sets the operating point) x the
  four fuels of the study (Jet-A1, HEFA-SPK, FT-SPK, ATJ-SPK). Each condition
  records the turbine inlet state, m_dot, cp, R, gamma and the shaft work the
  cycle demands. Written to ``outputs/turbine_envelope_v5.csv``.
* Split: 20 % of engine MODEL NAMES held out (seed 42); all modes and fuels of
  a held-out model are held out together.
* Loss = path term + physics terms, all in the inlet-anchored normalised
  variables the network works in:
    - path (pressure-path term, P4.4(2)): MSE of [rho*, p*, T*](x) against the
      analytic expansion path, x in [0, 1], 33 collocation points;
    - EOS residual p* - rho* T*, monotonic-T penalty relu(dT*/dx), and the
      work-matching term (T*(1) - (1 - tau))^2 / tau^2 — the same three terms
      as ``compute_loss_components`` with the same weights (1, 0.1, 0.5).
* Adam, lr 1e-3, 3000 full-batch epochs, seed 42, CPU, float32.
* Output ``models/turbine_pinn_v5.pt`` (never overwrites; provenance recorded).

Usage::

    python scripts/validation/train_turbine_surrogate.py --seed 42 --device auto
    python scripts/validation/train_turbine_surrogate.py --evaluate-only   # re-score the checkpoint
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

with open(os.devnull, "w") as _devnull, contextlib.redirect_stdout(_devnull):
    from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY, LocalFuelBlend, part_power_state
    from simulation.fuels import HEFA_SPK, FT_SPK, ATJ_SPK
    from simulation.turbine.turbine import (
        NormalizedTurbinePINN, THERMO_REF, run_turbine_pinn, analytic_expansion_path,
    )

CALIBRATION = REPO_ROOT / "outputs" / "calibration_trent1000_ae3_v4.json"
ICAO_CSV = REPO_ROOT / "data" / "icao_engine_data.csv"
ENVELOPE_CSV = REPO_ROOT / "outputs" / "turbine_envelope_v5.csv"
FIDELITY_CSV = REPO_ROOT / "outputs" / "turbine_surrogate_fidelity_v5.csv"
CYCLE_CSV = REPO_ROOT / "outputs" / "turbine_surrogate_cycle_check_v5.csv"
SUMMARY_MD = REPO_ROOT / "outputs" / "turbine_surrogate_v5.md"
SAVE_PATH = REPO_ROOT / "models" / "turbine_pinn_v5.pt"


def _bind_attempt(n: int) -> None:
    """Select the registered attempt and its artifact paths."""
    global ATTEMPT, FIDELITY_CSV, CYCLE_CSV, SUMMARY_MD, SAVE_PATH
    ATTEMPT = ATTEMPTS[n]
    sfx = ATTEMPT["suffix"]
    FIDELITY_CSV = REPO_ROOT / "outputs" / f"turbine_surrogate_fidelity_v5{sfx}.csv"
    CYCLE_CSV = REPO_ROOT / "outputs" / f"turbine_surrogate_cycle_check_v5{sfx}.csv"
    SUMMARY_MD = REPO_ROOT / "outputs" / f"turbine_surrogate_v5{sfx}.md"
    SAVE_PATH = REPO_ROOT / ATTEMPT["out"]

MODE_MAP = {"TAKE-OFF": "phi_to", "APPROACH": "phi_app", "IDLE": "phi_idle"}

ATTEMPTS = {
    1: {
        "id": "P4.4-attempt-1",
        "registered": "2026-09-18",
        "claim": "surrogate of run_turbine_analytic (eta_poly 0.9); no independent accuracy",
        "n_epochs": 3000,
        "lr": 1e-3,
        "n_collocation": 33,
        "path_error": "absolute",       # MSE in inlet-anchored normalised units
        "loss_weights": {"path": 1.0, "eos": 1.0, "monotonic": 0.1, "work": 0.5},
        "holdout_fraction_models": 0.2,
        "seed": 42,
        "gate": {"p5_T5_max_rel_err": 0.01, "cycle_thrust_tsfc_max_rel_err": 0.01},
        "out": "models/turbine_pinn_v5.pt",
        "suffix": "",
    },
    # Registered 2026-09-18 AFTER attempt 1 missed the gate, with exactly one
    # diagnosed defect fixed: the absolute path MSE does not resolve the exit
    # pressure ratio (0.08-0.25 in normalised units) to 1 %; attempt 2 uses
    # a RELATIVE path error so every point of the path is weighted by its own
    # magnitude. Epochs doubled because the attempt-1 loss was still falling
    # at 3000. Everything else identical. Reported next to attempt 1.
    2: {
        "id": "P4.4-attempt-2",
        "registered": "2026-09-18",
        "claim": "surrogate of run_turbine_analytic (eta_poly 0.9); no independent accuracy",
        "n_epochs": 6000,
        "lr": 1e-3,
        "n_collocation": 33,
        "path_error": "relative",
        "loss_weights": {"path": 1.0, "eos": 1.0, "monotonic": 0.1, "work": 0.5},
        "holdout_fraction_models": 0.2,
        "seed": 42,
        "gate": {"p5_T5_max_rel_err": 0.01, "cycle_thrust_tsfc_max_rel_err": 0.01},
        "out": "models/turbine_pinn_v5_a2.pt",
        "suffix": "_a2",
    },
}
ATTEMPT = ATTEMPTS[1]   # rebound by main() from --attempt


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def study_fuels() -> list:
    return [
        FUEL_LIBRARY["Jet-A1"],
        LocalFuelBlend(HEFA_SPK.name, HEFA_SPK.species),
        LocalFuelBlend(FT_SPK.name, FT_SPK.species),
        LocalFuelBlend(ATJ_SPK.name, ATJ_SPK.species),
    ]


# ---------------------------------------------------------------------------
# 1. Envelope
# ---------------------------------------------------------------------------

def configure_engine(engine, calib, opr, power_fraction, thrust_ratio) -> None:
    best, fixed = calib["best_params"], calib["fixed_parameters"]
    pi_c, m_dot = part_power_state(
        power_fraction=power_fraction, pi_rated=opr,
        m_dot_rated=fixed["base_airflow_kg_s"] * thrust_ratio,
        k_pi=best["k_pi"], k_mdot=best["k_mdot"],
    )
    dp = engine.design_point
    dp["pi_c"] = pi_c
    dp["mass_flow_core"] = m_dot
    dp["combustor_pressure_loss"] = best["pressure_loss"]
    dp["combustor_air_fraction"] = fixed.get("combustor_air_fraction", 1.0)
    dp["fpr"] = 1.0 + (fixed.get("fpr_rated", 1.45) - 1.0) * power_fraction ** best["k_pi"]


def build_envelope(calib: dict, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(ICAO_CSV)
    df["model"] = df["Engine ID"].str.replace(r"\s*BYPASS RATIO.*", "", regex=True).str.strip()
    calib_rows = df[df["Unique ID"] == calib["icao_uid"]]
    calib_thrust = float(calib_rows["Rated Thrust (kN)"].iloc[0])
    engine = IntegratedTurbofanEngine()
    eta_b = calib["best_params"]["eta_combustor"]
    fuels = study_fuels()
    A_in = engine.design_point["A_combustor_exit"]

    rows = []
    n_uid = df["Unique ID"].nunique()
    for k, (uid, grp) in enumerate(df.groupby("Unique ID")):
        opr = float(grp["Pressure Ratio"].iloc[0])
        thrust = float(grp["Rated Thrust (kN)"].iloc[0])
        model = grp["model"].iloc[0]
        for _, row in grp.iterrows():
            mode = row["Mode"]
            if mode not in MODE_MAP:
                continue
            phi = calib["best_params"][MODE_MAP[mode]]
            configure_engine(engine, calib, opr, float(row["Power (%)"]) / 100.0, thrust / calib_thrust)
            for fuel in fuels:
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                    res = engine.run_full_cycle(fuel_blend=fuel, phi=phi, combustor_efficiency=eta_b,
                                                turbine_model="analytic")
                m_dot = res["performance"]["total_mass_flow"]
                inlet = engine._cantera_to_flow_state(cantera_out=res["combustor"], m_dot=m_dot, A_ref=A_in)
                turb = res["turbine"]
                rows.append({
                    "uid": uid, "model": model, "mode": mode, "fuel": fuel.name, "phi": phi,
                    "rho4": inlet["rho"], "u4": inlet["u"], "p4": inlet["p"], "T4": inlet["T"],
                    "cp": turb["cp"], "R": turb["R"], "gamma": turb["gamma"], "m_dot": m_dot,
                    "W": turb["work_total"], "tau": turb["work_total"] / (m_dot * turb["cp"] * inlet["T"]),
                    "p5_analytic": turb["p"], "T5_analytic": turb["T"],
                    "thrust_kN_analytic": res["performance"]["thrust_kN"],
                    "tsfc_analytic": res["performance"]["tsfc_mg_per_Ns"],
                })
        if verbose:
            print(f"  envelope: {k+1}/{n_uid} records ({model})", end="\r")
    if verbose:
        print()
    env = pd.DataFrame(rows)
    ENVELOPE_CSV.parent.mkdir(parents=True, exist_ok=True)
    env.to_csv(ENVELOPE_CSV, index=False)
    return env


# ---------------------------------------------------------------------------
# 2. Vectorised normalised training loss (same terms as compute_loss_components)
# ---------------------------------------------------------------------------

def normalised_batch(env: pd.DataFrame, n_col: int, device) -> dict:
    """Stack (condition x collocation) rows of the inlet-anchored problem."""
    x = torch.linspace(0.0, 1.0, n_col, dtype=torch.float32)
    C = len(env)
    xx = x.repeat(C).view(-1, 1)
    rep = lambda col: torch.tensor(env[col].to_numpy(dtype=np.float32)).repeat_interleave(n_col).view(-1, 1)  # noqa: E731
    return {
        "x": xx.to(device),
        "cp": (rep("cp") / THERMO_REF["cp"]).to(device),
        "R": (rep("R") / THERMO_REF["R"]).to(device),
        "gamma_n": (rep("gamma") / THERMO_REF["gamma"]).to(device),
        "gamma": rep("gamma").to(device),
        "tau": rep("tau").to(device),
        "n_col": n_col, "C": C,
    }


def losses(model, b: dict, eta_poly: float) -> dict:
    x = b["x"].clone().requires_grad_(True)
    inlet_norm = torch.ones(x.size(0), 4, device=x.device)      # inlet-anchored: [1, 1, 1, 1]
    out = model.forward(x, b["cp"], b["R"], b["gamma_n"], inlet_norm, m_dot=1.0,
                        A_func=lambda t: 1.0 + t, work_fraction=b["tau"])
    rho_n, p_n, T_n = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    # analytic path in normalised variables (T4, p4, rho4 all -> 1)
    T_a = 1.0 - x * b["tau"]
    p_a = T_a ** (b["gamma"] / (eta_poly * (b["gamma"] - 1.0)))
    rho_a = p_a / T_a
    if ATTEMPT["path_error"] == "relative":
        loss_path = (((rho_n - rho_a) / rho_a) ** 2 + ((p_n - p_a) / p_a) ** 2
                     + ((T_n - T_a) / T_a) ** 2).mean()
    else:
        loss_path = ((rho_n - rho_a) ** 2 + (p_n - p_a) ** 2 + (T_n - T_a) ** 2).mean()
    loss_eos = ((p_n - rho_n * T_n) ** 2).mean()
    T_x = torch.autograd.grad(T_n, x, torch.ones_like(T_n), create_graph=True)[0]
    loss_mono = torch.relu(T_x).mean()
    # work term at x = 1 (last collocation row of each condition)
    T_out = T_n.view(b["C"], b["n_col"])[:, -1:]
    tau_c = b["tau"].view(b["C"], b["n_col"])[:, -1:]
    loss_work = (((1.0 - T_out) - tau_c) / tau_c).pow(2).mean()
    return {"path": loss_path, "eos": loss_eos, "monotonic": loss_mono, "work": loss_work}


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

def train(env_train: pd.DataFrame, eta_poly: float, seed: int, device: str, n_epochs: int,
          lr: float, verbose: bool = True) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model = NormalizedTurbinePINN(use_work_fraction=True).to(dev)
    b = normalised_batch(env_train, ATTEMPT["n_collocation"], dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=100, min_lr=1e-6)
    w = ATTEMPT["loss_weights"]
    hist = {k: [] for k in ("total", "path", "eos", "monotonic", "work", "lr")}
    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()
        L = losses(model, b, eta_poly)
        total = w["path"] * L["path"] + w["eos"] * L["eos"] + w["monotonic"] * L["monotonic"] + w["work"] * L["work"]
        total.backward()
        opt.step()
        sched.step(total.item())
        for k in ("path", "eos", "monotonic", "work"):
            hist[k].append(float(L[k].item()))
        hist["total"].append(float(total.item()))
        hist["lr"].append(opt.param_groups[0]["lr"])
        if verbose and (ep % 250 == 0 or ep == n_epochs - 1):
            print(f"  Ep {ep:4d} | total {total.item():.3e} | path {L['path'].item():.3e} | "
                  f"eos {L['eos'].item():.3e} | mono {L['monotonic'].item():.3e} | "
                  f"work {L['work'].item():.3e} | lr {opt.param_groups[0]['lr']:.1e}")
    return model, hist


# ---------------------------------------------------------------------------
# 4. Gate evaluation
# ---------------------------------------------------------------------------

def fidelity(model_path: Path, env: pd.DataFrame, engine_geom: dict) -> pd.DataFrame:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = NormalizedTurbinePINN(use_work_fraction="work_fraction" in ckpt.get("input_features", []))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    rows = []
    for r in env.itertuples(index=False):
        thermo = {"cp": r.cp, "R": r.R, "gamma": r.gamma}
        inlet = {"rho": r.rho4, "u": r.u4, "p": r.p4, "T": r.T4}
        out = run_turbine_pinn(str(model_path), inlet, r.W, r.m_dot, engine_geom["A_inlet"],
                               engine_geom["A_outlet"], engine_geom["length"], thermo)
        # raw network T5 before the work adjustment
        scales = {"rho": r.rho4, "u": r.u4, "p": r.p4, "T": r.T4, "cp": THERMO_REF["cp"],
                  "R": THERMO_REF["R"], "gamma": THERMO_REF["gamma"], "L": engine_geom["length"]}
        with torch.no_grad():
            st = model.predict_physical(torch.tensor([[1.0]]), thermo, inlet, r.m_dot, engine_geom,
                                        scales, work_fraction=r.tau if model.use_work_fraction else None)
        T5_raw = float(st[0, 3])
        rows.append({
            "uid": r.uid, "model": r.model, "mode": r.mode, "fuel": r.fuel, "split": r.split,
            "tau": r.tau, "p5_analytic_bar": r.p5_analytic / 1e5, "p5_pinn_bar": out["p"] / 1e5,
            "p5_rel_err": (out["p"] - r.p5_analytic) / r.p5_analytic,
            "T5_analytic_K": r.T5_analytic, "T5_pinn_raw_K": T5_raw,
            "T5_rel_err": (T5_raw - r.T5_analytic) / r.T5_analytic,
        })
    return pd.DataFrame(rows)


def cycle_check(model_path: Path, calib: dict) -> pd.DataFrame:
    df = pd.read_csv(ICAO_CSV)
    calib_rows = df[df["Unique ID"] == calib["icao_uid"]]
    opr = float(calib_rows["Pressure Ratio"].iloc[0])
    eta_b = calib["best_params"]["eta_combustor"]
    engine = IntegratedTurbofanEngine(turbine_pinn_path=str(model_path))
    rows = []
    for _, row in calib_rows.iterrows():
        mode = row["Mode"]
        if mode not in MODE_MAP:
            continue
        configure_engine(engine, calib, opr, float(row["Power (%)"]) / 100.0, 1.0)
        phi = calib["best_params"][MODE_MAP[mode]]
        for fuel in study_fuels():
            res = {}
            for tm in ("analytic", "pinn"):
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                    res[tm] = engine.run_full_cycle(fuel_blend=fuel, phi=phi, combustor_efficiency=eta_b,
                                                    turbine_model=tm)["performance"]
            rows.append({
                "mode": mode, "fuel": fuel.name,
                "thrust_kN_analytic": res["analytic"]["thrust_kN"], "thrust_kN_pinn": res["pinn"]["thrust_kN"],
                "thrust_rel_err": res["pinn"]["thrust_kN"] / res["analytic"]["thrust_kN"] - 1.0,
                "tsfc_analytic": res["analytic"]["tsfc_mg_per_Ns"], "tsfc_pinn": res["pinn"]["tsfc_mg_per_Ns"],
                "tsfc_rel_err": res["pinn"]["tsfc_mg_per_Ns"] / res["analytic"]["tsfc_mg_per_Ns"] - 1.0,
            })
    return pd.DataFrame(rows)


def write_summary(fid: pd.DataFrame, cyc: pd.DataFrame, verdict: str, model_path: Path, ckpt: dict) -> None:
    g = ATTEMPT["gate"]
    ho = fid[fid["split"] == "holdout"]
    tr = fid[fid["split"] == "train"]
    L = [f"# Turbine PINN v5 — surrogate fidelity (P4.4) — {date.today().isoformat()}\n",
         f"Checkpoint `{model_path.relative_to(REPO_ROOT)}` (git `{ckpt.get('git_sha', '')[:8]}`, seed {ckpt.get('seed')}, "
         f"{ckpt.get('epochs_run')} epochs). Claim: **{ATTEMPT['claim']}**. Gate fixed before the run: "
         f"p5 and raw T5 within {100*g['p5_T5_max_rel_err']:.0f} % of the analytic path on held-out engine models "
         f"(max over conditions), and full-cycle thrust and TSFC within {100*g['cycle_thrust_tsfc_max_rel_err']:.0f} %.\n",
         f"## Verdict: **{verdict}**\n",
         "## Surrogate fidelity across the LTO envelope\n",
         "| Split | Conditions | max \\|Δp5\\|/p5 | mean \\|Δp5\\|/p5 | max \\|ΔT5\\|/T5 (raw) | mean \\|ΔT5\\|/T5 (raw) |\n|---|---|---|---|---|---|"]
    for name, part in (("held-out models", ho), ("training models", tr)):
        L.append(f"| {name} | {len(part)} | {100*part['p5_rel_err'].abs().max():.3f} % | {100*part['p5_rel_err'].abs().mean():.3f} % | "
                 f"{100*part['T5_rel_err'].abs().max():.3f} % | {100*part['T5_rel_err'].abs().mean():.3f} % |")
    worst = ho.loc[ho["p5_rel_err"].abs().idxmax()]
    L.append(f"\nWorst held-out p5 case: {worst['model']} / {worst['mode']} / {worst['fuel']} — "
             f"analytic {worst['p5_analytic_bar']:.3f} bar, surrogate {worst['p5_pinn_bar']:.3f} bar "
             f"({100*worst['p5_rel_err']:+.3f} %). Held-out τ range {ho['tau'].min():.3f}–{ho['tau'].max():.3f}.\n")
    L.append("## Full cycle, calibration engine, `turbine_model=\"pinn\"` vs analytic\n")
    L.append("| Mode | Fuel | Thrust analytic (kN) | Thrust surrogate (kN) | Δ | TSFC analytic | TSFC surrogate | Δ |\n|---|---|---|---|---|---|---|---|")
    for r in cyc.itertuples(index=False):
        L.append(f"| {r.mode} | {r.fuel} | {r.thrust_kN_analytic:.2f} | {r.thrust_kN_pinn:.2f} | {100*r.thrust_rel_err:+.3f} % | "
                 f"{r.tsfc_analytic:.3f} | {r.tsfc_pinn:.3f} | {100*r.tsfc_rel_err:+.3f} % |")
    L.append("\n## What this does and does not show\n")
    L.append("- The v5 turbine PINN reproduces the analytic work-consistent expansion it was trained on. "
             "That is a speed/differentiability claim about a surrogate of the production model.\n"
             "- It is **not** evidence that either model matches a real turbine: no turbine measurement or CFD "
             "exists in this repository, and none was used.\n"
             "- The legacy checkpoint `models/turbine_pinn.pt` remains as adjudicated in Phase 3 (p5 −41.5 %); "
             "it lacked the work-fraction input and could not represent the expansion ratio.\n")
    L.append(f"## Artifacts\n- `{FIDELITY_CSV.relative_to(REPO_ROOT)}`\n- `{CYCLE_CSV.relative_to(REPO_ROOT)}`\n"
             f"- `{ENVELOPE_CSV.relative_to(REPO_ROOT)}`\n")
    SUMMARY_MD.write_text("\n".join(L))


def evaluate(model_path: Path, calib: dict, env: pd.DataFrame, verbose: bool = True) -> str:
    engine = IntegratedTurbofanEngine()
    A_in = engine.design_point["A_combustor_exit"]
    geom = {"A_inlet": A_in, "A_outlet": A_in * 1.82, "length": 0.5}
    fid = fidelity(model_path, env, geom)
    fid.to_csv(FIDELITY_CSV, index=False)
    cyc = cycle_check(model_path, calib)
    cyc.to_csv(CYCLE_CSV, index=False)
    g = ATTEMPT["gate"]
    ho = fid[fid["split"] == "holdout"]
    fid_ok = (ho["p5_rel_err"].abs().max() < g["p5_T5_max_rel_err"]
              and ho["T5_rel_err"].abs().max() < g["p5_T5_max_rel_err"])
    cyc_ok = (cyc["thrust_rel_err"].abs().max() < g["cycle_thrust_tsfc_max_rel_err"]
              and cyc["tsfc_rel_err"].abs().max() < g["cycle_thrust_tsfc_max_rel_err"])
    verdict = ("GATE CLEARED — surrogate; label it as such" if fid_ok and cyc_ok
               else "GATE MISSED — retire the turbine PINN to an appendix; production stays analytic")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    write_summary(fid, cyc, verdict, model_path, ckpt)
    if verbose:
        print(f"  held-out p5 max |err| {100*ho['p5_rel_err'].abs().max():.3f} %, "
              f"raw T5 max |err| {100*ho['T5_rel_err'].abs().max():.3f} %  (gate {100*g['p5_T5_max_rel_err']:.0f} %)")
        print(f"  cycle thrust max |err| {100*cyc['thrust_rel_err'].abs().max():.3f} %, "
              f"TSFC max |err| {100*cyc['tsfc_rel_err'].abs().max():.3f} %  (gate {100*g['cycle_thrust_tsfc_max_rel_err']:.0f} %)")
        print(f"  VERDICT: {verdict}")
        print(f"  {SUMMARY_MD.relative_to(REPO_ROOT)}")
    return verdict


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="P4.4 turbine surrogate: train, then score the pre-registered gate")
    ap.add_argument("--attempt", type=int, default=1, choices=sorted(ATTEMPTS),
                    help="registered attempt to run (see ATTEMPTS)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--attempt-id", default=None)
    ap.add_argument("--evaluate-only", action="store_true", help="score an existing checkpoint against the gate")
    ap.add_argument("--rebuild-envelope", action="store_true")
    args = ap.parse_args()
    _bind_attempt(args.attempt)
    args.seed = ATTEMPT["seed"] if args.seed is None else args.seed
    args.epochs = ATTEMPT["n_epochs"] if args.epochs is None else args.epochs
    args.lr = ATTEMPT["lr"] if args.lr is None else args.lr
    args.attempt_id = ATTEMPT["id"] if args.attempt_id is None else args.attempt_id
    device = "cpu" if args.device == "auto" else args.device
    out = Path(args.out) if args.out else SAVE_PATH
    print(f"Registered attempt: {ATTEMPT['id']} (path error: {ATTEMPT['path_error']}, "
          f"{args.epochs} epochs, lr {args.lr}, seed {args.seed}) -> {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")

    calib = json.load(open(CALIBRATION))
    eta_poly = calib["fixed_parameters"]["eta_turbine_polytropic"]

    print("=" * 72)
    print("P4.4  TURBINE SURROGATE — claim: surrogate of the analytic work-consistent expansion")
    print("=" * 72)
    if ENVELOPE_CSV.exists() and not args.rebuild_envelope:
        env = pd.read_csv(ENVELOPE_CSV)
        print(f"[1/4] Envelope loaded: {len(env)} conditions from {ENVELOPE_CSV.name}")
    else:
        print("[1/4] Building the calibrated LTO envelope (v4 calibration x ICAO Trent 1000 records x 4 fuels) ...")
        env = build_envelope(calib)
        print(f"      {len(env)} conditions, {env['model'].nunique()} engine models, "
              f"tau in [{env['tau'].min():.3f}, {env['tau'].max():.3f}], p5/p4 in "
              f"[{(env['p5_analytic']/env['p4']).min():.3f}, {(env['p5_analytic']/env['p4']).max():.3f}]")

    rng = np.random.RandomState(args.seed)
    models = sorted(env["model"].unique())
    n_ho = max(1, int(round(ATTEMPT["holdout_fraction_models"] * len(models))))
    holdout_models = sorted(rng.choice(models, size=n_ho, replace=False).tolist())
    env["split"] = np.where(env["model"].isin(holdout_models), "holdout", "train")
    print(f"[2/4] Split: {len(models) - n_ho} training models / {n_ho} held-out models {holdout_models}")

    if not args.evaluate_only:
        if out.exists():
            raise FileExistsError(f"{out} exists; checkpoints are never overwritten. Choose --out.")
        print(f"[3/4] Training ({args.epochs} epochs, lr {args.lr}, seed {args.seed}, {device}) ...")
        model, hist = train(env[env["split"] == "train"], eta_poly, args.seed, device, args.epochs, args.lr)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_features": ["x", "cp", "R", "gamma", "work_fraction"],
            "thermo_ref": THERMO_REF,
            "attempt": {**ATTEMPT, "id": args.attempt_id, "n_epochs": args.epochs, "lr": args.lr, "seed": args.seed},
            "seed": args.seed, "device": device, "epochs_run": len(hist["total"]),
            "lr": args.lr, "loss_weights": ATTEMPT["loss_weights"],
            "eta_poly": eta_poly,
            "envelope_csv": str(ENVELOPE_CSV.relative_to(REPO_ROOT)), "envelope_sha256": sha256(ENVELOPE_CSV),
            "calibration": str(CALIBRATION.relative_to(REPO_ROOT)), "calibration_sha256": sha256(CALIBRATION),
            "split": {"holdout_models": holdout_models, "train_models": [m for m in models if m not in holdout_models],
                      "seed": args.seed},
            "final_losses": {k: v[-1] for k, v in hist.items()},
            "git_sha": git_sha(),
        }, out)
        print(f"      saved {out.relative_to(REPO_ROOT)}")
    print("[4/4] Scoring the pre-registered gate ...")
    evaluate(out, calib, env)


if __name__ == "__main__":
    main()
