"""
Phase 6 thrust-matched LTO formulation (calibration v5) — shared by
calibrate_lto.py (--tag v5), holdout_icao_validation.py (--tag _v5) and
identifiability_profile.py.

Registered in docs/phase6_p61_registration.md BEFORE any pilot or fit; the
numbers below must match outputs/phase6/p61_registration.json.

Per ICAO record r and LTO mode m (x = 'Power (%)' / 100):
    F_target      = x * F_rated,r                     (input: ICAO rated thrust)
    m_rated,r     = W_ref * (F_rated,r / F_REF)^a_thrust
    pi_c          = 1 + (OPR_r - 1) * x^k_pi          (OPR_r: ICAO rated OPR)
    m_core        = m_rated,r * x^k_mdot
    FPR           = 1 + (FPR_rated - 1) * x^k_pi
    BPR           = BPR_r                              (ICAO, constant across modes)
    phi           solved by IntegratedTurbofanEngine.run_at_thrust(F_target)
    fuel flow     = model output (compared with ICAO 'Fuel Flow (kg/s)')

Fitted (shared across the calibration group): W_ref, a_thrust, k_pi, k_mdot.
Fixed (P6.2 range rule): see FIXED / FIXED_RANGES.
"""

from __future__ import annotations

import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ICAO_CSV = ROOT / "data" / "icao_engine_data.csv"
SPLIT_JSON = ROOT / "outputs" / "phase6" / "split_p61.json"

MODE_X = {"TAKE-OFF": 1.00, "APPROACH": 0.30, "IDLE": 0.07}
MODES = tuple(MODE_X)
F_REF_KN = 310.9          # Trent 1000-AE3 rated thrust (reference for W_ref)
FAILED_ROW_ERROR = 1.0    # relative error assigned to a row whose thrust is unreachable

# ---- fitted parameters: search box (registered) ----
FIT_BOUNDS = {
    "W_ref": (60.0, 140.0),     # rated core airflow at F_REF [kg/s]
    "a_thrust": (0.0, 2.0),     # rated-thrust scaling exponent of core airflow
    "k_pi": (0.2, 1.5),         # part-power exponent of pi_c and FPR
    "k_mdot": (0.2, 1.5),       # part-power exponent of core airflow
}
FIT_ORDER = tuple(FIT_BOUNDS)   # Optuna suggest order (part of reproducibility)

# ---- heating values for eta_b from ICAO CO/HC (method in the registration) ----
Q_CO_MJ_KG = 10.1018      # CO + 1/2 O2 -> CO2 at 298.15 K, CRECK thermo
Q_FUEL_MJ_KG = 44.4620    # n-C12H26 LHV (H2O vapour) at 298.15 K, CRECK thermo


def eta_b_from_emissions(ei_co_g_kg, ei_hc_g_kg):
    """Combustion efficiency from the unburned-energy balance of CO and HC.

    eta_b = 1 - (EI_CO * Q_CO + EI_HC * Q_fuel) / (1000 * Q_fuel), HC counted at
    the fuel heating value (EIs in g per kg fuel).
    """
    return 1.0 - (np.asarray(ei_co_g_kg) * Q_CO_MJ_KG
                  + np.asarray(ei_hc_g_kg) * Q_FUEL_MJ_KG) / (1000.0 * Q_FUEL_MJ_KG)


# ---- fixed parameters: central values (P6.2; sources in FIXED_RANGES) ----
FIXED = {
    "combustor_pressure_loss": 0.045,
    "eta_compressor": 0.86,
    "eta_turbine_polytropic": 0.90,
    "eta_fan": 0.90,
    "fpr_rated": 1.45,
    "combustor_air_fraction": 0.80,
    "combustor_heat_loss_fraction": 0.0,
    # eta_b per mode: group-weighted mean over the CALIBRATION group of
    # eta_b_from_emissions(CO, HC); filled by calibration_eta_b() and frozen
    # in the registration JSON.
    "eta_b": None,
}

FIXED_RANGES = {
    "combustor_pressure_loss": {
        "range": (0.04, 0.05),
        "sources": [
            "NASA/TM-2017-219501 (Jones, Haller & Tong), p. 4: burner 4 % stagnation pressure drop",
            "NASA/CR-2005-213657 (Liew, Urip & Yang), Table 1, p. 10: main burner total pressure ratio 0.96",
            "NASA/TM-2007-214690 (Jones), p. 12: NPSS turbojet example, 5 % burner pressure loss",
        ],
        "basis": "span of cited public design values",
    },
    "eta_compressor": {
        "range": None,  # filled in isentropic terms at rated OPR, see compressor_isentropic_range()
        "polytropic_range": (0.89, 0.93),
        "sources": [
            "NASA/TM-2017-219501, p. 3: HPC nominal polytropic 91 %, N+3 HPC ~2 % below nominal; LPC ~93 %",
            "NASA/CR-2005-213657, Table 1, p. 10: e_lpc 0.9036, e_hpc 0.9066",
        ],
        "basis": "span of cited polytropic values, converted to the model's single "
                 "isentropic efficiency at the calibration engine's rated OPR (43.2)",
    },
    "eta_turbine_polytropic": {
        "range": (0.90, 0.92),
        "sources": [
            "NASA/TM-2017-219501, p. 3: HPT polytropic 91 %, 1 % higher than the N+2 level (0.90)",
            "NASA/TM-2017-219501, p. 4: LPT polytropic 92 %",
            "NASA/CR-2005-213657, Table 1, p. 10: e_hpt 0.9029, e_lpt 0.9174",
        ],
        "basis": "span of cited polytropic values; central 0.90 = N+2 level (Trent 1000 predates N+3)",
    },
    "eta_fan": {
        "range": (0.89, 0.965),
        "sources": [
            "NASA/CR-2005-213657, Table 1, p. 10: fan polytropic 0.8961",
            "NASA/TM-2017-219501, p. 3: fan polytropic 97 % (geared, FPR 1.3; 'may seem aggressive')",
        ],
        "basis": "cited polytropic span, lowered by ~0.005 at the top for isentropic at FPR ~1.45",
    },
    "fpr_rated": {
        "range": (1.3, 1.7),
        "sources": [
            "NASA/TM-2017-219501, Table 3, p. 12: FPR 1.7 (NASA CFM56 model) and 1.3 (N+3)",
        ],
        "basis": "span of cited design values",
    },
    "combustor_air_fraction": {
        "range": (0.70, 0.90),
        "sources": [],
        "basis": "ILLUSTRATIVE: no page-citable range for the burner-zone air fraction was found; "
                 "nearest public analogue is secondary (cooling) flow 15-19 % in NASA/TM-2017-219501 "
                 "Table 3, p. 12. Propagated over 0.70-0.90 and reported as an assumption.",
    },
    "combustor_heat_loss_fraction": {
        "range": (0.0, 0.0),
        "sources": ["scripts/validation/heat_loss_provenance.md (structural argument, xi = 0)"],
        "basis": "fixed by structural argument (P4.5), not varied",
    },
    "eta_b": {
        "range": None,  # per mode: min-max over calibration records
        "sources": ["data/icao_engine_data.csv CO and HC emission indices (calibration group only)"],
        "basis": "data-derived per mode via eta_b_from_emissions (energy balance; CRECK heating values)",
    },
}


def compressor_isentropic(e_poly: float, pi: float = 43.2, gamma: float = 1.4) -> float:
    """Isentropic efficiency equivalent to polytropic ``e_poly`` at pressure ratio ``pi``."""
    k = (gamma - 1.0) / gamma
    return (pi ** k - 1.0) / (pi ** (k / e_poly) - 1.0)


def compressor_isentropic_range() -> tuple[float, float]:
    lo, hi = FIXED_RANGES["eta_compressor"]["polytropic_range"]
    return (compressor_isentropic(lo), compressor_isentropic(hi))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def model_name(engine_id: str) -> str:
    return engine_id.split("BYPASS RATIO")[0].strip()


def load_split(path: Path = SPLIT_JSON) -> dict:
    import json
    return json.loads(Path(path).read_text())


def load_rows(records: list[str], with_targets: bool) -> pd.DataFrame:
    """One row per (record, mode). Target columns only when ``with_targets``."""
    cols = ["Mode", "Power (%)", "Engine ID", "Unique ID", "Pressure Ratio",
            "Bypass Ratio", "Rated Thrust (kN)"]
    if with_targets:
        cols += ["Fuel Flow (kg/s)", "CO (g/kg)", "HC (g/kg)", "NOx (g/kg)"]
    df = pd.read_csv(ICAO_CSV, usecols=cols)
    df = df[df["Unique ID"].isin(records)].copy()
    df["Model"] = df["Engine ID"].map(model_name)
    df["x"] = df["Mode"].map(MODE_X)
    if df["x"].isna().any() or not np.allclose(df["x"] * 100.0, df["Power (%)"]):
        raise ValueError("unexpected mode / power setting in the ICAO CSV")
    df["Target Thrust (kN)"] = df["x"] * df["Rated Thrust (kN)"]
    return df.drop(columns=["Engine ID"]).reset_index(drop=True)


def attach_groups(df: pd.DataFrame, groups: list[list[str]]) -> pd.DataFrame:
    gid = {m: i for i, g in enumerate(groups) for m in g}
    df = df.copy()
    df["Group"] = df["Model"].map(gid)
    if df["Group"].isna().any():
        raise ValueError("record model not in any split group")
    df["Group"] = df["Group"].astype(int)
    # weights: each group equal, records equal within a group, modes equal
    n_groups = df["Group"].nunique()
    n_rec = df.groupby("Group")["Unique ID"].transform("nunique")
    df["w"] = 1.0 / (n_groups * n_rec * len(MODES))
    return df


def calibration_eta_b(cal_rows: pd.DataFrame) -> dict:
    """Per-mode eta_b central (group-weighted mean) and range (min-max over records)."""
    eta = eta_b_from_emissions(cal_rows["CO (g/kg)"], cal_rows["HC (g/kg)"])
    out = {}
    for mode in MODES:
        m = cal_rows["Mode"] == mode
        w = cal_rows.loc[m, "w"]
        out[mode] = {"central": float(np.sum(w * eta[m]) / np.sum(w)),
                     "range": (float(eta[m].min()), float(eta[m].max()))}
    return out


# --------------------------------------------------------------------------
# Cycle evaluation (parallel over unique cycle inputs)
# --------------------------------------------------------------------------
_ENGINE = None


def _init_worker():
    global _ENGINE
    import logging
    logging.getLogger("cantera").setLevel(logging.ERROR)
    import sys
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from integrated_engine import IntegratedTurbofanEngine
    with contextlib.redirect_stdout(io.StringIO()):
        _ENGINE = IntegratedTurbofanEngine()


def mode_state(params: dict, fixed: dict, opr: float, bpr: float, f_rated: float, x: float) -> dict:
    m_rated = params["W_ref"] * (f_rated / F_REF_KN) ** params["a_thrust"]
    return {
        "pi_c": 1.0 + (opr - 1.0) * x ** params["k_pi"],
        "mass_flow_core": m_rated * x ** params["k_mdot"],
        "fpr": 1.0 + (fixed["fpr_rated"] - 1.0) * x ** params["k_pi"],
        "bypass_ratio": bpr,
        "combustor_pressure_loss": fixed["combustor_pressure_loss"],
        "combustor_air_fraction": fixed["combustor_air_fraction"],
        "combustor_heat_loss_fraction": fixed["combustor_heat_loss_fraction"],
        "eta_fan": fixed["eta_fan"],
    }


def solve_task(task: tuple) -> dict:
    """Worker: one thrust-matched cycle. task = (state, eta_c, eta_poly, eta_b, target, guess, fuel)."""
    from integrated_engine import FUEL_LIBRARY, ThrustTargetUnreachable
    state, eta_c, eta_poly, eta_b, target, guess, fuel = task
    e = _ENGINE
    e.design_point.update(state)
    e.compressor.eta_c = eta_c
    e.turbine_design["eta_polytropic"] = eta_poly
    try:
        r = e.run_at_thrust(target, FUEL_LIBRARY[fuel], combustor_efficiency=eta_b, phi_guess=guess)
    except ThrustTargetUnreachable as exc:
        # the only failure scored as an ordinary row penalty (FAILED_ROW_ERROR)
        return {"status": "unreachable", "reason": exc.reason, "ff": np.nan, "phi": np.nan}
    # Any other exception is a configuration/programming error: it is NOT caught
    # here, so pool.map re-raises it in the driver and the run stops (R6-B).
    p = r["performance"]
    return {
        "status": "converged", "reason": "",
        "ff": float(p["fuel_mass_flow"]), "phi": r["thrust_match"]["phi"],
        "thrust_kN": float(p["thrust_kN"]), "tsfc_mg_Ns": float(p["tsfc_mg_per_Ns"]),
        "T3": float(r["compressor"]["T_out"]), "T4": float(r["combustor"]["T_out"]),
        "T5": float(r["turbine"]["T"]), "p3_bar": float(r["compressor"]["p_out"]) / 1e5,
        "thrust_core_kN": float(p["thrust_core_kN"]), "thrust_bypass_kN": float(p["thrust_bypass_kN"]),
        "m_core": float(state["mass_flow_core"]), "pi_c": float(state["pi_c"]),
        "nox_corr_g_s": float(r["emissions"]["NOx_g_s"]),
    }


class V5Model:
    """Thrust-matched predictions for a set of rows; parallel, warm-started, deterministic."""

    def __init__(self, fixed: dict, n_workers: int = 6, fuel: str = "Jet-A1"):
        if fixed.get("eta_b") is None:
            raise ValueError("fixed['eta_b'] (per-mode) must be set")
        self.fixed = fixed
        self.fuel = fuel
        self.guess: dict = {}
        self.n_evals = 0
        self.pool = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)

    def close(self):
        self.pool.shutdown()

    def predict(self, params: dict, rows: pd.DataFrame) -> pd.DataFrame:
        keys = list(zip(rows["Pressure Ratio"], rows["Bypass Ratio"],
                        rows["Rated Thrust (kN)"], rows["Mode"]))
        uniq = list(dict.fromkeys(keys))
        tasks = []
        for (opr, bpr, fr, mode) in uniq:
            x = MODE_X[mode]
            st = mode_state(params, self.fixed, opr, bpr, fr, x)
            tasks.append((st, self.fixed["eta_compressor"], self.fixed["eta_turbine_polytropic"],
                          self.fixed["eta_b"][mode], x * fr, self.guess.get((opr, bpr, fr, mode)),
                          self.fuel))
        results = list(self.pool.map(solve_task, tasks))
        self.n_evals += 1
        by_key = {}
        for k, res in zip(uniq, results):
            by_key[k] = res
            if res["status"] == "converged":
                self.guess[k] = res["phi"]
        out = pd.DataFrame([by_key[k] for k in keys], index=rows.index)
        return out


def relative_errors(pred_ff: pd.Series, rows: pd.DataFrame, status: pd.Series | None = None) -> np.ndarray:
    """Relative fuel-flow errors; a row is penalised with FAILED_ROW_ERROR only
    when its status is 'unreachable'. Any other non-finite prediction raises."""
    e = (pred_ff.to_numpy() - rows["Fuel Flow (kg/s)"].to_numpy()) / rows["Fuel Flow (kg/s)"].to_numpy()
    bad = ~np.isfinite(e)
    if bad.any():
        unreachable = (np.zeros(len(e), bool) if status is None
                       else (status.to_numpy() == "unreachable"))
        if not np.all(unreachable[bad]):
            raise RuntimeError(f"{int((bad & ~unreachable).sum())} non-finite fuel-flow "
                               "predictions that are not registered unreachable rows")
    return np.where(bad, FAILED_ROW_ERROR, e)


def residual_vector(pred_ff: pd.Series, rows: pd.DataFrame, status: pd.Series | None = None) -> np.ndarray:
    """sqrt(w) * relative error; sum of squares = group-weighted mean squared relative error."""
    return np.sqrt(rows["w"].to_numpy()) * relative_errors(pred_ff, rows, status)


def weighted_mape(pred_ff: pd.Series, rows: pd.DataFrame, status: pd.Series | None = None) -> float:
    return float(100.0 * np.sum(rows["w"].to_numpy() * np.abs(relative_errors(pred_ff, rows, status))))
