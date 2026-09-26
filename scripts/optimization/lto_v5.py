"""
Phase 6 thrust-matched LTO formulation (calibration v5) — shared by
calibrate_lto.py (--tag v5), holdout_icao_validation.py (--tag _v5) and
identifiability_profile.py.

Registered in docs/phase6_p61_registration.md BEFORE any pilot or fit; the
numbers below must match outputs/phase6/p61_registration.json (amendment A1,
R6 review). The first registration (illustrative beta 0.8) was rejected before
any pilot and is archived under outputs/phase6/superseded/.

Per ICAO record r and LTO mode m (x = 'Power (%)' / 100):
    F_target      = x * F_rated,r                     (input: ICAO rated thrust)
    m_rated,r     = W_ref * (F_rated,r / F_REF)^a_thrust
    pi_c          = 1 + (OPR_r - 1) * x^k_pi          (OPR_r: ICAO rated OPR)
    m_core        = m_rated,r * x^k_mdot
    FPR           = 1 + (FPR_rated - 1) * x^k_pi
    BPR           = BPR_r                              (ICAO, constant across modes)
    phi           solved by IntegratedTurbofanEngine.run_at_thrust(F_target)
    fuel flow     = model output (compared with ICAO 'Fuel Flow (kg/s)')

Combustor structure (amendment A1): SINGLE-ZONE. All core air enters one HP
equilibrium at the overall phi (design_point combustor_air_fraction = 1.0,
which disables the Phase 3.4 burner/dilution split); the temperature rise is
scaled by eta_b. The split parameter beta has no citable range for the
quantity implemented (air frozen out of equilibrium and remixed at constant cp
before the turbine), so decision 2's drop option is exercised: it is neither
fitted nor fixed in v5. Frozen v2-v4 calibrations and reproduction paths keep
their own beta values. Turbine cooling air (injected into the turbine) is a
different quantity and is not modelled.

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

# ---- heating values for eta_b from ICAO CO/HC (reproduced by creck_heating_values) ----
Q_CO_MJ_KG = 10.1018      # CO + 1/2 O2 -> CO2 at 298.15 K, CRECK thermo
Q_FUEL_MJ_KG = 44.4620    # n-C12H26 LHV (H2O vapour) at 298.15 K, CRECK thermo
MECH_CRECK = ROOT / "data" / "creck_c1c16_full.yaml"


def creck_heating_values(T: float = 298.15) -> dict:
    """Reproduce Q_CO_MJ_KG and Q_FUEL_MJ_KG from the production CRECK thermo.

    Reaction enthalpies at T (pure species at 1 atm; ideal gas, so pressure-
    independent), in MJ per kg of the named reactant:
        CO + 1/2 O2 -> CO2                          (Q_CO)
        n-C12H26 + 18.5 O2 -> 12 CO2 + 13 H2O(g)    (Q_fuel: LHV, water as VAPOUR)
    """
    import cantera as ct
    g = ct.Solution(str(MECH_CRECK))

    def h(sp):
        g.TPX = T, ct.one_atm, f"{sp}:1"
        return g.enthalpy_mole

    def mw(sp):
        return g.molecular_weights[g.species_index(sp)]

    return {
        "Q_CO_MJ_KG": (h("CO") + 0.5 * h("O2") - h("CO2")) / mw("CO") / 1e6,
        "Q_FUEL_MJ_KG": (h("NC12H26") + 18.5 * h("O2") - 12 * h("CO2") - 13 * h("H2O"))
                        / mw("NC12H26") / 1e6,
    }


def eta_b_from_emissions(ei_co_g_kg, ei_hc_g_kg):
    """Combustion-efficiency PROXY from the unburned-energy balance of CO and HC.

    eta_b = 1 - (EI_CO * Q_CO + EI_HC * Q_fuel) / (1000 * Q_fuel), EIs in g per
    kg fuel. Assumptions: (i) the chemical energy not released is carried only
    by CO and unburned HC (soot, H2 and other species neglected); (ii) HC is
    counted as if it were unburned fuel, at the fuel LHV per unit mass (ICAO
    reports HC as a methane-equivalent mass; the heating value of the actual
    unburned species is unknown); (iii) heating values are lower heating values
    (H2O vapour) at 298.15 K from CRECK thermo for n-C12H26, the production
    Jet-A1 surrogate. It is an energy-efficiency proxy fixed from data and used
    as the model's temperature-rise scaling (T4 = T3 + eta_b (T_ad - T3)); it is
    NOT an inverse-cycle parameter identified from the fuel-flow objective.
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
                 "isentropic efficiency at the calibration engine's rated OPR (43.2) with the "
                 "constant-gamma (1.4) relation compressor_isentropic(); ENVELOPE, applied to all "
                 "engines and modes. Approximation: the production compressor uses variable-cp "
                 "Cantera air and a temperature-rise efficiency; the variable-cp conversion "
                 "(compressor_isentropic_variable_cp) is ~0.008-0.013 higher and is reported, "
                 "not substituted. The central 0.86 lies inside both conversions.",
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
        "range": None,  # envelope filled by fan_isentropic_envelope()
        "polytropic_range": (0.8961, 0.97),
        "sources": [
            "NASA/CR-2005-213657, Table 1, p. 10: fan polytropic 0.8961",
            "NASA/TM-2017-219501, p. 3: fan polytropic 97 % (geared, FPR 1.3; 'may seem aggressive')",
        ],
        "basis": "cited polytropic span converted EXACTLY to the fan model's isentropic "
                 "efficiency (simulation/fan.py uses constant gamma 1.4, so fan_isentropic() is the "
                 "model's own definition). TRANSFORMED PER DRAW in the P6.2 Monte Carlo: draw "
                 "e_poly and FPR_rated, eta_fan = fan_isentropic(e_poly, FPR_rated). 'range' is the "
                 "envelope over the polytropic span and FPR_rated 1.3-1.7 (reporting only). "
                 "Central 0.90 at FPR 1.45 corresponds to e_poly ~0.905, inside the span.",
    },
    "fpr_rated": {
        "range": (1.3, 1.7),
        "sources": [
            "NASA/TM-2017-219501, Table 3, p. 12: FPR 1.7 (NASA CFM56 model) and 1.3 (N+3)",
        ],
        "basis": "span of cited design values",
    },
    "combustor_heat_loss_fraction": {
        "range": (0.0, 0.0),
        "sources": ["scripts/validation/heat_loss_provenance.md (structural argument, xi = 0)"],
        "basis": "fixed by structural argument (P4.5), not varied",
    },
    "eta_b": {
        "range": None,  # per mode: min-max over calibration records
        "sources": ["data/icao_engine_data.csv CO and HC emission indices (calibration group only)"],
        "basis": "data-derived per mode via eta_b_from_emissions (energy-efficiency proxy; "
                 "assumptions in its docstring; heating values reproduced by creck_heating_values); "
                 "calibration-group records only",
    },
}

# Model structure (amendment A1): single-zone combustor. Not a parameter: the
# value 1.0 switches off the burner/dilution split in run_full_cycle.
SINGLE_ZONE_AIR_FRACTION = 1.0


def compressor_isentropic(e_poly: float, pi: float = 43.2, gamma: float = 1.4) -> float:
    """Isentropic efficiency equivalent to polytropic ``e_poly`` at pressure ratio ``pi``."""
    k = (gamma - 1.0) / gamma
    return (pi ** k - 1.0) / (pi ** (k / e_poly) - 1.0)


def compressor_isentropic_range() -> tuple[float, float]:
    lo, hi = FIXED_RANGES["eta_compressor"]["polytropic_range"]
    return (compressor_isentropic(lo), compressor_isentropic(hi))


def compressor_isentropic_variable_cp(e_poly: float, pi: float = 43.2, T_in: float = 288.15,
                                      p_in: float = 101325.0, n_steps: int = 4000) -> float:
    """Documentation check of the constant-gamma conversion: the production
    compressor's temperature-rise efficiency (T_s - T_in)/(T_out - T_in) for a
    polytropic compression of variable-cp CRECK air (n_steps small isentropic
    steps, each with enthalpy rise dh_s / e_poly)."""
    import cantera as ct
    g = ct.Solution(str(MECH_CRECK))
    g.TPX = T_in, p_in, "O2:0.21, N2:0.79"
    s0, h, p = g.entropy_mass, g.enthalpy_mass, p_in
    g.SP = s0, p_in * pi
    T_s = g.T
    g.TPX = T_in, p_in, "O2:0.21, N2:0.79"
    r = pi ** (1.0 / n_steps)
    for _ in range(n_steps):
        g.SP = g.entropy_mass, p * r
        h += (g.enthalpy_mass - h) / e_poly
        p *= r
        g.HP = h, p
    return (T_s - T_in) / (g.T - T_in)


def fan_isentropic(e_poly: float, fpr: float, gamma: float = 1.4) -> float:
    """Fan isentropic efficiency equivalent to polytropic ``e_poly`` at ``fpr``
    (exact for simulation/fan.py, which uses constant gamma = 1.4)."""
    k = (gamma - 1.0) / gamma
    return (fpr ** k - 1.0) / (fpr ** (k / e_poly) - 1.0)


def fan_isentropic_envelope() -> tuple[float, float]:
    """Envelope of fan_isentropic over the cited polytropic span and FPR_rated
    span (isentropic efficiency falls with FPR and rises with e_poly)."""
    e_lo, e_hi = FIXED_RANGES["eta_fan"]["polytropic_range"]
    f_lo, f_hi = FIXED_RANGES["fpr_rated"]["range"]
    return (fan_isentropic(e_lo, f_hi), fan_isentropic(e_hi, f_lo))


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


def _init_worker(nox_fit_exclude_models):
    """Worker engine. The NOx correlation is refit WITHOUT the given models
    (the held-out group), so no NOx value produced here uses held-out data."""
    global _ENGINE
    import logging
    logging.getLogger("cantera").setLevel(logging.ERROR)
    import sys
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from integrated_engine import EmissionsEstimator, IntegratedTurbofanEngine
    with contextlib.redirect_stdout(io.StringIO()):
        _ENGINE = IntegratedTurbofanEngine()
        _ENGINE.emissions = EmissionsEstimator(nox_fit_exclude_models=set(nox_fit_exclude_models))
    if _ENGINE.emissions.nox_fit_exclude_models != set(nox_fit_exclude_models):
        raise RuntimeError("NOx held-out exclusion not applied")


def mode_state(params: dict, fixed: dict, opr: float, bpr: float, f_rated: float, x: float) -> dict:
    m_rated = params["W_ref"] * (f_rated / F_REF_KN) ** params["a_thrust"]
    return {
        "pi_c": 1.0 + (opr - 1.0) * x ** params["k_pi"],
        "mass_flow_core": m_rated * x ** params["k_mdot"],
        "fpr": 1.0 + (fixed["fpr_rated"] - 1.0) * x ** params["k_pi"],
        "bypass_ratio": bpr,
        "combustor_pressure_loss": fixed["combustor_pressure_loss"],
        "combustor_air_fraction": SINGLE_ZONE_AIR_FRACTION,   # structure, not a parameter
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

    def __init__(self, fixed: dict, nox_fit_exclude_models, n_workers: int = 6,
                 fuel: str = "Jet-A1"):
        """``nox_fit_exclude_models`` is required (no default): pass the
        held-out model names so worker NOx never comes from a fit that saw them."""
        if fixed.get("eta_b") is None:
            raise ValueError("fixed['eta_b'] (per-mode) must be set")
        if "combustor_air_fraction" in fixed:
            raise ValueError("combustor_air_fraction is not a v5 parameter (single-zone, A1)")
        if not nox_fit_exclude_models:
            raise ValueError("nox_fit_exclude_models must list the held-out models")
        self.fixed = fixed
        self.fuel = fuel
        self.guess: dict = {}
        self.n_evals = 0
        self.nox_fit_exclude_models = sorted(nox_fit_exclude_models)
        self.pool = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                        initargs=(self.nox_fit_exclude_models,))

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


# --------------------------------------------------------------------------
# Registered fit and identifiability profile (outputs/phase6/p61_registration.json)
# --------------------------------------------------------------------------
REGISTRATION = ROOT / "outputs" / "phase6" / "p61_registration.json"
CHI2_1_95 = 3.841


def load_registration() -> dict:
    import json
    reg = json.loads(REGISTRATION.read_text())
    if reg.get("amendment") != "A1":
        raise ValueError("active registration is not amendment A1")
    return reg


def calibration_rows(split: dict | None = None) -> pd.DataFrame:
    split = split or load_split()
    return attach_groups(load_rows(split["calibration_records"], with_targets=True),
                         split["calibration_groups"])


class Objective:
    """Weighted relative fuel-flow residuals of the calibration rows; logs every evaluation."""

    def __init__(self, model: V5Model, rows: pd.DataFrame, fixed_fit: dict | None = None):
        self.model, self.rows = model, rows
        self.fixed_fit = dict(fixed_fit or {})     # fitted parameters moved to fixed
        self.log: list[dict] = []

    def params(self, free: list[str], x) -> dict:
        p = dict(self.fixed_fit)
        p.update({k: float(v) for k, v in zip(free, x)})
        return p

    def residuals_params(self, p: dict) -> np.ndarray:
        pred = self.model.predict(p, self.rows)
        r = residual_vector(pred["ff"], self.rows, pred["status"])
        self.log.append(dict(p, sse=float(r @ r), n_unreachable=int((pred["status"] == "unreachable").sum())))
        return r

    def residuals(self, free: list[str], x) -> np.ndarray:
        return self.residuals_params(self.params(free, x))


def _box(free):
    lo = np.array([FIT_BOUNDS[k][0] for k in free])
    hi = np.array([FIT_BOUNDS[k][1] for k in free])
    return lo, hi


def polish(obj: Objective, free: list[str], x0, max_nfev: int):
    """Registered local step: scipy least_squares (trf, bounds, x_scale = box widths, diff_step 1e-4)."""
    from scipy.optimize import least_squares
    lo, hi = _box(free)
    x0 = np.clip(np.asarray(x0, float), lo, hi)
    return least_squares(lambda x: obj.residuals(free, x), x0, bounds=(lo, hi), method="trf",
                         x_scale=hi - lo, diff_step=1e-4, max_nfev=max_nfev)


def fit(obj: Objective, free: list[str], n_trials: int, polish_max_nfev: int, seed: int = 42) -> dict:
    """Registered fit: Optuna TPE (seed) over the box in FIT_ORDER, then least_squares polish."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    free = [k for k in FIT_ORDER if k in free]

    def objective(trial):
        x = [trial.suggest_float(k, *FIT_BOUNDS[k]) for k in free]
        r = obj.residuals(free, x)
        return float(r @ r)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    x_best = [study.best_params[k] for k in free]
    res = polish(obj, free, x_best, polish_max_nfev)
    sse_polished = float(res.fun @ res.fun)
    use_polished = sse_polished <= study.best_value
    x_opt = res.x if use_polished else np.array(x_best)
    lo, hi = _box(free)
    js = res.jac * (hi - lo)
    return {
        "free": free,
        "best_trial": {"params": study.best_params, "sse": study.best_value,
                       "number": study.best_trial.number},
        "polish": {"status": int(res.status), "message": res.message, "nfev": int(res.nfev),
                   "sse": sse_polished, "accepted": bool(use_polished)},
        "params": obj.params(free, x_opt),
        "sse": min(sse_polished, study.best_value),
        "jtj_condition_box_scaled": float(np.linalg.cond(js.T @ js)),
        "n_evaluations": len(obj.log),
    }


def profile(obj: Objective, free: list[str], opt: dict, grid_points: int, inner_max_nfev: int,
            n_eff: int, progress=None) -> dict:
    """Registered profile likelihood: each fitted parameter on a uniform grid over its box,
    the others re-fit (least_squares, warm-started from the neighbouring grid point);
    D = n_eff ln(SSE_profile / SSE_min)."""
    rows = []
    for name in free:
        others = [k for k in free if k != name]
        grid = np.linspace(*FIT_BOUNDS[name], grid_points)
        start = int(np.argmin(np.abs(grid - opt["params"][name])))
        for direction in (range(start, grid_points), range(start - 1, -1, -1)):
            x_prev = [opt["params"][k] for k in others]
            for i in direction:
                obj.fixed_fit[name] = float(grid[i])
                res = polish(obj, others, x_prev, inner_max_nfev)
                x_prev = res.x
                rows.append(dict(param=name, i=i, value=float(grid[i]), sse=float(res.fun @ res.fun),
                                 nfev=int(res.nfev), status=int(res.status),
                                 **{f"fit_{k}": float(v) for k, v in zip(others, res.x)}))
                if progress:
                    progress(rows[-1])
        del obj.fixed_fit[name]
    df = pd.DataFrame(rows).sort_values(["param", "i"]).reset_index(drop=True)
    sse_min = min(opt["sse"], float(df["sse"].min()))
    df["D"] = n_eff * np.log(df["sse"] / sse_min)
    verdicts = {}
    for name in free:
        g = df[df["param"] == name].sort_values("value")
        v, d = g["value"].to_numpy(), g["D"].to_numpy()
        lo_b, hi_b = FIT_BOUNDS[name]
        edges_ok = bool(d[0] >= CHI2_1_95 and d[-1] >= CHI2_1_95)
        inside = np.where(d < CHI2_1_95)[0]
        if len(inside):
            i0, i1 = inside.min(), inside.max()
            a = v[i0] if i0 == 0 else np.interp(CHI2_1_95, [d[i0], d[i0 - 1]], [v[i0], v[i0 - 1]])
            b = v[i1] if i1 == len(v) - 1 else np.interp(CHI2_1_95, [d[i1], d[i1 + 1]], [v[i1], v[i1 + 1]])
        else:   # grid too coarse to resolve the interval: bracket around the grid minimum
            j = int(np.argmin(d))
            a, b = v[max(j - 1, 0)], v[min(j + 1, len(v) - 1)]
        width_frac = float((b - a) / (hi_b - lo_b))
        verdicts[name] = {
            "D_at_lower_edge": float(d[0]), "D_at_upper_edge": float(d[-1]),
            "interval_95": [float(a), float(b)], "interval_width_frac_of_box": width_frac,
            "interval_contains_no_grid_point": bool(len(inside) == 0),
            "grid_argmin": float(v[int(np.argmin(d))]),
            "IDENTIFIED": bool(edges_ok and width_frac <= 0.5),
        }
    return {"sse_min": sse_min, "sse_min_source": "fit" if sse_min == opt["sse"] else "profile",
            "n_eff": n_eff, "verdicts": verdicts, "table": df}


# --------------------------------------------------------------------------
# Drivers (entry points: calibrate_lto.py --tag v5 [--pilot];
# identifiability_profile.py --v5 {pilot,full})
# --------------------------------------------------------------------------
OUT_DIR = ROOT / "outputs" / "phase6"
PILOT_FIT = OUT_DIR / "p61_pilot_fit.json"
PILOT_PROFILE = OUT_DIR / "identifiability_profile_v5_pilot"
FULL_FIT = ROOT / "outputs" / "calibration_v5.json"
FULL_PROFILE = ROOT / "outputs" / "identifiability_profile_v5"


def _write_new(path: Path, text: str) -> None:
    with open(path, "x") as fh:    # never overwrite a frozen artifact
        fh.write(text)


def _json(obj) -> str:
    import json
    return json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"


def run_calibration(pilot: bool, free: list[str] | None = None, n_workers: int = 8) -> dict:
    """Pilot or full v5 fit on the calibration group only (held-out targets never read)."""
    import datetime as dt
    reg = load_registration()
    split = load_split()
    stage = "pilot" if pilot else "full"
    out = PILOT_FIT if pilot else FULL_FIT
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    free = [k for k in FIT_ORDER if k in (free or FIT_ORDER)]
    if not pilot:
        import json
        pp = json.loads((PILOT_PROFILE.with_suffix(".json")).read_text())
        if free != pp["full_fit_free_set"]:
            raise SystemExit(f"--free {free} differs from the pilot decision {pp['full_fit_free_set']}")
    rows = calibration_rows(split)
    model = V5Model(reg["fixed_central"], nox_fit_exclude_models=split["heldout_models"],
                    n_workers=n_workers)
    obj = Objective(model, rows)
    t0 = dt.datetime.now()
    try:
        res = fit(obj, free, reg["fit"][stage]["n_trials"], reg["fit"][stage]["polish_max_nfev"])
        pred = model.predict(res["params"], rows)
    finally:
        model.close()
    rows_out = rows.drop(columns=["CO (g/kg)", "HC (g/kg)", "NOx (g/kg)"]).join(pred)
    res.update(
        stage=stage, registration="outputs/phase6/p61_registration.json (A1)",
        started=t0.isoformat(timespec="seconds"),
        finished=dt.datetime.now().isoformat(timespec="seconds"),
        fixed_central=reg["fixed_central"], fit_bounds=FIT_BOUNDS,
        calibration_weighted_mape_pct=weighted_mape(pred["ff"], rows, pred["status"]),
        n_rows=len(rows), n_unreachable=int((pred["status"] == "unreachable").sum()),
        unreachable_rows=rows_out.loc[pred["status"] == "unreachable",
                                      ["Unique ID", "Model", "Mode", "Target Thrust (kN)", "reason"]
                                      ].to_dict("records"),
        m_rated_at_F_REF_kg_s=res["params"].get("W_ref"),
        note="in-sample calibration fit (calibration group only); not validation",
    )
    stem = out.with_suffix("")
    _write_new(out, _json(res))
    pd.DataFrame(obj.log).to_csv(f"{stem}_evaluations.csv", index=False)
    rows_out.to_csv(f"{stem}_rows.csv", index=False)
    return res


def run_profile(stage: str, n_workers: int = 8) -> dict:
    import json
    reg = load_registration()
    split = load_split()
    fit_path = PILOT_FIT if stage == "pilot" else FULL_FIT
    out = PILOT_PROFILE if stage == "pilot" else FULL_PROFILE
    if out.with_suffix(".json").exists():
        raise SystemExit(f"{out}.json exists; refusing to overwrite")
    opt = json.loads(fit_path.read_text())
    free = opt["free"]
    ident = reg["identifiability"]
    rows = calibration_rows(split)
    model = V5Model(reg["fixed_central"], nox_fit_exclude_models=split["heldout_models"],
                    n_workers=n_workers)
    obj = Objective(model, rows, fixed_fit={k: v for k, v in opt["params"].items() if k not in free})
    log_path = Path(f"{out}_progress.log")

    def progress(r):
        with open(log_path, "a") as fh:
            fh.write(f"{r['param']} i={r['i']} value={r['value']:.6g} sse={r['sse']:.6g} "
                     f"nfev={r['nfev']}\n")
    try:
        prof = profile(obj, free, opt, ident[stage]["grid_points"], ident[stage]["inner_max_nfev"],
                       ident["n_eff"], progress=progress)
    finally:
        model.close()
    table = prof.pop("table")
    identified = [k for k in free if prof["verdicts"][k]["IDENTIFIED"]]
    prof.update(stage=stage, fit=str(fit_path.relative_to(ROOT)), free=free,
                rule=ident["rule"], threshold=CHI2_1_95,
                identified=identified,
                not_identified=[k for k in free if k not in identified],
                jtj_condition_box_scaled_at_fit=opt["jtj_condition_box_scaled"])
    if stage == "pilot":
        prof["full_fit_free_set"] = identified
        prof["pilot_consequence"] = ident["pilot_consequence"]
    table.to_csv(f"{out}.csv", index=False)
    _write_new(out.with_suffix(".json"), _json(prof))
    return prof
