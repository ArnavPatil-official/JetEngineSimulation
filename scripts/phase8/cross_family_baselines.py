"""Registered P8-A1 B2 and family-cluster G2.1 comparison utilities.

The fitter accepts calibration rows only. Held-out predictions and the paired
bootstrap are separate calls, so callers can fit before the held-out split is
opened. Cross-family bootstrap clusters are whole engine families (the user's
2026-09-29 clarification), never individual modes, engines or rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

FEATURES = ("Pressure Ratio", "Bypass Ratio")
SEED = 20260929
REPLICATES = 10_000


@dataclass(frozen=True)
class B2Fit:
    """Per-mode beta0, beta_OPR, beta_BPR and weighted design rank."""

    coefficients: dict[str, tuple[float, float, float]]
    rank: dict[str, int]


def _numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column not in df:
            raise KeyError(f"missing {column!r}")
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column}: non-finite calibration value")


def fit_b2(calibration: pd.DataFrame) -> B2Fit:
    """Weighted least squares TSFC ~ 1 + OPR + BPR separately for each mode.

    The registered target is ICAO fuel flow / mode target thrust (kg/s/kN);
    `w` is the existing group-balanced row weight from `lto_v5.attach_groups`.
    A rank-deficient pool fails rather than silently using a non-unique fit.
    """
    _numeric(calibration, (*FEATURES, "Target Thrust (kN)", "Fuel Flow (kg/s)", "w"))
    if "Mode" not in calibration:
        raise KeyError("missing 'Mode'")
    if (calibration["w"] <= 0).any() or (calibration["Target Thrust (kN)"] <= 0).any() \
            or (calibration["Fuel Flow (kg/s)"] <= 0).any():
        raise ValueError("B2 needs positive weights, thrust and measured fuel flow")
    coefficients, rank = {}, {}
    for mode, group in calibration.groupby("Mode", sort=True):
        x = np.column_stack([np.ones(len(group)), group[FEATURES[0]].to_numpy(dtype=float),
                             group[FEATURES[1]].to_numpy(dtype=float)])
        y = group["Fuel Flow (kg/s)"].to_numpy(dtype=float) / \
            group["Target Thrust (kN)"].to_numpy(dtype=float)
        sw = np.sqrt(group["w"].to_numpy(dtype=float))
        beta, _, matrix_rank, _ = np.linalg.lstsq(sw[:, None] * x, sw * y, rcond=None)
        if matrix_rank != 3:
            raise ValueError(f"B2 {mode}: OPR/BPR calibration design rank {matrix_rank}, need 3")
        coefficients[str(mode)] = tuple(map(float, beta))
        rank[str(mode)] = int(matrix_rank)
    if not coefficients:
        raise ValueError("B2 calibration pool is empty")
    return B2Fit(coefficients, rank)


def predict_b2(fit: B2Fit, rows: pd.DataFrame) -> pd.Series:
    """FF = fitted mode TSFC * target thrust, with no post-fit clipping."""
    _numeric(rows, (*FEATURES, "Target Thrust (kN)"))
    if "Mode" not in rows:
        raise KeyError("missing 'Mode'")
    if (rows["Target Thrust (kN)"] <= 0).any():
        raise ValueError("target thrust must be positive")
    out = np.empty(len(rows), dtype=float)
    for mode in rows["Mode"].unique():
        if mode not in fit.coefficients:
            raise ValueError(f"B2 mode {mode!r} was absent from calibration")
        mask = (rows["Mode"] == mode).to_numpy()
        b = np.array(fit.coefficients[mode])
        x = np.column_stack([np.ones(mask.sum()), rows.loc[mask, FEATURES[0]],
                             rows.loc[mask, FEATURES[1]]])
        out[mask] = (x @ b) * rows.loc[mask, "Target Thrust (kN)"].to_numpy(dtype=float)
    return pd.Series(out, index=rows.index, name="B2 Fuel Flow (kg/s)")


def paired_family_bootstrap(rows: pd.DataFrame, predictions: Mapping[str, pd.Series | np.ndarray],
                            *, n_replicates: int = REPLICATES, seed: int = SEED) -> dict:
    """Paired equal-family bootstrap of group-weighted MAPE differences.

    Required keys: model, B0, B1, B2. Every draw samples whole held-out
    families with replacement and is reused for all predictions. The weight
    `w` is normalised *within* each family; the drawn families then have equal
    weight, exactly as P8-A1 specifies. Returned differences are baseline
    MAPE minus model MAPE in percentage points.
    """
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    required = {"model", "B0", "B1", "B2"}
    if set(predictions) != required:
        raise ValueError(f"predictions need exactly {sorted(required)}")
    _numeric(rows, ("Fuel Flow (kg/s)", "w"))
    if "Family" not in rows:
        raise KeyError("held-out rows need a registered Family column")
    if rows["Family"].isna().any() or (rows["Fuel Flow (kg/s)"] <= 0).any() or \
            (rows["w"] <= 0).any():
        raise ValueError("families, observed fuel flow and weights must be present/positive")
    families = sorted(map(str, rows["Family"].unique()))
    if len(families) < 8:
        raise ValueError(f"registered cross-family score needs >=8 held-out families, got {len(families)}")
    observed = rows["Fuel Flow (kg/s)"].to_numpy(dtype=float)
    family_values = rows["Family"].astype(str).to_numpy()
    weights = rows["w"].to_numpy(dtype=float)
    mean_ape = {}
    for name, pred in predictions.items():
        if isinstance(pred, pd.Series):
            if not pred.index.equals(rows.index):
                raise ValueError(f"{name}: prediction index differs from held-out rows")
            arr = pred.to_numpy(dtype=float)
        else:
            arr = np.asarray(pred, dtype=float)
        if arr.shape != observed.shape or not np.isfinite(arr).all():
            raise ValueError(f"{name}: predictions must be finite, one per held-out row")
        ape = 100 * np.abs(arr - observed) / observed
        mean_ape[name] = np.array([
            float(np.average(ape[family_values == f], weights=weights[family_values == f]))
            for f in families])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(families), size=(n_replicates, len(families)))
    point = {name: float(np.mean(value)) for name, value in mean_ape.items()}
    comparisons = {}
    for name in ("B0", "B1", "B2"):
        family_delta = mean_ape[name] - mean_ape["model"]
        sampled = np.mean(family_delta[draws], axis=1)
        lo, hi = np.quantile(sampled, [.025, .975], method="linear")
        delta = point[name] - point["model"]
        comparisons[name] = {"delta_pp": float(delta), "ci95_pp": [float(lo), float(hi)],
                             "passes_margin_and_ci": bool(delta >= .25 and lo > 0)}
    return {"n_families": len(families), "families": families,
            "n_replicates": n_replicates, "seed": seed, "mape_pct": point,
            "comparisons": comparisons,
            "G2_1_pass": comparisons["B1"]["passes_margin_and_ci"] and
            comparisons["B2"]["passes_margin_and_ci"]}
