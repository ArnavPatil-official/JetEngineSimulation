"""
Held-out validation of the ICAO NOx correlation (leave-one-engine-out).

WHY THIS EXISTS
---------------
NOx is one of the optimisation objectives, yet the only validated quantity in
the project was LTO fuel flow (scripts/validation/holdout_icao_validation.py).
Reviewer 2 made exactly this point: "fuel-flow agreement at four LTO points does
not validate any of the quantities the optimization actually depends on (NOx,
specific thrust differences between blends, lifecycle CO2) ... the
blend-discriminating outputs are themselves never validated."

The production NOx model (integrated_engine.EmissionsEstimator._fit_nox_model)
is a log-log regression

    EI_NOx = A * OPR^B * mdot_fuel^C

fitted across the ICAO certification databank. Before this script the ONLY
statistic reported for it was the in-sample R^2 printed at construction -- which
is what produced the withdrawn "R^2 = 0.9969" Highlight. An in-sample R^2 of a
3-parameter fit measures nothing.

WHAT THIS SCRIPT MEASURES
-------------------------
Leave-one-engine-model-out cross-validation. For each of the engine model names
in the databank: refit the correlation on every OTHER engine model, then predict
the held-out model's EI_NOx from its OPR and fuel flow. No record of a held-out
engine ever enters its own fit.

The fit is performed by the production class itself, via the
`nox_fit_exclude_models` argument, so this validates the shipped code path
rather than a re-implementation of it.

WHAT IT DOES *NOT* SHOW
-----------------------
FIRST, AND MOST IMPORTANT: every certification record in
data/icao_engine_data.csv is a Trent 1000 variant. Holding out one variant and
fitting on the other 27 is WITHIN-FAMILY generalisation, not cross-family. The
held-out MAPE therefore comes out almost identical to the in-sample MAPE, and
that similarity is a property of the dataset, not a demonstration of
transferability. Do not describe this as engine-independent validation. A
genuine test needs certification records from other manufacturers and cores.

The correlation also has no fuel-composition term. Held-out skill here is evidence
about the OPR / fuel-flow dependence across certificated engines burning
conventional Jet A. It is NOT evidence that the model can rank SAF blends on
NOx, and it must never be cited as such. The variance decomposition
(outputs/results/variance_decomposition.csv) and the three-path comparison
(outputs/nox_dual_path.csv) are the relevant evidence there.

OUTPUTS
-------
- outputs/nox_holdout_validation.csv          per-record predictions and errors
- outputs/nox_holdout_validation_summary.csv  per-mode and overall summary
- outputs/plots/nox_holdout_pred_vs_icao.png  predicted vs certificated EI_NOx
- outputs/plots/nox_holdout_error_boxplot.png per-mode percentage error

USAGE
-----
    python scripts/validation/nox_holdout_validation.py
"""

import sys
import os
import argparse
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from integrated_engine import EmissionsEstimator

ICAO_CSV = PROJECT_ROOT / "data" / "icao_engine_data.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUT_DIR / "plots"

# Fixed-order categorical palette, matching holdout_icao_validation.py
MODE_COLORS = {"TAKE-OFF": "#2a78d6", "APPROACH": "#1baf7a", "IDLE": "#eda100"}
MODE_ORDER = ["TAKE-OFF", "APPROACH", "IDLE"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Leave-one-engine-out validation of the ICAO NOx correlation")
    p.add_argument("--icao-csv", default=str(ICAO_CSV),
                   help="ICAO databank CSV (default: data/icao_engine_data.csv)")
    p.add_argument("--tag", default="",
                   help="Suffix appended to output filenames")
    return p.parse_args()


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Same filter the production fit applies, plus the engine-model column."""
    out = df[(df["Fuel Flow (kg/s)"] > 0) &
             (df["NOx (g/kg)"] > 0) &
             (df["Pressure Ratio"] > 1)].copy()
    out["EngineModel"] = out["Engine ID"].map(EmissionsEstimator.engine_model_name)
    return out


def fit_excluding(icao_csv: str, exclude_models):
    """Fit the PRODUCTION NOx model with `exclude_models` held out."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        est = EmissionsEstimator(icao_data_path=icao_csv,
                                 nox_fit_exclude_models=exclude_models)
    return est


def predict_ei(est, opr, m_dot_fuel):
    """EI_NOx [g/kg fuel] from the fitted correlation."""
    return est.nox_A * (opr ** est.nox_B) * (m_dot_fuel ** est.nox_C)


def main():
    args = parse_args()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.icao_csv)
    df = usable_rows(raw)
    models = sorted(df["EngineModel"].unique())

    print("=" * 78)
    print("NOx CORRELATION -- LEAVE-ONE-ENGINE-OUT VALIDATION")
    print("=" * 78)
    print(f"Usable records: {len(df)}   engine models: {len(models)}")

    # --- Reference: the production (all-data, in-sample) fit ----------------
    full = fit_excluding(args.icao_csv, None)
    df["EI_pred_insample"] = predict_ei(full, df["Pressure Ratio"].values,
                                        df["Fuel Flow (kg/s)"].values)
    in_sample_mape = float(
        np.mean(np.abs(df["EI_pred_insample"] - df["NOx (g/kg)"]) / df["NOx (g/kg)"]) * 100)
    print(f"\nProduction (all-data) fit: "
          f"EI_NOx = {full.nox_A:.4f} * OPR^{full.nox_B:.4f} * mdot^{full.nox_C:.4f}")
    print(f"  in-sample R^2   = {full.nox_fit_r2_in_sample:.4f}  (training diagnostic)")
    print(f"  in-sample MAPE  = {in_sample_mape:.2f}%")

    # --- Leave-one-engine-model-out ----------------------------------------
    records = []
    for i, model in enumerate(models, 1):
        est = fit_excluding(args.icao_csv, {model})
        held = df[df["EngineModel"] == model]
        pred = predict_ei(est, held["Pressure Ratio"].values,
                          held["Fuel Flow (kg/s)"].values)
        actual = held["NOx (g/kg)"].values
        for (_, row), p_ei, a_ei in zip(held.iterrows(), pred, actual):
            records.append({
                "EngineModel": model,
                "UniqueID": row["Unique ID"],
                "Mode": row["Mode"],
                "OPR": float(row["Pressure Ratio"]),
                "FuelFlow_kg_s": float(row["Fuel Flow (kg/s)"]),
                "EI_NOx_icao_g_kg": float(a_ei),
                "EI_NOx_pred_g_kg": float(p_ei),
                "Error_pct": float((p_ei - a_ei) / a_ei * 100.0),
                "AbsError_pct": float(abs(p_ei - a_ei) / a_ei * 100.0),
                "n_fit_records": int(est.nox_fit_n),
                "n_fit_models": int(est.nox_fit_n_models),
            })
        print(f"  [{i:2d}/{len(models)}] held out {model:<28} "
              f"n={len(held):2d}  MAPE={np.mean(np.abs((pred-actual)/actual))*100:6.2f}%")

    res = pd.DataFrame.from_records(records)

    # --- Summary ------------------------------------------------------------
    rows = []
    for mode in MODE_ORDER:
        sub = res[res["Mode"] == mode]
        if sub.empty:
            continue
        rows.append({
            "Scope": mode, "n": len(sub),
            "MAPE_pct": sub["AbsError_pct"].mean(),
            "MedianAbsError_pct": sub["AbsError_pct"].median(),
            "MeanBias_pct": sub["Error_pct"].mean(),
            "P90AbsError_pct": sub["AbsError_pct"].quantile(0.90),
            "MaxAbsError_pct": sub["AbsError_pct"].max(),
        })
    rows.append({
        "Scope": "ALL (held-out)", "n": len(res),
        "MAPE_pct": res["AbsError_pct"].mean(),
        "MedianAbsError_pct": res["AbsError_pct"].median(),
        "MeanBias_pct": res["Error_pct"].mean(),
        "P90AbsError_pct": res["AbsError_pct"].quantile(0.90),
        "MaxAbsError_pct": res["AbsError_pct"].max(),
    })
    rows.append({
        "Scope": "ALL (in-sample, reference)", "n": len(df),
        "MAPE_pct": in_sample_mape,
        "MedianAbsError_pct": float(np.median(
            np.abs(df["EI_pred_insample"] - df["NOx (g/kg)"]) / df["NOx (g/kg)"] * 100)),
        "MeanBias_pct": float(np.mean(
            (df["EI_pred_insample"] - df["NOx (g/kg)"]) / df["NOx (g/kg)"] * 100)),
        "P90AbsError_pct": float(np.quantile(
            np.abs(df["EI_pred_insample"] - df["NOx (g/kg)"]) / df["NOx (g/kg)"] * 100, 0.90)),
        "MaxAbsError_pct": float(np.max(
            np.abs(df["EI_pred_insample"] - df["NOx (g/kg)"]) / df["NOx (g/kg)"] * 100)),
    })
    summary = pd.DataFrame(rows)

    per_row_path = OUT_DIR / f"nox_holdout_validation{args.tag}.csv"
    summary_path = OUT_DIR / f"nox_holdout_validation_summary{args.tag}.csv"
    res.to_csv(per_row_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n" + "-" * 78)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("-" * 78)
    n_family = res["EngineModel"].str.startswith("Trent 1000").sum()
    print(f"CAVEAT 1 (dataset): {n_family}/{len(res)} held-out records are Trent 1000")
    print("      variants. This is WITHIN-FAMILY generalisation; held-out and in-sample")
    print("      MAPE agreeing is a property of the databank, not proof of transfer.")
    print("CAVEAT 2 (scope): the correlation has no fuel-composition term, so this")
    print("      does NOT validate blend NOx ranking under any circumstances.")

    # --- Plots --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for mode in MODE_ORDER:
        sub = res[res["Mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(sub["EI_NOx_icao_g_kg"], sub["EI_NOx_pred_g_kg"], s=26,
                   color=MODE_COLORS[mode], edgecolor="white", linewidth=0.5,
                   label=mode, zorder=3)
    lim_hi = max(res["EI_NOx_icao_g_kg"].max(), res["EI_NOx_pred_g_kg"].max()) * 1.08
    ax.plot([0, lim_hi], [0, lim_hi], color="#888888", lw=1.0, ls="--",
            zorder=2, label="1:1")
    ax.set_xlim(0, lim_hi)
    ax.set_ylim(0, lim_hi)
    ax.set_xlabel("ICAO certificated EI$_{NOx}$ (g/kg fuel)")
    ax.set_ylabel("Held-out predicted EI$_{NOx}$ (g/kg fuel)")
    ax.set_title("NOx correlation: leave-one-engine-out prediction\n"
                 f"{len(res)} records, {len(models)} engine models, "
                 f"MAPE {res['AbsError_pct'].mean():.1f}%", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    scatter_path = PLOTS_DIR / f"nox_holdout_pred_vs_icao{args.tag}.png"
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    data = [res[res["Mode"] == m]["Error_pct"].values for m in MODE_ORDER]
    bp = ax.boxplot(data, labels=MODE_ORDER, patch_artist=True, widths=0.55)
    for patch, mode in zip(bp["boxes"], MODE_ORDER):
        patch.set_facecolor(MODE_COLORS[mode])
        patch.set_alpha(0.55)
        patch.set_edgecolor(MODE_COLORS[mode])
    for med in bp["medians"]:
        med.set_color("#222222")
    ax.axhline(0.0, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.set_ylabel("Held-out error (%)")
    ax.set_title("NOx correlation held-out error by LTO mode", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    box_path = PLOTS_DIR / f"nox_holdout_error_boxplot{args.tag}.png"
    fig.savefig(box_path, dpi=200)
    plt.close(fig)

    print(f"\nWrote: {per_row_path.relative_to(PROJECT_ROOT)}")
    print(f"       {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"       {scatter_path.relative_to(PROJECT_ROOT)}")
    print(f"       {box_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
