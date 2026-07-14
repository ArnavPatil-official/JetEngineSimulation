"""
Held-out cross-engine validation of the LTO fuel-flow model (Phase 1.1).

The engine model is calibrated ONCE against the ICAO certification record of the
Trent 1000-AE3 (UID 02P23RR126) by scripts/optimization/calibrate_lto.py, which
freezes its best parameters in outputs/calibration_trent1000_ae3.json. This
script loads that frozen calibration — it never re-tunes — and predicts LTO
fuel flow for every OTHER Trent 1000 certification record in
data/icao_engine_data.csv, scaling core airflow by rated-thrust ratio and
setting the compressor pressure ratio from each record's OPR column.

Data-reality notes (vs. the original plan wording):
- The CSV contains 3 LTO modes per record (TAKE-OFF, APPROACH, IDLE). There are
  NO CLIMB rows, so the calibration's Climb target (2.050 kg/s) is not traceable
  to this dataset and climb cannot be validated here.
- The CSV holds 60 certification records (Unique IDs) covering 28 engine model
  names; several models were certified multiple times. The headline held-out
  MAPE excludes every record whose model name is "Trent 1000-AE3" (the
  calibration engine under any certification), and the with-AE3 figure is also
  reported.

Outputs:
- outputs/holdout_icao_validation.csv          (per record x mode predictions)
- outputs/holdout_icao_validation_summary.csv  (per-mode and overall MAPE)
- outputs/plots/holdout_pred_vs_icao_scatter.png
- outputs/plots/holdout_mode_error_boxplot.png
"""

import sys
import json
import contextlib
import os
from pathlib import Path

# Add project root to sys.path so imports resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from integrated_engine import IntegratedTurbofanEngine, FUEL_LIBRARY

CALIBRATION_JSON = PROJECT_ROOT / "outputs" / "calibration_trent1000_ae3.json"
ICAO_CSV = PROJECT_ROOT / "data" / "icao_engine_data.csv"
OUT_CSV = PROJECT_ROOT / "outputs" / "holdout_icao_validation.csv"
OUT_SUMMARY = PROJECT_ROOT / "outputs" / "holdout_icao_validation_summary.csv"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

# CSV mode label -> calibration parameter names (Climb is absent from the CSV)
MODE_MAP = {
    "TAKE-OFF": {"phi_key": "phi_to", "scale_key": "Takeoff"},
    "APPROACH": {"phi_key": "phi_app", "scale_key": "Approach"},
    "IDLE": {"phi_key": "phi_idle", "scale_key": "Idle"},
}

# Fixed-order categorical palette (dataviz reference palette, slots 1-3)
MODE_COLORS = {"TAKE-OFF": "#2a78d6", "APPROACH": "#1baf7a", "IDLE": "#eda100"}
MODE_ORDER = ["TAKE-OFF", "APPROACH", "IDLE"]


def load_calibration() -> dict:
    if not CALIBRATION_JSON.exists():
        raise FileNotFoundError(
            f"{CALIBRATION_JSON} not found. Run "
            "scripts/optimization/calibrate_lto.py first to freeze the calibration."
        )
    with open(CALIBRATION_JSON) as fh:
        return json.load(fh)


def main() -> None:
    calib = load_calibration()
    best = calib["best_params"]
    fixed = calib["fixed_parameters"]
    eta_b = best["eta_combustor"]
    base_airflow = fixed["base_airflow_kg_s"]
    mode_scales = fixed["mode_scales"]
    calib_uid = calib["icao_uid"]

    df = pd.read_csv(ICAO_CSV)
    df["model"] = (
        df["Engine ID"].str.replace(r"\s*BYPASS RATIO.*", "", regex=True).str.strip()
    )

    calib_rows = df[df["Unique ID"] == calib_uid]
    if calib_rows.empty:
        raise ValueError(f"Calibration UID {calib_uid} not found in {ICAO_CSV}")
    calib_thrust = float(calib_rows["Rated Thrust (kN)"].iloc[0])
    calib_model = calib_rows["model"].iloc[0]

    holdout = df[df["Unique ID"] != calib_uid].copy()
    print("=" * 70)
    print("HELD-OUT CROSS-ENGINE VALIDATION (frozen calibration, no re-tuning)")
    print("=" * 70)
    print(f"Calibration engine: {calib_model} (UID {calib_uid}, "
          f"{calib_thrust:.1f} kN, eta_comb={eta_b:.4f})")
    print(f"Held-out records: {holdout['Unique ID'].nunique()} "
          f"({holdout['model'].nunique()} model names), "
          f"modes present in CSV: {sorted(df['Mode'].unique())}")
    print("NOTE: CSV has no CLIMB rows; climb cannot be validated from this data.\n")

    engine = IntegratedTurbofanEngine()

    records = []
    for uid, grp in holdout.groupby("Unique ID"):
        opr = float(grp["Pressure Ratio"].iloc[0])
        thrust = float(grp["Rated Thrust (kN)"].iloc[0])
        model = grp["model"].iloc[0]
        thrust_ratio = thrust / calib_thrust

        for _, row in grp.iterrows():
            mode = row["Mode"]
            if mode not in MODE_MAP:
                continue
            phi = best[MODE_MAP[mode]["phi_key"]]
            airflow_scale = mode_scales[MODE_MAP[mode]["scale_key"]]["airflow_scale"]

            # Frozen-calibration prediction: airflow scaled by rated-thrust
            # ratio, compressor OPR set from the CSV (constant across modes,
            # matching how the calibration effectively ran — see JSON notes).
            engine.compressor.pi_c = opr
            engine.design_point["mass_flow_core"] = (
                base_airflow * thrust_ratio * airflow_scale
            )

            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                res = engine.run_full_cycle(
                    fuel_blend=FUEL_LIBRARY["Jet-A1"],
                    phi=phi,
                    combustor_efficiency=eta_b,
                )
            pred_ff = res["performance"]["fuel_mass_flow"]
            icao_ff = float(row["Fuel Flow (kg/s)"])

            records.append({
                "Unique ID": uid,
                "Model": model,
                "Is_AE3_Model": model == calib_model,
                "Mode": mode,
                "OPR": opr,
                "Rated Thrust (kN)": thrust,
                "Thrust Ratio": thrust_ratio,
                "Phi": phi,
                "Airflow (kg/s)": base_airflow * thrust_ratio * airflow_scale,
                "ICAO Fuel Flow (kg/s)": icao_ff,
                "Predicted Fuel Flow (kg/s)": pred_ff,
                "Abs Pct Error": abs(pred_ff - icao_ff) / icao_ff * 100.0,
            })
        print(f"  {uid:<12} {model:<18} OPR={opr:4.1f} "
              f"thrust={thrust:5.1f} kN ... done")

    res_df = pd.DataFrame(records)
    res_df.to_csv(OUT_CSV, index=False)

    strict = res_df[~res_df["Is_AE3_Model"]]
    summary_rows = []
    for label, sub in [("held-out (excl. AE3 models)", strict),
                       ("held-out (incl. AE3 re-certifications)", res_df)]:
        for mode in MODE_ORDER:
            errs = sub.loc[sub["Mode"] == mode, "Abs Pct Error"]
            summary_rows.append({
                "Set": label, "Mode": mode, "N": len(errs),
                "MAPE (%)": errs.mean(), "Median APE (%)": errs.median(),
                "P25 (%)": errs.quantile(0.25), "P75 (%)": errs.quantile(0.75),
                "Max APE (%)": errs.max(),
            })
        summary_rows.append({
            "Set": label, "Mode": "ALL", "N": len(sub),
            "MAPE (%)": sub["Abs Pct Error"].mean(),
            "Median APE (%)": sub["Abs Pct Error"].median(),
            "P25 (%)": sub["Abs Pct Error"].quantile(0.25),
            "P75 (%)": sub["Abs Pct Error"].quantile(0.75),
            "Max APE (%)": sub["Abs Pct Error"].max(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 70)
    print("HELD-OUT VALIDATION SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ----- Plot 1: predicted vs ICAO scatter with identity line -----
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    lim_hi = 1.1 * max(res_df["ICAO Fuel Flow (kg/s)"].max(),
                       res_df["Predicted Fuel Flow (kg/s)"].max())
    ax.plot([0, lim_hi], [0, lim_hi], ls="--", lw=1, color="#9a9a94", zorder=1)
    ax.annotate("identity", xy=(0.86 * lim_hi, 0.88 * lim_hi), color="#6b6b66",
                fontsize=9, rotation=45, ha="center", va="center")
    for mode in MODE_ORDER:
        sub = res_df[res_df["Mode"] == mode]
        ax.scatter(sub["ICAO Fuel Flow (kg/s)"], sub["Predicted Fuel Flow (kg/s)"],
                   s=34, color=MODE_COLORS[mode], edgecolors="white",
                   linewidths=0.8, label=mode.title(), zorder=2)
    ax.set_xlim(0, lim_hi)
    ax.set_ylim(0, lim_hi)
    ax.set_xlabel("ICAO certification fuel flow [kg/s]")
    ax.set_ylabel("Predicted fuel flow [kg/s]")
    ax.set_title("Held-out fuel-flow prediction vs. ICAO data\n"
                 f"(calibrated on {calib_model} {calib_uid} only; "
                 f"{holdout['Unique ID'].nunique()} held-out records)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "holdout_pred_vs_icao_scatter.png", dpi=300)
    plt.close(fig)

    # ----- Plot 2: per-mode absolute-percent-error box plot -----
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [strict.loc[strict["Mode"] == m, "Abs Pct Error"] for m in MODE_ORDER]
    bp = ax.boxplot(data, tick_labels=[m.title() for m in MODE_ORDER],
                    patch_artist=True, widths=0.5,
                    medianprops={"color": "#1a1a19", "lw": 1.4},
                    flierprops={"marker": "o", "markersize": 4,
                                "markerfacecolor": "#9a9a94",
                                "markeredgecolor": "none"})
    for patch, mode in zip(bp["boxes"], MODE_ORDER):
        patch.set_facecolor(MODE_COLORS[mode])
        patch.set_alpha(0.75)
        patch.set_edgecolor("white")
    for i, errs in enumerate(data, start=1):
        ax.annotate(f"MAPE {errs.mean():.1f}%", xy=(i, errs.median()),
                    xytext=(i + 0.28, errs.median()), fontsize=9,
                    color="#1a1a19", va="center")
    ax.set_ylabel("Absolute fuel-flow error [%]")
    ax.set_title("Held-out error by LTO mode "
                 "(excl. AE3 re-certifications; CSV has no Climb mode)")
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "holdout_mode_error_boxplot.png", dpi=300)
    plt.close(fig)

    overall = strict["Abs Pct Error"].mean()
    print(f"\nHeadline held-out MAPE (excl. AE3 models): {overall:.2f}%")
    print(f"Saved: {OUT_CSV}\n       {OUT_SUMMARY}")
    print(f"       {PLOTS_DIR / 'holdout_pred_vs_icao_scatter.png'}")
    print(f"       {PLOTS_DIR / 'holdout_mode_error_boxplot.png'}")


if __name__ == "__main__":
    main()
