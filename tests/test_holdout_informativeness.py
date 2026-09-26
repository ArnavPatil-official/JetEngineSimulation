"""
Phase 6 P6.1 Step 1 — finding F-A: is the held-out fuel-flow validation informative?

The v4 held-out validation (scripts/validation/holdout_icao_validation.py
--tag _v4) set each held-out engine's core airflow to base_airflow x
thrust_ratio. With phi fixed per mode, fuel flow is phi*f_st*beta*m_core*x^k_mdot,
so the "prediction" is AE3's calibrated fuel flow times the rated-thrust ratio:
OPR and BPR never enter it. The first two tests pin that historical diagnosis
on the frozen v4 artifact (which is never modified). The last test is the
requirement on the replacement: the thrust-matched v5 held-out predictions must
depend on the cycle, not just on rated thrust. It fails until
outputs/holdout_icao_validation_v5.csv exists (docs/plan.md P6.1 Step 1).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
V4_CSV = ROOT / "outputs" / "archive" / "pre_phase6" / "holdout_icao_validation_v4.csv"  # archived P6.6
V5_CSV = ROOT / "outputs" / "holdout_icao_validation_v5.csv"
ICAO_CSV = ROOT / "data" / "icao_engine_data.csv"
AE3_UID = "02P23RR126"


@pytest.fixture(scope="module")
def v4():
    df = pd.read_csv(V4_CSV)
    icao = pd.read_csv(ICAO_CSV)
    ae3 = icao[icao["Unique ID"] == AE3_UID].set_index("Mode")["Fuel Flow (kg/s)"]
    df["naive_ff"] = df["Mode"].map(ae3) * df["Thrust Ratio"]
    df["naive_ape"] = ((df["naive_ff"] - df["ICAO Fuel Flow (kg/s)"]).abs()
                       / df["ICAO Fuel Flow (kg/s)"] * 100.0)
    return df


def test_v4_prediction_is_a_rated_thrust_rescaling(v4):
    """Historical diagnosis (F-A): predicted fuel flow / thrust ratio is constant per mode."""
    ratio = v4["Predicted Fuel Flow (kg/s)"] / v4["Thrust Ratio"]
    expected = {"TAKE-OFF": 2.318207, "APPROACH": 0.618863, "IDLE": 0.242230}
    for mode, grp in ratio.groupby(v4["Mode"]):
        assert len(grp) == 59
        assert grp.std() / grp.mean() < 1e-12, mode
        assert grp.mean() == pytest.approx(expected[mode], abs=5e-7)


def test_v4_model_vs_naive_ae3_ratio_baseline(v4):
    """F-A numbers, kept distinct for the 177-row and 171-row (excl. AE3 models) subsets."""
    assert len(v4) == 177
    assert v4["Abs Pct Error"].mean() == pytest.approx(2.46016229, abs=1e-7)
    assert v4["naive_ape"].mean() == pytest.approx(3.21571920, abs=1e-7)
    strict = v4[~v4["Is_AE3_Model"]]
    assert len(strict) == 171
    assert strict["Abs Pct Error"].mean() == pytest.approx(2.50278929, abs=1e-7)
    assert strict["naive_ape"].mean() == pytest.approx(3.24190221, abs=1e-7)
    # at take-off the model-free baseline is the better of the two
    to = v4[v4["Mode"] == "TAKE-OFF"]
    assert to["naive_ape"].mean() < to["Abs Pct Error"].mean()


def test_v5_holdout_prediction_depends_on_the_cycle():
    """The replacement validation must not be a rescaling rule (fails until v5 exists)."""
    assert V5_CSV.exists(), f"{V5_CSV.name} not produced yet (P6.1 Step 5)"
    df = pd.read_csv(V5_CSV)
    for col in ("Model", "Mode", "OPR", "BPR", "Rated Thrust (kN)",
                "Target Thrust (kN)", "Predicted Fuel Flow (kg/s)",
                "B0 Fuel Flow (kg/s)", "B1 Fuel Flow (kg/s)"):
        assert col in df.columns, col
    ok = df["Predicted Fuel Flow (kg/s)"].notna()
    assert ok.all(), "every held-out row must have a prediction or be reported as a failure"
    tsfc = df["Predicted Fuel Flow (kg/s)"] / df["Target Thrust (kN)"]
    for mode, grp in tsfc.groupby(df["Mode"]):
        # not proportional to thrust: predicted TSFC varies across held-out engines
        assert grp.std() / grp.mean() > 1e-4, mode
    # engines with the same rated thrust but different OPR/BPR get different predictions
    per = df.assign(tsfc=tsfc).groupby(["Mode", "Rated Thrust (kN)"])
    n_pairs = 0
    for _, grp in per:
        if grp["OPR"].nunique() > 1:
            n_pairs += 1
            assert grp["tsfc"].nunique() > 1
    assert n_pairs > 0, "held-out group has no equal-thrust, different-OPR engines to test"
