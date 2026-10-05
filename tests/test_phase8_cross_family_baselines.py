"""P8-A1 B2 and paired family bootstrap, without opening held-out targets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))
from cross_family_baselines import (SEED, fit_b2, paired_family_bootstrap,  # noqa: E402
                                    predict_b2)


def synthetic_calibration():
    rows = []
    for mode, beta in (("TAKE-OFF", [0.002, -0.00002, 0.00003]),
                       ("APPROACH", [0.0015, -0.00001, 0.00005])):
        for i in range(12):
            opr = 18.0 + 2.0 * i
            bpr = 3.0 + ((i * 7) % 9) / 2.0
            thrust = (80.0 + i * 15) * (1.0 if mode == "TAKE-OFF" else 0.3)
            tsfc = beta[0] + beta[1] * opr + beta[2] * bpr
            rows.append({"Mode": mode, "Pressure Ratio": opr, "Bypass Ratio": bpr,
                         "Target Thrust (kN)": thrust, "Fuel Flow (kg/s)": tsfc * thrust,
                         "w": 1.0 / (1 + i)})
    return pd.DataFrame(rows)


def test_b2_recovers_registered_per_mode_weighted_linear_fit():
    cal = synthetic_calibration()
    fitted = fit_b2(cal)
    assert fitted.rank == {"APPROACH": 3, "TAKE-OFF": 3}
    assert fitted.coefficients["TAKE-OFF"] == pytest.approx((.002, -.00002, .00003), abs=1e-14)
    assert fitted.coefficients["APPROACH"] == pytest.approx((.0015, -.00001, .00005), abs=1e-14)
    predicted = predict_b2(fitted, cal)
    assert predicted.index.equals(cal.index)
    assert np.allclose(predicted, cal["Fuel Flow (kg/s)"], rtol=1e-12)


def test_b2_rejects_rank_deficiency_and_unseen_mode():
    cal = synthetic_calibration()
    cal["Bypass Ratio"] = 2 * cal["Pressure Ratio"]
    with pytest.raises(ValueError, match="rank 2"):
        fit_b2(cal)
    fit = fit_b2(synthetic_calibration())
    other = synthetic_calibration().iloc[:1].copy()
    other["Mode"] = "IDLE"
    with pytest.raises(ValueError, match="absent from calibration"):
        predict_b2(fit, other)


def test_b2_can_fit_existing_trent_calibration_pool_only():
    """Read only the registered calibration rows; never open held-out rows."""
    sys.path.insert(0, str(ROOT / "scripts" / "optimization"))
    import lto_v5  # noqa: E402
    calibration = lto_v5.calibration_rows()
    assert len(calibration) == 93
    fit = fit_b2(calibration)
    assert set(fit.rank.values()) == {3}
    pred = predict_b2(fit, calibration)
    assert len(pred) == 93 and np.isfinite(pred).all() and (pred > 0).all()


def bootstrap_rows():
    data = []
    for family in range(8):
        for row in range(2):
            data.append({"Family": f"F{family}", "Fuel Flow (kg/s)": 1.0,
                         "w": 1.0 if row == 0 else 3.0})
    rows = pd.DataFrame(data)
    model = np.array([1.01 + .002 * i + .001 * j for i in range(8) for j in range(2)])
    b0 = np.array([1.08 + .003 * i + .001 * j for i in range(8) for j in range(2)])
    b1 = np.array([1.05 + .002 * i + .002 * j for i in range(8) for j in range(2)])
    b2 = np.array([1.035 + .001 * i + .002 * j for i in range(8) for j in range(2)])
    return rows, {"model": model, "B0": b0, "B1": b1, "B2": b2}


def test_bootstrap_uses_same_family_draw_for_every_arm_and_registered_seed():
    rows, preds = bootstrap_rows()
    result = paired_family_bootstrap(rows, preds, n_replicates=1000)
    assert result["n_families"] == 8 and result["seed"] == SEED
    assert result["G2_1_pass"]
    family_delta_b2 = []
    for i in range(8):
        sl = slice(2 * i, 2 * i + 2)
        family_delta_b2.append(np.average(100 * (preds["B2"][sl] - preds["model"][sl]),
                                           weights=[1, 3]))
    rng = np.random.default_rng(SEED)
    draw = rng.integers(8, size=(1000, 8))
    expected = np.quantile(np.mean(np.array(family_delta_b2)[draw], axis=1), [.025, .975])
    assert result["comparisons"]["B2"]["ci95_pp"] == pytest.approx(expected)
    assert result == paired_family_bootstrap(rows, preds, n_replicates=1000)
    assert result["comparisons"]["B0"]["delta_pp"] > \
        result["comparisons"]["B1"]["delta_pp"] > \
        result["comparisons"]["B2"]["delta_pp"]


def test_bootstrap_requires_eight_families_and_aligned_predictions():
    rows, preds = bootstrap_rows()
    with pytest.raises(ValueError, match=">=8"):
        paired_family_bootstrap(rows[rows["Family"] != "F7"],
                                {k: v[:-2] for k, v in preds.items()}, n_replicates=10)
    wrong_index = pd.Series(preds["model"], index=range(1, len(rows) + 1))
    with pytest.raises(ValueError, match="index differs"):
        paired_family_bootstrap(rows, {**preds, "model": wrong_index}, n_replicates=10)
