"""Streaming, single-pass quantitative scores for frozen paired checkpoints."""
from __future__ import annotations

import csv
import math
import statistics

from . import model

FIELDS = ("rho", "u", "T", "p")
RESIDUALS = ("mass", "momentum", "energy", "eos")


def metrics(case, xs, prediction, exact, residual, bars):
    """Each bar applies to each condition; no pooled score replaces these gates."""
    import numpy as np
    finite = bool(np.isfinite(prediction).all() and np.isfinite(residual).all())
    positive = bool(finite and (prediction > 0).all())
    result = {"finite": finite, "positive": positive, "accuracy_pass": False}
    if not positive:
        return result, None
    relative = (prediction - exact) / exact
    if not np.isfinite(relative).all() or not np.isfinite(relative**2).all() or not np.isfinite(residual**2).all():
        result["finite"] = False
        return result, None
    profile_max, profile_rms = np.max(np.abs(relative), axis=0), np.sqrt(np.mean(relative**2, axis=0))
    residual_max, residual_rms = np.max(np.abs(residual), axis=0), np.sqrt(np.mean(residual**2, axis=0))
    for i, name in enumerate(FIELDS):
        result[f"{name}_relative_max"] = float(profile_max[i])
        result[f"{name}_relative_rms"] = float(profile_rms[i])
    for i, name in enumerate(RESIDUALS):
        result[f"residual_{name}_max"] = float(residual_max[i])
        result[f"residual_{name}_rms"] = float(residual_rms[i])
    rho, u, temperature, pressure = prediction.T
    ratio_R, gamma = case["R"] / 287.0, case["gamma"]
    cp = gamma * ratio_R / (gamma - 1)
    mach = u / np.sqrt(gamma * ratio_R * temperature)
    area = 1 + .5 * np.asarray(xs)**2
    mass_reference = exact[:, 0] * exact[:, 1] * area
    result["mass_flow_relative_max"] = float(np.max(np.abs(rho*u*area/mass_reference - 1)))
    result["enthalpy_relative_max"] = float(np.max(np.abs((cp*temperature+.5*u*u)/cp - 1)))
    with np.errstate(over="ignore", invalid="ignore"):
        recovered_p0 = pressure * (1+.5*(gamma-1)*mach**2)**(gamma/(gamma-1))
    if not np.isfinite(recovered_p0).all():
        result["finite"] = False
        return result, None
    result["total_pressure_relative_max"] = float(np.max(np.abs(recovered_p0 - 1)))
    result["eos_max_abs"] = float(np.max(np.abs(pressure-ratio_R*rho*temperature)))
    if not all(math.isfinite(value) for key,value in result.items() if key.endswith(("_max", "_rms", "_abs"))):
        result["finite"] = False
        return {key:(value if not isinstance(value,float) or math.isfinite(value) else None) for key,value in result.items()}, None
    if case["regime"] == "smooth_subcritical":
        result["branch_pass"] = bool((mach < 1).all())
        result["exit_pressure_relative"] = float(abs(pressure[-1]*case["NPR"] - 1))
        branch = result["branch_pass"] and result["exit_pressure_relative"] <= bars["subcritical_exit_pressure_relative_max"]
    else:
        coordinates = np.asarray(xs)
        result["branch_pass"] = bool((mach[coordinates <= -.05] < 1).all() and (mach[coordinates >= .05] > 1).all())
        throat = np.flatnonzero(coordinates == 0)
        if len(throat) != 1:
            raise ValueError("registered scoring grid must contain exactly one throat")
        result["throat_mach_absolute"] = float(abs(mach[throat[0]]-1))
        branch = result["branch_pass"] and result["throat_mach_absolute"] <= bars["choked_throat_mach_abs_max"]
    result["accuracy_pass"] = bool(branch
        and (profile_max <= bars["profile_max_relative_error_max"]).all()
        and (profile_rms <= bars["profile_rms_relative_error_max"]).all()
        and (residual_max <= bars["residual_max_abs_max_each"]).all()
        and (residual_rms <= bars["residual_rms_max_each"]).all()
        and result["mass_flow_relative_max"] <= bars["mass_flow_vs_exact_relative_max"]
        and result["enthalpy_relative_max"] <= bars["total_enthalpy_vs_cpT0_relative_max"]
        and result["total_pressure_relative_max"] <= bars["recovered_total_pressure_vs_p0_relative_max"]
        and result["eos_max_abs"] <= bars["eos_dimensionless_max_abs"])
    return result, float(np.sum(relative**2))


def paired_decisions(aggregates, reg):
    """All predeclared seed pairs participate; a zero control cannot show benefit."""
    seeds = reg["models"]["paired_seeds"]
    pairs, accuracy = [], True
    for panel in ("synthetic_test", "product_test"):
        for regime in ("smooth_subcritical", "smooth_choked"):
            ratios = []
            for seed in seeds:
                physics = aggregates[(panel, regime, seed, "physics_on")]
                control = aggregates[(panel, regime, seed, "data_only")]
                accuracy = accuracy and physics["accuracy_pass"]
                num, den = physics["relative_rmse"], control["relative_rmse"]
                ratios.append(num/den if num is not None and den is not None and den > 0 else None)
            valid = all(value is not None and math.isfinite(value) for value in ratios)
            median = statistics.median(ratios) if valid else None
            pairs.append({"panel":panel, "regime":regime, "paired_ratios":ratios,
                          "median_ratio":median,
                          "physics_benefit_pass":bool(valid and median <= .8 and sum(value < 1 for value in ratios) >= 2)})
    benefit = all(row["physics_benefit_pass"] for row in pairs)
    return {"physics_on_accuracy_pass":bool(accuracy), "physics_benefit_pass":benefit,
            "registered_comparison_pass":bool(accuracy and benefit), "paired_comparisons":pairs}


def fieldnames():
    base = ["panel", "regime", "case_id", "seed", "arm", "NPR", "gamma", "R", "finite", "positive", "accuracy_pass"]
    base += [f"{name}_relative_{stat}" for name in FIELDS for stat in ("max", "rms")]
    base += [f"residual_{name}_{stat}" for name in RESIDUALS for stat in ("max", "rms")]
    return base + ["mass_flow_relative_max", "enthalpy_relative_max", "total_pressure_relative_max", "eos_max_abs",
                   "branch_pass", "exit_pressure_relative", "throat_mach_absolute", "NPR_invariance_relative"]


def score_panels(reg, panels, networks, reference, out, assert_current, *, final_test):
    """Predictions and references are produced once in bounded deterministic batches."""
    import numpy as np
    import torch
    xs = np.linspace(-1, 1, 161)
    aggregates, invariance = {}, {}
    scores_path = out / ("test_scores.csv" if final_test else "validation_scores.csv")
    prediction_path = out / "test_predictions.csv"
    with scores_path.open("x", newline="") as scores:
        writer = csv.DictWriter(scores, fieldnames=fieldnames())
        writer.writeheader()
        predictions = prediction_path.open("x", newline="") if final_test else None
        try:
            if predictions is not None:
                pfields = ["panel", "regime", "case_id", "seed", "arm", "NPR", "gamma", "R", "xi"]
                pfields += [f"{prefix}_{name}_hat" for prefix in ("prediction", "exact") for name in FIELDS]
                pwrite = csv.writer(predictions)
                pwrite.writerow(pfields)
            for panel, cases in panels.items():
                for offset in range(0, len(cases), 32):
                    assert_current()
                    batch = cases[offset:offset+32]
                    exact = np.stack([reference.profile(case, xs)[0] for case in batch])
                    for (seed, arm), network in networks.items():
                        raw = model.features(batch, xs, requires_grad=True)
                        predicted = model.evaluate(network, raw)
                        residual = model.residuals(predicted, raw)
                        predicted = predicted.detach().numpy().reshape(len(batch), 161, 4)
                        residual = residual.detach().numpy().reshape(len(batch), 161, 4)
                        for index, case in enumerate(batch):
                            row, squared = metrics(case, xs, predicted[index], exact[index], residual[index], reg["acceptance"])
                            prefix = {"panel":panel, "regime":case["regime"], "case_id":case["case_id"], "seed":seed, "arm":arm,
                                      "NPR":case["NPR"], "gamma":case["gamma"], "R":case["R"]}
                            score_row = {**prefix, **row}
                            key = (panel, case["regime"], seed, arm)
                            aggregate = aggregates.setdefault(key, {"cases":0, "points":0, "squared":0.0, "finite":True, "accuracy_pass":True})
                            aggregate["cases"] += 1
                            aggregate["points"] += 161*4
                            aggregate["finite"] = aggregate["finite"] and squared is not None
                            if squared is not None:
                                aggregate["squared"] += squared
                            aggregate["accuracy_pass"] = aggregate["accuracy_pass"] and row["accuracy_pass"]
                            if panel == "product_test" and case["regime"] == "smooth_choked":
                                ikey = (case["source"], case["source_id"], seed, arm)
                                stored = invariance.get(ikey)
                                current = predicted[index]
                                if stored is None:
                                    invariance[ikey] = {"lo":current.copy(), "hi":current.copy(), "nprs":[case["NPR"]], "rows":[score_row]}
                                else:
                                    stored["lo"] = np.minimum(stored["lo"], current)
                                    stored["hi"] = np.maximum(stored["hi"], current)
                                    stored["nprs"].append(case["NPR"])
                                    stored["rows"].append(score_row)
                                    if len(stored["nprs"]) == 3:
                                        if stored["nprs"] != reg["splits"]["product_test"]["NPR_choked"]:
                                            raise ValueError("incomplete or reordered choked NPR triplet")
                                        spread = (stored["hi"]-stored["lo"])/exact[index]
                                        value = float(np.max(spread)) if np.isfinite(spread).all() else None
                                        row["NPR_invariance_relative"] = value
                                        invariant_ok = value is not None and value <= reg["acceptance"]["choked_NPR_invariance_relative_max"]
                                        aggregate["accuracy_pass"] = bool(aggregate["accuracy_pass"] and invariant_ok)
                                        for record in stored["rows"]:
                                            record["NPR_invariance_relative"] = value
                                            record["accuracy_pass"] = bool(record["accuracy_pass"] and invariant_ok)
                                        writer.writerows(stored["rows"])
                                        del invariance[ikey]
                            else:
                                writer.writerow(score_row)
                            if predictions is not None:
                                values = [prefix[name] for name in pfields[:8]]
                                for j, x in enumerate(xs):
                                    pwrite.writerow(values+[float(x)]+predicted[index,j].tolist()+exact[index,j].tolist())
                        del raw, residual, predicted
                    scores.flush()
                    if predictions is not None:
                        predictions.flush()
            if invariance:
                raise ValueError("incomplete choked NPR invariance coverage")
        finally:
            if predictions is not None:
                predictions.close()
    for values in aggregates.values():
        values["relative_rmse"] = math.sqrt(values.pop("squared")/values["points"]) if values["finite"] else None
    return aggregates
