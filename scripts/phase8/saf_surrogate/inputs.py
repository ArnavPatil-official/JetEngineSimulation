"""Input-only readers, deterministic query designs and v6 public state algebra."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .registration import json_bytes, sha256_file

FUELS = ("JetA", "HEFA", "FT", "ATJ")
FEATURES = ("f_HEFA", "f_FT", "f_ATJ", "thrust_fraction", "combustor_pressure_loss",
            "eta_compressor", "eta_turbine_polytropic", "fpr_rated", "eta_fan",
            "eta_b_IDLE", "eta_b_APPROACH", "eta_b_TAKE-OFF")
MODES = (("TAKE-OFF", 1.0), ("APPROACH", .30), ("IDLE", .07), ("CLIMB85", .85))


def finite_positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Public engine inputs must be finite and positive")
    return value


def load_public_inputs(root, reg):
    """Whitelisted columns; no fuel-flow or emission target is decoded."""
    path = Path(root) / "data/icao_engine_data.csv"
    expected = reg["provenance"]["historical_metadata_input_sha256"][str(path.relative_to(root))]
    if sha256_file(path) != expected:
        raise ValueError("Public input edition drift")
    allowed = reg["inputs"]["ae3_columns"]
    rows = []
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {key: header.index(key) for key in allowed}
        uid_index = indices["Unique ID"]
        for raw in reader:
            if raw[uid_index] != reg["scope"]["engine_uid"]:
                continue
            rows.append({key: raw[index] for key, index in indices.items()})
    if not rows:
        raise ValueError("AE3 public input record missing")
    tuples = {(finite_positive(row["Pressure Ratio"]), finite_positive(row["Bypass Ratio"]),
               finite_positive(row["Rated Thrust (kN)"])) for row in rows}
    if len(tuples) != 1:
        raise ValueError("AE3 public design inputs disagree across modes")
    opr, bpr, rated = tuples.pop()
    return {"uid": reg["scope"]["engine_uid"], "opr": opr, "bpr": bpr,
            "rated_kN": rated, "source_sha256": expected, "fit": reg["scope"]["fit_parameters"]}


def load_fixed_draws(root, reg):
    path = Path(root) / reg["inputs"]["draw_file"]
    if sha256_file(path) != reg["provenance"]["historical_metadata_input_sha256"][reg["inputs"]["draw_file"]]:
        raise ValueError("Fixed-draw source drift")
    allowed = reg["inputs"]["draw_columns"]
    expected = set(reg["fuel"]["draw_ids"])
    output = {}
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {key: header.index(key) for key in allowed}
        for raw in reader:
            key = raw[indices["case"]]
            if key not in expected:
                continue
            if key in output:
                raise ValueError("Duplicate fixed draw")
            output[key] = {column.removeprefix("fixed_"): finite_positive(raw[index])
                           for column, index in indices.items() if column != "case"}
    if set(output) != expected:
        raise ValueError("The exact 64 fixed draws are required")
    central = reg["scope"]["fixed_central"]
    output["central"] = {key: central[key] for key in (
        "combustor_pressure_loss", "eta_compressor", "eta_turbine_polytropic", "fpr_rated", "eta_fan")}
    output["central"].update({f"eta_b_{mode}": central["eta_b"][mode]
                              for mode in ("IDLE", "APPROACH", "TAKE-OFF")})
    for fixed in output.values():
        if not 0 < fixed["combustor_pressure_loss"] < 1 or fixed["fpr_rated"] <= 1:
            raise ValueError("Invalid fixed pressure loss or fan pressure ratio")
        if any(not 0 < fixed[key] <= 1 for key in fixed if key.startswith("eta_")):
            raise ValueError("Fixed efficiencies must be in (0,1]")
    return output


def canonical_query(query, draws, public):
    fractions = [float(query[f"f_{fuel}"]) for fuel in FUELS]
    if any(not math.isfinite(f) or f < 0 for f in fractions):
        raise ValueError("Fractions must be finite and nonnegative")
    total = math.fsum(fractions)
    if abs(total - 1) > 1e-12:
        raise ValueError("Mass fractions must sum to one")
    x = float(query["thrust_fraction"])
    if not math.isfinite(x) or not .07 <= x <= 1:
        raise ValueError("Thrust fraction outside registered domain")
    draw = query["draw_id"]
    if draw not in draws:
        raise ValueError("Unregistered fixed draw")
    result = {f"f_{fuel}": f / total for fuel, f in zip(FUELS, fractions)}
    result.update(thrust_fraction=x, draw_id=draw)
    identity = {key: (value.hex() if isinstance(value, float) else value)
                for key, value in result.items()}
    identity["fixed"] = {key: float(value).hex() for key, value in draws[draw].items()}
    identity["public"] = public
    result["input_sha256"] = hashlib.sha256(json_bytes(identity)).hexdigest()
    return result


def state_for(query, public, draws):
    q = canonical_query(query, draws, public)
    x, fixed, fit = q["thrust_fraction"], draws[q["draw_id"]], public["fit"]
    knots = (.07, .30, .85, 1.0)
    values = (fixed["eta_b_IDLE"], fixed["eta_b_APPROACH"],
              fixed["eta_b_TAKE-OFF"], fixed["eta_b_TAKE-OFF"])
    for i in range(3):
        if x <= knots[i + 1]:
            eta_b = values[i] + (values[i + 1] - values[i]) * (x - knots[i]) / (knots[i + 1] - knots[i])
            break
    m_rated = fit["W_ref"] * (public["rated_kN"] / 310.9) ** fit["a_thrust"]
    return {"ma": m_rated * x ** fit["k_mdot"], "pi_c": 1 + (public["opr"] - 1) * x ** fit["k_pi"],
            "fpr": 1 + (fixed["fpr_rated"] - 1) * x ** fit["k_pi"],
            "target_kN": public["rated_kN"] * x, "bpr": public["bpr"], "eta_b": eta_b,
            "eta_c": fixed["eta_compressor"], "eta_t": fixed["eta_turbine_polytropic"],
            "eta_fan": fixed["eta_fan"], "pressure_loss": fixed["combustor_pressure_loss"]}


def feature_rows(queries, draws, public):
    import numpy as np
    rows = []
    for query in queries:
        q = canonical_query(query, draws, public)
        rows.append([q.get(key, draws[q["draw_id"]].get(key)) for key in FEATURES])
    return np.asarray(rows, dtype=np.float64)


def simplex(u):
    values = sorted(float(v) for v in u[:3])
    return (values[0], values[1] - values[0], values[2] - values[1], 1 - values[2])


def query_designs(reg, draws, public):
    from scipy.stats import qmc
    output = {}
    definitions = (("train", reg["sampling"]["train_seed"], 4096),
                   ("validation", reg["sampling"]["validation"]["seed"], 1024),
                   ("test", reg["sampling"]["test"]["seed"], 2048),
                   ("physics", reg["sampling"]["physics_collocation"]["seed"], 2048))
    for name, seed, count in definitions:
        points = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(int(math.log2(count)))
        rows = []
        for i, point in enumerate(points):
            query = {f"f_{fuel}": value for fuel, value in zip(FUELS, simplex(point))}
            query.update(thrust_fraction=.07 + .93 * float(point[3]),
                         draw_id=f"draw_{int(64 * point[4]):02d}")
            query = canonical_query(query, draws, public)
            query.update(split=name, design_id=f"{name}_{i:06d}", prefix_index=i)
            rows.append(query)
        output[name] = rows
    for size in reg["sampling"]["train_sizes"]:
        counts = {draw: 0 for draw in reg["fuel"]["draw_ids"]}
        for row in output["train"][:size]:
            counts[row["draw_id"]] += 1
        if set(counts.values()) != {size // 64}:
            raise ValueError("Sobol prefix does not balance every fixed draw")
    for name, seed, count, power in (("ranking_test", reg["sampling"]["ranking_test"]["seed"], 64, 6),
                                     ("study", reg["sampling"]["screening_study"]["seed"], 10000, 14)):
        points = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(power)[:count]
        rows = []
        for i, point in enumerate(points):
            for draw in reg["fuel"]["draw_ids"]:
                query = {f"f_{fuel}": value for fuel, value in zip(FUELS, simplex(point))}
                query.update(thrust_fraction=1.0, draw_id=draw)
                query = canonical_query(query, draws, public)
                query.update(split=name, design_id=f"{name}_{i:06d}", prefix_index=i)
                rows.append(query)
        output[name] = rows
    seen = set()
    for name, rows in output.items():
        for row in rows:
            if row["input_sha256"] in seen:
                raise ValueError(f"Duplicate canonical query in {name}")
            seen.add(row["input_sha256"])
    return output


def named_queries(draws, public):
    fuels = {"JetA": {"JetA_dooley2012": 1.0}}
    mapping = {"HEFA": "HEFA_liang2025", "FT": "FT_dooley2012", "ATJ": "ATJ_C1matched"}
    for pathway in ("HEFA", "FT", "ATJ"):
        for percent in (10, 20, 30, 50, 100):
            fuels[f"{pathway}-{percent}"] = {"JetA_dooley2012": 1 - percent / 100,
                                            mapping[pathway]: percent / 100}
    fuels["JetA_dooley2010"] = {"JetA_dooley2010": 1.0}
    rows = []
    for fuel in sorted(fuels):
        for op, x in MODES:
            in_api = fuel != "JetA_dooley2010"
            row = {"named_case_id": f"{fuel}|{op}", "fuel": fuel, "op": op,
                   "thrust_fraction": x, "draw_id": "central", "in_product_API": in_api,
                   "fuel_parts": {k: v for k, v in fuels[fuel].items() if v > 0}}
            if in_api:
                row.update({f"f_{name}": 0.0 for name in FUELS})
                row["f_JetA"] = fuels[fuel].get("JetA_dooley2012", 0.0)
                if fuel != "JetA":
                    row[f"f_{fuel.split('-')[0]}"] = int(fuel.split('-')[1]) / 100
                row.update(canonical_query(row, draws, public))
            else:
                row.update({f"f_{name}": None for name in FUELS})
                row["input_sha256"] = hashlib.sha256(json_bytes({"named_case_id": row["named_case_id"],
                    "fuel_parts": row["fuel_parts"], "thrust_fraction": x.hex(),
                    "fixed": draws["central"], "public": public})).hexdigest()
            rows.append(row)
    if len(rows) != 68:
        raise ValueError("Exact named central property count violated")
    return rows
