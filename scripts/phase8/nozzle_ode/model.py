"""Shared-backend smooth-nozzle models and neutral CPU64 scoring algebra."""
from __future__ import annotations

import json
import math


def selected_backend(name=None):
    from simulation.ml_backend import get_backend
    return get_backend(name, device="cpu")


def build(reg, seed, backend=None):
    selected = selected_backend(backend)
    return selected.mlp(4, tuple(reg["models"]["architecture"]["hidden"]), 4,
                        activation="tanh", seed=seed)


def features(cases, xs, *, requires_grad=False, backend=None):
    selected = selected_backend(backend)
    value = selected.array([[x, case["NPR"], case["gamma"], case["R"]] for case in cases for x in xs])
    if requires_grad and selected.name == "torch":
        value.requires_grad_(True)
    return value


def evaluate(network, raw, backend=None):
    selected = selected_backend(backend)
    xp = selected.ops
    normalized = xp.stack((raw[:, 0],
        2*(xp.log(raw[:, 1])-math.log(1.02))/(math.log(12)-math.log(1.02))-1,
        2*(raw[:, 2]-1.2)/.2-1, 2*(raw[:, 3]-260)/70-1), axis=1)
    return xp.softplus(selected.forward(network, normalized))+1e-12


def residuals(output, raw, backend=None):
    """Function-based gradients retain the parameter graph on both backends."""
    selected = selected_backend(backend)
    xp = selected.ops
    if callable(output):
        field = output
        values = field(raw)
        def derivative(fn):
            return selected.input_grad(fn, raw)[:, 0]
        mass_gradient = derivative(lambda x: field(x)[:, 0]*field(x)[:, 1]*(1+.5*x[:, 0]**2))
        du = derivative(lambda x: field(x)[:, 1])
        dp = derivative(lambda x: field(x)[:, 3])
        dt = derivative(lambda x: field(x)[:, 2])
    else:
        # Existing Torch manufactured residual fixtures may pass a graph tensor.
        if selected.name != "torch":
            raise TypeError("MLX residuals require a function of the input coordinates")
        import torch
        values = output
        def derivative(value):
            return torch.autograd.grad(value, raw, torch.ones_like(value), create_graph=True)[0][:, 0]
        mass_gradient = derivative(values[:, 0]*values[:, 1]*(1+.5*raw[:, 0]**2))
        du, dp, dt = (derivative(values[:, i]) for i in (1, 3, 2))
    rho, u, temperature, pressure = (values[:, i] for i in range(4))
    ratio_R = raw[:, 3]/287
    cp = raw[:, 2]*ratio_R/(raw[:, 2]-1)
    return xp.stack((mass_gradient, rho*u*du+dp, cp*dt+u*du,
                     pressure-ratio_R*rho*temperature), axis=1)


def neutral_parameters(network, backend=None):
    if isinstance(network, dict) and "layers.0.weight" in network:
        return network
    selected = selected_backend(backend)
    return {key:selected.to_numpy(value).copy() for key,value in selected.parameters(network).items()}


def predict_numpy64(params, raw):
    """Analytic MLP/softplus derivative in NumPy64; no scoring-time ML runtime."""
    import numpy as np
    raw = np.asarray(raw, dtype=np.float64)
    value = np.column_stack((raw[:, 0],
        2*(np.log(raw[:, 1])-math.log(1.02))/(math.log(12)-math.log(1.02))-1,
        2*(raw[:, 2]-1.2)/.2-1, 2*(raw[:, 3]-260)/70-1))
    derivative = np.zeros_like(value); derivative[:, 0] = 1
    count = sum(key.endswith(".weight") for key in params)
    for i in range(count):
        weight = np.asarray(params[f"layers.{i}.weight"], dtype=np.float64)
        value = value@weight.T+np.asarray(params[f"layers.{i}.bias"], dtype=np.float64)
        derivative = derivative@weight.T
        if i+1 < count:
            value = np.tanh(value)
            derivative = derivative*(1-value**2)
    positive = value >= 0
    sigmoid = np.empty_like(value)
    sigmoid[positive] = 1/(1+np.exp(-value[positive]))
    exp_value = np.exp(value[~positive]); sigmoid[~positive] = exp_value/(1+exp_value)
    derivative *= sigmoid
    value = np.log1p(np.exp(-np.abs(value)))+np.maximum(value, 0)+1e-12
    rho, u, temperature, pressure = value.T
    drho, du, dt, dp = derivative.T
    area = 1+.5*raw[:, 0]**2
    ratio_R = raw[:, 3]/287
    cp = raw[:, 2]*ratio_R/(raw[:, 2]-1)
    residual = np.column_stack((drho*u*area+rho*du*area+rho*u*raw[:, 0],
        rho*u*du+dp, cp*dt+u*du, pressure-ratio_R*rho*temperature))
    return value, residual


def train_pair(reg, seed, cases, reference, out, assert_current, checkpoint_identity, *, backend=None):
    import numpy as np
    selected = selected_backend(backend)
    xp = selected.ops
    cfg = reg["models"]["optimizer"]
    interior = reg["splits"]["training"]["interior_label_x"]
    boundary = reg["splits"]["training"]["boundary_label_x"]
    collocation = np.linspace(-1, 1, 65)
    data_input = features(cases, interior, backend=selected.name)
    boundary_input = features(cases, boundary, backend=selected.name)
    data_target = selected.array(np.asarray([q for case in cases for q in reference.profile(case, interior)[0]]))
    boundary_target = selected.array(np.asarray([q for case in cases for q in reference.profile(case, boundary)[0]]))
    initial = neutral_parameters(build(reg, seed, selected.name), selected.name)
    results = {}
    for arm, weights in reg["models"]["arms"].items():
        assert_current()
        network = build(reg, seed, selected.name)
        selected.set_parameters(network, initial)
        collocation_input = features(cases, collocation, requires_grad=True, backend=selected.name)
        path = out/"training_logs"/f"{arm}-seed{seed}.jsonl"
        counts = {"adam_steps":0, "lbfgs_evaluations":0}
        parts = {}
        def loss(current):
            data = xp.mean((evaluate(current, data_input, selected.name)-data_target)**2)
            bc = xp.mean((evaluate(current, boundary_input, selected.name)-boundary_target)**2)
            phys = xp.mean(residuals(lambda x:evaluate(current, x, selected.name), collocation_input,
                                    selected.name)**2) if weights["lambda_physics"] else selected.array(0.0)
            parts.update(data=data, bc=bc, phys=phys)
            return weights["lambda_data"]*data+weights["lambda_boundary"]*bc+weights["lambda_physics"]*phys
        def scalar(value):
            return float(selected.to_numpy(value))
        with path.open("x") as log:
            adam = selected.adam(network, lr=cfg["adam_lr"], betas=tuple(cfg["adam_betas"]),
                                 eps=cfg["adam_eps"], weight_decay=cfg["adam_weight_decay"])
            for step in range(cfg["adam_steps"]):
                if step%25 == 0: assert_current()
                value = selected.step(network, adam, loss)
                if not math.isfinite(scalar(value)):
                    raise ValueError("nonfinite training loss")
                counts["adam_steps"] += 1
                if step%25 == 0 or step+1 == cfg["adam_steps"]:
                    log.write(json.dumps({"stage":"adam", "step":step+1, "loss":scalar(value),
                        "components":[scalar(parts[key]) for key in ("data", "bc", "phys")]})+"\n")
                    log.flush()
            if selected.name == "torch":
                lbfgs = selected.lbfgs(network, lr=cfg["lbfgs_lr"], max_iter=cfg["lbfgs_max_iter"],
                    max_eval=cfg["lbfgs_max_eval"], history_size=cfg["lbfgs_history_size"],
                    line_search_fn=cfg["lbfgs_line_search"], tolerance_grad=cfg["lbfgs_tolerance_grad"],
                    tolerance_change=cfg["lbfgs_tolerance_change"])
                def closure():
                    if counts["lbfgs_evaluations"]%25 == 0: assert_current()
                    lbfgs.zero_grad(set_to_none=True)
                    collocation_input.grad = None
                    value = loss(network)
                    value.backward()
                    counts["lbfgs_evaluations"] += 1
                    log.write(json.dumps({"stage":"lbfgs", "evaluation":counts["lbfgs_evaluations"],
                        "loss":scalar(value), "components":[scalar(parts[key]) for key in ("data", "bc", "phys")]})+"\n")
                    log.flush()
                    return value
                lbfgs.step(closure)
            else:
                log.write(json.dumps({"stage":"lbfgs", "status":"UNAVAILABLE",
                    "reason":"Optional MLX uses the prospectively disclosed Adam-only procedure"})+"\n")
        assert_current()
        params = neutral_parameters(network, selected.name)
        checkpoint = out/"checkpoints"/f"{arm}-seed{seed}.npz"
        metadata = {"seed":seed, "arm":arm, "counts":counts, "identity":checkpoint_identity,
                    "backend":selected.name, "dtype":str(next(iter(params.values())).dtype),
                    "score_dtype":"float64", "optimizer":cfg,
                    "lbfgs":"Torch registered polish" if selected.name == "torch" else "UNAVAILABLE",
                    "primary_scientific_procedure":selected.name == "torch"}
        selected.save_npz(network, checkpoint, metadata=metadata)
        with checkpoint.with_suffix(".json").open("x") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False); stream.write("\n")
        results[arm] = params
    return results
