"""Matched CPU64 model and explicit dimensionless quasi-1D residuals."""
from __future__ import annotations

import copy
import math


def build(reg, seed):
    import torch
    torch.manual_seed(seed)
    layers, width = [], 4
    for hidden in reg["models"]["architecture"]["hidden"]:
        layers.extend([torch.nn.Linear(width, hidden, dtype=torch.float64), torch.nn.Tanh()])
        width = hidden
    layers.append(torch.nn.Linear(width, 4, dtype=torch.float64))
    return torch.nn.Sequential(*layers)


def features(cases, xs, *, requires_grad=False):
    import torch
    rows = [[x, case["NPR"], case["gamma"], case["R"]] for case in cases for x in xs]
    return torch.tensor(rows, dtype=torch.float64, device="cpu", requires_grad=requires_grad)


def evaluate(network, raw):
    import torch
    normalized = torch.stack((raw[:, 0],
        2 * (torch.log(raw[:, 1]) - math.log(1.02)) / (math.log(12) - math.log(1.02)) - 1,
        2 * (raw[:, 2] - 1.2) / .2 - 1, 2 * (raw[:, 3] - 260) / 70 - 1), dim=1)
    return torch.nn.functional.softplus(network(normalized)) + 1e-12


def residuals(output, raw):
    import torch
    rho, u, temperature, pressure = output.unbind(dim=1)
    ratio_R = raw[:, 3] / 287
    cp = raw[:, 2] * ratio_R / (raw[:, 2] - 1)
    area = 1 + .5 * raw[:, 0] ** 2
    def derivative(value):
        return torch.autograd.grad(value, raw, torch.ones_like(value), create_graph=True)[0][:, 0]
    return torch.stack((derivative(rho * u * area), rho * u * derivative(u) + derivative(pressure),
                        cp * derivative(temperature) + u * derivative(u),
                        pressure - ratio_R * rho * temperature), dim=1)


def train_pair(reg, seed, cases, reference, out, assert_current, checkpoint_identity):
    import torch
    cfg = reg["models"]["optimizer"]
    interior = reg["splits"]["training"]["interior_label_x"]
    boundary = reg["splits"]["training"]["boundary_label_x"]
    collocation = [float(x) for x in torch.linspace(-1, 1, 65, dtype=torch.float64)]
    data_input, boundary_input = features(cases, interior), features(cases, boundary)
    data_target = torch.tensor([q for case in cases for q in reference.profile(case, interior)[0]], dtype=torch.float64)
    boundary_target = torch.tensor([q for case in cases for q in reference.profile(case, boundary)[0]], dtype=torch.float64)
    initial = copy.deepcopy(build(reg, seed).state_dict())
    results = {}
    for arm, weights in reg["models"]["arms"].items():
        assert_current()
        network = build(reg, seed)
        network.load_state_dict(initial)
        collocation_input = features(cases, collocation, requires_grad=True)
        path = out / "training_logs" / f"{arm}-seed{seed}.jsonl"
        counts = {"adam_steps": 0, "lbfgs_evaluations": 0}
        def loss():
            data = (evaluate(network, data_input) - data_target).square().mean()
            bc = (evaluate(network, boundary_input) - boundary_target).square().mean()
            phys = (residuals(evaluate(network, collocation_input), collocation_input).square().mean()
                    if weights["lambda_physics"] else data.new_zeros(()))
            total = weights["lambda_data"] * data + weights["lambda_boundary"] * bc + weights["lambda_physics"] * phys
            if not torch.isfinite(total):
                raise ValueError("nonfinite training loss")
            return total, [float(t.detach()) for t in (data, bc, phys)]
        with path.open("x") as log:
            import json
            adam = torch.optim.Adam(network.parameters(), lr=cfg["adam_lr"], betas=tuple(cfg["adam_betas"]),
                                    eps=cfg["adam_eps"], weight_decay=cfg["adam_weight_decay"])
            for step in range(cfg["adam_steps"]):
                if step % 25 == 0:
                    assert_current()
                adam.zero_grad(set_to_none=True)
                collocation_input.grad = None
                value, parts = loss()
                value.backward()
                adam.step()
                counts["adam_steps"] += 1
                if step % 25 == 0 or step + 1 == cfg["adam_steps"]:
                    log.write(json.dumps({"stage": "adam", "step": step + 1, "loss": float(value.detach()), "components": parts}) + "\n")
                    log.flush()
            lbfgs = torch.optim.LBFGS(network.parameters(), lr=cfg["lbfgs_lr"], max_iter=cfg["lbfgs_max_iter"],
                 max_eval=cfg["lbfgs_max_eval"], history_size=cfg["lbfgs_history_size"],
                 line_search_fn=cfg["lbfgs_line_search"], tolerance_grad=cfg["lbfgs_tolerance_grad"],
                 tolerance_change=cfg["lbfgs_tolerance_change"])
            def closure():
                if counts["lbfgs_evaluations"] % 25 == 0:
                    assert_current()
                lbfgs.zero_grad(set_to_none=True)
                collocation_input.grad = None
                value, parts = loss()
                value.backward()
                counts["lbfgs_evaluations"] += 1
                log.write(json.dumps({"stage": "lbfgs", "evaluation": counts["lbfgs_evaluations"], "loss": float(value.detach()), "components": parts}) + "\n")
                log.flush()
                return value
            lbfgs.step(closure)
        assert_current()
        checkpoint = out / "checkpoints" / f"{arm}-seed{seed}.pt"
        with checkpoint.open("xb") as stream:
            torch.save({"state_dict": network.state_dict(), "adam": adam.state_dict(), "lbfgs": lbfgs.state_dict(),
                        "seed": seed, "arm": arm, "counts": counts, "identity": checkpoint_identity}, stream)
        results[arm] = network
    return results
