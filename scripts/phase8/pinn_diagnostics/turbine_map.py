"""Track 4a: four-input turbine pressure-ratio map, CPU float64.

Target (registered): the analytic work-consistent polytropic expansion that
production uses,

    log(p5/p4) = gamma / (eta_p (gamma - 1)) * log1p(-tau),

over the (tau, gamma) box of ``outputs/turbine_envelope_v5.csv``. The network
receives the four requested features [tau, gamma, eta_p, cp/R], but eta_p is
fixed and cp/R = gamma/(gamma - 1) is derived from gamma, so this envelope has
only **two varying independent dimensions**. The eta feature is constant
(eta/reference - 1 = 0) and carries no information here.

Training loss: mean squared error of the predicted log pressure ratio against
the registered target (the registration fixes the target, budgets and
optimizers; MSE in the target variable is the plain regression loss). Stopping
uses training quantities only (step budgets and the L-BFGS tolerances); the
score grid is evaluated exactly once, after training.

Score: max |expm1(pred_log - exact_log)| = max |p5_pred/p5_exact - 1| over the
65x65 grid, gate strictly below 0.001. The relative error of a pressure ratio
is invariant to the inlet pressure scale p4.

Diagnostic of representability only: not a turbine model, not a PINN (no
physics residual), and not a validation of anything against data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

DTYPE = torch.float64

TARGET_EXPR = "gamma/(eta_p*(gamma-1))*log1p(-tau)"
CP_OVER_R_EXPR = "gamma/(gamma-1)"
METRIC_EXPR = "max(abs(expm1(pred_log-exact_log)))"
FEATURES = ["tau", "gamma", "eta_p", "cp_over_R"]


# ---------------------------------------------------------------------------
# Analytic target and domain
# ---------------------------------------------------------------------------

def _check_domain(tau, gamma, eta_p) -> None:
    tau, gamma = np.asarray(tau, dtype=float), np.asarray(gamma, dtype=float)
    if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(gamma)) and math.isfinite(eta_p)):
        raise ValueError("non-finite turbine input")
    if np.any(tau <= 0.0) or np.any(tau >= 1.0):
        raise ValueError("work fraction tau must lie in (0, 1)")
    if np.any(gamma <= 1.0):
        raise ValueError("gamma must exceed 1")
    if not 0.0 < eta_p <= 1.0:
        raise ValueError("polytropic efficiency must lie in (0, 1]")


def cp_over_r(gamma):
    """cp/R of a calorically perfect gas, derived from gamma."""
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma <= 1.0):
        raise ValueError("gamma must exceed 1")
    return gamma / (gamma - 1.0)


def polytropic_log_pressure_ratio(tau, gamma, eta_p):
    """log(p5/p4) of the work-consistent polytropic expansion."""
    _check_domain(tau, gamma, eta_p)
    tau, gamma = np.asarray(tau, dtype=float), np.asarray(gamma, dtype=float)
    return gamma / (eta_p * (gamma - 1.0)) * np.log1p(-tau)


@dataclass(frozen=True)
class TurbineBox:
    tau: tuple
    gamma: tuple
    eta_p: float
    eta_reference: float

    @classmethod
    def from_registration(cls, turbine: dict) -> "TurbineBox":
        if turbine["target"] != TARGET_EXPR or turbine["cp_over_R"] != CP_OVER_R_EXPR \
                or turbine["metric"] != METRIC_EXPR or turbine["features"] != FEATURES:
            raise ValueError("registration target/metric/features differ from the implemented ones")
        m = re.search(r"reference\s*([0-9.]+)", turbine["feature_scaling"])
        if m is None:
            raise ValueError("eta reference not found in registered feature_scaling")
        box = cls(tuple(turbine["tau"]), tuple(turbine["gamma"]), float(turbine["eta_p"]), float(m.group(1)))
        _check_domain(np.array(box.tau), np.array(box.gamma), box.eta_p)
        if not (box.tau[0] < box.tau[1] and box.gamma[0] < box.gamma[1]):
            raise ValueError("empty turbine box")
        return box

    @property
    def cp_over_r_bounds(self) -> tuple:
        # cp/R = gamma/(gamma - 1) decreases with gamma
        return (float(cp_over_r(self.gamma[1])), float(cp_over_r(self.gamma[0])))

    def check_inside(self, tau, gamma) -> None:
        tau, gamma = np.asarray(tau, dtype=float), np.asarray(gamma, dtype=float)
        if np.any(tau < self.tau[0]) or np.any(tau > self.tau[1]) \
                or np.any(gamma < self.gamma[0]) or np.any(gamma > self.gamma[1]):
            raise ValueError("input outside the registered turbine box")


def _to_unit_interval(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


def features(tau, gamma, box: TurbineBox) -> torch.Tensor:
    """(N, 4) float64 features [tau, gamma, eta_p, cp/R], scaled as registered."""
    box.check_inside(tau, gamma)
    _check_domain(tau, gamma, box.eta_p)
    tau, gamma = np.asarray(tau, dtype=float), np.asarray(gamma, dtype=float)
    lo, hi = box.cp_over_r_bounds
    cols = [
        _to_unit_interval(tau, *box.tau),
        _to_unit_interval(gamma, *box.gamma),
        np.full_like(tau, box.eta_p / box.eta_reference - 1.0),
        _to_unit_interval(cp_over_r(gamma), lo, hi),
    ]
    return torch.as_tensor(np.stack(cols, axis=1), dtype=DTYPE)


# ---------------------------------------------------------------------------
# Inputs: training samples and the score grid
# ---------------------------------------------------------------------------

def sobol_base2_exponent(n: int) -> int:
    """m with 2**m == n; the registered draw is ``draw_base2`` (powers of two only)."""
    n = int(n)
    m = n.bit_length() - 1
    if n <= 0 or (1 << m) != n:
        raise ValueError(f"Sobol sample count {n} is not a power of two (draw_base2)")
    return m


def sobol_implementation(n: int, seed: int) -> str:
    """The registered Sobol call written out for these arguments."""
    return (f"torch.quasirandom.SobolEngine(dimension=2,scramble=True,seed={int(seed)})"
            f".draw_base2(m={sobol_base2_exponent(n)},dtype=torch.float64)")


def sobol_samples(n: int, seed: int, box: TurbineBox) -> tuple:
    """Scrambled 2D Sobol points (registered ``draw_base2``) mapped into the (tau, gamma) box."""
    m = sobol_base2_exponent(n)
    u = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed).draw_base2(m, dtype=DTYPE).numpy()
    tau = box.tau[0] + u[:, 0] * (box.tau[1] - box.tau[0])
    gamma = box.gamma[0] + u[:, 1] * (box.gamma[1] - box.gamma[0])
    return tau, gamma


def score_grid(box: TurbineBox, n_per_axis: int) -> tuple:
    """Tensor-product grid, endpoints included (n_per_axis**2 points)."""
    t = np.linspace(box.tau[0], box.tau[1], n_per_axis)
    g = np.linspace(box.gamma[0], box.gamma[1], n_per_axis)
    tt, gg = np.meshgrid(t, g, indexing="ij")
    return tt.ravel(), gg.ravel()


def count_overlap(tau_tr, gamma_tr, tau_s, gamma_s) -> int:
    """Number of score-grid inputs that are exactly a training input."""
    train_pts = set(zip(np.asarray(tau_tr).tolist(), np.asarray(gamma_tr).tolist()))
    return sum((t, g) in train_pts for t, g in zip(np.asarray(tau_s).tolist(), np.asarray(gamma_s).tolist()))


# ---------------------------------------------------------------------------
# Network, training and scoring
# ---------------------------------------------------------------------------

def build_mlp(hidden: list, activation: str, seed: int) -> nn.Sequential:
    acts = {"tanh": nn.Tanh, "silu": nn.SiLU}
    if activation not in acts:
        raise ValueError(f"unsupported activation {activation!r}")
    torch.manual_seed(seed)
    layers, width = [], len(FEATURES)
    for h in hidden:
        layers += [nn.Linear(width, h, dtype=DTYPE), acts[activation]()]
        width = h
    layers.append(nn.Linear(width, 1, dtype=DTYPE))
    return nn.Sequential(*layers)


def relative_errors(pred_log, exact_log) -> np.ndarray:
    """|p_pred/p_exact - 1| computed from log pressure ratios (exact, no cancellation)."""
    return np.abs(np.expm1(np.asarray(pred_log, dtype=float) - np.asarray(exact_log, dtype=float)))


def relative_errors_from_pressures(p_pred, p_exact) -> np.ndarray:
    return np.abs(np.asarray(p_pred, dtype=float) / np.asarray(p_exact, dtype=float) - 1.0)


def gate(errors, threshold: float) -> dict:
    """Strict max gate: PASS only if every error is strictly below the threshold."""
    e = np.asarray(errors, dtype=float)
    if e.size == 0 or not np.all(np.isfinite(e)):
        raise ValueError("gate needs finite errors")
    return {"max": float(e.max()), "mean": float(e.mean()), "threshold": float(threshold),
            "pass": bool(e.max() < threshold)}


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, cfg: dict, log=print) -> dict:
    """Adam then one L-BFGS call, budgets from the registration; training data only."""
    if X.dtype != DTYPE or y.dtype != DTYPE:
        raise TypeError("training tensors must be float64")
    loss_fn = lambda: ((model(X) - y) ** 2).mean()  # noqa: E731
    adam = torch.optim.Adam(model.parameters(), lr=cfg["adam_lr"], betas=tuple(cfg["adam_betas"]),
                            eps=cfg["adam_eps"], weight_decay=cfg["adam_weight_decay"])
    hist = {"adam": []}
    for step in range(cfg["adam_steps"]):
        adam.zero_grad()
        loss = loss_fn()
        loss.backward()
        adam.step()
        if step % 250 == 0 or step == cfg["adam_steps"] - 1:
            hist["adam"].append([step, float(loss.item())])
            log(f"adam step {step} train_mse {loss.item():.6e}")
    adam_final = float(loss_fn().item())

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=cfg["lbfgs_lr"], max_iter=cfg["lbfgs_max_iter"],
                              max_eval=cfg["lbfgs_max_eval"], tolerance_grad=cfg["lbfgs_tolerance_grad"],
                              tolerance_change=cfg["lbfgs_tolerance_change"],
                              history_size=cfg["lbfgs_history_size"], line_search_fn=cfg["lbfgs_line_search"])

    def closure():
        lbfgs.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    lbfgs.step(closure)
    state = lbfgs.state[lbfgs._params[0]]
    final = float(loss_fn().item())
    log(f"lbfgs n_iter {state.get('n_iter')} func_evals {state.get('func_evals')} train_mse {final:.6e}")
    return {"adam_checkpoints": hist["adam"], "train_mse_after_adam": adam_final,
            "lbfgs_n_iter": int(state.get("n_iter", 0)), "lbfgs_func_evals": int(state.get("func_evals", 0)),
            "train_mse_final": final, "train_rmse_log_final": math.sqrt(final)}


def predict_log(model: nn.Module, tau, gamma, box: TurbineBox) -> np.ndarray:
    with torch.no_grad():
        return model(features(tau, gamma, box)).numpy().ravel()


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    """Write-once checkpoint (refuses an existing file)."""
    with open(path, "xb") as fh:
        torch.save({"model_state_dict": model.state_dict(), "meta": meta}, fh)


def load_checkpoint(path: Path, hidden: list, activation: str) -> tuple:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_mlp(hidden, activation, seed=0)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["meta"]


def run(reg: dict, out_dir: Path, log=print, identity: dict | None = None) -> dict:
    """Train once, save the checkpoint, score the grid once. Returns the result record.

    ``identity`` is the runner's start identity (HEAD, registration and source
    hashes, envelope); it is stored unchanged in the checkpoint metadata.
    """
    cfg = reg["turbine"]
    box = TurbineBox.from_registration(cfg)
    seed = int(reg["seed"])
    if cfg["sobol_implementation"] != sobol_implementation(cfg["training_samples"], seed):
        raise ValueError("registered Sobol implementation differs from the implemented draw_base2 call")
    tau_tr, gamma_tr = sobol_samples(cfg["training_samples"], seed, box)
    tau_s, gamma_s = score_grid(box, cfg["score_grid_points_per_axis"])
    # a contaminated grid is refused before any training or scoring
    n_overlap = count_overlap(tau_tr, gamma_tr, tau_s, gamma_s)
    if n_overlap:
        raise ValueError(f"{n_overlap} score-grid inputs equal training inputs; not training or scoring")
    X = features(tau_tr, gamma_tr, box)
    y = torch.as_tensor(polytropic_log_pressure_ratio(tau_tr, gamma_tr, box.eta_p), dtype=DTYPE).view(-1, 1)
    model = build_mlp(cfg["hidden"], cfg["activation"], seed)
    log(f"turbine: {cfg['training_samples']} Sobol samples, MLP {cfg['hidden']} {cfg['activation']}, seed {seed}")
    hist = train(model, X, y, cfg, log)
    ckpt_path = out_dir / "turbine_checkpoint.pt"
    save_checkpoint(ckpt_path, model, {"registration_id": reg["id"], "seed": seed, "features": FEATURES,
                                       "hidden": cfg["hidden"], "activation": cfg["activation"],
                                       "dtype": "float64", "train_history": hist, "start_identity": identity})

    # the single scoring pass
    exact = polytropic_log_pressure_ratio(tau_s, gamma_s, box.eta_p)
    pred = predict_log(model, tau_s, gamma_s, box)
    err = relative_errors(pred, exact)
    g = gate(err, cfg["pass_strictly_below"])
    k = int(np.argmax(err))
    result = {
        "status": "PASS" if g["pass"] else "FAIL",
        "metric": METRIC_EXPR,
        "max_relative_pressure_error": g["max"],
        "mean_relative_pressure_error": g["mean"],
        "threshold_strictly_below": g["threshold"],
        "worst": {"tau": float(tau_s[k]), "gamma": float(gamma_s[k]), "exact_log": float(exact[k]),
                  "pred_log": float(pred[k])},
        "grid_points": int(err.size),
        "grid_training_overlap": int(n_overlap),
        "train": hist,
        "historical_retired_max_relative_pct": cfg["historic_retired_max_relative_pct"],
        "independent_varying_dimensions": 2,
        "checkpoint": ckpt_path.name,
    }
    log(f"turbine score (once): max rel {g['max']:.6e}, mean {g['mean']:.6e} -> {result['status']}")
    return result
