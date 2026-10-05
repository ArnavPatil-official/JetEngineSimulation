"""Phase 8 Track D3: MLX scaffolding and MLX-vs-PyTorch parity (gate G2 infra).

Runs only inside the catjet-mlx env (envs/mlx/); in the main .venv, which
has no mlx, the whole module is skipped by importorskip.

    ~/miniforge3/envs/catjet-mlx/bin/python -m pytest tests/test_phase8_mlx_parity.py -v

All data here are synthetic (seeded random numbers, a toy nozzle, a toy
analytic function). No project data; no model is trained on project data
before gate G2.
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "phase8"))

from ml import parity  # noqa: E402
from ml.models_mlx import (Ensemble, bootstrap_group_indices, composite_loss, flat_grad_vector,  # noqa: E402
                           make_m1, make_m3, make_m4, member_seeds, physics_terms, per_term_grads)
from ml.models_torch import ma_weights_torch  # noqa: E402
from ml.models_mlx import ma_weights_mx  # noqa: E402
from ml.nondim import Scaler  # noqa: E402
from ml.score64 import (ensemble_predict64, export_params64, forward64, load_params64,  # noqa: E402
                        predict64, save_params64, score64)
from ml.spec import max_rel_diff, m4_spec  # noqa: E402
from ml.train_mlx import load_checkpoint, max_sample_sd, save_checkpoint, train  # noqa: E402
from ml.weighting import (GradNormWeights, ReLoBRaLoWeights, ma_weights)  # noqa: E402

TOL = 1e-5          # registered parity tolerance (max-normalised relative)
GPU_TOL = 1e-4      # GPU forward is reported; loose sanity bound only
SWEEP_TOL = 5e-5    # extra seeds, float32-conditioning bound (see README)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def _restore_device():
    old = mx.default_device()
    yield
    mx.set_default_device(old)


def _report(name, d):
    print(f"\n[{name}] " + ", ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in d.items()))


# --------------------------------------------------------------------------
# environment and layout
# --------------------------------------------------------------------------
def test_pinned_versions():
    import mlx
    assert mx.__version__ == "0.32.3"
    assert torch.__version__.split("+")[0] == "2.9.1"
    assert mlx is not None


def test_mlx_linear_layout_matches_torch():
    mx.set_default_device(mx.cpu)
    lin = nn.Linear(3, 5)
    assert tuple(lin.weight.shape) == (5, 3) == tuple(torch.nn.Linear(3, 5).weight.shape)
    x = mx.array(np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32))
    manual = np.array(x) @ np.array(lin.weight).T + np.array(lin.bias)
    assert np.allclose(np.array(lin(x)), manual, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# M1 / M4 parity: forward, loss, parameter gradients (CPU float32), GPU report
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["M1", "M4"])
def test_mlp_parity(kind):
    r = parity.mlp_parity(kind, seed=0)
    _report(kind, r)
    assert r["forward_cpu"] < TOL
    assert r["loss_cpu"] < TOL
    assert r["grad_cpu"] < TOL
    assert r["forward_gpu_vs_torch_cpu"] < GPU_TOL
    assert r["forward_gpu_vs_mlx_cpu"] < GPU_TOL


@pytest.mark.parametrize("kind", ["M1", "M4"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mlp_parity_other_seeds(kind, seed):
    r = parity.mlp_parity(kind, seed=seed)
    assert max(r["forward_cpu"], r["loss_cpu"], r["grad_cpu"]) < TOL


def test_m1_output_is_bounded():
    m = make_m1(4, 2, bound=0.05, seed=0)
    x = mx.array(100.0 * np.random.default_rng(0).standard_normal((512, 4)).astype(np.float32))
    y = np.array(m(x))
    assert np.all(np.abs(y) <= 0.05 + 1e-7)
    assert [l.weight.shape for l in m.layers] == [(64, 4), (64, 64), (64, 64), (2, 64)]


def test_m4_architecture():
    m = make_m4(7, 3, seed=0)
    assert [tuple(l.weight.shape) for l in m.layers] == [(128, 7), (128, 128), (128, 128), (128, 128), (3, 128)]
    assert m.spec.activation == "silu" and m.spec.output == "linear"


# --------------------------------------------------------------------------
# M3 physics-loss parity (incl. second derivatives)
# --------------------------------------------------------------------------
def test_m3_physics_parity_cpu():
    r = parity.m3_parity(seed=0)
    _report("M3 fields/derivs", {k: r[k] for k in ("fields_cpu", "dF_dx_cpu", "d2F_dx2_cpu",
                                                   "entropy_hinge_active_points")})
    _report("M3 loss", r["loss_cpu"])
    _report("M3 grad", r["grad_cpu"])
    assert r["fields_cpu"] < TOL
    assert r["dF_dx_cpu"] < TOL
    assert r["d2F_dx2_cpu"] < TOL
    active, n = map(int, r["entropy_hinge_active_points"].split("/"))
    assert 0 < active < n  # the second-law hinge is exercised on both sides
    for k, v in r["loss_cpu"].items():
        assert v < TOL, (k, v)
    for k, v in r["grad_cpu"].items():
        assert v < TOL, (k, v)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_m3_physics_parity_seed_sweep(seed):
    """Other seeds: the momentum residual dF/dx - p dA/dx and the entropy
    hinge s_in - s are differences of O(1) float32 numbers, so their
    parameter gradients carry float32 cancellation error ~ eps32 * kappa.
    Bound: SWEEP_TOL against torch-f32 and against a float64 reference."""
    r = parity.m3_parity(seed=seed)
    worst = max(r["loss_cpu_max"], r["grad_cpu_max"], r["d2F_dx2_cpu"])
    worst64 = max(v["mlx32"] for v in r["grad_vs_float64"].values())
    _report(f"M3 seed {seed}", {"worst_vs_torch32": worst, "worst_mlx32_vs_f64": worst64})
    assert worst < SWEEP_TOL
    assert worst64 < SWEEP_TOL


def test_m3_gpu_reported():
    r = parity.m3_parity(seed=0)
    d = {k: r[k] for k in ("fields_gpu_vs_torch_cpu", "dF_dx_gpu_vs_torch_cpu",
                           "loss_gpu_vs_torch_cpu_max", "loss_gpu_vs_mlx_cpu_max")}
    _report("M3 GPU", d)
    assert all(v < GPU_TOL for v in d.values())


# --------------------------------------------------------------------------
# weighting schemes
# --------------------------------------------------------------------------
def test_ma_weights_hand_values():
    # all losses equal: data -> .1+.9*sigmoid(1), phys = bc -> .1+.9*0.5
    s1 = 1.0 / (1.0 + math.exp(-1.0))
    w = ma_weights(1.0, 1.0, 1.0)
    assert w["data"] == pytest.approx(0.1 + 0.9 * s1, abs=1e-8)
    assert w["data"] == pytest.approx(0.7579527, abs=1e-7)
    assert w["phys"] == pytest.approx(0.55, abs=1e-12)
    assert w["bc"] == pytest.approx(0.55, abs=1e-12)
    # L_data=0.2, L_phys=0.05, L_bc=0.01: arguments -0.7, 3, 19
    w = ma_weights(0.2, 0.05, 0.01)
    assert w["data"] == pytest.approx(0.1 + 0.9 / (1 + math.exp(0.7)), abs=1e-7)
    assert w["data"] == pytest.approx(0.3986310, abs=1e-7)
    assert w["phys"] == pytest.approx(0.9573167, abs=1e-7)
    assert w["bc"] == pytest.approx(0.99999999, abs=1e-7)
    for v in w.values():
        assert 0.1 < v <= 1.0
    # in-graph MLX and torch versions agree with numpy
    wm = ma_weights_mx(mx.array(0.2), mx.array(0.05), mx.array(0.01))
    wt = ma_weights_torch(torch.tensor(0.2), torch.tensor(0.05), torch.tensor(0.01))
    for k in w:
        assert float(wm[k]) == pytest.approx(w[k], abs=1e-6)
        assert float(wt[k]) == pytest.approx(w[k], abs=1e-6)


def _toy_m3():
    prob, x_col, bc, data = parity.toy_nozzle(0)
    net = make_m3(hidden=(16, 16), out_ref=(1.0, 0.3, 1.0), seed=0)
    xm = mx.array(x_col)
    bcm = tuple(mx.array(a) for a in bc)
    dm = tuple(mx.array(a) for a in data)
    return net, (lambda m: physics_terms(m, prob, xm, bcm, dm))


def test_ma_weights_are_detached():
    """grad of sum lam_g(L) L_g with in-graph Ma weights equals
    sum lam_g grad L_g with lam frozen; without stop_gradient it would not."""
    mx.set_default_device(mx.cpu)
    net, terms_fn = _toy_m3()
    terms = {k: float(v) for k, v in terms_fn(net).items()}
    lam = ma_weights(terms["data"], terms["phys"], terms["bc"])
    _, g_ma = nn.value_and_grad(net, lambda m: composite_loss(terms_fn(m), "ma_sigmoid"))(net)
    _, g_fix = nn.value_and_grad(net, lambda m: composite_loss(terms_fn(m), lam))(net)
    a, b = flat_grad_vector(g_ma), flat_grad_vector(g_fix)
    assert max_rel_diff(a, b) < 1e-5

    def undetached(m):
        t = terms_fn(m)
        Ld, Lp, Lb = t["data"], t["phys"], t["bc"]
        w = {"data": 0.1 + 0.9 * mx.sigmoid((Lp + Lb - Ld) / (Ld + 1e-8)),
             "phys": 0.1 + 0.9 * mx.sigmoid((Ld - Lp) / (Lp + 1e-8)),
             "bc": 0.1 + 0.9 * mx.sigmoid((Ld - Lb) / (Lb + 1e-8))}
        return sum(w[g] * t[g] for g in ("data", "phys", "bc"))

    _, g_und = nn.value_and_grad(net, undetached)(net)
    assert max_rel_diff(flat_grad_vector(g_und), b) > 1e-3  # the check is sensitive


def test_gradnorm_toy():
    mx.set_default_device(mx.cpu)
    net, terms_fn = _toy_m3()
    scheme = GradNormWeights(ref="phys", alpha=0.9)
    grads = per_term_grads(net, terms_fn)
    lam = scheme.update(grads)
    gmax = np.max(np.abs(grads["phys"]))
    for k in ("data", "bc"):
        expect = 0.1 * 1.0 + 0.9 * gmax / np.mean(np.abs(grads[k]))
        assert lam[k] == pytest.approx(expect, rel=1e-10)
    assert lam["phys"] == 1.0
    assert all(isinstance(v, float) and v > 0 and np.isfinite(v) for v in lam.values())
    _run_weighted_toy(net, terms_fn, lambda n: scheme.update(per_term_grads(n, terms_fn)))


def test_relobralo_toy():
    mx.set_default_device(mx.cpu)
    net, terms_fn = _toy_m3()
    scheme = ReLoBRaLoWeights(alpha=0.999, temperature=0.1, expected_rho=0.999, seed=0)
    L0 = {"data": 0.4, "phys": 2.0, "bc": 0.1}
    assert scheme.update(L0) == {"data": 1.0, "phys": 1.0, "bc": 1.0}
    L1 = {"data": 0.2, "phys": 1.0, "bc": 0.09}
    lam = scheme.update(L1)
    # hand formula
    Lv0, Lv1 = np.array([0.4, 2.0, 0.1]), np.array([0.2, 1.0, 0.09])

    def bal(Lt, Lr):
        e = np.exp(Lt / (0.1 * Lr))
        return 3 * e / e.sum()

    rho = scheme.last_rho
    hist = rho * np.ones(3) + (1 - rho) * bal(Lv1, Lv0)
    expect = 0.999 * hist + 0.001 * bal(Lv1, Lv0)  # t-1 == 0 at the first update
    assert np.allclose([lam[k] for k in ("data", "phys", "bc")], expect, rtol=1e-12)
    assert np.isclose(bal(Lv1, Lv0).sum(), 3.0)
    assert all(isinstance(v, float) and v > 0 for v in lam.values())

    fresh = ReLoBRaLoWeights(seed=0)
    _run_weighted_toy(net, terms_fn, lambda n: fresh.update({k: float(v) for k, v in terms_fn(n).items()}))


def _run_weighted_toy(net, terms_fn, weight_fn, steps=60):
    """A few Adam steps on the toy nozzle with scheme-updated weights:
    weights stay positive/finite, composite loss decreases."""
    import mlx.optimizers as optim
    opt = optim.Adam(learning_rate=3e-3, bias_correction=True)
    first = last = None
    for _ in range(steps):
        lam = weight_fn(net)
        assert all(isinstance(v, float) and v > 0 and np.isfinite(v) for v in lam.values())
        L, g = nn.value_and_grad(net, lambda m: composite_loss(terms_fn(m), lam))(net)
        # weights are constants: gradient equals sum lam_g grad L_g
        opt.update(net, g)
        mx.eval(net.parameters(), opt.state)
        first = float(L) if first is None else first
        last = float(L)
    unweighted_now = sum(float(v) for k, v in terms_fn(net).items() if k in ("data", "phys", "bc"))
    assert np.isfinite(unweighted_now)
    assert last < first


# --------------------------------------------------------------------------
# nondim, score64, training, checkpoints, ensemble
# --------------------------------------------------------------------------
def test_nondim_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X_train = np.column_stack([rng.normal(1.2e5, 3e4, 200), rng.normal(1500.0, 80.0, 200),
                               rng.normal(0.02, 1e-3, 200), np.full(200, 7.0)])
    X_test = X_train[:50] * 1.1
    s = Scaler(names=["p", "T", "far", "const"]).fit(X_train)
    Z = s.transform(X_train)
    assert Z.dtype == np.float64
    assert np.allclose(Z[:, :3].mean(0), 0, atol=1e-12) and np.allclose(Z[:, :3].std(0), 1, atol=1e-12)
    assert s.constant_columns == [3] and s.scale[3] == 1.0
    for X in (X_train, X_test):
        back = s.inverse_transform(s.transform(X))
        assert np.max(np.abs(back - X) / np.abs(X)) < 1e-14
    with pytest.raises(RuntimeError):
        s.fit(X_test)  # frozen after the single training fit
    s.save(tmp_path / "s.json")
    s2 = Scaler.load(tmp_path / "s.json")
    assert np.array_equal(s2.mean, s.mean) and np.array_equal(s2.scale, s.scale)
    assert s2.fit_sha256 == s.fit_sha256 and s2.n_fit == 200
    assert np.array_equal(s2.transform(X_test), s.transform(X_test))


@pytest.mark.parametrize("which", ["M1", "M4", "M3"])
def test_score64_matches_mlx_float32(which, tmp_path):
    rng = np.random.default_rng(5)
    if which == "M1":
        m, X = make_m1(6, 1, seed=3), rng.standard_normal((300, 6))
    elif which == "M4":
        m, X = make_m4(6, 4, seed=3), rng.standard_normal((300, 6))
    else:
        m, X = make_m3(seed=3, out_ref=(1.0, 0.3, 1.0)), rng.uniform(0, 1, (300, 1))
    p64 = export_params64(m)
    Y64 = forward64(p64, m.spec, X)
    assert Y64.dtype == np.float64
    out = {}
    for dev in (mx.cpu, mx.gpu):
        mx.set_default_device(dev)
        Y32 = np.array(m(mx.array(X.astype(np.float32))), dtype=np.float64)
        out[str(dev)] = max_rel_diff(Y32, Y64)
        assert out[str(dev)] < TOL
    _report(f"score64 {which}", out)
    save_params64(tmp_path / "w.npz", p64, m.spec)
    p2, spec2 = load_params64(tmp_path / "w.npz")
    assert spec2 == m.spec and all(np.array_equal(p2[k], p64[k]) for k in p64)


def _toy_function(n, seed):
    """Synthetic analytic target (not project data)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(200.0, 400.0, n), rng.uniform(1e5, 3e5, n)])
    a = (X[:, 0] - 300.0) / 100.0
    b = (X[:, 1] - 2e5) / 1e5
    Y = (1000.0 + 50.0 * np.sin(np.pi * a) + 30.0 * b ** 2)[:, None]
    return X, Y


def _fit_toy(seed, epochs=150):
    X, Y = _toy_function(512, 0)
    Xt, Yt = _toy_function(256, 99)
    xs, ys = Scaler().fit(X), Scaler().fit(Y)
    m = make_m4(2, 1, seed=seed)
    hist = train(m, xs.transform(X), ys.transform(Y), epochs=epochs, lr=3e-3, batch_size=128, seed=seed)
    sc = score64(export_params64(m), m.spec, Xt, Yt, xs, ys)
    return m, hist, sc


def test_three_seed_toy_training_spread():
    runs = [_fit_toy(s) for s in (0, 1, 2)]
    metrics = np.array([[r[2]["rmse"], r[2]["mape_pct"]] for r in runs])
    spread = max_sample_sd(metrics)
    _report("3-seed toy", {"rmse": metrics[:, 0].tolist().__repr__(), "mape_pct": metrics[:, 1].tolist().__repr__(),
                           "max_sample_sd": spread})
    for _, hist, sc in runs:
        assert len(hist["train_loss"]) == 150 and not hist["stopped_early"]  # no early stopping by default
        assert hist["train_loss"][-1] < 0.05 * hist["train_loss"][0]
        assert sc["mape_pct"] < 1.0  # learned the toy function to < 1 % in float64 scoring
    assert np.isfinite(spread) and spread > 0
    assert spread == pytest.approx(max(np.std(metrics[:, 0], ddof=1), np.std(metrics[:, 1], ddof=1)))
    # same seed reproduces exactly; different seeds differ
    again = _fit_toy(0)
    assert again[1]["train_loss"] == runs[0][1]["train_loss"]
    assert again[2] == runs[0][2]
    assert runs[0][1]["train_loss"][-1] != runs[1][1]["train_loss"][-1]


def test_compiled_step_matches_eager_first_step():
    X, Y = _toy_function(64, 0)
    X = Scaler().fit(X).transform(X)
    Y = Scaler().fit(Y).transform(Y)
    a, b = make_m4(2, 1, seed=4), make_m4(2, 1, seed=4)
    ha = train(a, X, Y, epochs=1, seed=4, compile=True)
    hb = train(b, X, Y, epochs=1, seed=4, compile=False)
    assert ha["train_loss"][0] == pytest.approx(hb["train_loss"][0], rel=1e-6)
    pa, pb = dict(tree_flatten(a.parameters())), dict(tree_flatten(b.parameters()))
    assert max(max_rel_diff(np.array(pa[k]), np.array(pb[k])) for k in pa) < 1e-5


def test_early_stopping_opt_in():
    X, Y = _toy_function(128, 0)
    X = Scaler().fit(X).transform(X)
    Y = Scaler().fit(Y).transform(Y)
    m = make_m4(2, 1, seed=0)
    h = train(m, X[:96], Y[:96], epochs=40, lr=0.3, seed=0, patience=2, X_val=X[96:], Y_val=Y[96:])
    assert h["stopped_early"] and len(h["train_loss"]) < 40
    with pytest.raises(ValueError):
        train(m, X, Y, epochs=1, patience=3)


def test_checkpoint_sha256(tmp_path):
    m = make_m4(3, 2, seed=11)
    sha = save_checkpoint(m, tmp_path / "ck", meta={"note": "toy"})
    m2 = make_m4(3, 2, seed=12)
    side = load_checkpoint(m2, tmp_path / "ck", expected_sha256=sha)
    assert side["sha256"] == sha and side["meta"] == {"note": "toy"}
    x = mx.array(np.ones((2, 3), np.float32))
    assert np.array_equal(np.array(m(x)), np.array(m2(x)))
    with pytest.raises(ValueError):
        load_checkpoint(m2, tmp_path / "ck", expected_sha256="0" * 64)
    w = tmp_path / "ck.safetensors"
    raw = bytearray(w.read_bytes())
    raw[-1] ^= 0x01
    w.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        load_checkpoint(m2, tmp_path / "ck")
    save_checkpoint(m, tmp_path / "ck1")
    with pytest.raises(ValueError):
        load_checkpoint(make_m1(3, 2, seed=0), tmp_path / "ck1")  # spec mismatch


def test_ensemble_and_group_bootstrap():
    seeds = member_seeds(7, 5)
    assert len(set(seeds)) == 5 and seeds == member_seeds(7, 5)
    ens = Ensemble(make_m1(4, 1).spec, 5, base_seed=7)
    w0 = [np.array(m.layers[0].weight) for m in ens.members]
    assert all(not np.array_equal(w0[0], w) for w in w0[1:])
    x = mx.array(np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32))
    mean, sd = ens.predict(x)
    assert mean.shape == (10, 1) and sd.shape == (10, 1) and float(sd.min()) > 0
    m64, sd64, _ = ensemble_predict64([export_params64(m) for m in ens.members], ens.spec,
                                      np.array(x, dtype=np.float64))
    assert max_rel_diff(np.array(mean), m64) < TOL and max_rel_diff(np.array(sd), sd64) < 1e-4

    groups = np.array(["A"] * 3 + ["B"] * 2 + ["C"] * 4 + ["D"] * 1)
    boots = bootstrap_group_indices(groups, 20, seed=3)
    assert all(np.array_equal(a[0], b[0]) and a[1] == b[1]
               for a, b in zip(boots, bootstrap_group_indices(groups, 20, seed=3)))
    for idx, oob in boots:
        drawn = groups[idx]
        for g in np.unique(drawn):  # whole groups only, in multiples of the group size
            assert np.sum(drawn == g) % np.sum(groups == g) == 0
        assert set(oob) == set(np.unique(groups)) - set(np.unique(drawn))
        assert sum(np.sum(drawn == g) // np.sum(groups == g) for g in np.unique(drawn)) == 4
    assert any(len(oob) > 0 for _, oob in boots)


def test_compiled_m3_physics_step():
    """mx.compile around nn.value_and_grad of a loss that itself contains
    mx.vmap(mx.grad(.)) w.r.t. the network input (the momentum residual):
    compiled and eager training agree."""
    prob, x_col, bc, data = parity.toy_nozzle(0)
    bcm = tuple(mx.array(a) for a in bc)
    dm = tuple(mx.array(a) for a in data)
    out = []
    for comp in (True, False):
        net = make_m3(hidden=(16, 16), out_ref=(1.0, 0.3, 1.0), seed=0)
        lf = lambda m, x, y: composite_loss(physics_terms(m, prob, x, bcm, dm), "ma_sigmoid")
        h = train(net, x_col, np.zeros_like(x_col), loss_fn=lf, epochs=20, lr=3e-3, seed=0, compile=comp)
        out.append(h["train_loss"])
    assert out[0][0] == pytest.approx(out[1][0], rel=1e-6)
    assert out[0][-1] < out[0][0]
    assert max_rel_diff(np.array(out[0]), np.array(out[1])) < 1e-4
