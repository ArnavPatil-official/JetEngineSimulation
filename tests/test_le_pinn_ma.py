"""
P7.4 contracts, checked BEFORE any study training (simulation/nozzle/le_pinn_ma.py,
scripts/validation/train_sajben_ma.py, report_sajben_ma.py):

* the residual on manufactured differentiable fields equals an independent
  symbolic (sympy) evaluation of the registered equations, variable-mu gradient
  terms included;
* uniform-flow and constant-viscosity limits; turbulence counted once;
  unit/coordinate scaling; exact chain rule through the model's normalisation;
* wall normals of the curved wall and the adiabatic normal-flux term;
* the current-loss weight equations, detached;
* withheld labels cannot enter training; paired arms share subsets and init;
* resume after interruption continues the identical trajectory; the report
  refuses an incomplete batch.

The smoke trainings here are NON-STUDY implementation checks with temporary
outputs; their losses are not compared across arms and no score is computed.
"""

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from simulation.nozzle import le_pinn_ma as lm  # noqa: E402

REG_PATH = ROOT / "outputs" / "phase7" / "p74_registration.json"
REF = {"rho_ref": 1.45, "a_ref": 335.0, "T_ref": 280.0, "gamma": 1.4, "R": 287.0}
L_REF = 0.044


# --------------------------------------------------------------------------
# Manufactured fields (torch and sympy versions of the same expressions)
# --------------------------------------------------------------------------
def _expr(lib, x, y, uniform=False, mut_scale=1.0):
    sin, cos = lib.sin, lib.cos
    if uniform:
        rho, u, v, T = 1.2 + 0 * x, 150.0 + 0 * x, 5.0 + 0 * x, 290.0 + 0 * x
        return {"rho": rho, "u": u, "v": v, "T": T, "p": rho * REF["R"] * T, "mu_t": 2e-3 + 0 * x}
    rho = 1.2 + 0.1 * sin(30 * x) * cos(20 * y)
    u = 100 + 20 * sin(20 * x + 10 * y)
    v = 10 * cos(10 * x - 20 * y)
    T = 280 + 15 * sin(10 * x + 10 * y)
    p = rho * REF["R"] * T * (1 + 0.01 * sin(10 * x))
    mut = mut_scale * 1e-3 * (1 + 0.5 * sin(20 * x) * cos(10 * y))
    return {"rho": rho, "u": u, "v": v, "T": T, "p": p, "mu_t": mut}


def _torch_fields(uniform=False, mut_scale=1.0):
    return lambda xy: _expr(torch, xy[:, 0], xy[:, 1], uniform, mut_scale)


def _sym_residuals(mu_const=None, freeze_mu_gradient=False, pr=lm.PR, pr_t=lm.PR_T):
    """Registered equations, written out independently with sympy."""
    x, y = sp.symbols("x y")
    f = _expr(sp, x, y)
    g, R = REF["gamma"], REF["R"]
    cp = g * R / (g - 1)
    cv = cp - R
    mu = (lm.SUTH_C1 * f["T"] ** sp.Rational(3, 2) / (f["T"] + lm.SUTH_S)) if mu_const is None else mu_const
    mut = f["mu_t"]
    if freeze_mu_gradient:     # treat mu_eff as a local constant inside the divergence
        m0, mt0 = sp.symbols("m0 mt0")
        mu_s, mut_s = m0, mt0
    else:
        mu_s, mut_s = mu, mut
    mue = mu_s + mut_s
    u, v, T, rho, p = f["u"], f["v"], f["T"], f["rho"], f["p"]
    ux, uy, vx, vy = sp.diff(u, x), sp.diff(u, y), sp.diff(v, x), sp.diff(v, y)
    div = ux + vy
    txx = mue * (2 * ux - sp.Rational(2, 3) * div)
    tyy = mue * (2 * vy - sp.Rational(2, 3) * div)
    txy = mue * (uy + vx)
    k = cp * (mu_s / pr + mut_s / pr_t)
    qx, qy = -k * sp.diff(T, x), -k * sp.diff(T, y)
    H = cv * T + (u ** 2 + v ** 2) / 2 + p / rho
    rr, ar = REF["rho_ref"], REF["a_ref"]
    res = {
        "continuity": (sp.diff(rho * u, x) + sp.diff(rho * v, y)) / (rr * ar / L_REF),
        "x_mom": (sp.diff(rho * u * u + p - txx, x) + sp.diff(rho * u * v - txy, y)) / (rr * ar ** 2 / L_REF),
        "y_mom": (sp.diff(rho * u * v - txy, x) + sp.diff(rho * v * v + p - tyy, y)) / (rr * ar ** 2 / L_REF),
        "energy": (sp.diff(rho * u * H - u * txx - v * txy + qx, x)
                   + sp.diff(rho * v * H - u * txy - v * tyy + qy, y)) / (rr * ar ** 3 / L_REF),
        "eos": (p - rho * R * T) / (rr * ar ** 2),
    }
    if freeze_mu_gradient:
        res = {k2: e.subs({sp.Symbol("m0"): mu, sp.Symbol("mt0"): mut}) for k2, e in res.items()}
    return {k2: sp.lambdify((x, y), e, "numpy") for k2, e in res.items()}


@pytest.fixture(scope="module")
def pts():
    rng = np.random.default_rng(0)
    return rng.uniform([-0.1, 0.0], [0.3, 0.05], size=(64, 2))


def _auto(pts, fields, **kw):
    xy = torch.tensor(pts, dtype=torch.float64, requires_grad=True)
    return {k: v.detach().numpy() for k, v in lm.rans_residuals(xy, fields, REF, L_REF, **kw).items()}


def test_residuals_equal_symbolic_equations(pts):
    auto = _auto(pts, _torch_fields())
    sym = _sym_residuals()
    for k in ("continuity", "x_mom", "y_mom", "energy", "eos"):
        ref = sym[k](pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(auto[k], ref, rtol=1e-9, atol=1e-9 * np.abs(ref).max(), err_msg=k)


def test_variable_mu_gradient_terms_survive(pts):
    auto = _auto(pts, _torch_fields(mut_scale=50.0))
    x, y = pts[:, 0], pts[:, 1]
    # symbolic residual WITH grad(mu_eff) . grad(u) terms equals autograd to
    # ~1e-12 relative; the form WITHOUT them differs by orders of magnitude more
    sym_full = _sym_residuals_scaled(50.0, freeze=False)
    sym_frozen = _sym_residuals_scaled(50.0, freeze=True)
    for k in ("x_mom", "y_mom", "energy"):
        full, frozen = sym_full[k](x, y), sym_frozen[k](x, y)
        err_full = np.abs(auto[k] - full).max()
        err_frozen = np.abs(auto[k] - frozen).max()
        assert err_full < 1e-10 * np.abs(full).max(), k
        assert err_frozen > 1e4 * max(err_full, 1e-16), k


def _sym_residuals_scaled(scale, freeze):
    global _expr
    orig = _expr

    def scaled(lib, x, y, uniform=False, mut_scale=1.0):
        return orig(lib, x, y, uniform, scale)
    _expr = scaled
    try:
        return _sym_residuals(freeze_mu_gradient=freeze)
    finally:
        _expr = orig


def test_uniform_flow_has_zero_residual(pts):
    auto = _auto(pts, _torch_fields(uniform=True))
    for k, v in auto.items():
        assert np.abs(v).max() < 1e-10, k


def test_constant_viscosity_limit(pts):
    """mu_eff constant: the viscous x-momentum term is mu (lap u + 1/3 d(div u)/dx)."""
    mu0 = 3e-3
    f = _torch_fields()

    def no_mut(xy):
        d = f(xy)
        d["mu_t"] = 0 * d["mu_t"]
        return d
    a = _auto(pts, no_mut, mu_fn=lambda T: mu0 + 0 * T)
    x, y = sp.symbols("x y")
    e = _expr(sp, x, y)
    u, v, rho, p = e["u"], e["v"], e["rho"], e["p"]
    visc = mu0 * (sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(sp.diff(u, x) + sp.diff(v, y), x) / 3)
    inv = sp.diff(rho * u * u + p, x) + sp.diff(rho * u * v, y)
    ref = sp.lambdify((x, y), (inv - visc) / (REF["rho_ref"] * REF["a_ref"] ** 2 / L_REF))(pts[:, 0], pts[:, 1])
    np.testing.assert_allclose(a["x_mom"], ref, rtol=1e-9)


def test_turbulence_counted_once(pts):
    """(mu = m0, mu_t = m1) and (mu = m0 + m1, mu_t = 0) give the same momentum
    residual; with Pr = Pr_t the energy residual matches too."""
    m0, m1 = 2e-5, 4e-3
    f = _torch_fields(uniform=False)

    def with_mut(val):
        def g(xy):
            d = f(xy)
            d["mu_t"] = val + 0 * d["mu_t"]
            return d
        return g
    a = _auto(pts, with_mut(m1), mu_fn=lambda T: m0 + 0 * T, pr=0.9, pr_t=0.9)
    b = _auto(pts, with_mut(0.0), mu_fn=lambda T: (m0 + m1) + 0 * T, pr=0.9, pr_t=0.9)
    for k in ("x_mom", "y_mom", "energy", "continuity"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-11, atol=1e-12)
    c = _auto(pts, with_mut(m1), mu_fn=lambda T: 2 * m0 + 0 * T, pr=0.9, pr_t=0.9)
    assert np.abs(c["x_mom"] - a["x_mom"]).max() > 0     # the molecular part is not ignored


def test_unit_and_coordinate_scaling(pts):
    """Field f on length L vs the same field stretched by c on length cL with
    viscosity scaled by c (same Reynolds number): identical scaled residuals."""
    c = 1000.0      # e.g. metres -> millimetres
    f = _torch_fields()

    def stretched(xy):
        d = f(xy / c)
        d["mu_t"] = d["mu_t"] * c
        return d
    a = _auto(pts, f)
    xy = torch.tensor(pts * c, dtype=torch.float64, requires_grad=True)
    b = lm.rans_residuals(xy, stretched, REF, L_REF * c,
                          mu_fn=lambda T: lm.sutherland(T) * c)
    for k in a:
        np.testing.assert_allclose(b[k].detach().numpy(), a[k], rtol=1e-9, atol=1e-12, err_msg=k)


@pytest.fixture(scope="module")
def wind():
    return lm.load_wind()


@pytest.fixture(scope="module")
def geom(wind):
    return lm.geometry_from(wind)


def test_model_chain_rule_through_normalisation(geom):
    torch.manual_seed(0)
    sc = {"mean": [1.2, 200.0, 5.0, 1e5, 280.0, 1.0], "std": [0.2, 80.0, 10.0, 2e4, 20.0, 1.5]}
    m = lm.MaLEPINN(geom, sc, width=32, n_hidden=2, b_width=16, b_hidden=2).double()
    xy = torch.tensor([[0.05, 0.02], [0.2, 0.03], [-0.1, 0.01]], dtype=torch.float64, requires_grad=True)
    u = m.fields(xy)["u"]
    g = torch.autograd.grad(u.sum(), xy)[0].numpy()
    h = 1e-7
    for j in range(2):
        e = torch.zeros(3, 2, dtype=torch.float64)
        e[:, j] = h
        fd = ((m.fields(xy.detach() + e)["u"] - m.fields(xy.detach() - e)["u"]) / (2 * h)).detach().numpy()
        np.testing.assert_allclose(g[:, j], fd, rtol=1e-5)


def test_wall_normals_and_adiabatic_flux(geom):
    wxy, wn = lm.wall_points(geom)
    n = len(wxy) // 2
    np.testing.assert_allclose(np.linalg.norm(wn, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(wn[:n], np.tile([0.0, 1.0], (n, 1)), atol=1e-12)
    # upper wall normal is perpendicular to the wall tangent (curved wall)
    xs, yu = geom.x_nodes, geom.y_upper
    dy = np.gradient(yu, xs)[1:]
    tangent = np.column_stack([np.ones_like(dy), dy])
    assert np.abs((tangent * wn[n:]).sum(axis=1)).max() < 1e-12
    assert np.abs(wn[n:, 0]).max() > 0.05                     # genuinely curved
    # dT/dn: T = T0 + a y  ->  dT/dn = a * n_y on each wall
    a = 1000.0

    def fields(xy):
        z = 0 * xy[:, 0]
        return {"u": z, "v": z, "T": 280.0 + a * xy[:, 1], "rho": 1 + z, "p": 1 + z, "mu_t": z}
    xyw = torch.tensor(wxy, dtype=torch.float64, requires_grad=True)
    _L, terms = lm.wall_bc_loss(xyw, torch.tensor(wn), fields, REF, L_REF)
    expect = np.mean((a * wn[:, 1] * L_REF / REF["T_ref"]) ** 2)
    assert float(terms["dTdn"]) == pytest.approx(expect, rel=1e-12)
    assert float(terms["u"]) == 0.0 and float(terms["v"]) == 0.0


def test_current_loss_weights():
    Ld = torch.tensor(1.0, requires_grad=True)
    Lp = torch.tensor(2.0, requires_grad=True)
    Lb = torch.tensor(3.0, requires_grad=True)
    w = lm.ma_weights(Ld, Lp, Lb, eps=0.0)
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    assert float(w["data"]) == pytest.approx(0.1 + 0.9 * sig((2 + 3 - 1) / 1))
    assert float(w["phys"]) == pytest.approx(0.1 + 0.9 * sig((1 - 2) / 2))
    assert float(w["bc"]) == pytest.approx(0.1 + 0.9 * sig((1 - 3) / 3))
    assert not any(v.requires_grad for v in w.values())
    total = w["data"] * Ld + w["phys"] * Lp + w["bc"] * Lb
    total.backward()
    assert float(Lp.grad) == pytest.approx(float(w["phys"]))   # weight not differentiated


# --------------------------------------------------------------------------
# Registration, split, leakage, pairing
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reg():
    import train_sajben_ma as t
    return t.load_registration(REG_PATH)


def test_registration_design(reg):
    split = json.loads((ROOT / reg["split"]["file"]).read_text())
    test, pool = set(split["test"]), set(split["pool"])
    assert len(test) == 816 and len(pool) == 3264 and not (test & pool)
    assert len(test | pool) == 4080 and all(g % 81 != 0 for g in test | pool)   # inflow column excluded
    assert len(reg["runs"]) == 30
    sizes = {"f002": 65, "f005": 163, "f010": 326, "f025": 816, "f100": 3264}
    for s in ("42", "43", "44"):
        sub = split["subsets"][s]
        assert {k: len(v) for k, v in sub.items()} == sizes
        keys = list(sizes)
        for a, b in zip(keys, keys[1:]):
            assert set(sub[a]) <= set(sub[b])                                   # nested
        assert set(sub["f100"]) == pool
    assert reg["training"]["epochs"] == 5000 and reg["training"]["collocation_batch"] is None
    for r in reg["runs"]:
        assert r["init_seed"] == r["seed"]
        twin = [q for q in reg["runs"] if q["seed"] == r["seed"] and q["fraction_key"] == r["fraction_key"]]
        assert len(twin) == 2 and twin[0]["train_ids_sha256"] == twin[1]["train_ids_sha256"]


def test_withheld_labels_cannot_enter_training(reg, wind):
    import train_sajben_ma as t
    run = next(r for r in reg["runs"] if r["run_id"] == "p74_s43_f005_physics")
    b1 = t.build_training_batch(reg, run, sol=copy.deepcopy(wind))
    split = json.loads((ROOT / reg["split"]["file"]).read_text())
    train = set(int(i) for i in b1["train_ids"])
    assert not train & set(split["test"])
    # scramble every label outside this run's training rows (held-out AND unused pool rows)
    poisoned = copy.deepcopy(wind)
    rng = np.random.default_rng(1)
    mask = np.ones(wind.nj * wind.ni, bool)
    mask[list(train)] = False
    for k in ("rho", "u", "v", "p", "T", "mu_l", "mu_t"):
        a = getattr(poisoned, k).ravel().copy()
        a[mask] = rng.normal(size=mask.sum()) * 1e6
        setattr(poisoned, k, a.reshape(wind.nj, wind.ni))
    b2 = t.build_training_batch(reg, run, sol=poisoned)
    for k in ("xy", "labels", "colloc", "wall_xy", "wall_n"):
        assert torch.equal(b1[k], b2[k]), k
    assert b1["scalers"] == b2["scalers"]
    assert set(b1) == {"geom", "train_ids", "scalers", "xy", "labels", "colloc", "wall_xy", "wall_n"}
    assert b1["labels"].shape == (163, 6)                  # values only, no gradient targets
    exp = np.column_stack([getattr(wind, k).ravel()[sorted(train)] for k in ("rho", "u", "v", "p", "T", "mu_t")])
    np.testing.assert_allclose(b1["labels"].numpy(), exp.astype(np.float32))


def test_paired_arms_share_initialisation(reg, wind):
    import train_sajben_ma as t
    ph = next(r for r in reg["runs"] if r["run_id"] == "p74_s44_f010_physics")
    do = next(r for r in reg["runs"] if r["run_id"] == "p74_s44_f010_dataonly")
    bp, bd = t.build_training_batch(reg, ph, sol=wind), t.build_training_batch(reg, do, sol=wind)
    mp, md = t.make_model(bp, reg, ph["init_seed"]), t.make_model(bd, reg, do["init_seed"])
    for (k1, v1), (k2, v2) in zip(mp.state_dict().items(), md.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2), k1


# --------------------------------------------------------------------------
# NON-STUDY implementation checks (temporary outputs)
# --------------------------------------------------------------------------
def _tmp_registration(tmp_path, **training):
    reg = json.loads(REG_PATH.read_text())
    reg["training"].update(training)
    p = tmp_path / "p74_registration.json"
    p.write_text(json.dumps(reg))
    return p


def test_resume_continues_the_identical_trajectory(tmp_path, wind, monkeypatch):
    import train_sajben_ma as t
    monkeypatch.setenv("P7_TORCH_THREADS", "2")
    regp = _tmp_registration(tmp_path, checkpoint_every=2)
    rid = "p74_s42_f002_dataonly"
    a = t.train(rid, regp, tmp_path / "A", smoke_epochs=6, sol=wind, verbose=False)
    calls = {"n": 0}
    orig = t.losses

    def dying(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt("simulated interruption")
        return orig(*args, **kw)
    monkeypatch.setattr(t, "losses", dying)
    with pytest.raises(KeyboardInterrupt):
        t.train(rid, regp, tmp_path / "B", smoke_epochs=6, sol=wind, verbose=False)
    monkeypatch.setattr(t, "losses", orig)
    b = t.train(rid, regp, tmp_path / "B", smoke_epochs=6, sol=wind, verbose=False)
    assert b["resumed_from_epoch"] == 4
    sa = torch.load(tmp_path / "A" / a["final_checkpoint"], weights_only=False)["model_state_dict"]
    sb = torch.load(tmp_path / "B" / b["final_checkpoint"], weights_only=False)["model_state_dict"]
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    with pytest.raises(SystemExit):                     # write-once final outputs
        t.train(rid, regp, tmp_path / "B", smoke_epochs=6, sol=wind, verbose=False)


def test_physics_arm_trains_and_records_provenance(tmp_path, wind, monkeypatch):
    import train_sajben_ma as t
    monkeypatch.setenv("P7_TORCH_THREADS", "2")
    regp = _tmp_registration(tmp_path, collocation_batch=64)
    rec = t.train("p74_s42_f002_physics", regp, tmp_path, smoke_epochs=2, sol=wind, verbose=False)
    assert rec["smoke"] and rec["arm"] == "physics" and rec["n_train_rows"] == 65
    assert rec["history"]["lam_phys"][0] is not None and 0.1 <= rec["history"]["lam_phys"][0] <= 1.0
    assert rec["final_checkpoint_sha256"] == t.sha256(tmp_path / rec["final_checkpoint"])
    assert "commit" in rec["source"]


def test_report_refuses_an_incomplete_batch(tmp_path):
    import report_sajben_ma as rep
    reg = json.loads(REG_PATH.read_text())
    with pytest.raises(SystemExit, match="refused"):
        rep.verify_complete(reg, tmp_path, require_runner=True)


def test_claim_rule_arithmetic():
    import pandas as pd
    import report_sajben_ma as rep
    rows = []
    for fk, f, ph, do in (("f002", .02, [.20, .21, .22], [.30, .31, .32]),
                          ("f005", .05, [.20, .21, .22], [.215, .225, .235]),
                          ("f010", .10, [.20, .21, .22], [.30, .30, .31])):
        for s, a, b in zip((42, 43, 44), ph, do):
            rows.append({"fraction_key": fk, "fraction": f, "seed": s, "arm": "physics", "primary_worse_wall": a})
            rows.append({"fraction_key": fk, "fraction": f, "seed": s, "arm": "dataonly", "primary_worse_wall": b})
    t, v = rep.claim_table(pd.DataFrame(rows))
    assert list(t["benefit"]) == [True, False, True]
    assert v["physics_benefit_claimed"] and v["n_claim_fractions_with_benefit"] == 2
    r = t.iloc[0]
    assert r["seed_spread"] == pytest.approx(max(np.std([.20, .21, .22], ddof=1), np.std([.30, .31, .32], ddof=1)))
