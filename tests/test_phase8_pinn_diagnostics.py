"""Phase 8 Track 4: numerical contracts of the isolated PINN diagnostics.

Synthetic fixtures only. Manufactured-solution and nozzle tests use their own
coefficients/geometry values (not the registered ones), so the registered
diagnostic is computed only by the committed runner. No training beyond a
few optimizer steps, no production simulation, no empirical data.
"""
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase8.pinn_diagnostics import nozzle_verification as nv  # noqa: E402
from scripts.phase8.pinn_diagnostics import run_diagnostics as rd  # noqa: E402
from scripts.phase8.pinn_diagnostics import turbine_map as tm  # noqa: E402

REG = json.loads((ROOT / "docs" / "phase8_track4_registration.json").read_text())
BOX = tm.TurbineBox.from_registration(REG["turbine"])

# test-only manufactured fixture (differs from the registered coefficients)
MMS = {
    "coordinates": [-0.5, 1.5], "grid_x": 7, "grid_y": 5, "R": 0.7, "cp": 2.9, "Pr": 0.9,
    "fields": nv.FIELDS_EXPR, "literal_thermal_coefficient": nv.THERMAL_EXPR,
    "interpretation": "test fixture",
    "coefficients": {
        "rho": [1.2, 0.15, 0.8, -0.4], "u": [0.4, 0.12, -0.5, 0.6], "v": [-0.1, 0.07, 0.3, 0.9],
        "T": [1.3, 0.1, 0.5, 0.4], "mu": [0.03, 0.005, -0.7, 0.3], "conductivity": [0.05, 0.006, 0.4, 0.5],
        "mu_t": [0.02, 0.003, -0.2, 0.7], "UU": [0.02, 0.005, 0.6, -0.3], "VV": [0.03, 0.004, 0.2, 0.5],
        "UV": [0.004, 0.002, -0.6, 0.4]},
    "absolute_residual_max": 1e-10, "relative_forcing_disagreement_max": 1e-9,
    "negative_control_min_discrepancy": 1e-7,
}


# ---------------------------------------------------------------------------
# Turbine map
# ---------------------------------------------------------------------------

def test_known_value_polytropic_pressure():
    assert tm.polytropic_log_pressure_ratio(0.5, 1.4, 1.0) == pytest.approx(3.5 * math.log(0.5), rel=1e-15)
    got = math.exp(tm.polytropic_log_pressure_ratio(0.5, 1.4, 0.9))
    assert got == pytest.approx(0.5 ** (1.4 / (0.9 * 0.4)), rel=1e-14)


@pytest.mark.parametrize("tau,gamma,eta", [(0.0, 1.3, 0.9), (1.0, 1.3, 0.9), (-0.1, 1.3, 0.9),
                                           (0.3, 1.0, 0.9), (0.3, 1.3, 0.0), (float("nan"), 1.3, 0.9)])
def test_invalid_turbine_domain_rejected(tau, gamma, eta):
    with pytest.raises(ValueError):
        tm.polytropic_log_pressure_ratio(tau, gamma, eta)


def test_features_reject_points_outside_box():
    with pytest.raises(ValueError):
        tm.features([BOX.tau[1] + 1e-6], [BOX.gamma[0]], BOX)
    with pytest.raises(ValueError):
        tm.features([BOX.tau[0]], [BOX.gamma[0] - 1e-6], BOX)


def test_registration_mismatch_rejected():
    bad = dict(REG["turbine"], target="gamma/(gamma-1)*log1p(-tau)")
    with pytest.raises(ValueError):
        tm.TurbineBox.from_registration(bad)


def test_gate_is_strict_max_not_mean():
    errs = np.array([0.0, 0.0, 0.0, 0.0011])          # mean 2.75e-4 < 1e-3, max above
    assert not tm.gate(errs, 1e-3)["pass"]
    assert not tm.gate([1e-3], 1e-3)["pass"]          # equality fails: strictly below
    assert tm.gate([9.99e-4, 0.0], 1e-3)["pass"]


def test_relative_log_scoring_exact_and_pressure_scale_invariant():
    exact = np.array([-1.3, -1.8, -2.3])
    delta = np.array([1e-4, -3e-4, 2e-12])
    pred = exact + delta
    err = tm.relative_errors(pred, exact)
    np.testing.assert_array_equal(err, np.abs(np.expm1(pred - exact)))
    np.testing.assert_allclose(err[:2], np.abs(np.expm1(delta[:2])), rtol=1e-10)
    assert err[2] == pytest.approx(2e-12, rel=1e-3)   # no cancellation at tiny errors
    for p4 in (1.0, 101325.0, 4.2e6):
        e_p = tm.relative_errors_from_pressures(p4 * np.exp(pred), p4 * np.exp(exact))
        np.testing.assert_allclose(e_p[:2], err[:2], rtol=1e-9)


def test_constant_eta_feature_and_scaled_corners():
    tau = np.array([BOX.tau[0], BOX.tau[1], BOX.tau[0], BOX.tau[1]])
    gam = np.array([BOX.gamma[0], BOX.gamma[0], BOX.gamma[1], BOX.gamma[1]])
    X = tm.features(tau, gam, BOX).numpy()
    assert X.shape == (4, 4)
    assert np.all(X[:, 2] == 0.0)                     # eta/reference - 1 with eta = reference
    np.testing.assert_allclose(X[:, 0], [-1, 1, -1, 1], atol=1e-15)
    np.testing.assert_allclose(X[:, 1], [-1, -1, 1, 1], atol=1e-15)
    np.testing.assert_allclose(X[:, 3], [1, 1, -1, -1], atol=1e-14)   # cp/R falls as gamma rises
    np.testing.assert_allclose(tm.cp_over_r(gam), gam / (gam - 1.0), rtol=1e-15)


def test_float64_gradients_and_checkpoint_roundtrip(tmp_path):
    model = tm.build_mlp([8, 8], "tanh", seed=3)
    X = tm.features(*tm.sobol_samples(16, 3, BOX), BOX)
    assert X.dtype == torch.float64
    model(X).pow(2).mean().backward()
    assert all(p.dtype == torch.float64 and p.grad.dtype == torch.float64 for p in model.parameters())
    path = tmp_path / "ckpt.pt"
    tm.save_checkpoint(path, model, {"seed": 3})
    loaded, meta = tm.load_checkpoint(path, [8, 8], "tanh")
    with torch.no_grad():
        assert torch.equal(model(X), loaded(X))
    assert meta == {"seed": 3}
    with pytest.raises(FileExistsError):
        tm.save_checkpoint(path, model, {"seed": 3})


def test_training_and_score_inputs_are_disjoint():
    c = REG["turbine"]
    tr = set(zip(*(a.tolist() for a in tm.sobol_samples(c["training_samples"], REG["seed"], BOX))))
    grid = list(zip(*(a.tolist() for a in tm.score_grid(BOX, c["score_grid_points_per_axis"]))))
    assert len(grid) == 4225 and len(set(grid)) == 4225
    assert not tr.intersection(grid)


def test_sobol_is_the_registered_draw_base2():
    c = REG["turbine"]
    assert tm.sobol_implementation(c["training_samples"], REG["seed"]) == c["sobol_implementation"]
    u = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=7).draw_base2(4, dtype=torch.float64).numpy()
    tau, gam = tm.sobol_samples(16, 7, BOX)
    np.testing.assert_array_equal(tau, BOX.tau[0] + u[:, 0] * (BOX.tau[1] - BOX.tau[0]))
    np.testing.assert_array_equal(gam, BOX.gamma[0] + u[:, 1] * (BOX.gamma[1] - BOX.gamma[0]))
    for n in (0, 3, 100, 4095):
        with pytest.raises(ValueError):
            tm.sobol_samples(n, 7, BOX)


def test_contaminated_score_grid_refused_before_training(tmp_path, monkeypatch):
    """Mocked collision: no training, checkpoint or score may happen."""
    tau_tr, gam_tr = tm.sobol_samples(REG["turbine"]["training_samples"], REG["seed"], BOX)
    real_grid = tm.score_grid

    def grid_with_collision(box, n):
        t, g = (a.copy() for a in real_grid(box, n))
        t[5], g[5] = tau_tr[0], gam_tr[0]
        return t, g

    def forbidden(*_a, **_k):
        raise AssertionError("contaminated grid reached training or scoring")

    monkeypatch.setattr(tm, "score_grid", grid_with_collision)
    for name in ("features", "build_mlp", "train", "save_checkpoint", "predict_log", "gate"):
        monkeypatch.setattr(tm, name, forbidden)
    with pytest.raises(ValueError, match="score-grid inputs equal training inputs"):
        tm.run(REG, tmp_path, log=lambda _m: None)
    assert list(tmp_path.iterdir()) == []


def test_generated_fixtures_repeat():
    a, b = tm.sobol_samples(64, 11, BOX), tm.sobol_samples(64, 11, BOX)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    m1, m2 = tm.build_mlp([8], "silu", 5), tm.build_mlp([8], "silu", 5)
    assert all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters()))
    assert all(np.array_equal(u, v) for u, v in zip(nv.manufactured_grid(MMS), nv.manufactured_grid(MMS)))


def test_tiny_training_reduces_loss():
    cfg = dict(REG["turbine"], adam_steps=20, lbfgs_max_iter=5, lbfgs_max_eval=8)
    tau, gam = tm.sobol_samples(32, 1, BOX)
    X = tm.features(tau, gam, BOX)
    y = torch.as_tensor(tm.polytropic_log_pressure_ratio(tau, gam, BOX.eta_p)).view(-1, 1)
    model = tm.build_mlp([8], "tanh", 1)
    with torch.no_grad():
        before = float(((model(X) - y) ** 2).mean())
    hist = tm.train(model, X, y, cfg, log=lambda _m: None)
    assert hist["train_mse_final"] < before


# ---------------------------------------------------------------------------
# Activations, MMS, residual limits, chain rule, Ma weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tanh", "silu"])
def test_closed_form_activation_derivatives_match_autograd(name):
    z = torch.linspace(-4.0, 4.0, 33, dtype=torch.float64, requires_grad=True)
    h = nv.activation_torch(name)(z)
    d1 = torch.autograd.grad(h.sum(), z, create_graph=True)[0]
    d2 = torch.autograd.grad(d1.sum(), z)[0]
    H, H1, H2 = nv.activation_numpy(name, z.detach().numpy())
    np.testing.assert_allclose(H, h.detach().numpy(), atol=1e-15)
    np.testing.assert_allclose(H1, d1.detach().numpy(), atol=1e-14)
    np.testing.assert_allclose(H2, d2.numpy(), atol=1e-14)


def test_unknown_activation_rejected():
    with pytest.raises(ValueError):
        nv.activation_numpy("relu", 0.0)


@pytest.mark.parametrize("name", ["tanh", "silu"])
def test_mms_forcing_and_all_four_negative_controls(name):
    r = nv.verify_manufactured(MMS, name)
    assert r["status"] == "PASS"
    assert r["max_abs_forced_residual"] <= 1e-10
    assert r["max_relative_forcing_disagreement"] <= 1e-9
    assert set(r["negative_control_discrepancy"]) == set(nv.OMISSIONS)
    assert all(v > 1e-7 for v in r["negative_control_discrepancy"].values())


def test_mms_detects_wrong_forcing():
    bad = json.loads(json.dumps(MMS))
    bad["cp"] = MMS["cp"] * (1 + 1e-3)          # forcing built with a wrong cp
    xs, ys = nv.manufactured_grid(bad)
    f_bad = nv.analytic_forcing(xs, ys, MMS["coefficients"], "tanh", MMS["R"], bad["cp"], MMS["Pr"])
    x = torch.tensor(xs, requires_grad=True)
    y = torch.tensor(ys, requires_grad=True)
    f = nv.manufactured_fields_torch(x, y, MMS["coefficients"], "tanh", MMS["R"])
    res = nv.ma_residuals(x, y, f, MMS["R"], MMS["cp"], MMS["Pr"])
    assert np.max(np.abs(res["energy"].detach().numpy() - f_bad["energy"])) > MMS["absolute_residual_max"]


def test_uniform_flow_and_eos_limits():
    coeffs = {k: [v[0], 0.0, v[2], v[3]] for k, v in MMS["coefficients"].items()}
    xs, ys = nv.manufactured_grid(MMS)
    x = torch.tensor(xs, requires_grad=True)
    y = torch.tensor(ys, requires_grad=True)
    f = nv.manufactured_fields_torch(x, y, coeffs, "tanh", MMS["R"])
    res = nv.ma_residuals(x, y, f, MMS["R"], MMS["cp"], MMS["Pr"])
    for eq in nv.EQUATIONS:
        assert torch.max(torch.abs(res[eq])).item() <= 1e-15
    forcing = nv.analytic_forcing(xs, ys, coeffs, "tanh", MMS["R"], MMS["cp"], MMS["Pr"])
    assert all(np.max(np.abs(v)) <= 1e-15 for v in forcing.values())
    f["p"] = f["p"] + 0.1                        # violate the equation of state by a known amount
    res = nv.ma_residuals(x, y, f, MMS["R"], MMS["cp"], MMS["Pr"])
    np.testing.assert_allclose(res["eos"].detach().numpy(), 0.1, rtol=1e-14)


def test_unknown_omission_rejected():
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    f = nv.manufactured_fields_torch(x, x, MMS["coefficients"], "tanh", 1.0)
    with pytest.raises(ValueError):
        nv.ma_residuals(x, x, f, 1.0, 1.0, 1.0, omit=("pressure",))


def test_physical_coordinate_chain_rule():
    lo, hi, a = 2.0, 5.0, 1.3
    x = torch.linspace(lo, hi, 9, dtype=torch.float64, requires_grad=True)
    q = torch.tanh(a * nv.min_max_normalize(x, lo, hi))
    d1 = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
    d2 = torch.autograd.grad(d1.sum(), x)[0]
    _, h1, h2 = nv.activation_numpy("tanh", a * (x.detach().numpy() - lo) / (hi - lo))
    np.testing.assert_allclose(d1.detach().numpy(), a / (hi - lo) * h1, atol=1e-15)
    np.testing.assert_allclose(d2.numpy(), (a / (hi - lo)) ** 2 * h2, atol=1e-15)


def test_ma_weights_are_asymmetric_and_detached():
    Ld = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    Lp = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    Lb = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    eps = REG["ma_loss_weights_epsilon"]
    w = nv.ma_loss_weights(Ld, Lp, Lb, eps)
    sig = lambda t: 1.0 / (1.0 + math.exp(-t))  # noqa: E731
    assert w["data"].item() == pytest.approx(0.1 + 0.9 * sig((2 + 3 - 1) / (1 + eps)), rel=1e-15)
    assert w["phys"].item() == pytest.approx(0.1 + 0.9 * sig((1 - 2) / (2 + eps)), rel=1e-15)
    assert w["bc"].item() == pytest.approx(0.1 + 0.9 * sig((1 - 3) / (3 + eps)), rel=1e-15)
    symmetric_phys = 0.1 + 0.9 * sig((1 + 3 - 2) / (2 + eps))   # the old "sum of the other two" shorthand
    assert abs(w["phys"].item() - symmetric_phys) > 0.1
    assert not any(v.requires_grad for v in w.values())
    assert all(0.1 < v.item() < 1.0 for v in w.values())


def test_ma_weights_from_python_scalars_are_cpu_float64():
    eps = REG["ma_loss_weights_epsilon"]
    w = nv.ma_loss_weights(1e-2, 1e-4, 1e-3, eps)        # the guide's E7 example, plain floats
    assert all(v.dtype == torch.float64 and v.device.type == "cpu" for v in w.values())
    sig = lambda t: 1.0 / (1.0 + math.exp(-t))  # noqa: E731
    assert w["data"].item() == pytest.approx(0.1 + 0.9 * sig((1e-4 + 1e-3 - 1e-2) / (1e-2 + eps)), rel=1e-14)
    assert w["phys"].item() == pytest.approx(0.1 + 0.9 * sig((1e-2 - 1e-4) / (1e-4 + eps)), rel=1e-14)
    assert w["bc"].item() == pytest.approx(0.1 + 0.9 * sig((1e-2 - 1e-3) / (1e-3 + eps)), rel=1e-14)
    assert w["data"].item() == pytest.approx(0.362, abs=5e-4)          # about 0.36, 1.00, 1.00
    assert w["phys"].item() == pytest.approx(1.0, abs=1e-12) and w["bc"].item() == pytest.approx(0.99989, abs=1e-5)
    w32 = nv.ma_loss_weights(torch.tensor(1e-2, dtype=torch.float32), 1e-4, 1e-3, eps)
    assert all(v.dtype == torch.float64 for v in w32.values())


# ---------------------------------------------------------------------------
# Exact quasi-1D nozzle (test geometry: gamma 1.4, 81 points)
# ---------------------------------------------------------------------------

NZ = nv.QuasiOneD(1.4, 1.0, 1.0, 1.0, (-1.0, 1.0), 81, 1e-13)


@pytest.mark.parametrize("M,branch", [(0.05, "subsonic"), (0.3, "subsonic"), (0.7, "subsonic"),
                                      (0.99, "subsonic"), (1.01, "supersonic"), (1.5, "supersonic"),
                                      (2.5, "supersonic"), (4.0, "supersonic")])
def test_exact_branch_inversion(M, branch):
    ratio = float(nv.area_mach_ratio(M, 1.4))
    assert nv.mach_from_area_ratio(ratio, 1.4, branch, 1e-13) == pytest.approx(M, rel=1e-11)


def test_choking_limit():
    assert nv.mach_from_area_ratio(1.0, 1.33, "subsonic", 1e-13) == 1.0
    assert nv.mach_from_area_ratio(1.0, 1.33, "supersonic", 1e-13) == 1.0
    prev = None
    for d in (1e-2, 1e-4, 1e-6, 1e-8):
        sub = nv.mach_from_area_ratio(1.0 + d, 1.33, "subsonic", 1e-13)
        sup = nv.mach_from_area_ratio(1.0 + d, 1.33, "supersonic", 1e-13)
        assert sub < 1.0 < sup
        gap = max(1.0 - sub, sup - 1.0)
        assert gap <= 3.0 * math.sqrt(d)
        assert prev is None or gap < prev
        prev = gap
    with pytest.raises(ValueError):
        nv.mach_from_area_ratio(0.999, 1.33, "subsonic", 1e-13)


def test_mass_and_enthalpy_conservation_isentropic():
    sm = NZ.smooth_subsonic(0.25)["errors"]
    ch = NZ.choked_isentropic()["errors"]
    for e in (sm, ch):
        assert e["total_enthalpy_rel"] <= 1e-13 and e["total_pressure_rel"] <= 1e-13
    assert sm["mass_flow_rel"] <= 1e-11 and sm["max_mach"] < 1.0
    assert ch["mass_flow_vs_choked_rel"] <= 1e-11 and ch["branches_ok"] and ch["throat_mach_abs"] == 0.0


def test_smooth_case_refuses_choking_inlet_mach():
    with pytest.raises(ValueError):
        NZ.smooth_subsonic(0.6)


def test_rational_normal_shock_oracle():
    s = nv.normal_shock(2.0, 1.4)
    assert s["M2"] == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-15)
    assert s["p_ratio"] == pytest.approx(float(Fraction(9, 2)), rel=1e-15)
    assert s["rho_ratio"] == pytest.approx(float(Fraction(8, 3)), rel=1e-15)
    assert s["T_ratio"] == pytest.approx(float(Fraction(27, 16)), rel=1e-15)
    assert nv.shock_oracle(REG["nozzle"]["oracle"], 1e-14)["status"] == "PASS"
    with pytest.raises(ValueError):
        nv.shock_oracle(dict(REG["nozzle"]["oracle"], pressure_ratio=4.4), 1e-14)
    with pytest.raises(ValueError):
        nv.normal_shock(0.9, 1.4)
    assert nv.normal_shock(1.0, 1.33) == {"M2": 1.0, "p_ratio": 1.0, "rho_ratio": 1.0, "T_ratio": 1.0,
                                          "p0_ratio": 1.0}


def test_back_pressure_inversion_recovers_shock():
    for xr in (0.2, 0.5, 0.9):
        pb = NZ.exit_pressure(xr)
        xs = NZ.invert_back_pressure(pb)
        sc = nv.shock_scores(NZ, xs, pb, xr)
        assert sc["position_abs_error"] <= 1e-9
        for k in nv.SHOCK_ERROR_KEYS:
            assert sc[k] <= 1e-10, k
        assert 0.0 < sc["total_pressure_loss"] < 1.0 and sc["M2_branch"] < 1.0 < sc["M1"]


def test_back_pressure_range_endpoints_and_jump_convention():
    lo, hi = NZ.internal_shock_range()
    assert NZ.invert_back_pressure(hi) == 0.0 and NZ.invert_back_pressure(lo) == NZ.x_exit
    for xs, pb in ((0.0, hi), (NZ.x_exit, lo)):
        sc = nv.shock_scores(NZ, NZ.invert_back_pressure(pb), pb, xs)
        assert sc["position_abs_error"] == 0.0
        for k in nv.SHOCK_ERROR_KEYS:
            assert sc[k] <= 1e-10, (xs, k)
    throat = nv.shock_scores(NZ, 0.0, hi, 0.0)                      # sonic null shock: no loss
    assert throat["p02_over_p01"] == 1.0 and throat["total_pressure_loss"] == 0.0
    prof = NZ.shock_profile(NZ.x_exit)                              # a point at x == xs is downstream
    st, down = prof["state"], prof["jump"]["down"]
    assert st["M"][-1] < 1.0 < st["M"][-2]
    assert st["p"][-1] == pytest.approx(float(down["p"]), rel=1e-12) and st["p"][-1] > st["p"][-2]
    assert prof["p0_local"][-1] == pytest.approx(NZ.p0 * prof["jump"]["shock"]["p0_ratio"], rel=1e-15)


def test_higher_back_pressure_moves_shock_upstream():
    pe = [NZ.exit_pressure(x) for x in np.linspace(0.0, 1.0, 21)]
    assert np.all(np.diff(pe) < 0.0)
    lo, hi = NZ.internal_shock_range()
    pbs = np.linspace(lo, hi, 7)[1:-1]
    xs = [NZ.invert_back_pressure(p) for p in pbs]
    assert np.all(np.diff(xs) < 0.0)


@pytest.mark.parametrize("factor", [1.0 + 1e-6, 1.5])
def test_invalid_back_pressure_rejected(factor):
    lo, hi = NZ.internal_shock_range()
    with pytest.raises(ValueError):
        NZ.invert_back_pressure(hi * factor)
    with pytest.raises(ValueError):
        NZ.invert_back_pressure(lo / factor)
    with pytest.raises(ValueError):
        NZ.invert_back_pressure(float("nan"))


@pytest.mark.parametrize("g", [1.0, 0.9, float("nan")])
def test_invalid_gamma_rejected(g):
    with pytest.raises(ValueError):
        nv.QuasiOneD(g, 1.0, 1.0, 1.0, (-1.0, 1.0), 11, 1e-13)


def test_complete_input_identity_and_duplicates():
    mk = lambda pb, name="c": nv.ShockCase(name, 1.4, 1.0, 1.0, 1.0, nv.AREA_EXPR, (-1.0, 1.0), 81, pb)  # noqa: E731
    pb = NZ.exit_pressure(0.5)
    a, b = mk(pb), mk(pb)
    assert a.identity() == b.identity() and a.identity() != mk(pb * (1 + 1e-15)).identity()
    assert len(nv.validate_cases([a, b])) == 1                      # consistent duplicate collapses
    with pytest.raises(ValueError):
        nv.validate_cases([a, mk(pb * 1.01)])                       # same case name, different input
    with pytest.raises(ValueError):
        nv.validate_cases([mk(-1.0)])
    x1 = nv.QuasiOneD.from_case(a, 1e-13).invert_back_pressure(a.back_pressure)
    x2 = nv.QuasiOneD.from_case(b, 1e-13).invert_back_pressure(b.back_pressure)
    assert x1 == x2


# ---------------------------------------------------------------------------
# Runner: resource refusal and write-once behaviour
# ---------------------------------------------------------------------------

AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; charging"
BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging"
MAC_PY = "/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
# ``ps -Ao pid=,args=`` text; a comm column would truncate MAC_PY to "/Library/Framewo"
PS = ("  101 /Users/x/p/.venv/bin/python scripts/phase8/benchmark.py --arm 2\n"
      "  102 bash -c until grep -q \"queue done\" q.log; do sleep 120; done; "
      ".venv/bin/python scripts/phase8/ablation_ladder.py --step A2\n"
      "  103 caffeinate -i bash -c .venv/bin/python scripts/phase8/ablation_ladder.py --step A2\n"
      "  104 /usr/bin/python3 -m http.server\n"
      f"  105 {MAC_PY} /Users/x/p/scripts/phase8/ablation_ladder.py --step A2\n"
      "  106 /Users/x/p/.venv/bin/python3.12 -u -X faulthandler -m scripts.phase8.benchmark --arm 3\n"
      f"  107 {MAC_PY} /Users/x/p/scripts/dispatch_claude.py\n"
      "  108 .venv/bin/python -m pytest tests/test_calibration_v6.py -q\n"
      "  109 /Users/x/p/.venv/bin/python scripts/optimization/calibrate_lto.py\n"
      "  110 /Applications/Code Helper (Plugin).app/Contents/MacOS/Code Helper (Plugin) --type=x\n")
FAKE_HEAD = "0123456789abcdef0123456789abcdef01234567"
REG_ARG = ["--registration", "docs/phase8_track4_registration.json"]


def test_resource_checks_distinguish_active_jobs_from_parked_shells():
    assert rd.on_mains(AC) and not rd.on_mains(BATT) and not rd.on_mains(None)
    hits = [h.split()[0] for h in rd.heavy_python_processes(PS, own_pid=999)]
    assert hits == ["101", "105", "106", "109"]          # script, Mac framework Python, module form, calibration
    assert [h.split()[0] for h in rd.heavy_python_processes(PS, own_pid=101)] == ["105", "106", "109"]
    assert rd.python_target(f"{MAC_PY} /a/b.py") == "/a/b.py"
    assert rd.python_target("bash -c .venv/bin/python scripts/phase8/benchmark.py") is None
    quiet = "\n".join(line for line in PS.splitlines() if line.split()[0] in ("102", "103", "104", "107", "108"))
    assert rd.resource_blockers(AC, quiet, 999, 15, 15) == []
    assert len(rd.resource_blockers(BATT, PS, 999, 0, 15)) == 3


def test_resource_checks_fail_closed_on_unreadable_commands(monkeypatch):
    for ps in (None, "", "   \n"):
        assert any("process list unreadable" in b for b in rd.resource_blockers(AC, ps, 999, 15, 15))
    calls = []
    monkeypatch.setattr(rd, "_cmd", lambda args: calls.append(args) or None)
    blockers = rd.live_blockers(15)
    assert any("mains" in b for b in blockers) and any("process list unreadable" in b for b in blockers)
    assert ["ps", "-Ao", "pid=,args="] in calls                     # no truncating comm column
    assert rd.git_head() is None
    assert rd.dirty_sources(ROOT / "docs" / "phase8_track4_registration.json") is None
    monkeypatch.setattr(rd, "_cmd", lambda args: "fatal: not a git repository\n")
    assert rd.git_head() is None


def test_runner_refuses_when_blocked_and_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "attempt"
    args = REG_ARG + ["--output-dir", str(out)]
    monkeypatch.setattr(rd, "git_head", lambda: FAKE_HEAD)
    monkeypatch.setattr(rd, "live_blockers", lambda nice: ["not on mains"])
    monkeypatch.setattr(rd, "dirty_sources", lambda p: "")
    assert rd.main(args) == 3 and not out.exists()
    monkeypatch.setattr(rd, "live_blockers", lambda nice: [])
    monkeypatch.setattr(rd, "dirty_sources", lambda p: "?? scripts/phase8/pinn_diagnostics/x.py\n")
    assert rd.main(args) == 3 and not out.exists()
    monkeypatch.setattr(rd, "dirty_sources", lambda p: None)          # git status unreadable
    assert rd.main(args) == 3 and not out.exists()
    monkeypatch.setattr(rd, "dirty_sources", lambda p: "")
    monkeypatch.setattr(rd, "git_head", lambda: None)                 # HEAD missing
    assert rd.main(args) == 3 and not out.exists()
    monkeypatch.undo()
    monkeypatch.setattr(rd, "_cmd", lambda args: None)                # pmset, ps and git all fail
    assert rd.main(args) == 3 and not out.exists()


def test_runner_refuses_existing_output(tmp_path, monkeypatch):
    out = tmp_path / "attempt"
    out.mkdir()
    (out / "report.json").write_text("{}")
    monkeypatch.setattr(rd, "git_head", lambda: FAKE_HEAD)
    monkeypatch.setattr(rd, "live_blockers", lambda nice: [])
    monkeypatch.setattr(rd, "dirty_sources", lambda p: "")
    assert rd.main(REG_ARG + ["--output-dir", str(out)]) == 2
    assert (out / "report.json").read_text() == "{}"


def test_write_once_helpers(tmp_path):
    d = rd.create_output_dir(tmp_path / "a" / "b")
    with pytest.raises(FileExistsError):
        rd.create_output_dir(d)
    rd.write_once(d / "x.txt", "1")
    with pytest.raises(FileExistsError):
        rd.write_once(d / "x.txt", "2")
    assert (d / "x.txt").read_text() == "1"


def test_envelope_read_round_trip_from_start_bytes():
    raw = (ROOT / REG["inputs"]["synthetic_envelope"]).read_bytes()
    assert rd.check_inputs(REG, raw)["box_equals_envelope_extent"]
    with pytest.raises(ValueError):
        rd.check_inputs(REG, raw + b"\n")                            # not the registered bytes
    with pytest.raises(ValueError):
        rd.check_inputs(REG, None)


def _identity():
    return {"git_head": FAKE_HEAD, "registration_path": "docs/phase8_track4_registration.json",
            "registration_sha256": "r", "source_sha256": {s: "s" for s in rd.SOURCES},
            "synthetic_envelope": REG["inputs"]["synthetic_envelope"], "synthetic_envelope_sha256": "e",
            "synthetic_envelope_error": None}


def test_aggregate_status_and_identity_drift():
    assert rd.aggregate_status(["PASS", "PASS"]) == "PASS"
    assert rd.aggregate_status(["PASS", "BLOCKED"]) == "BLOCKED"
    assert rd.aggregate_status(["BLOCKED", "FAIL"]) == "FAIL"
    assert rd.aggregate_status(["FAIL", "ERROR", "PASS"]) == "ERROR"
    assert rd.aggregate_status([]) == "ERROR" and rd.aggregate_status(["PASS", None]) == "ERROR"
    start = _identity()
    assert rd.identity_drift(start, dict(start)) == []
    moved = dict(start, git_head="f" * 40, source_sha256=dict(start["source_sha256"], **{"turbine_map.py": "b"}))
    assert rd.identity_drift(start, moved) == ["git_head", "source_sha256.turbine_map.py"]
    assert rd.identity_drift(start, {"error": "gone"}) == ["gone"]


def _mms(status):
    return {"status": status, "max_abs_forced_residual": 0.0, "max_relative_forcing_disagreement": 0.0,
            "per_equation": {}, "negative_control_discrepancy": {"dissipation": 1.0}}


def _raise(exc):
    def f(*_a, **_k):
        raise exc
    return f


@pytest.fixture
def no_global_torch_state(monkeypatch):
    """execute() sets thread counts and deterministic mode; keep them out of the test session."""
    for name in ("set_num_threads", "set_num_interop_threads", "use_deterministic_algorithms"):
        monkeypatch.setattr(torch, name, lambda *_a, **_k: None)


REG_PATH = ROOT / "docs" / "phase8_track4_registration.json"


def test_execute_non_pass_exits_nonzero_and_blocks_nozzle_ladder(tmp_path, monkeypatch, no_global_torch_state):
    """Mocked rungs only: a turbine error is confined, a failed MMS activation blocks the ladder."""
    ident = _identity()
    monkeypatch.setattr(rd, "end_identity", lambda p, reg: dict(ident))
    monkeypatch.setattr(tm, "run", _raise(RuntimeError("mock turbine")))
    monkeypatch.setattr(nv, "verify_manufactured", lambda m, act: _mms("PASS" if act == "tanh" else "FAIL"))
    monkeypatch.setattr(nv, "run_ladder", _raise(AssertionError("ladder ran")))
    out = rd.create_output_dir(tmp_path / "a")
    assert rd.execute(REG, REG_PATH, ident, b"", out) == 1
    rep = json.loads((out / "report.json").read_text())
    assert rep["status"] == "ERROR" and rep["turbine"]["status"] == "ERROR"
    assert rep["manufactured"]["silu"]["status"] == "FAIL"
    assert set(rep["nozzle"]) == set(nv.RUNGS) and all(r["status"] == "BLOCKED" for r in rep["nozzle"].values())
    assert rep["start_identity"] == ident and rep["identity_drift"] == []
    assert json.loads((out / "config.json").read_text())["start_identity"] == ident
    assert "Aggregate status: ERROR" in (out / "report.md").read_text()
    assert "EXIT code 1" in (out / "run_log.txt").read_text() and (out / "hashes.json").exists()


def test_execute_pass_only_when_every_rung_passes_and_drift_is_error(tmp_path, monkeypatch, no_global_torch_state):
    ident = _identity()
    monkeypatch.setattr(rd, "check_inputs", lambda reg, b: {})
    monkeypatch.setattr(rd, "markdown_report", lambda rep: "mock")
    monkeypatch.setattr(tm, "run", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(nv, "verify_manufactured", lambda m, act: _mms("PASS"))
    monkeypatch.setattr(nv, "run_ladder", lambda reg: {k: {"status": "PASS"} for k in nv.RUNGS})
    monkeypatch.setattr(rd, "end_identity", lambda p, reg: dict(ident))
    assert rd.execute(REG, REG_PATH, ident, b"", rd.create_output_dir(tmp_path / "ok")) == 0
    monkeypatch.setattr(rd, "end_identity", lambda p, reg: dict(ident, registration_sha256="changed"))
    out = rd.create_output_dir(tmp_path / "drift")
    assert rd.execute(REG, REG_PATH, ident, b"", out) == 1
    rep = json.loads((out / "report.json").read_text())
    assert rep["status"] == "ERROR" and rep["identity_drift"] == ["registration_sha256"]
    assert rep["start_identity"]["registration_sha256"] == "r"           # start identity is kept


def test_execute_setup_failure_is_logged_and_hashed(tmp_path, monkeypatch, no_global_torch_state):
    monkeypatch.setattr(rd, "environment", _raise(OSError("mock environment failure")))
    out = rd.create_output_dir(tmp_path / "a")
    assert rd.execute(REG, REG_PATH, _identity(), None, out) == 1
    log = (out / "run_log.txt").read_text()
    assert "ERROR OSError('mock environment failure')" in log and "EXIT code 1" in log
    hashes = json.loads((out / "hashes.json").read_text())
    assert set(hashes) == {"config.json", "run_log.txt"} and not (out / "report.json").exists()
