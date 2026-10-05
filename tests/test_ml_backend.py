"""Tiny portable fixtures; no study datasets or project training jobs."""
import json

import numpy as np
import pytest

from simulation.ml_backend import get_backend, mse_loss, resolve_backend, weighted_loss
from simulation.ml_backend.interface import MLPSpec, seeded_parameters, validate_parameters
from simulation.ml_backend.numpy_reference import forward, input_derivatives, mse_parameter_gradients


@pytest.fixture
def torch_backend():
    pytest.importorskip("torch")
    return get_backend("torch")


@pytest.fixture
def mlx_backend():
    pytest.importorskip("mlx.core")
    return get_backend("mlx", device="cpu")


def fixture_data():
    rng = np.random.default_rng(831)
    return rng.normal(size=(4, 3)), rng.normal(size=(4, 2))


def assert_relative_error(actual, reference, tolerance):
    """A family norm bound, independent of supplementary elementwise checks."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    reference_norm = np.linalg.norm(reference)
    error_norm = np.linalg.norm(actual-reference)
    error = error_norm/max(reference_norm, np.finfo(np.float64).tiny)
    assert error <= tolerance, f"relative norm error {error:.17g} exceeds {tolerance:.17g}"
    return error


def test_selector_explicit_environment_and_default(monkeypatch):
    monkeypatch.delenv("CATJET_ML_BACKEND", raising=False)
    assert resolve_backend() == "torch"
    monkeypatch.setenv("CATJET_ML_BACKEND", "MLX")
    assert resolve_backend() == "mlx"
    assert resolve_backend("torch") == "torch"
    with pytest.raises(ValueError, match="Unknown ML backend"):
        resolve_backend("unknown")


def test_parameter_validation_and_shared_weights():
    spec = MLPSpec(3, (5, 4), 2, "silu")
    params = seeded_parameters(spec, 7)
    assert validate_parameters(params, spec) == [(3, 5), (5, 4), (4, 2)]
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            MLPSpec(size, (4,), 2)
    malformed = dict(params)
    malformed.pop("layers.0.bias")
    with pytest.raises(ValueError, match="canonical"):
        validate_parameters(malformed)
    malformed = dict(params, **{"layers.0.bias": np.zeros(6)})
    with pytest.raises(ValueError, match="dimensions"):
        validate_parameters(malformed)
    malformed = dict(params)
    malformed["layers.0.weight"] = params["layers.0.weight"].copy()
    malformed["layers.0.weight"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_parameters(malformed)


@pytest.mark.parametrize("activation", ["silu", "tanh"])
def test_torch64_independent_forward_loss_parameter_and_input_gradients(torch_backend, activation):
    b = torch_backend
    x, target = fixture_data()
    model = b.mlp(3, (5, 4), 2, activation=activation, seed=17)
    params = {name: b.to_numpy(value) for name, value in b.parameters(model).items()}
    assert all(value.dtype == np.float64 for value in params.values())
    reference = forward(params, x, activation=activation)
    reference_loss, reference_grads = mse_parameter_gradients(params, x, target, activation=activation)
    prediction = b.forward(model, x)
    loss, grads = b.value_and_grad(model, lambda m: mse_loss(m(b.array(x)), b.array(target), b.xp))
    np.testing.assert_allclose(b.to_numpy(prediction), reference, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b.to_numpy(loss), reference_loss, rtol=1e-10, atol=1e-10)
    assert_relative_error(b.to_numpy(prediction), reference, 1e-10)
    assert_relative_error(b.to_numpy(loss), reference_loss, 1e-10)
    for name in reference_grads:
        np.testing.assert_allclose(b.to_numpy(grads[name]), reference_grads[name], rtol=1e-10, atol=1e-10)
        assert_relative_error(b.to_numpy(grads[name]), reference_grads[name], 1e-10)
    jac, hessian = input_derivatives(params, x, activation=activation)
    first = b.input_grad(lambda a: model(a)[:, 0], b.array(x))
    second = b.input_grad(lambda a: model(a)[:, 0], b.array(x), order=2)
    np.testing.assert_allclose(b.to_numpy(first), jac[:, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b.to_numpy(second), np.diagonal(hessian[:, 0], axis1=-2, axis2=-1), rtol=1e-10, atol=1e-10)
    assert_relative_error(b.to_numpy(first), jac[:, 0], 1e-10)
    assert_relative_error(b.to_numpy(second), np.diagonal(hessian[:, 0], axis1=-2, axis2=-1), 1e-10)
    point_jac = b.jacobian(model, b.array(x[0]))
    point_hessian = b.jacobian(lambda a: b.jacobian(model, a), b.array(x[0]))
    np.testing.assert_allclose(b.to_numpy(point_jac), jac[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b.to_numpy(point_hessian), hessian[0], rtol=1e-10, atol=1e-10)
    assert_relative_error(b.to_numpy(point_jac), jac[0], 1e-10)
    assert_relative_error(b.to_numpy(point_hessian), hessian[0], 1e-10)


def test_torch_second_derivative_diagonal_for_coupled_features(torch_backend):
    b = torch_backend
    x = b.array([.25, -.4])
    fn = lambda a: a[0]**2*a[1]+a[0]*a[1]**2
    np.testing.assert_allclose(b.to_numpy(b.input_grad(fn, x, order=2)), [-.8, .5], atol=1e-14)
    np.testing.assert_array_equal(b.to_numpy(b.input_grad(lambda a: b.array(3.), x, order=2)), [0, 0])


def test_torch_parameter_graph_survives_input_second_derivatives(torch_backend):
    b = torch_backend
    model = b.mlp(2, (3,), 1, activation="tanh", seed=19)
    x = np.asarray([[.2, -.3], [.4, .1]])
    loss_fn = lambda m: b.xp.mean(b.input_grad(lambda a: m(a)[:, 0], b.array(x), order=2)**2)
    value, grads = b.value_and_grad(model, loss_fn)
    params = {name: b.to_numpy(parameter) for name, parameter in b.parameters(model).items()}
    def independent_loss(weights):
        _, hess = input_derivatives(weights, x, activation="tanh")
        return np.mean(np.diagonal(hess[:, 0], axis1=-2, axis2=-1)**2)
    assert float(b.to_numpy(value)) > 0
    for name, gradient in grads.items():
        numerical = np.zeros_like(params[name])
        for index in np.ndindex(numerical.shape):
            plus = {k: v.copy() for k, v in params.items()}
            minus = {k: v.copy() for k, v in params.items()}
            plus[name][index] += 1e-6; minus[name][index] -= 1e-6
            numerical[index] = (independent_loss(plus)-independent_loss(minus))/2e-6
        np.testing.assert_allclose(b.to_numpy(gradient), numerical, rtol=1e-6, atol=1e-9)


def test_torch_seeded_native_model_does_not_change_global_rng(torch_backend):
    b = torch_backend
    state = b.torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    model_a = b.mlp(3, (5,), 2, seed=13)
    model_b = b.mlp(3, (5,), 2, seed=13)
    assert b.torch.equal(state, b.torch.random.get_rng_state())
    new_numpy_state = np.random.get_state()
    assert numpy_state[0] == new_numpy_state[0]
    np.testing.assert_array_equal(numpy_state[1], new_numpy_state[1])
    assert numpy_state[2:] == new_numpy_state[2:]
    for name, value in b.parameters(model_a).items():
        np.testing.assert_array_equal(b.to_numpy(value), b.to_numpy(b.parameters(model_b)[name]))


def test_neutral_export_preserves_torch64_and_cpu64_score(torch_backend, tmp_path):
    b = torch_backend
    x, _ = fixture_data()
    model = b.mlp(3, (5, 4), 2, activation="tanh", seed=5)
    path = b.save_npz(model, tmp_path/"weights.npz", metadata={"seed": 5})
    with np.load(path, allow_pickle=False) as weights:
        assert all(weights[name].dtype == np.float64 for name in weights.files if name.startswith("layers."))
        record = json.loads(weights["__metadata__"].item())
        assert record["activation"] == "tanh" and record["dtype"] == "float64"
        assert record["metadata"]["seed"] == 5
    restored = b.load_npz(path)
    np.testing.assert_array_equal(b.to_numpy(b.forward(restored, x)), b.to_numpy(b.forward(model, x)))
    score = b.score_numpy64(restored, x)
    assert score.dtype == np.float64
    np.testing.assert_allclose(score, b.to_numpy(b.forward(model, x)), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(b.score64(restored, x), score, atol=1e-10, rtol=1e-10)
    params = {n: b.to_numpy(v) for n, v in b.parameters(restored).items()}
    np.testing.assert_allclose(b.score64(params, x, activation="tanh"), score, atol=1e-10, rtol=1e-10)
    assert b.info()["score_backend"] == "torch"
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        b.save_npz(model, path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="activation"):
        b.load_npz(path, activation="silu")


def test_neutral_load_refuses_broken_parameters(torch_backend, tmp_path):
    params = seeded_parameters(MLPSpec(2, (3,), 1), 0)
    np.savez(tmp_path/"extra.npz", **params, unrelated=np.zeros(2))
    with pytest.raises(ValueError, match="Unknown"):
        torch_backend.load_npz(tmp_path/"extra.npz")
    params["layers.1.weight"] = np.ones((1, 4))
    np.savez(tmp_path/"broken.npz", **params)
    with pytest.raises(ValueError, match="connect"):
        torch_backend.load_npz(tmp_path/"broken.npz")


def test_torch_adam_and_optional_lbfgs(torch_backend):
    b = torch_backend
    x, target = fixture_data()
    def tiny_run():
        model = b.mlp(3, (4,), 2, seed=23)
        fn = lambda m: b.mse(m(b.array(x)), b.array(target))
        optimizer = b.adam(model, lr=.01, betas=(.8, .95), eps=1e-7, weight_decay=0., amsgrad=False)
        before = float(b.to_numpy(fn(model)))
        for _ in range(3):
            b.step(model, optimizer, fn)
        assert float(b.to_numpy(fn(model))) < before
        return model
    a, c = tiny_run(), tiny_run()
    for name in b.parameters(a):
        np.testing.assert_array_equal(b.to_numpy(b.parameters(a)[name]), b.to_numpy(b.parameters(c)[name]))
    calls = []
    fn = lambda m: (calls.append(1), b.mse(m(b.array(x)), b.array(target)))[1]
    before = float(b.to_numpy(fn(a)))
    b.step(a, b.lbfgs(a, lr=.1, max_iter=3, max_eval=5, history_size=3), fn)
    assert len(calls) > 1
    assert float(b.to_numpy(fn(a))) < before


def test_shared_loss_weights(torch_backend):
    b = torch_backend
    terms = {"a": b.array(2.), "b": b.array(3.)}
    assert float(weighted_loss(terms, {"a": .5, "b": 2.})) == 7
    with pytest.raises(ValueError):
        weighted_loss(terms, {"a": 1})
    values = b.array([-1000., 0., 25., 1000.])
    np.testing.assert_allclose(b.to_numpy(b.xp.softplus(values)), np.logaddexp(b.to_numpy(values), 0), rtol=1e-10, atol=1e-10)
    assert b.xp.training_dtype == b.torch.float64


def test_explicit_native_and_numpy_seed(torch_backend):
    b = torch_backend
    b.seed(73)
    native = b.torch.rand(5)
    numpy = np.random.uniform(size=5)
    b.seed(73)
    np.testing.assert_array_equal(b.to_numpy(native), b.to_numpy(b.torch.rand(5)))
    np.testing.assert_array_equal(numpy, np.random.uniform(size=5))
    with pytest.raises(ValueError, match="Seed"):
        b.seed(-1)


@pytest.mark.parametrize("activation", ["silu", "tanh"])
def test_real_mlx_torch32_shared_outputs_losses_parameter_and_input_derivatives(mlx_backend, activation):
    pytest.importorskip("torch")
    torch = get_backend("torch", dtype="float32")
    mlx = mlx_backend
    x, target = fixture_data()
    models = [b.mlp(3, (5, 4), 2, activation=activation, seed=37) for b in (torch, mlx)]
    results = []
    for b, model in zip((torch, mlx), models):
        fn = lambda m: b.mse(m(b.array(x)), b.array(target))
        loss, grads = b.value_and_grad(model, fn)
        first = b.input_grad(lambda a: model(a)[:, 0], b.array(x))
        second = b.input_grad(lambda a: model(a)[:, 0], b.array(x), order=2)
        jac = b.jacobian(model, b.array(x[0]))
        hess = b.jacobian(lambda a: b.jacobian(model, a), b.array(x[0]))
        results.append((b.to_numpy(model(b.array(x))), b.to_numpy(loss),
                        {n: b.to_numpy(g) for n, g in grads.items()},
                        b.to_numpy(first), b.to_numpy(second), b.to_numpy(jac), b.to_numpy(hess)))
    for a, c in zip(results[0], results[1]):
        if isinstance(a, dict):
            assert set(a) == set(c)
            for name in a:
                np.testing.assert_allclose(a[name], c[name], rtol=1e-5, atol=1e-5)
                assert_relative_error(a[name], c[name], 1e-5)
        else:
            np.testing.assert_allclose(a, c, rtol=1e-5, atol=1e-5)
            assert_relative_error(a, c, 1e-5)
    for name in torch.parameters(models[0]):
        np.testing.assert_array_equal(torch.to_numpy(torch.parameters(models[0])[name]), mlx.to_numpy(mlx.parameters(models[1])[name]))


def test_real_mlx_torch32_second_derivative_loss_parameter_gradients(mlx_backend):
    pytest.importorskip("torch")
    results = []
    for b in (get_backend("torch", dtype="float32"), mlx_backend):
        model = b.mlp(2, (4,), 1, activation="tanh", seed=39)
        x = b.array([[.2, -.3], [.4, .1]])
        value, gradients = b.value_and_grad(model, lambda m: b.xp.mean(b.input_grad(lambda a: m(a)[:, 0], x, order=2)**2))
        results.append((b.to_numpy(value), {n: b.to_numpy(g) for n, g in gradients.items()}))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-5)
    assert_relative_error(results[0][0], results[1][0], 1e-5)
    for name in results[0][1]:
        np.testing.assert_allclose(results[0][1][name], results[1][1][name], rtol=1e-5, atol=1e-5)
        assert_relative_error(results[0][1][name], results[1][1][name], 1e-5)


def test_real_mlx_neutral_export_and_shared_adam(mlx_backend, tmp_path):
    pytest.importorskip("torch")
    torch = get_backend("torch", dtype="float32")
    mlx = mlx_backend
    x, target = fixture_data()
    models = [b.mlp(3, (4,), 2, seed=41) for b in (torch, mlx)]
    for b, model in zip((torch, mlx), models):
        optimizer = b.adam(model, lr=.01, betas=(.8, .95), eps=1e-7, weight_decay=0., amsgrad=False)
        for _ in range(3):
            b.step(model, optimizer, lambda m: b.mse(m(b.array(x)), b.array(target)))
    for name in torch.parameters(models[0]):
        np.testing.assert_allclose(torch.to_numpy(torch.parameters(models[0])[name]), mlx.to_numpy(mlx.parameters(models[1])[name]), rtol=1e-5, atol=1e-5)
        assert_relative_error(torch.to_numpy(torch.parameters(models[0])[name]), mlx.to_numpy(mlx.parameters(models[1])[name]), 1e-5)
    path = mlx.save_npz(models[1], tmp_path/"mlx.npz")
    with np.load(path, allow_pickle=False) as weights:
        assert all(weights[n].dtype == np.float32 for n in weights.files if n.startswith("layers."))
    torch64 = get_backend("torch")
    restored = torch64.load_npz(path)
    assert all(v.dtype == torch64.torch.float64 for v in torch64.parameters(restored).values())
    np.testing.assert_allclose(mlx.score_numpy64(models[1], x), torch64.to_numpy(torch64.forward(restored, x)), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mlx.score64(models[1], x), torch64.score64(restored, x), rtol=1e-10, atol=1e-10)
    params = {n:mlx.to_numpy(v) for n,v in mlx.parameters(models[1]).items()}
    np.testing.assert_array_equal(mlx.score64(models[1], x), mlx.score64(params, x))
    with pytest.raises(ValueError, match="float32"):
        get_backend("mlx", dtype="float64")
    with pytest.raises(NotImplementedError, match="weight_decay"):
        mlx.adam(models[1], weight_decay=.1)


def test_real_mlx_vector_second_derivative_diagonal(mlx_backend):
    b = mlx_backend
    x = b.array([.25, -.4])
    fn = lambda a: a[0]**2*a[1]+a[0]*a[1]**2
    np.testing.assert_allclose(b.to_numpy(b.input_grad(fn, x, order=2)), [-.8, .5], atol=1e-6)
    values = b.array([-1000., 0., 25., 1000.])
    np.testing.assert_allclose(b.to_numpy(b.xp.softplus(values)), np.logaddexp(b.to_numpy(values), 0), rtol=1e-5, atol=1e-5)
    b.seed(71)
    first = b.to_numpy(b.mx.random.uniform(shape=(5,)))
    numpy_first = np.random.uniform(size=5)
    b.seed(71)
    np.testing.assert_array_equal(first, b.to_numpy(b.mx.random.uniform(shape=(5,))))
    np.testing.assert_array_equal(numpy_first, np.random.uniform(size=5))
