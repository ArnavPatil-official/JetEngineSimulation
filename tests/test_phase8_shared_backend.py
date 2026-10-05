"""Manufactured integration checks; no project labels or scientific fits."""
from __future__ import annotations

import json
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from simulation.ml_backend import get_backend
from scripts.phase8.saf_surrogate import models
from scripts.phase8.saf_surrogate.train_torch import safetensors_bytes, read_safetensors
from scripts.phase8.nozzle_ode import model as nozzle


def test_default_primary_and_explicit_score_selection(monkeypatch):
    monkeypatch.setenv("CATJET_ML_BACKEND", "mlx")
    network = models.make_model(42, "torch")
    assert next(network.parameters()).dtype == torch.float64
    params = {name:value.detach().numpy() for name,value in network.named_parameters()}
    features = np.random.default_rng(4).normal(size=(2, 12))
    reference = models.forward64(params, features)
    primary = models.forward_cpu64(params, features, "torch", model=network)
    optional = models.forward_cpu64(params, features)  # env selects NumPy64 without importing MLX
    for actual, expected in zip(primary, reference):
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)
    for actual, expected in zip(optional, reference):
        np.testing.assert_array_equal(actual, expected)


def test_torch_neutral_precision_and_crossload(tmp_path):
    selected = get_backend("torch")
    network = selected.mlp(4, (5,), 4, activation="tanh", seed=9)
    with torch.no_grad():
        network.layers[0].weight[0,0] = 1+2**-40
    archive = tmp_path/"neutral.npz"
    selected.save_npz(network, archive)
    restored = selected.load_npz(archive)
    assert restored.spec.activation == "tanh"
    params = {name:selected.to_numpy(value) for name,value in selected.parameters(network).items()}
    for name,value in selected.parameters(restored).items():
        np.testing.assert_array_equal(selected.to_numpy(value), params[name])
    tensors = read_safetensors(safetensors_bytes(params))
    assert tensors["layers.0.weight"][0,0] == 1+2**-40
    assert all(value.dtype == np.float64 for value in tensors.values())


def test_nozzle_function_graph_matches_independent_cpu64_algebra():
    selected = get_backend("torch")
    network = selected.mlp(4, (5, 5), 4, activation="tanh", seed=19)
    raw = selected.array([[-.45,1.07,1.29,287],[.32,10.2,1.34,300]], requires_grad=True)
    field = lambda x:nozzle.evaluate(network, x, "torch")
    actual = nozzle.residuals(field, raw, "torch")
    expected, residual = nozzle.predict_numpy64(nozzle.neutral_parameters(network, "torch"), selected.to_numpy(raw))
    np.testing.assert_allclose(selected.to_numpy(field(raw)), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(selected.to_numpy(actual), residual, rtol=1e-12, atol=1e-12)
    _, gradients = selected.value_and_grad(network, lambda current:(nozzle.residuals(lambda x:nozzle.evaluate(current,x,"torch"),raw,"torch")**2).mean())
    assert all(torch.isfinite(value).all() for value in gradients.values())
    assert any(value.abs().max()>0 for value in gradients.values())
    second = selected.input_grad(lambda x:field(x)[:,0], raw, order=2)
    assert second.shape == raw.shape and second.requires_grad and torch.isfinite(second).all()


@pytest.mark.parametrize("backend",["torch",pytest.param("mlx",marks=pytest.mark.skipif(importlib.util.find_spec("mlx") is None,reason="Optional MLX unavailable"))])
def test_tiny_nozzle_pair_retains_native_counts_and_portable_metadata(tmp_path,backend):
    reg = {"models":{"architecture":{"hidden":[5]},
        "optimizer":{"adam_lr":.001,"adam_betas":[.9,.999],"adam_eps":1e-8,"adam_weight_decay":0,
            "adam_steps":1,"lbfgs_lr":1,"lbfgs_max_iter":1,"lbfgs_max_eval":2,"lbfgs_history_size":3,
            "lbfgs_line_search":"strong_wolfe","lbfgs_tolerance_grad":1e-9,"lbfgs_tolerance_change":1e-12},
        "arms":{"data_only":{"lambda_data":1,"lambda_boundary":1,"lambda_physics":0},
                "physics_on":{"lambda_data":1,"lambda_boundary":1,"lambda_physics":1}}},
        "splits":{"training":{"interior_label_x":[-.5,.5],"boundary_label_x":[-1,1]}}}
    class Reference:
        def profile(self,case,xs):
            return np.tile([1,.5,1,1],(len(xs),1)), {}
    (tmp_path/"training_logs").mkdir();(tmp_path/"checkpoints").mkdir()
    results = nozzle.train_pair(reg, 42, [{"NPR":1.05,"gamma":1.3,"R":287}], Reference(),
        tmp_path, lambda:None, {"fixture":True}, backend=backend)
    assert set(results)=={"data_only","physics_on"}
    for arm in results:
        metadata=json.loads((tmp_path/"checkpoints"/f"{arm}-seed42.json").read_text())
        assert metadata["counts"]["adam_steps"]==1
        assert metadata["counts"]["lbfgs_evaluations"]>=1 if backend=="torch" else metadata["counts"]["lbfgs_evaluations"]==0
        assert metadata["dtype"]==("float64" if backend=="torch" else "float32")
        assert metadata["primary_scientific_procedure"]==(backend=="torch")
        loaded=get_backend("torch").load_npz(tmp_path/"checkpoints"/f"{arm}-seed42.npz")
        assert loaded.spec.activation=="tanh"
        for name,value in get_backend("torch").parameters(loaded).items():
            np.testing.assert_array_equal(value.detach().numpy(),results[arm][name])
