"""Torch backend toy checks on manufactured thermochemistry; no study inputs.

Nothing here reads registered splits, labels, mechanisms or the simulator.
MLX-only checks run where MLX is installed and are skipped elsewhere.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.phase8.saf_surrogate.models import forward64, make_model, output_map
from scripts.phase8.saf_surrogate.thermo import Thermo, TorchOps
from scripts.phase8.saf_surrogate.train import fit_loop, registered_loss
from scripts.phase8.saf_surrogate import train_torch
from scripts.phase8.saf_surrogate.train_torch import (LAYER_SHAPES, TorchBackend, read_safetensors,
                                                      resolve_device)

ROOT = Path(__file__).resolve().parents[1]
HAS_MLX = importlib.util.find_spec("mlx") is not None
K = 492


def manufactured_properties():
    """Fixed synthetic 492-species NASA7 set with nontrivial elements."""
    rng = np.random.default_rng(20261004)
    names = ["N2", "O2", "CO2", "H2O"] + [f"S{i:03d}" for i in range(4, K)]
    mw = np.concatenate(([28.0, 32.0, 44.0, 18.0], rng.uniform(10, 200, K-4)))
    low = np.zeros((K, 7)); high = np.zeros((K, 7))
    low[:, 0] = rng.uniform(2.5, 6.0, K); low[:, 1] = rng.uniform(0, 1e-3, K)
    low[:, 5] = rng.uniform(-5e4, 5e4, K); low[:, 6] = rng.uniform(-5, 20, K)
    high[:] = low; high[:, 0] += rng.uniform(0, .5, K)
    high[:, 5] = low[:, 5]-1000*(high[:, 0]-low[:, 0])  # continuous h at 1000 K
    elements = ["C", "H", "O", "N"]
    matrix = rng.dirichlet(np.ones(4), K)
    matrix[:4] = [[0, 0, 0, 1], [0, 0, 1, 0], [12/44, 0, 32/44, 0], [0, 2/18, 16/18, 0]]
    def simplex(indices):
        y = np.zeros(K); y[indices] = rng.dirichlet(np.ones(len(indices))); return y.tolist()
    air = np.zeros(K); air[0], air[1] = .767, .233
    fuels = {name: {"Y": simplex(rng.choice(np.arange(4, K), 6, replace=False))}
             for name in ("JetA_s", "HEFA_s", "FT_s", "ATJ_s")}
    return {"species_order": names, "molecular_weights_kg_kmol": mw.tolist(),
            "atomic_weights_kg_kmol": [12.011, 1.008, 15.999, 14.007], "elements": elements,
            "element_mass_matrix": matrix.tolist(), "gas_constant_J_kmol_K": 8314.46,
            "coefficients": np.stack((low, high), axis=1).tolist(),
            "temperature_bounds_K": [[200.0, 1000.0, 6000.0]]*K,
            "burner_air_Y": air.tolist(), "compressor_air_Y": air.tolist(),
            "surrogates": {"JetA": "JetA_s", "HEFA": "HEFA_s", "FT": "FT_s", "ATJ": "ATJ_s"},
            "fuels": fuels}


@pytest.fixture(scope="module")
def thermo():
    return Thermo(manufactured_properties())


def toy_states(thermo, rows, seed=7):
    rng = np.random.default_rng(seed)
    fuel = np.zeros((rows, K)); fuel[:, 4:10] = rng.dirichlet(np.ones(6), rows)
    return {"ma": rng.uniform(20, 100, rows), "eta_b": rng.uniform(.998, 1, rows),
            "T3": rng.uniform(600, 900, rows), "fuel_Y": fuel}


def toy_arrays(rows=40, physics=128, seed=11):
    rng = np.random.default_rng(seed)
    Y = rng.dirichlet(np.ones(K)*.05, rows)
    target = np.column_stack((np.log(rng.uniform(.3, 2, rows)), np.log(rng.uniform(1.1, 1.7, rows))))
    P = rng.standard_normal((physics, 12))
    return (rng.standard_normal((rows, 12)), target, Y, P, P-.01, P+.01), np.full(physics, .001)


def numpy_logits(params, features):
    """Pre-output-map float64 forward, the same chain as forward64."""
    value = np.asarray(features, dtype=np.float64)
    for i in range(5):
        value = value @ params[f"layers.{i}.weight"].T + params[f"layers.{i}.bias"]
        if i < 4:
            value = value/(1+np.exp(-value))
    return value


def params64(model):
    return {key: value.detach().cpu().numpy().astype(np.float64) for key, value in model.state_dict().items()}


def torch_inputs(arrays, widths, states, index, pindex):
    X, target, Y, P, Pminus, Pplus = arrays
    to = lambda value: torch.as_tensor(np.asarray(value, dtype=np.float64))
    return ([to(v) for v in (X[index], target[index], Y[index], P[pindex], Pminus[pindex], Pplus[pindex])]
            + [{key: to(value[pindex]) for key, value in states.items()}, to(widths[pindex])])


def test_output_map_uses_amax_and_matches_cpu64():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((9, 494)).astype(np.float32); z[:, 2:] *= 40; z[0, 7] = 900  # overflow-safe softmax
    actual = output_map(torch.as_tensor(z), TorchOps())
    expected = output_map(z.astype(np.float64), np)
    for a, b in zip(actual[:2], expected[:2]):
        assert np.allclose(a.numpy(), b, rtol=1e-6)
    species = actual[2].numpy()
    assert np.isfinite(species).all() and np.allclose(species.sum(axis=1), 1, atol=1e-6)
    assert np.abs(species-expected[2]).sum(axis=1).max() <= 1e-5


def test_residuals_match_numpy64_preserve_graph_and_device(thermo):
    rng = np.random.default_rng(5)
    rows = 32; states = toy_states(thermo, rows)
    ff = rng.uniform(.3, 2, rows); T4 = rng.uniform(1100, 1700, rows); Y = rng.dirichlet(np.ones(K)*.05, rows)
    energy64, element64 = thermo.residuals(ff, T4, Y, states)
    ops = TorchOps("cpu", dtype=torch.float32)  # explicit optional float32 graph diagnostic
    ff_t, T4_t, Y_t = (torch.tensor(v, dtype=torch.float32, requires_grad=True) for v in (ff, T4, Y))
    energy, element = thermo.residuals(ff_t, T4_t, Y_t, {k: torch.as_tensor(v, dtype=torch.float32) for k, v in states.items()}, ops)
    assert energy.device.type == "cpu" and energy.dtype == torch.float32 and energy.requires_grad
    # Float32 tolerance scaled by the largest canceling enthalpy flow term.
    href = thermo.species_h(298.15)
    flow = np.abs(ff+states["ma"])*np.abs(Y*(thermo.species_h(T4[:, None])-href)).sum(axis=-1)
    flow += np.abs(ff+states["ma"])*np.abs(Y*href).sum(axis=-1)
    tolerance = 64*np.finfo(np.float32).eps*flow/(states["ma"]*1e6)
    assert (np.abs(energy.detach().numpy()-energy64) <= tolerance).all()
    assert np.allclose(element.detach().numpy(), element64, atol=1e-6)
    assert np.abs(energy64).max() > 1e-3 and np.abs(element64).max() > 1e-3  # nontrivial residuals
    (energy**2).sum().add((element**2).sum()).backward()
    for value in (ff_t, T4_t, Y_t):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().max() > 0


@pytest.mark.parametrize("arm", ["Mdata", "Mphys"])
def test_registered_loss_parity_with_numpy64(thermo, arm):
    model = make_model(42, "torch", "cpu"); params = params64(model)
    arrays, widths = toy_arrays(); states = toy_states(thermo, 128)
    index, pindex = np.arange(40), np.arange(64)
    with torch.no_grad():
        actual = float(registered_loss(model, arm, thermo, TorchOps(), *torch_inputs(arrays, widths, states, index, pindex)))
    X, target, Y, P, Pminus, Pplus = arrays
    # Primary Torch and the independent reference both use the original float64 inputs.
    cast = lambda value: np.asarray(value, dtype=np.float64)
    args = [cast(v) for v in (X[index], target[index], Y[index], P[pindex], Pminus[pindex], Pplus[pindex])]
    args += [{key: cast(value[pindex]) for key, value in states.items()}, cast(widths[pindex])]
    expected = registered_loss(lambda x: numpy_logits(params, x), arm, thermo, np, *args)
    assert actual == pytest.approx(float(expected), rel=1e-4)
    if arm == "Mphys":  # the physics terms are active, not identically zero
        assert float(expected) > float(registered_loss(lambda x: numpy_logits(params, x), "Mdata", thermo, np, *args))


def test_physics_parameter_gradients_finite_and_match_central_difference(thermo):
    model = make_model(43, "torch", "cpu"); params = params64(model)
    arrays, widths = toy_arrays(); states = toy_states(thermo, 128)
    index, pindex = np.arange(40), np.arange(64)
    physics = lambda m, xp, *args: registered_loss(m, "Mphys", thermo, xp, *args)-registered_loss(m, "Mdata", thermo, xp, *args)
    inputs = torch_inputs(arrays, widths, states, index, pindex)
    physics(model, TorchOps(), *inputs).backward()
    grads = {name: value.grad.numpy().astype(np.float64) for name, value in model.named_parameters()}
    assert set(grads) == set(LAYER_SHAPES)
    assert all(np.isfinite(g).all() and np.abs(g).max() > 0 for g in grads.values())
    # Directional central difference of the physics-only loss in NumPy64.
    direction = {key: np.random.default_rng(9).standard_normal(value.shape) for key, value in params.items()}
    X, target, Y, P, Pminus, Pplus = arrays
    cast = lambda value: np.asarray(value, dtype=np.float64)
    args = [cast(v) for v in (X[index], target[index], Y[index], P[pindex], Pminus[pindex], Pplus[pindex])]
    args += [{key: cast(value[pindex]) for key, value in states.items()}, cast(widths[pindex])]
    def loss64(step):
        moved = {key: params[key]+step*direction[key] for key in params}
        return float(physics(lambda x: numpy_logits(moved, x), np, *args))
    analytic = sum(float((grads[key]*direction[key]).sum()) for key in params)
    for h in (1e-3, 1e-2):
        finite = (loss64(h)-loss64(-h))/(2*h)
        assert abs(analytic-finite)/max(1e-5, abs(analytic), abs(finite)) <= .02


def fit_toy(seed, tmp_path, name, thermo, *, arm="Mphys", epochs=4, backend=None):
    arrays, widths = toy_arrays(); states = toy_states(thermo, 128)
    checks = []
    backend = backend or TorchBackend("cpu")
    model, steps = backend.fit(arm, seed, arrays, states, widths, thermo, checks.append, tmp_path/f"{name}.jsonl",
                               epochs=epochs, physics_rows=128)
    log = [json.loads(line) for line in (tmp_path/f"{name}.jsonl").read_text().splitlines()]
    return model, steps, checks, log


def test_toy_training_updates_weights_and_is_seeded_repeatable(tmp_path, thermo):
    initial = params64(make_model(44, "torch", "cpu"))
    model, steps, checks, log = fit_toy(44, tmp_path, "a", thermo, epochs=30)
    assert steps == 30 and checks == list(range(30)) and [row["epoch"] for row in log] == list(range(1, 31))
    assert log[-1]["physics_cycle"] == 14 and log[-1]["physics_position"] == 128  # 30 x 64 rows over 128, lazy refill
    trained = params64(model)
    assert all(not np.array_equal(trained[key], initial[key]) for key in initial)
    assert log[-1]["loss"] < log[0]["loss"]
    again = params64(fit_toy(44, tmp_path, "b", thermo, epochs=30)[0])
    assert all(np.array_equal(trained[key], again[key]) for key in trained)
    other = params64(fit_toy(45, tmp_path, "c", thermo, epochs=30)[0])
    assert not np.array_equal(trained["layers.0.weight"], other["layers.0.weight"])
    assert [json.loads(line)["loss"] for line in (tmp_path/"b.jsonl").read_text().splitlines()] == [row["loss"] for row in log]


def test_shared_loop_order_matches_registered_rng_contract(tmp_path):
    seen = []
    steps = fit_loop(300, 42, lambda index, pindex: seen.append((index.copy(), pindex.copy())) or 1.0,
                     lambda epoch: None, tmp_path/"log.jsonl", epochs=2, physics_rows=2048)
    generator, physics = np.random.default_rng(42), np.random.default_rng(42+100000)
    order, porder = [generator.permutation(300) for _ in range(2)], physics.permutation(2048)
    assert steps == 4 and [len(i) for i, _ in seen] == [256, 44, 256, 44]
    assert np.array_equal(np.concatenate([i for i, _ in seen[:2]]), order[0])
    assert np.array_equal(np.concatenate([i for i, _ in seen[2:]]), order[1])
    assert np.array_equal(np.concatenate([p for _, p in seen]), porder[:256])


def test_export_writes_canonical_once_and_cpu64_inference_matches(tmp_path, thermo):
    backend = TorchBackend("cpu")
    model = fit_toy(42, tmp_path, "fit", thermo, epochs=2, backend=backend)[0]
    metadata = {"arm": "Mphys", "N": 64, "seed": 42, "training_backend": backend.info()}
    params, npz = backend.save(model, tmp_path, "Mphys", 64, 42, metadata)
    assert npz == "models/Mphys/N64/seed42.npz"
    with np.load(tmp_path/npz, allow_pickle=False) as archive:
        assert list(archive.files) == list(LAYER_SHAPES)
        stored = {key: archive[key] for key in archive.files}
    assert all(stored[key].dtype == np.float64 and stored[key].shape == shape for key, shape in LAYER_SHAPES.items())
    checkpoint = read_safetensors((tmp_path/"models/Mphys/N64/seed42.safetensors").read_bytes())
    assert all(np.array_equal(checkpoint[key].astype(np.float64), stored[key]) for key in LAYER_SHAPES)
    record = json.loads((tmp_path/"models/Mphys/N64/seed42.json").read_text())
    assert record["training_backend"]["backend"] == "torch" and record["training_backend"]["device"] == "cpu"
    assert record["training_backend"]["dtype"] == "float64" and record["training_backend"]["version"] == torch.__version__
    with pytest.raises(FileExistsError):
        backend.save(model, tmp_path, "Mphys", 64, 42, metadata)
    features = np.random.default_rng(39001).standard_normal((33, 12)).astype(np.float64)
    with torch.no_grad():
        actual = [v.numpy().astype(np.float64) for v in output_map(model(torch.as_tensor(features)), TorchOps())]
    expected = forward64(stored, features.astype(np.float64))
    relative = [np.abs(a-b).max()/np.abs(b).max() for a, b in zip(actual[:2], expected[:2])]
    assert max(relative) <= 5e-5 and np.abs(actual[2]-expected[2]).sum(axis=1).max() <= 1e-4


def test_unavailable_cuda_is_rejected_before_any_gate(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        resolve_device("cuda")
    with pytest.raises(ValueError):
        resolve_device("mps")
    from scripts.phase8.saf_surrogate import run
    monkeypatch.setattr(run, "load_registration", lambda *a: pytest.fail("gate reached"))
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        run.main(["run", "--backend", "torch", "--device", "cuda"])
    with pytest.raises(SystemExit):
        run.main(["run", "--backend", "mlx", "--device", "cpu"])


class _GateReached(Exception):
    pass


def test_thread_env_validated_before_any_torch_import(tmp_path):
    import subprocess
    code = ("import os, sys\n"
            "os.environ['OMP_NUM_THREADS'] = '4'\n"
            "from scripts.phase8.saf_surrogate import run\n"
            "try:\n"
            "    run.pipeline(str(sys.argv[1]), 'outputs/_unused', run.REGISTRATION, backend='torch', device='cpu')\n"
            "except ValueError as error:\n"
            "    assert 'one-thread-per-worker' in str(error)\n"
            "    assert 'torch' not in sys.modules\n"
            "else:\n"
            "    raise SystemExit('expected ValueError before any torch import')\n")
    subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=ROOT, check=True)


def test_preimported_torch_thread_count_forced_to_one(monkeypatch):
    from scripts.phase8.saf_surrogate import run
    for key in run.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(4)  # a Torch already imported elsewhere with a different count
    try:
        assert torch.get_num_threads() == 4
        def reached_gate(*a):
            raise _GateReached
        monkeypatch.setattr(run, "load_registration", reached_gate)
        with pytest.raises(_GateReached):
            run.main(["run", "--backend", "torch", "--device", "cpu"])
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)


def test_gpu_metadata_reports_actual_selected_cuda_device_not_index0(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"FakeGPU{index}")
    backend = TorchBackend("cuda")
    assert backend.device_label() == "cuda:1"
    assert backend.info()["device_name"] == "FakeGPU1"


@pytest.fixture
def no_mlx(monkeypatch):
    for name in [name for name in sys.modules if name == "mlx" or name.startswith("mlx.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mlx", None)  # any `import mlx...` now fails


def fake_source_runtime(monkeypatch, tmp_path, properties):
    """Manufactured Cantera/core stand-ins for the registered source checks."""
    thermo = Thermo(properties)
    class Gas:
        molecular_weights = thermo.mw
        def __init__(self, path): self.T = 300.0
        @property
        def TP(self): return self.T, 101325
        @TP.setter
        def TP(self, value): self.T = value[0]
        def _rt(self, field): return thermo._species(self.T, field)*thermo.mw/thermo.Ru
        standard_enthalpies_RT = property(lambda self: self._rt("h")/self.T)
        standard_cp_R = property(lambda self: self._rt("cp"))
        standard_entropies_R = property(lambda self: self._rt("s"))
    cantera = ModuleType("cantera"); cantera.Solution = Gas; cantera.gas_constant = thermo.Ru
    monkeypatch.setitem(sys.modules, "cantera", cantera)
    class Engine:
        def __init__(self, path): self.config = SimpleNamespace(pi_c=None, eta_c=None)
        def run_compressor(self, T, p): return {"T_out": thermo.compressor_temperature(self.config.pi_c, self.config.eta_c)}
    from scripts.phase8.saf_surrogate import teacher
    monkeypatch.setattr(teacher, "load_selected_core", lambda path, sha: SimpleNamespace(V6Engine=Engine))
    reg = json.loads((ROOT/"docs/phase8_saf_surrogate_registration.json").read_text())
    fixed = reg["scope"]["fixed_central"]
    row = {key: fixed[key] for key in ("combustor_pressure_loss", "eta_compressor", "eta_turbine_polytropic", "fpr_rated", "eta_fan")}
    row.update({"eta_b_"+mode: fixed["eta_b"][mode] for mode in ("IDLE", "APPROACH", "TAKE-OFF")})
    draws = {key: dict(row) for key in ("central", *[f"draw_{i:02d}" for i in range(64)])}
    public = {"opr": 18.65, "bpr": 5.6, "rated_kN": 42.6, "fit": reg["scope"]["fit_parameters"]}
    for name, value in (("frozen_properties.json", properties), ("public_inputs.json", public), ("fixed_draws.json", draws)):
        (tmp_path/name).write_text(json.dumps(value))
    return SimpleNamespace(binary_path=tmp_path/"core.so", binary_sha256="0"*64), SimpleNamespace(assert_current=lambda: None)


def test_torch_source_checks_and_gpu_receipt_never_import_mlx(monkeypatch, tmp_path, no_mlx):
    from scripts.phase8.saf_surrogate.timing import gpu_measurements, source_diagnostics
    context, run = fake_source_runtime(monkeypatch, tmp_path, manufactured_properties())
    source_diagnostics(ROOT, tmp_path, {}, context, run, backend="torch")
    receipt = json.loads((tmp_path/"precision.json").read_text())
    assert receipt["state"] == "PASS" and receipt["training_backend"] == "torch"
    assert max(receipt["gradient_normalized_errors"].values()) <= .02
    assert min(receipt["negative_control_disagreements"].values()) > .02
    gpu = gpu_measurements(tmp_path, None, None, run, backend="torch")
    assert gpu["state"] == "UNAVAILABLE" and gpu["available"] is False and "--backend torch" in gpu["reason"]
    assert json.loads((tmp_path/"mlx_gpu_timing.json").read_text())["state"] == "UNAVAILABLE"
    assert sys.modules["mlx"] is None


def test_torch_cuda_gpu_timing_reports_unavailable_without_real_cuda(tmp_path, no_mlx, monkeypatch):
    """Unavailable-device branch works on both CPU and actual CUDA hosts."""
    from scripts.phase8.saf_surrogate.timing import gpu_measurements
    run = SimpleNamespace(assert_current=lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gpu = gpu_measurements(tmp_path, None, None, run, backend="torch", device="cuda")
    assert gpu["state"] == "UNAVAILABLE" and gpu["available"] is False
    assert gpu["reason"] == "Torch CUDA GPU is unavailable"
    assert json.loads((tmp_path/"mlx_gpu_timing.json").read_text())["state"] == "UNAVAILABLE"
    assert sys.modules["mlx"] is None


@pytest.mark.parametrize("change", ["matching", "wrong_forward", "nonfinite"])
def test_selected_device_forward_parity_rejects_wrong_and_nonfinite_outputs(monkeypatch, change):
    """CPU toy tensors stand in for device tensors; no CUDA or project inputs."""
    from scripts.phase8.saf_surrogate import inputs, timing
    models = [make_model(seed, "torch", "cpu", dtype="float32") for seed in (42, 43, 44)]  # optional GPU32 diagnostic
    product = SimpleNamespace(bundle={"members": [{"seed": seed} for seed in (42, 43, 44)]},
        params=[params64(model) for model in models],
        scalers=[(np.zeros(12), np.ones(12)) for _ in models], draws={}, public={},
        canonical_queries=lambda queries: queries)
    features = np.random.default_rng(39001).standard_normal((50, 12))
    queries = [{"toy_index": i} for i in range(50)]
    seen = []
    def toy_features(batch, draws, public):
        seen.extend(query["toy_index"] for query in batch)
        return features[[query["toy_index"] for query in batch]]
    monkeypatch.setattr(inputs, "feature_rows", toy_features)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    if change != "matching":
        with torch.no_grad():
            models[0].layers[-1].bias[0] += .1 if change == "wrong_forward" else float("nan")
    result = timing._torch_cuda_precision(product, queries, models, "cpu")
    assert seen == list(range(33)) and result["rows"] == 33 and result["device"] == "cpu"
    assert [row["seed"] for row in result["members"]] == [42, 43, 44]
    if change == "matching":
        assert result["state"] == "PASS"
        assert max(result["ff_max_normalized_error"], result["T4_max_normalized_error"]) <= 5e-5
        assert result["species_L1_max_error"] <= 1e-4
    else:
        assert result["state"] == "FAIL" and result["members"][0]["state"] == "FAIL"
        if change == "nonfinite":
            assert result["members"][0]["finite"] is False
            assert result["ff_max_normalized_error"] is None
        else:
            assert result["ff_max_normalized_error"] > 5e-5
    json.dumps(result, allow_nan=False)  # Receipt stays valid strict JSON.


def test_gpu_complete_refused_before_benchmarks_when_selected_export_parity_fails(tmp_path, monkeypatch):
    from scripts.phase8.saf_surrogate import timing
    members = [{"seed": seed, "weights_path": f"seed{seed}.npz"} for seed in (42, 43, 44)]
    for member in members:
        (tmp_path/member["weights_path"].replace(".npz", ".safetensors")).write_bytes(b"fixture checkpoint")
    product = SimpleNamespace(output=tmp_path, bundle={"members": members})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(timing, "make_model", lambda *a, **kw: SimpleNamespace(load_state_dict=lambda state: None))
    monkeypatch.setattr(train_torch, "read_safetensors", lambda data: {})
    failed = {"state": "FAIL", "rows": 33, "device": "cuda:1", "ff_max_normalized_error": .1}
    monkeypatch.setattr(timing, "_torch_cuda_precision", lambda *a: failed)
    monkeypatch.setattr(timing, "_gpu_measure_torch", lambda *a: pytest.fail("benchmark ran after precision FAIL"))
    run = SimpleNamespace(assert_current=lambda: None)
    result = timing.gpu_measurements(tmp_path, product, [], run, backend="torch", device="cuda")
    assert result["state"] == "FAIL" and result["available"] is False
    assert result["failure_stage"] == "cuda_cpu64_parity" and result["cuda_cpu64_parity"] == failed
    assert "CPU64 export" in result["reason"]
    assert json.loads((tmp_path/"mlx_gpu_timing.json").read_text()) == result


def test_torch_training_runs_with_mlx_import_blocked(tmp_path, thermo, no_mlx):
    model, steps, _, _ = fit_toy(42, tmp_path, "blocked", thermo, epochs=2)
    assert steps == 2 and sys.modules["mlx"] is None
    with pytest.raises(ImportError):
        make_model(42,"mlx")  # the optional explicit MLX backend needs MLX


@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed in this environment")
def test_mlx_default_path_matches_torch_loss_and_reads_torch_checkpoint(tmp_path, thermo):
    import mlx.core as mx
    from scripts.phase8.saf_surrogate.train import MLXBackend
    # MLX's default device, as in registered training (its CPU JIT needs a host compiler).
    backend = TorchBackend("cpu")
    model = fit_toy(42, tmp_path, "fit", thermo, epochs=1, backend=backend)[0]
    backend.save(model, tmp_path, "Mphys", 64, 42, {})
    from simulation.ml_backend import get_backend
    twin = get_backend("mlx",device="auto").load_npz(tmp_path/"models/Mphys/N64/seed42.npz")
    arrays, widths = toy_arrays(); states = toy_states(thermo, 128)
    index, pindex = np.arange(40), np.arange(64)
    inputs = torch_inputs(arrays, widths, states, index, pindex)
    as_mx = lambda value: mx.array(value.numpy().astype(np.float32))  # optional native float32 diagnostic
    mlx_inputs = [as_mx(v) for v in inputs[:6]] + [{k: as_mx(v) for k, v in inputs[6].items()}, as_mx(inputs[7])]
    for arm in ("Mdata", "Mphys"):
        with torch.no_grad():
            expected = float(registered_loss(model, arm, thermo, TorchOps(), *inputs))
        assert float(registered_loss(twin, arm, thermo, mx, *mlx_inputs)) == pytest.approx(expected, rel=1e-4)
    first = fit_toy(42, tmp_path, "m1", thermo, epochs=3, backend=MLXBackend())
    second = fit_toy(42, tmp_path, "m2", thermo, epochs=3, backend=MLXBackend())
    assert first[1] == 3 and [r["loss"] for r in first[3]] == [r["loss"] for r in second[3]]


def test_training_imports_without_posix_resource_module():
    import subprocess
    code = ("import sys; sys.modules['resource'] = None\n"
            "from scripts.phase8.saf_surrogate.train import peak_rss_bytes\n"
            "from scripts.phase8.saf_surrogate.train_torch import TorchBackend\n"
            "assert peak_rss_bytes() is None and TorchBackend('cpu').memory()['process_peak_rss_bytes'] is None\n"
            "assert 'mlx' not in sys.modules\n")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
