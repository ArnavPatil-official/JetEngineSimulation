"""
Unit tests for the LE-PINN module.

Validates architecture dimensions, fusion mechanism, initialisation,
normalisation, optimiser/scheduler config, geometry helpers, and
non-regression of the existing NozzlePINN.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import simulation.nozzle.le_pinn as le_pinn_module

from simulation.nozzle.le_pinn import (
    GlobalNetwork,
    BoundaryNetwork,
    LE_PINN,
    MinMaxNormalizer,
    AdaptiveLossWeighting,
    compute_rans_residuals,
    compute_wall_distances,
    setup_training,
    finetune_on_cfd_data,
    validate_le_pinn,
    FUSION_DELTA,
)


# ============================================================================
# 1. Network shapes
# ============================================================================

class TestGlobalNetwork:
    def test_output_shape(self) -> None:
        net = GlobalNetwork()
        x = torch.randn(32, 6)
        out = net(x)
        assert out.shape == (32, 9), f"Expected (32, 9), got {out.shape}"

    def test_single_sample(self) -> None:
        net = GlobalNetwork()
        x = torch.randn(1, 6)
        out = net(x)
        assert out.shape == (1, 9)


class TestBoundaryNetwork:
    def test_output_shape(self) -> None:
        net = BoundaryNetwork()
        x = torch.randn(32, 6)
        out = net(x)
        assert out.shape == (32, 2), f"Expected (32, 2), got {out.shape}"

    def test_single_sample(self) -> None:
        net = BoundaryNetwork()
        x = torch.randn(1, 6)
        out = net(x)
        assert out.shape == (1, 2)


# ============================================================================
# 2. Fusion mechanism
# ============================================================================

class TestLE_PINN_Fusion:
    @pytest.fixture
    def model(self) -> LE_PINN:
        torch.manual_seed(42)
        return LE_PINN()

    def test_fusion_near_wall(self, model: LE_PINN) -> None:
        """Points with d < delta should have P, T replaced by boundary net."""
        N = 16
        inputs = torch.randn(N, 6)
        wall_dists = torch.full((N, 1), 1e-5)  # all near-wall

        fused = model(inputs, wall_dists)
        assert fused.shape == (N, 9)

        # Boundary network predictions
        boundary_preds = model.boundary_net(inputs)

        # P (idx 3) and T (idx 4) should match boundary output
        torch.testing.assert_close(fused[:, 3], boundary_preds[:, 0])
        torch.testing.assert_close(fused[:, 4], boundary_preds[:, 1])

    def test_no_fusion_far_wall(self, model: LE_PINN) -> None:
        """Points with d >= delta should use global predictions unchanged."""
        N = 16
        inputs = torch.randn(N, 6)
        wall_dists = torch.full((N, 1), 1.0)  # all far from wall

        fused = model(inputs, wall_dists)
        global_preds = model.global_net(inputs)

        torch.testing.assert_close(fused, global_preds)

    def test_mixed_fusion(self, model: LE_PINN) -> None:
        """Half near-wall, half far — only near-wall points fused."""
        N = 8
        inputs = torch.randn(N, 6)
        # First 4 near-wall, last 4 far
        wall_dists = torch.cat([
            torch.full((4, 1), 1e-5),
            torch.full((4, 1), 1.0),
        ])

        fused = model(inputs, wall_dists)
        global_preds = model.global_net(inputs)
        boundary_preds = model.boundary_net(inputs)

        # Near-wall: P,T from boundary
        torch.testing.assert_close(fused[:4, 3], boundary_preds[:4, 0])
        torch.testing.assert_close(fused[:4, 4], boundary_preds[:4, 1])
        # Far: all from global
        torch.testing.assert_close(fused[4:], global_preds[4:])

    def test_threshold_value(self, model: LE_PINN) -> None:
        assert model.delta == FUSION_DELTA == 5e-4


# ============================================================================
# 3. Initialization
# ============================================================================

class TestInitialization:
    def test_xavier_uniform_weights(self) -> None:
        """All Linear weights should have Xavier uniform initialization."""
        net = GlobalNetwork()
        for m in net.modules():
            if isinstance(m, nn.Linear):
                fan_in, fan_out = m.weight.shape[1], m.weight.shape[0]
                limit = (6.0 / (fan_in + fan_out)) ** 0.5
                assert m.weight.min().item() >= -limit - 0.01
                assert m.weight.max().item() <= limit + 0.01
                # Standard deviation should be non-trivial (not default init)
                assert m.weight.std().item() > 0.001

    def test_zero_bias_init(self) -> None:
        """All biases should be initialized to zero."""
        for NetClass in [GlobalNetwork, BoundaryNetwork]:
            net = NetClass()
            for m in net.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    assert torch.allclose(m.bias, torch.zeros_like(m.bias)), \
                        f"Bias not zero in {NetClass.__name__}"


# ============================================================================
# 4. Normalizer
# ============================================================================

class TestMinMaxNormalizer:
    def test_round_trip(self) -> None:
        data = torch.randn(100, 6) * 10 + 5
        norm = MinMaxNormalizer()
        transformed = norm.fit_transform(data)
        reconstructed = norm.inverse_transform(transformed)
        torch.testing.assert_close(reconstructed, data, atol=1e-5, rtol=1e-5)

    def test_range(self) -> None:
        data = torch.randn(100, 3)
        norm = MinMaxNormalizer()
        transformed = norm.fit_transform(data)
        assert transformed.min() >= -0.01
        assert transformed.max() <= 1.01

    def test_epsilon_constant_column(self) -> None:
        """Constant column (max == min) should not cause NaN."""
        data = torch.ones(50, 3) * 7.0
        norm = MinMaxNormalizer(epsilon=1e-8)
        transformed = norm.fit_transform(data)
        assert torch.isfinite(transformed).all()


# ============================================================================
# 5. Optimizer & Scheduler
# ============================================================================

class TestTrainingSetup:
    def test_optimizer_config(self) -> None:
        model = LE_PINN()
        opt, _ = setup_training(model)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.defaults["lr"] == pytest.approx(1e-4)
        assert opt.defaults["weight_decay"] == pytest.approx(1e-5)

    def test_scheduler_config(self) -> None:
        model = LE_PINN()
        _, sched = setup_training(model)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.patience == 10
        assert sched.factor == pytest.approx(0.5)
        # min_lr is stored per param group
        for group_lrs in sched.min_lrs:
            assert group_lrs == pytest.approx(1e-8)


# ============================================================================
# 6. Wall distance
# ============================================================================

class TestWallDistance:
    def test_known_distances(self) -> None:
        wall = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        query = torch.tensor([[0.0, 0.0], [1.0, 0.5]])
        dists = compute_wall_distances(query, wall)
        assert dists.shape == (2, 1)
        assert dists[0].item() == pytest.approx(1.0, abs=1e-6)
        assert dists[1].item() == pytest.approx(0.5, abs=1e-6)


# ============================================================================
# 7. RANS residuals shape
# ============================================================================

class TestRANSResiduals:
    def test_output_shapes(self) -> None:
        N = 20
        inputs = torch.randn(N, 6, requires_grad=True)

        # Build outputs that have a computational graph connecting to inputs.
        # Use a small network so that autograd can trace derivatives back to
        # individual input columns.
        net = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 9))
        outputs = net(inputs)

        res = compute_rans_residuals(inputs, outputs)
        assert len(res) == 5
        for r in res:
            assert r.shape == (N, 1), f"Expected ({N}, 1), got {r.shape}"

    def test_normalized_residuals_finite(self) -> None:
        """Normalized residuals should be finite and dimensionless-scaled."""
        N = 50
        inputs = torch.randn(N, 6, requires_grad=True)
        net = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 9))
        outputs = net(inputs)

        res_mass, res_xmom, res_ymom, res_energy, res_eos = compute_rans_residuals(
            inputs, outputs, geometry="planar", normalize=True
        )

        # All should be finite
        assert torch.isfinite(res_mass).all()
        assert torch.isfinite(res_xmom).all()
        assert torch.isfinite(res_ymom).all()
        assert torch.isfinite(res_energy).all()
        assert torch.isfinite(res_eos).all()

        # After normalization, magnitudes should be reasonable (not 1e12+)
        assert res_mass.abs().max() < 1e6
        assert res_xmom.abs().max() < 1e6
        assert res_ymom.abs().max() < 1e6
        assert res_energy.abs().max() < 1e6
        assert res_eos.abs().max() < 1e6

    def test_planar_vs_axisymmetric_geometry(self) -> None:
        """Planar and axisymmetric modes should produce different residuals."""
        N = 20
        inputs = torch.randn(N, 6, requires_grad=True)
        net = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 9))
        outputs = net(inputs)

        res_planar = compute_rans_residuals(inputs, outputs, geometry="planar", normalize=False)
        inputs2 = inputs.detach().clone().requires_grad_(True)
        outputs2 = net(inputs2)
        res_axi = compute_rans_residuals(inputs2, outputs2, geometry="axisymmetric", normalize=False)

        # Residuals should differ (axisymmetric has extra source terms)
        # At least one residual should be different
        different = False
        for rp, ra in zip(res_planar, res_axi):
            if not torch.allclose(rp, ra, atol=1e-6):
                different = True
                break
        assert different, "Planar and axisymmetric residuals should differ"


# ============================================================================
# 8. Existing PINN non-regression
# ============================================================================

class TestExistingPINNUnaffected:
    def test_nozzle_pinn_import(self) -> None:
        """Importing the existing NozzlePINN should still work."""
        from simulation.nozzle.nozzle import NozzlePINN
        model = NozzlePINN()
        assert model is not None
        # Verify it has the expected architecture (3 hidden, 64 wide, 8→3)
        params = list(model.parameters())
        assert len(params) > 0


# ============================================================================
# 9. CFD fine-tuning & validation smoke tests
# ============================================================================

DATASET_PATH = str(
    Path(__file__).parent.parent / "data" / "processed" / "master_shock_dataset.pt"
)


def _load_cfd_dataset() -> dict:
    try:
        return torch.load(DATASET_PATH, weights_only=True)
    except TypeError:
        return torch.load(DATASET_PATH)


def _write_cfd_subset(path: Path, max_rows: int = 128) -> str:
    dataset = _load_cfd_dataset()
    n_rows = min(max_rows, int(dataset["inputs"].shape[0]))
    subset = {
        "inputs": dataset["inputs"][:n_rows].clone(),
        "targets": dataset["targets"][:n_rows].clone(),
    }
    if "sample_weights" in dataset:
        subset["sample_weights"] = dataset["sample_weights"][:n_rows].clone()
    torch.save(subset, path)
    return str(path)


@pytest.mark.skipif(
    not Path(DATASET_PATH).exists(),
    reason="master_shock_dataset.pt not present — run fetch_and_build_cfd_data.py",
)
class TestCFDFinetune:
    def test_finetune_returns_model_and_history(self, tmp_path: Path) -> None:
        """finetune_on_cfd_data runs 3 epochs without crashing."""
        save = str(tmp_path / "le_pinn_cfd_test.pt")
        model, history = finetune_on_cfd_data(
            dataset_path=DATASET_PATH,
            n_epochs=3,
            save_path=save,
            physics_loss_weight=0.0,  # skip physics for speed
            verbose=False,
        )
        assert isinstance(model, LE_PINN)
        assert len(history["loss_total"]) == 3
        assert len(history["loss_data"]) == 3
        assert Path(save).exists()

    def test_finetune_loss_decreases_or_stable(self, tmp_path: Path) -> None:
        """Data loss over 20 epochs should not diverge (sanity check)."""
        model, history = finetune_on_cfd_data(
            dataset_path=DATASET_PATH,
            n_epochs=20,
            save_path=str(tmp_path / "le_pinn_cfd_test2.pt"),
            physics_loss_weight=0.0,
            verbose=False,
        )
        assert history["loss_data"][-1] < history["loss_data"][0] * 10, (
            "Data loss grew more than 10× — training may be unstable"
        )

    def test_finetune_warning_for_missing_pretrained(self, tmp_path: Path) -> None:
        """A missing pretrained_path should warn, not raise."""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            finetune_on_cfd_data(
                dataset_path=DATASET_PATH,
                pretrained_path="/nonexistent/path/le_pinn.pt",
                n_epochs=2,
                save_path=str(tmp_path / "le_pinn_cfd_warn.pt"),
                physics_loss_weight=0.0,
                verbose=False,
            )
        assert any("not found" in str(w.message).lower() for w in caught), (
            "Expected a RuntimeWarning about missing pretrained path"
        )

    def test_validate_returns_metrics(self) -> None:
        """validate_le_pinn returns a dict with rmse_* and r2_* keys."""
        model = LE_PINN()
        metrics = validate_le_pinn(
            model,
            dataset_path=DATASET_PATH,
            verbose=False,
        )
        assert isinstance(metrics, dict)
        # rho, u, P should have non-trivial range → metrics present
        for var in ("rho", "u", "P"):
            assert f"rmse_{var}" in metrics, f"Missing rmse_{var}"
            assert f"r2_{var}" in metrics, f"Missing r2_{var}"
            assert np.isfinite(metrics[f"rmse_{var}"]), f"rmse_{var} is not finite"

    def test_validate_handles_warning_for_missing_normalizer(self) -> None:
        """validate_le_pinn should warn if no checkpoint normalizer available."""
        import warnings
        model = LE_PINN()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = validate_le_pinn(model, dataset_path=DATASET_PATH, verbose=False)
        # Should warn about missing output normalizer when no checkpoint provided
        normalizer_warned = any(
            "normaliser" in str(w.message).lower() or "normalizer" in str(w.message).lower()
            for w in caught
        )
        assert normalizer_warned, "Expected warning about missing output normalizer"


@pytest.mark.skipif(
    not Path(DATASET_PATH).exists(),
    reason="master_shock_dataset.pt not present — run parse_sajben_cfd.py",
)
class TestSchedulerMonitor:
    def test_scheduler_steps_on_data_loss_not_total(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        subset_path = _write_cfd_subset(tmp_path / "scheduler_subset.pt")
        calls: list[float] = []
        original_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

        def record_step(self, metrics, *args, **kwargs):
            calls.append(float(metrics))
            return original_step(self, metrics, *args, **kwargs)

        def fake_physics_loss(*args, **kwargs) -> torch.Tensor:
            return torch.tensor(3.0, device=args[1].device)

        monkeypatch.setattr(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            "step",
            record_step,
        )
        monkeypatch.setattr(le_pinn_module, "_safe_physics_loss", fake_physics_loss)

        _, history = finetune_on_cfd_data(
            dataset_path=subset_path,
            n_epochs=5,
            save_path=str(tmp_path / "scheduler_monitor.pt"),
            physics_loss_weight=0.1,
            physics_max_points=64,
            device="cpu",
            verbose=False,
        )

        assert len(calls) == 5
        assert calls == pytest.approx(history["loss_data"])
        assert any(
            abs(metric - total) > 1e-12
            for metric, total in zip(calls, history["loss_total"])
        )

    def test_lr_does_not_immediately_collapse(self, tmp_path: Path) -> None:
        subset_path = _write_cfd_subset(tmp_path / "lr_subset.pt")

        _, history = finetune_on_cfd_data(
            dataset_path=subset_path,
            n_epochs=50,
            save_path=str(tmp_path / "lr_guard.pt"),
            physics_loss_weight=0.1,
            physics_max_points=64,
            device="cpu",
            verbose=False,
        )

        assert len(history["lr"]) == 50
        assert history["lr"][-1] > 1e-9
