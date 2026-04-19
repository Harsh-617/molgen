"""
Tests for train_diffusion.py

Smoke tests only — we verify the training loop runs without errors
for a few steps, not that the model converges (that takes hours).

Run with: pytest tests/test_train_diffusion.py -v
"""

import torch
import pytest
from torch_geometric.data import Batch
from molgen.data.graph_builder import smiles_to_graph
from molgen.models.conditioning import PropertyConditioner
from molgen.models.denoiser import GraphDenoiser
from molgen.utils.noise_schedule import CosineNoiseSchedule
from molgen.training.train_diffusion import (
    diffusion_loss,
    build_bond_targets,
    run_epoch,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def schedule():
    return CosineNoiseSchedule(T=100)   # tiny T for speed


@pytest.fixture
def denoiser():
    return GraphDenoiser(
        node_dim=18, edge_dim=6, hidden_dim=32,
        num_layers=2, time_emb_dim=16, cond_dim=32,
    )


@pytest.fixture
def conditioner():
    return PropertyConditioner(
        num_properties=4, conditioning_dim=32, null_prob=0.2
    )


@pytest.fixture
def batch():
    mols = ["CCO", "c1ccccc1", "CC(=O)O"]
    data_list = [smiles_to_graph(s) for s in mols]
    # Attach fake normalised labels
    for d in data_list:
        d.y = torch.randn(1, 4)
    return Batch.from_data_list(data_list)


# ── diffusion_loss ─────────────────────────────────────────────────────────────

class TestDiffusionLoss:
    def test_returns_tensor_and_dict(self, batch):
        E = batch.edge_index.shape[1]
        N = batch.x.shape[0]
        eps_pred    = torch.randn(N, 18)
        eps_true    = torch.randn(N, 18)
        bond_logits = torch.randn(E, 1)
        bond_true   = torch.ones(E, 1)

        loss, comps = diffusion_loss(eps_pred, eps_true, bond_logits, bond_true)
        assert isinstance(loss, torch.Tensor)
        assert "node_loss" in comps
        assert "edge_loss" in comps

    def test_loss_is_positive(self, batch):
        E = batch.edge_index.shape[1]
        N = batch.x.shape[0]
        loss, _ = diffusion_loss(
            torch.randn(N, 18), torch.randn(N, 18),
            torch.randn(E, 1),  torch.ones(E, 1),
        )
        assert loss.item() > 0

    def test_perfect_node_prediction_lowers_loss(self, batch):
        """MSE node loss should be 0 when eps_pred == eps_true."""
        E = batch.edge_index.shape[1]
        N = batch.x.shape[0]
        eps = torch.randn(N, 18)
        loss_perfect, comps = diffusion_loss(
            eps, eps,                          # perfect node prediction
            torch.randn(E, 1), torch.ones(E, 1),
        )
        assert comps["node_loss"] < 1e-5

    def test_lambda_edge_scales_edge_loss(self, batch):
        """Higher lambda_edge should increase total loss."""
        E = batch.edge_index.shape[1]
        N = batch.x.shape[0]
        eps_pred    = torch.randn(N, 18)
        eps_true    = torch.randn(N, 18)
        bond_logits = torch.randn(E, 1)
        bond_true   = torch.ones(E, 1)

        loss1, _ = diffusion_loss(eps_pred, eps_true, bond_logits, bond_true,
                                  lambda_edge=1.0)
        loss2, _ = diffusion_loss(eps_pred, eps_true, bond_logits, bond_true,
                                  lambda_edge=5.0)
        assert loss2.item() > loss1.item()


# ── build_bond_targets ─────────────────────────────────────────────────────────

class TestBuildBondTargets:
    def test_shape(self, batch):
        E      = batch.edge_index.shape[1]
        result = build_bond_targets(batch.edge_index, batch.x.shape[0])
        assert result.shape == (E, 1)

    def test_all_ones(self, batch):
        result = build_bond_targets(batch.edge_index, batch.x.shape[0])
        assert (result == 1.0).all()


# ── run_epoch (smoke test) ─────────────────────────────────────────────────────

class TestRunEpoch:
    def _make_loader(self, batch):
        """Wrap a single batch into a list (acts as a one-batch loader)."""
        return [batch]

    def test_phase2_train_returns_finite_loss(
        self, denoiser, conditioner, schedule, batch, device
    ):
        """Phase 2 unconditional training step should not crash or NaN."""
        optimizer = torch.optim.Adam(
            list(denoiser.parameters()) + list(conditioner.parameters()),
            lr=1e-3,
        )
        loss, comps = run_epoch(
            denoiser, conditioner, schedule,
            loader=self._make_loader(batch),
            optimizer=optimizer,
            device=device,
            is_train=True,
            phase=2,
            lambda_edge=1.0,
            T=100,
        )
        assert isinstance(loss, float)
        assert loss == loss          # not NaN
        assert loss > 0
        assert "node_loss" in comps
        assert "edge_loss" in comps

    def test_phase3_train_returns_finite_loss(
        self, denoiser, conditioner, schedule, batch, device
    ):
        """Phase 3 conditional training step should not crash or NaN."""
        optimizer = torch.optim.Adam(
            list(denoiser.parameters()) + list(conditioner.parameters()),
            lr=1e-3,
        )
        loss, comps = run_epoch(
            denoiser, conditioner, schedule,
            loader=self._make_loader(batch),
            optimizer=optimizer,
            device=device,
            is_train=True,
            phase=3,
            lambda_edge=1.0,
            T=100,
        )
        assert isinstance(loss, float)
        assert loss == loss
        assert loss > 0

    def test_val_mode_no_grad_update(
        self, denoiser, conditioner, schedule, batch, device
    ):
        """Validation run should not change model parameters."""
        # Snapshot parameters before val run
        params_before = [p.clone() for p in denoiser.parameters()]

        run_epoch(
            denoiser, conditioner, schedule,
            loader=self._make_loader(batch),
            optimizer=None,          # no optimizer = no update
            device=device,
            is_train=False,
            phase=2,
            lambda_edge=1.0,
            T=100,
        )

        params_after = list(denoiser.parameters())
        for before, after in zip(params_before, params_after):
            assert torch.allclose(before, after), \
                "Val run should not modify parameters"

    def test_phase2_ignores_properties(
        self, denoiser, conditioner, schedule, batch, device
    ):
        """Phase 2 should use null conditioning regardless of batch.y."""
        optimizer = torch.optim.Adam(
            list(denoiser.parameters()) + list(conditioner.parameters()),
            lr=1e-3,
        )
        # Run twice with different y values — loss should be same
        # (because phase 2 ignores y entirely)
        torch.manual_seed(42)
        loss1, _ = run_epoch(
            denoiser, conditioner, schedule,
            loader=self._make_loader(batch),
            optimizer=optimizer, device=device,
            is_train=False, phase=2, lambda_edge=1.0, T=100,
        )

        batch.y = torch.randn_like(batch.y) * 100  # wildly different properties
        torch.manual_seed(42)
        loss2, _ = run_epoch(
            denoiser, conditioner, schedule,
            loader=self._make_loader(batch),
            optimizer=optimizer, device=device,
            is_train=False, phase=2, lambda_edge=1.0, T=100,
        )

        assert abs(loss1 - loss2) < 1e-5, \
            "Phase 2 loss should not depend on property values"