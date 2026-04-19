"""
Tests for conditioning.py and denoiser.py

Run with: pytest tests/test_diffusion.py -v
"""

import torch
import pytest
from torch_geometric.data import Batch
from molgen.models.conditioning import PropertyConditioner
from molgen.models.denoiser import GraphDenoiser, SinusoidalTimeEmbedding
from molgen.data.graph_builder import smiles_to_graph


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def conditioner():
    return PropertyConditioner(num_properties=4, conditioning_dim=64, null_prob=0.2)


@pytest.fixture
def denoiser():
    return GraphDenoiser(
        node_dim=18, edge_dim=6, hidden_dim=64,
        num_layers=2, time_emb_dim=32, cond_dim=64,
    )


@pytest.fixture
def batch():
    mols = ["CCO", "c1ccccc1", "CC(=O)O"]
    return Batch.from_data_list([smiles_to_graph(s) for s in mols])


# ── SinusoidalTimeEmbedding ────────────────────────────────────────────────────

class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(time_emb_dim=64)
        t   = torch.randint(0, 1000, (10,))
        out = emb(t)
        assert out.shape == (10, 64)

    def test_different_timesteps_different_embeddings(self):
        emb = SinusoidalTimeEmbedding(time_emb_dim=64)
        t1  = torch.tensor([100])
        t2  = torch.tensor([500])
        assert not torch.allclose(emb(t1), emb(t2))

    def test_output_is_finite(self):
        emb = SinusoidalTimeEmbedding(time_emb_dim=64)
        t   = torch.randint(0, 1000, (20,))
        assert torch.isfinite(emb(t)).all()


# ── PropertyConditioner ────────────────────────────────────────────────────────

class TestPropertyConditioner:
    def test_output_shape(self, conditioner):
        props = torch.randn(4, 4)   # batch of 4, 4 properties each
        out   = conditioner(props)
        assert out.shape == (4, 64)

    def test_force_null_returns_same_for_all(self, conditioner):
        """force_null=True should return the same embedding for every item."""
        conditioner.eval()
        props = torch.randn(5, 4)
        out   = conditioner(props, force_null=True)
        # All rows should be identical (all are the null embedding)
        assert torch.allclose(out[0], out[1])
        assert torch.allclose(out[0], out[4])

    def test_training_drops_some_conditioning(self, conditioner):
            """In training mode, some items should get null embedding."""
            conditioner.train()
            torch.manual_seed(0)
            props = torch.randn(200, 4)
            out   = conditioner(props)
            null  = conditioner.null_embedding.detach()
            is_null = [torch.allclose(out[i].detach(), null) for i in range(200)]
            assert any(is_null), "Expected some null conditioning during training"

    def test_eval_mode_no_dropout(self, conditioner):
        """In eval mode, conditioning should never be dropped."""
        conditioner.eval()
        props = torch.randn(50, 4)
        out   = conditioner(props)
        null  = conditioner.null_embedding.detach()
        is_null = [torch.allclose(out[i].detach(), null) for i in range(50)]
        assert not any(is_null), "Conditioning should not drop in eval mode"

    def test_output_is_finite(self, conditioner):
        props = torch.randn(8, 4)
        assert torch.isfinite(conditioner(props)).all()


# ── GraphDenoiser ──────────────────────────────────────────────────────────────

class TestGraphDenoiser:
    def test_output_shapes(self, denoiser, batch):
        """eps_pred must match node features, bond_logits must match edges."""
        B = 3   # 3 molecules in batch
        c = torch.randn(B, 64)
        t = torch.randint(1, 1000, (batch.x.shape[0],))

        eps_pred, bond_logits = denoiser(
            batch.x, batch.edge_index, batch.edge_attr, t, c, batch.batch
        )

        assert eps_pred.shape   == batch.x.shape,          \
            f"eps_pred shape mismatch: {eps_pred.shape}"
        assert bond_logits.shape == (batch.edge_index.shape[1], 1), \
            f"bond_logits shape mismatch: {bond_logits.shape}"

    def test_output_is_finite(self, denoiser, batch):
        B = 3
        c = torch.randn(B, 64)
        t = torch.randint(1, 1000, (batch.x.shape[0],))
        eps_pred, bond_logits = denoiser(
            batch.x, batch.edge_index, batch.edge_attr, t, c, batch.batch
        )
        assert torch.isfinite(eps_pred).all()
        assert torch.isfinite(bond_logits).all()

    def test_different_timesteps_different_output(self, denoiser, batch):
        """Same graph at different timesteps should produce different predictions."""
        denoiser.eval()
        B = 3
        c  = torch.randn(B, 64)
        N  = batch.x.shape[0]
        t1 = torch.full((N,), 100, dtype=torch.long)
        t2 = torch.full((N,), 900, dtype=torch.long)

        with torch.no_grad():
            eps1, _ = denoiser(
                batch.x, batch.edge_index, batch.edge_attr, t1, c, batch.batch
            )
            eps2, _ = denoiser(
                batch.x, batch.edge_index, batch.edge_attr, t2, c, batch.batch
            )
        assert not torch.allclose(eps1, eps2)

    def test_conditioning_affects_output(self, denoiser, batch):
        """Different conditioning vectors should give different predictions."""
        denoiser.eval()
        N  = batch.x.shape[0]
        t  = torch.full((N,), 500, dtype=torch.long)
        c1 = torch.zeros(3, 64)
        c2 = torch.ones(3, 64)

        with torch.no_grad():
            eps1, _ = denoiser(
                batch.x, batch.edge_index, batch.edge_attr, t, c1, batch.batch
            )
            eps2, _ = denoiser(
                batch.x, batch.edge_index, batch.edge_attr, t, c2, batch.batch
            )
        assert not torch.allclose(eps1, eps2)

    def test_gradients_flow(self, denoiser, batch):
        """Gradients must flow back through the full denoiser."""
        B  = 3
        c  = torch.randn(B, 64)
        t  = torch.randint(1, 1000, (batch.x.shape[0],))
        x  = batch.x.clone().requires_grad_(True)

        eps_pred, bond_logits = denoiser(
            x, batch.edge_index, batch.edge_attr, t, c, batch.batch
        )
        (eps_pred.sum() + bond_logits.sum()).backward()
        assert x.grad is not None

    def test_parameter_count(self, denoiser):
        """Denoiser should have more params than encoder (deeper network)."""
        n = sum(p.numel() for p in denoiser.parameters())
        assert n > 50_000, f"Too few parameters: {n}"