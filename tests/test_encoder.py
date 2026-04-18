"""
Tests for encoder.py

Run with: pytest tests/test_encoder.py -v
"""

import torch
import pytest
from torch_geometric.data import Batch
from molgen.models.encoder import MPNNEncoder, MPNNLayer, PropertyHead
from molgen.data.graph_builder import smiles_to_graph


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def encoder():
    return MPNNEncoder(node_dim=18, edge_dim=6, hidden_dim=128, num_layers=4)


@pytest.fixture
def single_batch():
    """A batch containing one molecule (ethanol)."""
    data = smiles_to_graph("CCO")
    return Batch.from_data_list([data])


@pytest.fixture
def multi_batch():
    """A batch of three molecules of different sizes."""
    mols = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    data_list = [smiles_to_graph(s) for s in mols]
    return Batch.from_data_list(data_list)


# ── MPNNLayer tests ────────────────────────────────────────────────────────────

class TestMPNNLayer:
    def test_output_shape_preserved(self):
        """Layer should not change the shape of node features."""
        layer = MPNNLayer(hidden_dim=64, edge_dim=6)
        h = torch.randn(5, 64)
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)
        edge_attr  = torch.randn(4, 6)
        out = layer(h, edge_index, edge_attr)
        assert out.shape == (5, 64)

    def test_gradients_flow(self):
        """Gradients must flow back through the layer."""
        layer = MPNNLayer(hidden_dim=32, edge_dim=6)
        h = torch.randn(4, 32, requires_grad=True)
        edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
        edge_attr  = torch.randn(4, 6)
        out = layer(h, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None


# ── MPNNEncoder tests ──────────────────────────────────────────────────────────

class TestMPNNEncoder:
    def test_output_shape_single(self, encoder, single_batch):
        """Single molecule → predictions shape [1, 4]."""
        out = encoder(single_batch.x, single_batch.edge_index,
                      single_batch.edge_attr, single_batch.batch)
        assert out.shape == (1, 4), f"Expected (1,4) got {out.shape}"

    def test_output_shape_batch(self, encoder, multi_batch):
        """Batch of 3 molecules → predictions shape [3, 4]."""
        out = encoder(multi_batch.x, multi_batch.edge_index,
                      multi_batch.edge_attr, multi_batch.batch)
        assert out.shape == (3, 4), f"Expected (3,4) got {out.shape}"

    def test_qed_in_zero_one(self, encoder, multi_batch):
        """QED column (index 1) must always be in [0, 1] due to sigmoid."""
        out = encoder(multi_batch.x, multi_batch.edge_index,
                      multi_batch.edge_attr, multi_batch.batch)
        qed = out[:, 1]
        assert (qed >= 0).all() and (qed <= 1).all(), \
            f"QED out of range: {qed}"

    def test_output_is_finite(self, encoder, multi_batch):
        """No NaN or Inf in predictions."""
        out = encoder(multi_batch.x, multi_batch.edge_index,
                      multi_batch.edge_attr, multi_batch.batch)
        assert torch.isfinite(out).all()

    def test_encode_shape(self, encoder, multi_batch):
        """encode() should return [B, hidden_dim] not [B, 4]."""
        h_G = encoder.encode(multi_batch.x, multi_batch.edge_index,
                             multi_batch.edge_attr, multi_batch.batch)
        assert h_G.shape == (3, 128)

    def test_different_molecules_different_embeddings(self, encoder, multi_batch):
        """Different molecules must produce different graph embeddings."""
        h_G = encoder.encode(multi_batch.x, multi_batch.edge_index,
                             multi_batch.edge_attr, multi_batch.batch)
        # Pairwise differences should all be non-zero
        assert not torch.allclose(h_G[0], h_G[1])
        assert not torch.allclose(h_G[0], h_G[2])

    def test_gradients_flow_end_to_end(self, encoder, multi_batch):
        """Gradients must flow all the way back to input features."""
        x = multi_batch.x.clone().requires_grad_(True)
        out = encoder(x, multi_batch.edge_index,
                      multi_batch.edge_attr, multi_batch.batch)
        out.sum().backward()
        assert x.grad is not None

    def test_parameter_count_reasonable(self, encoder):
        """Model should have between 100k and 1M parameters."""
        n_params = sum(p.numel() for p in encoder.parameters())
        assert 100_000 < n_params < 1_000_000, \
            f"Unexpected parameter count: {n_params:,}"