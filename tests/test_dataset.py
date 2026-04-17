"""
Tests for dataset.py

These tests use max_mols=50 so they run fast locally.
Full QM9 loading is tested implicitly during training.

Run with: pytest tests/test_dataset.py -v
"""

import pytest
import torch
from molgen.data.dataset import MolGenDataset, get_dataloaders


@pytest.fixture(scope="module")
def tiny_train():
    """Load 50 molecules — fast enough for CI."""
    return MolGenDataset(root="data/test_cache", split="train", max_mols=50)


class TestMolGenDataset:
    def test_dataset_nonempty(self, tiny_train):
        assert len(tiny_train) > 0

    def test_data_object_fields(self, tiny_train):
        """Each item must have x, edge_index, edge_attr, y, smiles."""
        item = tiny_train[0]
        assert hasattr(item, "x")
        assert hasattr(item, "edge_index")
        assert hasattr(item, "edge_attr")
        assert hasattr(item, "y")
        assert hasattr(item, "smiles")

    def test_node_feature_dim(self, tiny_train):
        """Node features must be 18-dimensional."""
        item = tiny_train[0]
        assert item.x.shape[1] == 18

    def test_edge_feature_dim(self, tiny_train):
        """Edge features must be 6-dimensional."""
        item = tiny_train[0]
        assert item.edge_attr.shape[1] == 6

    def test_label_shape(self, tiny_train):
        """Labels must be shape [1, 4]."""
        item = tiny_train[0]
        assert item.y.shape == (1, 4)

    def test_labels_are_finite(self, tiny_train):
        """No NaN or Inf in labels."""
        for item in tiny_train:
            assert torch.isfinite(item.y).all()


class TestDataLoaders:
    def test_get_dataloaders_returns_three(self):
        train, val, test = get_dataloaders(
            root="data/test_cache", batch_size=16, max_mols=50
        )
        assert train is not None
        assert val   is not None
        assert test  is not None

    def test_batch_shapes(self):
        train, _, _ = get_dataloaders(
            root="data/test_cache", batch_size=16, max_mols=50
        )
        batch = next(iter(train))
        # x should be [total_atoms_in_batch, 18]
        assert batch.x.shape[1] == 18
        # y should be [num_graphs_in_batch, 4]
        assert batch.y.shape[1] == 4