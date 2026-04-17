"""
Tests for graph_builder.py

Run with: pytest tests/test_graph_builder.py -v
"""

import pytest
import torch
from molgen.data.graph_builder import smiles_to_graph, atom_features, bond_features, ATOM_TYPES


class TestOneHotAndFeatures:
    def test_atom_feature_length(self):
        """Every atom must produce exactly 18 features."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        for atom in mol.GetAtoms():
            feats = atom_features(atom)
            assert len(feats) == 18, f"Expected 18, got {len(feats)}"

    def test_bond_feature_length(self):
        """Every bond must produce exactly 6 features."""
        from rdkit import Chem
        from molgen.data.graph_builder import bond_features
        mol = Chem.MolFromSmiles("CCO")
        for bond in mol.GetBonds():
            feats = bond_features(bond)
            assert len(feats) == 6, f"Expected 6, got {len(feats)}"

    def test_unknown_atom_uses_other_slot(self):
        """Atom type not in vocab should set last element to 1."""
        from molgen.data.graph_builder import one_hot
        enc = one_hot("Xe", ATOM_TYPES)   # Xenon not in our vocab
        assert enc[-1] == 1.0
        assert sum(enc) == 1.0            # exactly one 1


class TestSmilestoGraph:
    def test_ethanol_shapes(self):
        """Ethanol (CCO): 3 heavy atoms, 2 bonds → check tensor shapes."""
        data = smiles_to_graph("CCO")
        assert data is not None

        # 3 heavy atoms (C, C, O) — H atoms are implicit
        assert data.x.shape == (3, 18)

        # 2 bonds × 2 directions = 4 edges
        assert data.edge_index.shape == (2, 4)
        assert data.edge_attr.shape  == (4, 6)

    def test_benzene_shapes(self):
        """Benzene (c1ccccc1): 6 atoms, 6 bonds (ring)."""
        data = smiles_to_graph("c1ccccc1")
        assert data is not None
        assert data.x.shape[0] == 6          # 6 carbon atoms
        assert data.edge_index.shape[1] == 12 # 6 bonds × 2 directions

    def test_invalid_smiles_returns_none(self):
        """Garbage SMILES should return None, not crash."""
        data = smiles_to_graph("not_a_molecule_$$$$")
        assert data is None

    def test_canonical_smiles_stored(self):
        """Data object should store the canonical SMILES string."""
        data = smiles_to_graph("OCC")   # same as ethanol but different order
        assert hasattr(data, "smiles")
        assert data.smiles == "CCO"     # canonical form

    def test_with_label(self):
        """Passing y should attach it to the Data object."""
        y = torch.tensor([[2.3, 0.7, 3.1, 5.0]])   # fake property values
        data = smiles_to_graph("CCO", y=y)
        assert data.y.shape == (1, 4)

    def test_feature_values_are_finite(self):
        """No NaN or Inf in any feature tensor."""
        data = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert torch.isfinite(data.x).all()
        assert torch.isfinite(data.edge_attr).all()

    def test_edge_index_valid_range(self):
        """All node indices in edge_index must be within [0, N-1]."""
        data = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        n_atoms = data.x.shape[0]
        assert data.edge_index.min() >= 0
        assert data.edge_index.max() < n_atoms