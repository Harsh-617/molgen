"""
graph_builder.py

Converts a SMILES string into a PyTorch Geometric Data object.
This is the entry point for all molecular data in MolGen.
"""

from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data


# ── Vocabulary ────────────────────────────────────────────────────────────────

# All atom types we recognise. Anything outside this list → 'other' (last slot)
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]

# Hybridisation states we encode
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

# Bond types we encode
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


# ── Helper: one-hot encoding ───────────────────────────────────────────────────

def one_hot(value: object, vocab: list) -> list[float]:
    """
    Return a one-hot list of length len(vocab).
    If value is not in vocab, the last element is set to 1 (catch-all 'other').

    Example:
        one_hot("C", ATOM_TYPES) → [1,0,0,0,0,0,0,0,0,0]
        one_hot("X", ATOM_TYPES) → [0,0,0,0,0,0,0,0,0,1]
    """
    encoding = [0.0] * len(vocab)
    if value in vocab:
        encoding[vocab.index(value)] = 1.0
    else:
        encoding[-1] = 1.0          # 'other' slot
    return encoding


# ── Atom featurisation ─────────────────────────────────────────────────────────

def atom_features(atom: Chem.Atom) -> list[float]:
    """
    Build an 18-dimensional feature vector for a single RDKit atom.

    Breakdown:
        [0:10]  Atom type one-hot          (10 dims)
        [10]    Formal charge, normalised   ( 1 dim )
        [11:16] Hybridisation one-hot       ( 5 dims)
        [16]    Is aromatic (binary)        ( 1 dim )
        [17]    Num implicit Hs, normalised ( 1 dim )
    Total: 18 dims
    """
    return (
        one_hot(atom.GetSymbol(), ATOM_TYPES)                   # 10
        + [float(atom.GetFormalCharge()) / 4.0]                 #  1  (divide by 4: range ±4)
        + one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES) #  5
        + [float(atom.GetIsAromatic())]                         #  1
        + [float(atom.GetTotalNumHs()) / 4.0]                   #  1  (divide by 4: max ~4 Hs)
    )  # total: 18


# ── Bond featurisation ─────────────────────────────────────────────────────────

def bond_features(bond: Chem.Bond) -> list[float]:
    """
    Build a 6-dimensional feature vector for a single RDKit bond.

    Breakdown:
        [0:4]  Bond type one-hot   (4 dims)
        [4]    Is conjugated       (1 dim)
        [5]    Is in ring          (1 dim)
    Total: 6 dims
    """
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)  # 4
        + [float(bond.GetIsConjugated())]        # 1
        + [float(bond.IsInRing())]               # 1
    )  # total: 6


# ── Main builder ───────────────────────────────────────────────────────────────

def smiles_to_graph(
    smiles: str,
    y: Optional[torch.Tensor] = None,
) -> Optional[Data]:
    """
    Convert a SMILES string into a PyTorch Geometric Data object.

    Args:
        smiles: A valid SMILES string, e.g. "CCO" for ethanol.
        y:      Optional graph-level label tensor of shape [1, num_properties].
                Pass None during inference when labels are not available.

    Returns:
        A PyG Data object with:
            x          [N, 18]      node (atom) feature matrix
            edge_index [2, 2*E]     bond connectivity (both directions)
            edge_attr  [2*E, 6]     edge (bond) feature matrix
            y          [1, P]       graph-level property labels (if provided)
            smiles     str          the canonical SMILES (for deduplication later)

        Returns None if the SMILES is invalid or cannot be sanitised.
    """
    # ── Parse + sanitise ──────────────────────────────────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None                 # invalid SMILES → skip

    # Add hydrogens implicitly (they are encoded in H-count feature)
    # We do NOT add explicit H atoms as nodes — keeps graphs small
    Chem.SanitizeMol(mol)

    # ── Node features ─────────────────────────────────────────────────────────
    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)  # [N, 18]

    # ── Edge features ─────────────────────────────────────────────────────────
    # We store each bond twice (both directions) so messages flow both ways
    sources, targets, edge_feats = [], [], []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)

        sources  += [i, j]          # forward + reverse
        targets  += [j, i]
        edge_feats += [feat, feat]  # same features for both directions

    edge_index = torch.tensor([sources, targets], dtype=torch.long)   # [2, 2E]
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float)          # [2E, 6]

    # ── Canonical SMILES ──────────────────────────────────────────────────────
    # Canonical = a single unique string for each molecule regardless of how
    # the atoms were ordered in the input. Used for deduplication in evaluation.
    canonical = Chem.MolToSmiles(mol, canonical=True)

    # ── Assemble Data object ──────────────────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=canonical,
    )

    if y is not None:
        data.y = y

    return data