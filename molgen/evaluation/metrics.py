"""
metrics.py

Core evaluation metrics for generated molecules.

Validity   — can RDKit parse and sanitise the molecule?
Uniqueness — are the generated molecules all different from each other?
Novelty    — are they different from molecules in the training set?
FCD        — Fréchet ChemNet Distance (distribution-level quality)
"""

import os
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Silence RDKit warnings during evaluation
RDLogger.DisableLog("rdApp.*")


# ── Validity ───────────────────────────────────────────────────────────────────

def is_valid(smiles: str) -> bool:
    """
    Return True if SMILES can be parsed and sanitised by RDKit.
    A molecule that fails this check cannot be used in a lab or simulation.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def canonicalise(smiles: str) -> Optional[str]:
    """
    Return canonical SMILES, or None if invalid.
    Canonical form removes atom ordering artefacts so
    CCO and OCC both become CCO.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def compute_validity(smiles_list: list[str]) -> float:
    """
    Fraction of generated SMILES that are chemically valid.

    Args:
        smiles_list: Raw SMILES strings from the generator.

    Returns:
        validity: float in [0, 1]
    """
    if not smiles_list:
        return 0.0
    n_valid = sum(1 for s in smiles_list if is_valid(s))
    return n_valid / len(smiles_list)


# ── Uniqueness ─────────────────────────────────────────────────────────────────

def compute_uniqueness(smiles_list: list[str]) -> tuple[float, list[str]]:
    """
    Fraction of valid molecules that are unique (no duplicates).

    Args:
        smiles_list: SMILES strings (may include invalid ones).

    Returns:
        (uniqueness, unique_valid_smiles)
        uniqueness:          float in [0, 1]
        unique_valid_smiles: deduplicated list of valid canonical SMILES
    """
    valid = [canonicalise(s) for s in smiles_list if is_valid(s)]
    if not valid:
        return 0.0, []

    unique = list(set(valid))
    return len(unique) / len(valid), unique


# ── Morgan Fingerprints + Tanimoto ────────────────────────────────────────────

def morgan_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048):
    """
    Compute Morgan (circular) fingerprint for a SMILES string.
    Returns None if the molecule is invalid.

    Morgan fingerprints are the standard for molecular similarity —
    each bit represents the presence of a particular circular substructure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto_similarity(fp1, fp2) -> float:
    """
    Tanimoto (Jaccard) similarity between two Morgan fingerprints.
    Range [0, 1]. 1.0 = identical, 0.0 = no shared substructures.

    Threshold for "novel": Tanimoto < 0.4 vs all training molecules.
    """
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def max_tanimoto_to_set(fp, reference_fps: list) -> float:
    """
    Maximum Tanimoto similarity between one fingerprint and a reference set.
    Used to check if a generated molecule is genuinely novel.
    """
    if not reference_fps:
        return 0.0
    from rdkit import DataStructs
    similarities = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    return max(similarities)


# ── Novelty ────────────────────────────────────────────────────────────────────

def compute_novelty(
    unique_smiles:   list[str],
    training_smiles: list[str],
    threshold:       float = 0.4,
) -> float:
    """
    Fraction of unique generated molecules that are novel vs training set.

    A molecule is novel if its maximum Tanimoto similarity to any training
    molecule is below `threshold` (default 0.4).

    Tanimoto < 0.4 = genuinely new scaffold
    Tanimoto 0.4-0.7 = similar but not identical
    Tanimoto > 0.7 = likely just a minor variant

    Args:
        unique_smiles:   Deduplicated valid generated SMILES.
        training_smiles: All SMILES in the training set.
        threshold:       Similarity cutoff for novelty (default 0.4).

    Returns:
        novelty: float in [0, 1]
    """
    if not unique_smiles or not training_smiles:
        return 0.0

    # Precompute training fingerprints
    train_fps = []
    for s in training_smiles:
        fp = morgan_fingerprint(s)
        if fp is not None:
            train_fps.append(fp)

    if not train_fps:
        return 0.0

    n_novel = 0
    for s in unique_smiles:
        fp = morgan_fingerprint(s)
        if fp is None:
            continue
        max_sim = max_tanimoto_to_set(fp, train_fps)
        if max_sim < threshold:
            n_novel += 1

    return n_novel / len(unique_smiles)


# ── FCD — Fréchet ChemNet Distance ─────────────────────────────────────────────

def compute_fcd(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> Optional[float]:
    """
    Fréchet ChemNet Distance between generated and reference molecules.

    FCD measures how similar the *distribution* of generated molecules is
    to real drug-like molecules, using embeddings from a pretrained ChemNet
    neural network. Lower = better.

    FCD < 1.0  — excellent (state of the art)
    FCD < 5.0  — good
    FCD > 10.0 — poor

    Returns None if fcd_torch is not installed or inputs are too small.

    Args:
        generated_smiles: Valid generated SMILES (at least 2).
        reference_smiles: Reference SMILES (training set or drug database).
    """
    try:
        import fcd
    except ImportError:
        print("fcd-torch not installed — skipping FCD computation.")
        return None

    # FCD needs at least 2 molecules
    gen_valid = [s for s in generated_smiles if is_valid(s)]
    ref_valid = [s for s in reference_smiles  if is_valid(s)]

    if len(gen_valid) < 2 or len(ref_valid) < 2:
        return None

    try:
        fcd_score = fcd.get_fcd(gen_valid, ref_valid)
        return float(fcd_score)
    except Exception as e:
        print(f"FCD computation failed: {e}")
        return None