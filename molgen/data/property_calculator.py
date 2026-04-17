"""
property_calculator.py

Computes molecular properties for QM9 molecules using RDKit.
Handles LogP, QED, and SA Score. HOMO-LUMO gap comes from QM9 labels directly.

All properties are normalised to [0, 1] using training-set statistics
so the model sees consistent input scale regardless of property units.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import RDConfig
import os
import sys

# SA Score is in an RDKit contrib script — not part of the main API
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore


# ── Raw property computation ───────────────────────────────────────────────────

def compute_logp(mol: Chem.Mol) -> float:
    """
    Wildman-Crippen LogP — lipophilicity / water-solubility proxy.
    Typical drug range: -1 to 5. No hard bounds.
    """
    return Descriptors.MolLogP(mol)


def compute_qed(mol: Chem.Mol) -> float:
    """
    Quantitative Estimate of Drug-likeness. Always in [0, 1].
    Combines 8 drug-likeness rules into one score.
    """
    return QED.qed(mol)


def compute_sa_score(mol: Chem.Mol) -> float:
    """
    Synthetic Accessibility Score. Range [1, 10].
    1 = trivially easy to synthesise, 10 = nearly impossible.
    """
    return sascorer.calculateScore(mol)


def compute_properties(smiles: str) -> Optional[dict[str, float]]:
    """
    Compute all three RDKit properties for a given SMILES string.

    Args:
        smiles: A SMILES string.

    Returns:
        Dict with keys 'logp', 'qed', 'sa_score', or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "logp":     compute_logp(mol),
        "qed":      compute_qed(mol),
        "sa_score": compute_sa_score(mol),
    }


# ── Normalisation ──────────────────────────────────────────────────────────────

@dataclass
class PropertyStats:
    """
    Stores mean and std for each property computed over the training set.
    Used to normalise properties to zero mean, unit variance.

    We use z-score normalisation (not min-max) because:
    - LogP has no hard bounds, so min-max would break on out-of-range molecules
    - z-score is more robust to outliers
    - The model just needs consistent scale, not [0,1] range

    QM9 approximate statistics (computed from the full dataset):
    These will be recomputed from actual training data in dataset.py —
    these defaults are just reasonable fallbacks.
    """
    logp_mean:      float = 2.536
    logp_std:       float = 1.423
    qed_mean:       float = 0.478
    qed_std:        float = 0.210
    sa_mean:        float = 3.051
    sa_std:         float = 0.830
    homo_lumo_mean: float = 0.274   # eV, from QM9
    homo_lumo_std:  float = 0.073


class PropertyNormaliser:
    """
    Normalises and denormalises molecular properties using z-score scaling.

    Usage:
        normaliser = PropertyNormaliser(stats)
        normed = normaliser.normalise({"logp": 2.3, "qed": 0.7, ...})
        original = normaliser.denormalise(normed)
    """

    def __init__(self, stats: Optional[PropertyStats] = None) -> None:
        self.stats = stats or PropertyStats()

    def normalise(self, props: dict[str, float]) -> dict[str, float]:
        """Scale raw property values to zero mean, unit variance."""
        s = self.stats
        result = {}
        if "logp" in props:
            result["logp"]      = (props["logp"]     - s.logp_mean)      / s.logp_std
        if "qed" in props:
            result["qed"]       = (props["qed"]      - s.qed_mean)       / s.qed_std
        if "sa_score" in props:
            result["sa_score"]  = (props["sa_score"] - s.sa_mean)        / s.sa_std
        if "homo_lumo" in props:
            result["homo_lumo"] = (props["homo_lumo"]- s.homo_lumo_mean) / s.homo_lumo_std
        return result

    def denormalise(self, props: dict[str, float]) -> dict[str, float]:
        """Invert normalisation — convert model outputs back to real units."""
        s = self.stats
        result = {}
        if "logp" in props:
            result["logp"]      = props["logp"]      * s.logp_std  + s.logp_mean
        if "qed" in props:
            result["qed"]       = props["qed"]       * s.qed_std   + s.qed_mean
        if "sa_score" in props:
            result["sa_score"]  = props["sa_score"]  * s.sa_std    + s.sa_mean
        if "homo_lumo" in props:
            result["homo_lumo"] = props["homo_lumo"] * s.homo_lumo_std + s.homo_lumo_mean
        return result

    def update_stats_from_dataset(self, all_props: list[dict[str, float]]) -> None:
        """
        Recompute mean and std from actual training data.
        Call this once after loading QM9, before any normalisation.

        Args:
            all_props: List of property dicts, one per molecule in training set.
        """
        logps     = [p["logp"]     for p in all_props if "logp"     in p]
        qeds      = [p["qed"]      for p in all_props if "qed"      in p]
        sas       = [p["sa_score"] for p in all_props if "sa_score" in p]
        homo_lumos= [p["homo_lumo"]for p in all_props if "homo_lumo"in p]

        if logps:
            self.stats.logp_mean = float(np.mean(logps))
            self.stats.logp_std  = float(np.std(logps)) or 1.0

        if qeds:
            self.stats.qed_mean = float(np.mean(qeds))
            self.stats.qed_std  = float(np.std(qeds)) or 1.0

        if sas:
            self.stats.sa_mean = float(np.mean(sas))
            self.stats.sa_std  = float(np.std(sas)) or 1.0

        if homo_lumos:
            self.stats.homo_lumo_mean = float(np.mean(homo_lumos))
            self.stats.homo_lumo_std  = float(np.std(homo_lumos)) or 1.0