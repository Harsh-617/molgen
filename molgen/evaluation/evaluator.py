"""
evaluator.py

Main evaluation class — runs all metrics on a list of generated SMILES
and returns a structured results dict.

Usage:
    evaluator = MolGenEvaluator(training_smiles)
    results   = evaluator.evaluate(generated_smiles, target_properties)
    print(results)
"""

from typing import Optional
import numpy as np

from molgen.evaluation.metrics import (
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    compute_fcd,
    is_valid,
    canonicalise,
)
from molgen.data.property_calculator import compute_properties


class MolGenEvaluator:
    """
    Evaluates a set of generated molecules against multiple metrics.

    Metrics computed:
        validity      — fraction of chemically valid molecules
        uniqueness    — fraction of valid molecules that are unique
        novelty       — fraction of unique molecules not in training set
        property_mae  — mean absolute error vs target properties (per property)
        fcd           — Fréchet ChemNet Distance vs training set

    Args:
        training_smiles: List of SMILES from the training set.
                         Used for novelty and FCD computation.
                         Can be None to skip those metrics.
    """

    def __init__(self, training_smiles: Optional[list[str]] = None) -> None:
        self.training_smiles = training_smiles or []

    def evaluate(
        self,
        generated_smiles:  list[str],
        target_properties: Optional[dict[str, float]] = None,
        compute_fcd_score: bool = True,
    ) -> dict:
        """
        Run full evaluation on a list of generated SMILES.

        Args:
            generated_smiles:  Raw SMILES from the generator (may include invalid).
            target_properties: Dict of target values e.g.
                               {'logp': 2.0, 'qed': 0.8, 'sa_score': 3.0}
                               Used to compute property MAE.
                               Pass None to skip property accuracy.
            compute_fcd_score: Whether to compute FCD (slow for large sets).

        Returns:
            results dict with keys:
                n_generated, n_valid, n_unique, n_novel,
                validity, uniqueness, novelty,
                property_mae (dict), fcd,
                valid_smiles, unique_smiles
        """
        results = {
            "n_generated": len(generated_smiles),
            "validity":    0.0,
            "uniqueness":  0.0,
            "novelty":     0.0,
            "n_valid":     0,
            "n_unique":    0,
            "n_novel":     0,
            "property_mae": {},
            "fcd":          None,
            "valid_smiles":  [],
            "unique_smiles": [],
        }

        if not generated_smiles:
            return results

        # ── Validity ──────────────────────────────────────────────────────────
        results["validity"] = compute_validity(generated_smiles)
        results["n_valid"]  = int(results["validity"] * len(generated_smiles))

        # ── Uniqueness ────────────────────────────────────────────────────────
        uniqueness, unique_smiles = compute_uniqueness(generated_smiles)
        results["uniqueness"]    = uniqueness
        results["unique_smiles"] = unique_smiles
        results["n_unique"]      = len(unique_smiles)

        # ── Novelty ───────────────────────────────────────────────────────────
        if self.training_smiles and unique_smiles:
            results["novelty"] = compute_novelty(unique_smiles, self.training_smiles)
            results["n_novel"] = int(results["novelty"] * len(unique_smiles))

        # ── Property MAE ──────────────────────────────────────────────────────
        if target_properties and unique_smiles:
            results["property_mae"] = self._compute_property_mae(
                unique_smiles, target_properties
            )

        # ── FCD ───────────────────────────────────────────────────────────────
        if compute_fcd_score and self.training_smiles and len(unique_smiles) >= 2:
            results["fcd"] = compute_fcd(unique_smiles, self.training_smiles)

        results["valid_smiles"] = [
            canonicalise(s) for s in generated_smiles if is_valid(s)
        ]

        return results

    def _compute_property_mae(
        self,
        smiles_list:       list[str],
        target_properties: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute mean absolute error between predicted and target properties.

        Uses RDKit to compute actual properties of generated molecules,
        then compares to the user-specified targets.

        Args:
            smiles_list:       Valid generated SMILES.
            target_properties: Target property values (unnormalised).

        Returns:
            Dict of MAE per property, e.g.
            {'logp': 0.3, 'qed': 0.05, 'sa_score': 0.4}
        """
        property_errors: dict[str, list[float]] = {
            k: [] for k in target_properties
        }

        for smiles in smiles_list:
            props = compute_properties(smiles)
            if props is None:
                continue
            for prop_name, target_val in target_properties.items():
                if prop_name in props:
                    error = abs(props[prop_name] - target_val)
                    property_errors[prop_name].append(error)

        return {
            prop: float(np.mean(errors)) if errors else float("nan")
            for prop, errors in property_errors.items()
        }

    def summary(self, results: dict) -> str:
        """Format results as a human-readable string."""
        lines = [
            f"Generated:  {results['n_generated']}",
            f"Valid:      {results['n_valid']} "
            f"({results['validity']*100:.1f}%)",
            f"Unique:     {results['n_unique']} "
            f"({results['uniqueness']*100:.1f}% of valid)",
            f"Novel:      {results['n_novel']} "
            f"({results['novelty']*100:.1f}% of unique)",
        ]
        if results["property_mae"]:
            lines.append("Property MAE:")
            for prop, mae in results["property_mae"].items():
                lines.append(f"  {prop}: {mae:.3f}")
        if results["fcd"] is not None:
            lines.append(f"FCD:        {results['fcd']:.3f}")
        return "\n".join(lines)