"""
Tests for property_calculator.py

Run with: pytest tests/test_property_calculator.py -v
"""

import pytest
from molgen.data.property_calculator import (
    compute_properties,
    PropertyNormaliser,
    PropertyStats,
)


class TestComputeProperties:
    def test_ethanol_returns_dict(self):
        """compute_properties should return a dict with 3 keys."""
        props = compute_properties("CCO")
        assert props is not None
        assert set(props.keys()) == {"logp", "qed", "sa_score"}

    def test_logp_is_float(self):
        props = compute_properties("CCO")
        assert isinstance(props["logp"], float)

    def test_qed_in_range(self):
        """QED is always between 0 and 1 by definition."""
        for smiles in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]:
            props = compute_properties(smiles)
            assert 0.0 <= props["qed"] <= 1.0, f"QED out of range for {smiles}"

    def test_sa_score_in_range(self):
        """SA Score is always between 1 and 10 by definition."""
        for smiles in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]:
            props = compute_properties(smiles)
            assert 1.0 <= props["sa_score"] <= 10.0

    def test_invalid_smiles_returns_none(self):
        assert compute_properties("not_valid_$$") is None


class TestPropertyNormaliser:
    def test_normalise_then_denormalise_roundtrip(self):
        """Normalise then denormalise should recover original values."""
        normaliser = PropertyNormaliser()
        original = {"logp": 2.3, "qed": 0.7, "sa_score": 3.5, "homo_lumo": 0.28}
        normed   = normaliser.normalise(original)
        recovered = normaliser.denormalise(normed)

        for key in original:
            assert abs(recovered[key] - original[key]) < 1e-5, \
                f"Roundtrip failed for {key}: {recovered[key]} != {original[key]}"

    def test_normalised_values_are_finite(self):
        """Normalised values should not be NaN or Inf."""
        import math
        normaliser = PropertyNormaliser()
        props = compute_properties("CC(=O)Oc1ccccc1C(=O)O")
        normed = normaliser.normalise(props)
        for k, v in normed.items():
            assert math.isfinite(v), f"Non-finite value for {k}: {v}"

    def test_update_stats_from_dataset(self):
        """Stats should update when we pass training data."""
        normaliser = PropertyNormaliser()
        fake_data = [
            {"logp": 1.0, "qed": 0.5, "sa_score": 2.0, "homo_lumo": 0.3},
            {"logp": 3.0, "qed": 0.7, "sa_score": 4.0, "homo_lumo": 0.2},
        ]
        normaliser.update_stats_from_dataset(fake_data)
        assert abs(normaliser.stats.logp_mean - 2.0) < 1e-5   # mean of [1.0, 3.0]
        assert abs(normaliser.stats.qed_mean  - 0.6) < 1e-5   # mean of [0.5, 0.7]