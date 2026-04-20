"""
Tests for metrics.py and evaluator.py

Run with: pytest tests/test_evaluation.py -v
"""

import pytest
from molgen.evaluation.metrics import (
    is_valid,
    canonicalise,
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    morgan_fingerprint,
    tanimoto_similarity,
)
from molgen.evaluation.evaluator import MolGenEvaluator


# ── Sample data ────────────────────────────────────────────────────────────────

VALID_SMILES = [
    "CCO",                          # ethanol
    "c1ccccc1",                     # benzene
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",# caffeine
]
INVALID_SMILES = ["not_a_molecule", "$$$$", "", "XYZ123###"]
DUPLICATE_SMILES = ["CCO", "OCC", "C(C)O"]   # all ethanol, different orderings


# ── is_valid ───────────────────────────────────────────────────────────────────

class TestIsValid:
    def test_valid_molecules(self):
        for s in VALID_SMILES:
            assert is_valid(s), f"Expected valid: {s}"

    def test_invalid_molecules(self):
        for s in INVALID_SMILES:
            assert not is_valid(s), f"Expected invalid: {s}"

    def test_empty_string(self):
        assert not is_valid("")

    def test_none_like(self):
        assert not is_valid(None)


# ── canonicalise ──────────────────────────────────────────────────────────────

class TestCanonicalise:
    def test_same_molecule_same_canonical(self):
        """Different SMILES for ethanol should all canonicalise to CCO."""
        results = {canonicalise(s) for s in DUPLICATE_SMILES}
        assert len(results) == 1, f"Expected 1 canonical form, got {results}"

    def test_invalid_returns_none(self):
        assert canonicalise("not_valid_$$") is None


# ── compute_validity ───────────────────────────────────────────────────────────

class TestComputeValidity:
    def test_all_valid(self):
        assert compute_validity(VALID_SMILES) == 1.0

    def test_all_invalid(self):
        assert compute_validity(INVALID_SMILES) == 0.0

    def test_mixed(self):
        mixed = VALID_SMILES[:2] + INVALID_SMILES[:2]  # 2 valid, 2 invalid
        assert abs(compute_validity(mixed) - 0.5) < 1e-6

    def test_empty_list(self):
        assert compute_validity([]) == 0.0


# ── compute_uniqueness ─────────────────────────────────────────────────────────

class TestComputeUniqueness:
    def test_all_unique(self):
        uniqueness, unique = compute_uniqueness(VALID_SMILES)
        assert uniqueness == 1.0
        assert len(unique) == len(VALID_SMILES)

    def test_all_duplicates(self):
        """All ethanol variants should collapse to one unique molecule."""
        uniqueness, unique = compute_uniqueness(DUPLICATE_SMILES)
        assert len(unique) == 1
        assert uniqueness < 1.0

    def test_empty_list(self):
        uniqueness, unique = compute_uniqueness([])
        assert uniqueness == 0.0
        assert unique == []

    def test_returns_canonical_smiles(self):
        """Unique list should contain canonical SMILES, not raw input."""
        _, unique = compute_uniqueness(DUPLICATE_SMILES)
        assert unique[0] == "CCO"


# ── compute_novelty ────────────────────────────────────────────────────────────

class TestComputeNovelty:
    def test_all_novel(self):
        """Generated molecules completely different from training."""
        generated = ["CCO", "c1ccccc1"]
        training  = ["CC(=O)O", "CCC"]   # acetic acid, propane
        novelty   = compute_novelty(generated, training, threshold=0.4)
        assert novelty > 0.0

    def test_none_novel(self):
        """If generated == training, novelty should be 0."""
        smiles  = ["CCO", "c1ccccc1"]
        novelty = compute_novelty(smiles, smiles, threshold=0.99)
        assert novelty == 0.0

    def test_empty_generated(self):
        assert compute_novelty([], ["CCO"]) == 0.0

    def test_empty_training(self):
        assert compute_novelty(["CCO"], []) == 0.0


# ── Morgan fingerprints ────────────────────────────────────────────────────────

class TestMorganFingerprint:
    def test_valid_molecule_returns_fingerprint(self):
        fp = morgan_fingerprint("CCO")
        assert fp is not None

    def test_invalid_molecule_returns_none(self):
        fp = morgan_fingerprint("not_valid_$$")
        assert fp is None

    def test_identical_molecules_identical_fingerprints(self):
        fp1 = morgan_fingerprint("CCO")
        fp2 = morgan_fingerprint("OCC")   # same molecule
        assert tanimoto_similarity(fp1, fp2) == pytest.approx(1.0)

    def test_different_molecules_lower_similarity(self):
        fp1 = morgan_fingerprint("CCO")       # ethanol
        fp2 = morgan_fingerprint("c1ccccc1")  # benzene
        assert tanimoto_similarity(fp1, fp2) < 0.5


# ── MolGenEvaluator ────────────────────────────────────────────────────────────

class TestMolGenEvaluator:
    @pytest.fixture
    def evaluator(self):
        training = ["CC(=O)O", "CCC", "CCCC", "c1ccccc1O"]
        return MolGenEvaluator(training_smiles=training)

    def test_evaluate_returns_all_keys(self, evaluator):
        results = evaluator.evaluate(VALID_SMILES, compute_fcd_score=False)
        for key in ["validity", "uniqueness", "novelty", "n_generated",
                    "n_valid", "n_unique", "property_mae", "fcd"]:
            assert key in results, f"Missing key: {key}"

    def test_validity_correct(self, evaluator):
        mixed   = VALID_SMILES[:2] + INVALID_SMILES[:2]
        results = evaluator.evaluate(mixed, compute_fcd_score=False)
        assert abs(results["validity"] - 0.5) < 1e-6

    def test_uniqueness_deduplicates(self, evaluator):
        results = evaluator.evaluate(
            DUPLICATE_SMILES + VALID_SMILES[:2], compute_fcd_score=False
        )
        assert results["n_unique"] <= results["n_valid"]

    def test_property_mae_computed(self, evaluator):
        targets = {"logp": 2.0, "qed": 0.7, "sa_score": 3.0}
        results = evaluator.evaluate(
            VALID_SMILES, target_properties=targets, compute_fcd_score=False
        )
        assert "logp"     in results["property_mae"]
        assert "qed"      in results["property_mae"]
        assert "sa_score" in results["property_mae"]
        for mae in results["property_mae"].values():
            assert mae >= 0.0

    def test_empty_input(self, evaluator):
        results = evaluator.evaluate([], compute_fcd_score=False)
        assert results["validity"]   == 0.0
        assert results["uniqueness"] == 0.0
        assert results["n_generated"] == 0

    def test_summary_returns_string(self, evaluator):
        results = evaluator.evaluate(VALID_SMILES, compute_fcd_score=False)
        summary = evaluator.summary(results)
        assert isinstance(summary, str)
        assert "Valid" in summary
        assert "Unique" in summary