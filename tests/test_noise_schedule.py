"""
Tests for noise_schedule.py

Run with: pytest tests/test_noise_schedule.py -v
"""

import torch
import pytest
from molgen.utils.noise_schedule import CosineNoiseSchedule


@pytest.fixture
def schedule():
    return CosineNoiseSchedule(T=1000)


class TestCosineNoiseSchedule:
    def test_alphas_bar_starts_at_one(self, schedule):
        """α̅_0 should be 1.0 — no corruption at t=0."""
        assert abs(float(schedule.alphas_bar[0]) - 1.0) < 1e-4

    def test_alphas_bar_ends_near_zero(self, schedule):
        """α̅_T should be close to 0 — full corruption at t=T."""
        assert float(schedule.alphas_bar[-1]) < 0.01

    def test_alphas_bar_monotonically_decreasing(self, schedule):
        """Signal retention must strictly decrease over time."""
        diffs = schedule.alphas_bar[1:] - schedule.alphas_bar[:-1]
        assert (diffs <= 0).all()

    def test_q_sample_shape(self, schedule):
        """q_sample must return tensors of same shape as input."""
        x0 = torch.randn(10, 18)
        t  = torch.full((10,), 500, dtype=torch.long)
        x_t, eps = schedule.q_sample(x0, t)
        assert x_t.shape == x0.shape
        assert eps.shape == x0.shape

    def test_q_sample_t0_is_clean(self, schedule):
        """At t=0, x_t should equal x_0 (no noise added)."""
        x0  = torch.randn(5, 18)
        t   = torch.zeros(5, dtype=torch.long)
        eps = torch.zeros_like(x0)   # pass zero noise
        x_t, _ = schedule.q_sample(x0, t, eps=eps)
        # sqrt(α̅_0)=1, sqrt(1-α̅_0)≈0, so x_t ≈ x0
        assert torch.allclose(x_t, x0, atol=1e-3)

    def test_q_sample_tT_is_noisy(self, schedule):
        """At t=T, x_t should be dominated by noise, not signal."""
        x0  = torch.zeros(100, 18)   # clean signal is all zeros
        t   = torch.full((100,), 999, dtype=torch.long)
        x_t, _ = schedule.q_sample(x0, t)
        # If dominated by noise, std should be > 0.5
        assert x_t.std() > 0.5

    def test_predict_x0_roundtrip(self, schedule):
        """predict_x0_from_eps should recover x_0 from (x_t, eps)."""
        x0  = torch.randn(8, 18)
        t   = torch.full((8,), 250, dtype=torch.long)
        x_t, eps = schedule.q_sample(x0, t)
        x0_hat   = schedule.predict_x0_from_eps(x_t, t, eps)
        assert torch.allclose(x0_hat, x0, atol=1e-4)

    def test_corrupt_adjacency_shape(self, schedule):
        """Corrupted adjacency must have same shape as input."""
        A0  = torch.randint(0, 2, (9, 9))
        A_t = schedule.corrupt_adjacency(A0, t=500)
        assert A_t.shape == A0.shape

    def test_corrupt_adjacency_t0_unchanged(self, schedule):
        """At t=0, adjacency should not be flipped (p_flip≈0)."""
        A0  = torch.zeros(9, 9, dtype=torch.long)
        A_t = schedule.corrupt_adjacency(A0, t=0)
        assert torch.equal(A_t, A0)

    def test_all_values_finite(self, schedule):
        """No NaN or Inf in any precomputed tensor."""
        for name in ["alphas_bar", "alphas", "betas",
                     "sqrt_ab", "sqrt_one_minus_ab", "posterior_variance"]:
            tensor = getattr(schedule, name)
            assert torch.isfinite(tensor).all(), f"Non-finite values in {name}"