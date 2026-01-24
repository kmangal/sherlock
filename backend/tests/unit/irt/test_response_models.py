"""
Tests for NRM item parameters and response sampling.

The NRM model uses sum-to-zero identification: Σa_k = 0, Σb_k = 0.
This constrains the model to have K-1 free parameters per type.
"""

import numpy as np
import pytest

from analysis_service.core.utils import get_rng
from analysis_service.irt import (
    NRMItemParameters,
    sample_response,
    sample_responses_batch,
)


class TestNRMProbabilities:
    """Tests for NRMItemParameters.compute_probabilities."""

    def test_probabilities_sum_to_one(self) -> None:
        """Probabilities must sum to 1 across all categories."""
        item = NRMItemParameters(
            item_id=0,
            discriminations=(0.5, 0.3, -0.4, -0.4),  # Sum = 0
            intercepts=(0.2, -0.1, 0.1, -0.2),  # Sum = 0
        )

        for ability in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            theta = np.array([ability], dtype=np.float64)
            probs = item.compute_probabilities(theta)[0]
            assert np.isclose(probs.sum(), 1.0), (
                f"Probs don't sum to 1 at θ={ability}"
            )
            assert np.all(probs >= 0), "Negative probabilities"
            assert np.all(probs <= 1), "Probabilities > 1"

    def test_batch_probabilities_shape(self) -> None:
        """Batch computation should return correct shape."""
        item = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.5, -0.5),
            intercepts=(0.0, 0.0, 0.0),
        )

        theta = np.array([-2.0, 0.0, 0.2, 2.0], dtype=np.float64)
        probs = item.compute_probabilities(theta)

        assert probs.shape == (4, 3), f"Wrong shape: {probs.shape}"
        for i in range(4):
            assert np.isclose(probs[i].sum(), 1.0)

    def test_discrimination_ordering_effect(self) -> None:
        """Higher discrimination category should dominate at extreme theta."""
        # Category 1 has highest discrimination
        item = NRMItemParameters(
            item_id=0,
            discriminations=(-0.5, 1.5, -1.0),  # Sum = 0
            intercepts=(0.0, 0.0, 0.0),
        )

        # At high theta, category with highest a should have highest prob
        theta_high = np.array([3.0], dtype=np.float64)
        probs_high = item.compute_probabilities(theta_high)[0]
        assert probs_high[1] > probs_high[0]
        assert probs_high[1] > probs_high[2]

        # At low theta, category with lowest a should have highest prob
        theta_low = np.array([-3.0], dtype=np.float64)
        probs_low = item.compute_probabilities(theta_low)[0]
        assert probs_low[2] > probs_low[0]
        assert probs_low[2] > probs_low[1]

    def test_intercept_effect_at_origin(self) -> None:
        """At theta=0, probabilities should depend only on intercepts."""
        # Zero discriminations, varying intercepts
        item = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.0, 0.0),
            intercepts=(1.0, 0.0, -1.0),  # Sum = 0
        )

        theta = np.array([0.0], dtype=np.float64)
        probs = item.compute_probabilities(theta)[0]

        # Higher intercept -> higher probability
        assert probs[0] > probs[1] > probs[2]

    def test_uniform_at_equal_params(self) -> None:
        """Equal parameters should give uniform probabilities."""
        item = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.0, 0.0, 0.0),
            intercepts=(0.0, 0.0, 0.0, 0.0),
        )

        theta = np.array([0.0], dtype=np.float64)
        probs = item.compute_probabilities(theta)[0]

        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_allclose(probs, expected, rtol=1e-10)


class TestNRMNumericalStability:
    """Tests for numerical stability of probability computations."""

    def test_extreme_theta_no_overflow(self) -> None:
        """Should handle extreme theta without overflow."""
        item = NRMItemParameters(
            item_id=0,
            discriminations=(1.0, 0.5, -1.5),
            intercepts=(0.5, -0.5, 0.0),
        )

        for theta_val in [-20.0, -10.0, 10.0, 20.0]:
            theta = np.array([theta_val], dtype=np.float64)
            probs = item.compute_probabilities(theta)[0]

            assert np.isfinite(probs).all(), f"Non-finite at θ={theta_val}"
            assert np.isclose(probs.sum(), 1.0), f"Sum != 1 at θ={theta_val}"

    def test_extreme_discriminations(self) -> None:
        """Should handle large discrimination values."""
        item = NRMItemParameters(
            item_id=0,
            discriminations=(3.0, 2.0, -5.0),  # Large values, sum = 0
            intercepts=(0.0, 0.0, 0.0),
        )

        theta = np.array([2.0], dtype=np.float64)
        probs = item.compute_probabilities(theta)[0]

        assert np.isfinite(probs).all()
        assert np.isclose(probs.sum(), 1.0)


class TestSampling:
    """Tests for response sampling functions."""

    def test_sample_response_valid_range(self) -> None:
        """Sampled response should be valid category index."""
        item = NRMItemParameters(
            item_id=0,
            discriminations=(0.3, 0.2, -0.25, -0.25),
            intercepts=(0.1, 0.0, -0.05, -0.05),
        )
        rng = get_rng(42)

        for _ in range(100):
            response = sample_response(0.0, item, rng=rng)
            assert 0 <= response < 4

    def test_sample_response_respects_probabilities(self) -> None:
        """Sampling should respect probability distribution."""
        # Category 0 should be very likely
        item = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.0, 0.0),
            intercepts=(5.0, -2.5, -2.5),  # Category 0 much more likely
        )
        rng = get_rng(42)

        counts = [0, 0, 0]
        n_samples = 1000
        for _ in range(n_samples):
            response = sample_response(0.0, item, rng=rng)
            counts[response] += 1

        # Category 0 should have most samples
        assert counts[0] > 0.9 * n_samples

    def test_sample_responses_batch_shape(self) -> None:
        """Batch sampling should return correct shape."""
        abilities = np.array([0.0, 1.0, -1.0])
        item_params = [
            NRMItemParameters(
                item_id=i,
                discriminations=(0.0, 0.3, -0.3),
                intercepts=(0.0, 0.0, 0.0),
            )
            for i in range(5)
        ]
        rng = get_rng(42)

        responses = sample_responses_batch(abilities, item_params, rng=rng)

        assert responses.shape == (3, 5)
        assert responses.dtype == np.int8

    def test_sample_responses_batch_valid_indices(self) -> None:
        """All batch responses should be valid indices."""
        abilities = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        n_categories = 4
        item_params = [
            NRMItemParameters(
                item_id=i,
                discriminations=(0.2, 0.1, -0.15, -0.15),
                intercepts=(0.1, 0.0, -0.05, -0.05),
            )
            for i in range(10)
        ]
        rng = get_rng(42)

        responses = sample_responses_batch(abilities, item_params, rng=rng)

        assert np.all(responses >= 0)
        assert np.all(responses < n_categories)

    def test_high_ability_prefers_high_discrimination(self) -> None:
        """High ability candidates should prefer high-discrimination categories."""
        # Category 0 has highest discrimination
        item = NRMItemParameters(
            item_id=0,
            discriminations=(2.0, 0.0, -1.0, -1.0),  # Sum = 0
            intercepts=(0.0, 0.0, 0.0, 0.0),
        )
        rng = get_rng(42)

        counts = [0, 0, 0, 0]
        n_samples = 500
        high_ability = 3.0

        for _ in range(n_samples):
            response = sample_response(high_ability, item, rng=rng)
            counts[response] += 1

        # Category 0 should dominate at high ability
        assert counts[0] > 0.8 * n_samples


class TestSumToZeroConstraint:
    """Tests verifying sum-to-zero identification is respected."""

    def test_create_default_satisfies_constraint(self) -> None:
        """Default parameters should satisfy sum-to-zero."""
        params = NRMItemParameters.create_default(item_id=0, n_categories=4)

        assert sum(params.discriminations) == pytest.approx(0.0)
        assert sum(params.intercepts) == pytest.approx(0.0)

    def test_from_array_enforces_constraint(self) -> None:
        """from_array should always produce sum-to-zero parameters."""
        # Arbitrary free parameters
        arr = np.array([1.0, -0.5, 0.3, 0.2, -0.1, 0.4])

        params = NRMItemParameters.from_array(
            item_id=0, arr=arr, n_categories=4
        )

        assert sum(params.discriminations) == pytest.approx(0.0)
        assert sum(params.intercepts) == pytest.approx(0.0)

    def test_to_array_from_array_roundtrip(self) -> None:
        """Roundtrip should preserve sum-to-zero parameters."""
        # Start with sum-to-zero params
        original = NRMItemParameters(
            item_id=0,
            discriminations=(0.5, 0.3, -0.8),  # Sum = 0
            intercepts=(0.2, -0.1, -0.1),  # Sum = 0
        )

        arr = original.to_array()
        reconstructed = NRMItemParameters.from_array(
            item_id=0, arr=arr, n_categories=3
        )

        np.testing.assert_allclose(
            original.discriminations, reconstructed.discriminations, rtol=1e-10
        )
        np.testing.assert_allclose(
            original.intercepts, reconstructed.intercepts, rtol=1e-10
        )
