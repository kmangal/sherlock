"""
Tests for NRM gradient computations.

Verifies analytical gradients match numerical gradients.
"""

import numpy as np

from analysis_service.irt.estimation.gradients import (
    compute_nrm_probabilities,
    nrm_negative_expected_log_likelihood,
    nrm_negative_expected_log_likelihood_gradient,
)


class TestNRMProbabilities:
    def test_probabilities_sum_to_one(self) -> None:
        """Probabilities should sum to 1 across categories."""
        theta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        discriminations = np.array([0.0, 0.5, 1.0, 1.5])
        intercepts = np.array([0.0, 0.3, -0.5, 0.2])

        probs = compute_nrm_probabilities(theta, discriminations, intercepts)

        # Shape check
        assert probs.shape == (5, 4)

        # Sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-10)

        # All positive
        assert (probs > 0).all()
        assert (probs < 1).all()

    def test_reference_category_at_origin(self) -> None:
        """At theta=0, probabilities depend only on intercepts."""
        theta = np.array([0.0])
        discriminations = np.array([0.0, 1.0, 2.0])
        intercepts = np.array([0.0, 0.0, 0.0])

        probs = compute_nrm_probabilities(theta, discriminations, intercepts)

        # With all intercepts = 0 and theta = 0, all categories should be equal
        np.testing.assert_allclose(probs[0], [1 / 3, 1 / 3, 1 / 3], rtol=1e-10)

    def test_discrimination_effect(self) -> None:
        """Higher discrimination should increase separation at extreme theta."""
        theta = np.array([2.0])
        # Category 2 has highest discrimination
        discriminations = np.array([0.0, 0.5, 2.0])
        intercepts = np.array([0.0, 0.0, 0.0])

        probs = compute_nrm_probabilities(theta, discriminations, intercepts)

        # Category 2 should have highest probability at high theta
        assert probs[0, 2] > probs[0, 1] > probs[0, 0]


class TestNRMGradients:
    def test_gradient_matches_numerical(self) -> None:
        """Analytical gradient should match numerical gradient."""
        n_categories = 4
        n_quadrature = 11
        n_candidates = 50

        # Setup
        theta = np.linspace(-3, 3, n_quadrature, dtype=np.float64)

        # Random posteriors (normalized per candidate, float32 per UPDATES.md)
        rng = np.random.default_rng(42)
        posteriors_raw = rng.random((n_candidates, n_quadrature))
        posteriors = (
            posteriors_raw / posteriors_raw.sum(axis=1, keepdims=True)
        ).astype(np.float32)

        # Random response indicators (one-hot per candidate)
        response_indicators = np.zeros((n_candidates, n_categories))
        responses = rng.integers(0, n_categories, size=n_candidates)
        response_indicators[np.arange(n_candidates), responses] = 1.0

        # Parameters to test
        params = np.array([0.5, 1.0, 0.8, 0.2, -0.3, 0.1])  # 3 a's, 3 b's

        # Analytical gradient
        grad_analytical = nrm_negative_expected_log_likelihood_gradient(
            params, theta, posteriors, response_indicators, n_categories
        )

        # Numerical gradient
        eps = 1e-6
        grad_numerical = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps

            f_plus = nrm_negative_expected_log_likelihood(
                params_plus,
                theta,
                posteriors,
                response_indicators,
                n_categories,
            )
            f_minus = nrm_negative_expected_log_likelihood(
                params_minus,
                theta,
                posteriors,
                response_indicators,
                n_categories,
            )

            grad_numerical[i] = (f_plus - f_minus) / (2 * eps)

        # Compare
        np.testing.assert_allclose(
            grad_analytical, grad_numerical, rtol=1e-4, atol=1e-6
        )

    def test_gradient_at_uniform_posteriors(self) -> None:
        """Test gradient with uniform posteriors."""
        n_categories = 3
        n_quadrature = 5
        n_candidates = 20

        theta = np.linspace(-2, 2, n_quadrature, dtype=np.float64)

        # Uniform posteriors (float32)
        posteriors = (
            np.ones((n_candidates, n_quadrature)) / n_quadrature
        ).astype(np.float32)

        # All respond with category 1
        response_indicators = np.zeros((n_candidates, n_categories))
        response_indicators[:, 1] = 1.0

        # Test at origin (all params = 0)
        params = np.zeros(2 * (n_categories - 1))

        grad = nrm_negative_expected_log_likelihood_gradient(
            params, theta, posteriors, response_indicators, n_categories
        )

        # Gradient should be non-zero (pushing toward observed responses)
        assert not np.allclose(grad, 0)

    def test_loss_decreases_along_gradient(self) -> None:
        """Loss should decrease when moving in negative gradient direction."""
        n_categories = 4
        n_quadrature = 21
        n_candidates = 100

        theta = np.linspace(-3, 3, n_quadrature, dtype=np.float64)

        rng = np.random.default_rng(123)
        posteriors_raw = rng.random((n_candidates, n_quadrature))
        posteriors = posteriors_raw / posteriors_raw.sum(axis=1, keepdims=True)

        response_indicators = np.zeros((n_candidates, n_categories))
        responses = rng.integers(0, n_categories, size=n_candidates)
        response_indicators[np.arange(n_candidates), responses] = 1.0

        # Start at some non-optimal point
        params = np.array([0.5, 1.0, 0.3, 0.2, -0.3, 0.1])

        # Get loss and gradient
        loss = nrm_negative_expected_log_likelihood(
            params, theta, posteriors, response_indicators, n_categories
        )
        grad = nrm_negative_expected_log_likelihood_gradient(
            params, theta, posteriors, response_indicators, n_categories
        )

        # Take a small step in negative gradient direction
        step_size = 0.01
        params_new = params - step_size * grad

        loss_new = nrm_negative_expected_log_likelihood(
            params_new, theta, posteriors, response_indicators, n_categories
        )

        # Loss should decrease (or stay same if already at minimum)
        assert loss_new <= loss + 1e-10


class TestNRMNumericalStability:
    def test_extreme_theta_values(self) -> None:
        """Should handle extreme theta values without overflow."""
        theta = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        discriminations = np.array([0.0, 2.0, 3.0])
        intercepts = np.array([0.0, 1.0, -1.0])

        probs = compute_nrm_probabilities(theta, discriminations, intercepts)

        # Should still sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-10)

        # Should not have NaN or Inf
        assert np.isfinite(probs).all()

    def test_extreme_parameters(self) -> None:
        """Should handle extreme parameter values."""
        theta = np.array([-2.0, 0.0, 2.0])
        discriminations = np.array([0.0, 4.0, -2.0])  # Large discrimination
        intercepts = np.array([0.0, 5.0, -5.0])  # Large intercepts

        probs = compute_nrm_probabilities(theta, discriminations, intercepts)

        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-10)
