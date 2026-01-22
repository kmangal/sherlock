"""
Tests for Gauss-Hermite quadrature.
"""

import numpy as np

from analysis_service.irt.estimation.config import QuadratureConfig
from analysis_service.irt.estimation.quadrature import get_quadrature


class TestGaussHermiteQuadrature:
    def test_weights_sum_to_one(self) -> None:
        """Quadrature weights should sum to 1."""
        config = QuadratureConfig(n_points=41)
        quad = get_quadrature(config)

        np.testing.assert_allclose(quad.weights.sum(), 1.0, rtol=1e-10)

    def test_weights_positive(self) -> None:
        """All weights should be positive."""
        config = QuadratureConfig(n_points=41)
        quad = get_quadrature(config)

        assert (quad.weights > 0).all()

    def test_points_symmetric(self) -> None:
        """Points should be symmetric around the mean."""
        config = QuadratureConfig(n_points=41, mean=0.0, std=1.0)
        quad = get_quadrature(config)

        # Mean of points should be close to specified mean
        np.testing.assert_allclose(
            np.sum(quad.points * quad.weights), 0.0, atol=1e-10
        )

    def test_variance_matches_std(self) -> None:
        """Variance of quadrature should match specified std^2."""
        config = QuadratureConfig(n_points=41, mean=0.0, std=1.0)
        quad = get_quadrature(config)

        # E[X^2] - E[X]^2
        mean = np.sum(quad.points * quad.weights)
        variance = np.sum(quad.points**2 * quad.weights) - mean**2

        np.testing.assert_allclose(variance, 1.0, rtol=1e-3)

    def test_scaled_distribution(self) -> None:
        """Should correctly scale to non-standard normal."""
        config = QuadratureConfig(n_points=41, mean=2.0, std=0.5)
        quad = get_quadrature(config)

        # Mean should be approximately 2.0
        mean = np.sum(quad.points * quad.weights)
        np.testing.assert_allclose(mean, 2.0, rtol=1e-3)

        # Variance should be approximately 0.25
        variance = np.sum(quad.points**2 * quad.weights) - mean**2
        np.testing.assert_allclose(variance, 0.25, rtol=1e-3)

    def test_different_n_points(self) -> None:
        """Should work with different numbers of points."""
        for n_points in [11, 21, 41, 61]:
            config = QuadratureConfig(n_points=n_points)
            quad = get_quadrature(config)

            assert quad.n_points == n_points
            assert len(quad.points) == n_points
            assert len(quad.weights) == n_points
            np.testing.assert_allclose(quad.weights.sum(), 1.0, rtol=1e-10)

    def test_integrates_polynomial_exactly(self) -> None:
        """Gauss-Hermite should integrate low-degree polynomials exactly."""
        config = QuadratureConfig(n_points=21, mean=0.0, std=1.0)
        quad = get_quadrature(config)

        # E[X^3] for standard normal should be 0
        third_moment = np.sum(quad.points**3 * quad.weights)
        np.testing.assert_allclose(third_moment, 0.0, atol=1e-10)

        # E[X^4] for standard normal should be 3
        fourth_moment = np.sum(quad.points**4 * quad.weights)
        np.testing.assert_allclose(fourth_moment, 3.0, rtol=1e-3)
