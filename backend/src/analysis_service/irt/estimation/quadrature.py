"""
Gauss-Hermite quadrature for latent variable integration.

This module provides quadrature points and weights for numerical integration
over the standard normal ability distribution, used in MML-EM estimation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis_service.irt.estimation.config import QuadratureConfig


@dataclass(frozen=True)
class GaussHermiteQuadrature:
    """
    Gauss-Hermite quadrature points and weights.

    The points and weights are scaled to integrate over N(mean, std^2).

    Attributes:
        points: Quadrature points (theta values), shape (n_points,).
        weights: Quadrature weights (probabilities), shape (n_points,).
            Weights sum to 1.
    """

    points: NDArray[np.float64]
    weights: NDArray[np.float64]

    @property
    def n_points(self) -> int:
        """Number of quadrature points."""
        return len(self.points)


def get_quadrature(config: QuadratureConfig) -> GaussHermiteQuadrature:
    """
    Generate Gauss-Hermite quadrature points and weights.

    Uses numpy's hermgauss function and transforms from physicists' Hermite
    polynomials (which integrate exp(-x^2)) to probabilists' convention
    (which integrates the standard normal distribution).

    The transformation is:
        - x_prob = sqrt(2) * x_phys
        - w_prob = w_phys / sqrt(pi)

    Then scales to N(mean, std^2):
        - theta = mean + std * x_prob
        - weights normalized to sum to 1

    Args:
        config: Quadrature configuration specifying number of points,
            mean, and standard deviation.

    Returns:
        GaussHermiteQuadrature with points and weights.
    """
    # Get physicists' Hermite-Gauss quadrature
    x_phys, w_phys = np.polynomial.hermite.hermgauss(config.n_points)

    # Transform to probabilists' convention
    x_prob = np.sqrt(2.0) * x_phys
    w_prob = w_phys / np.sqrt(np.pi)

    # Scale to N(mean, std^2)
    theta = config.mean + config.std * x_prob

    # Normalize weights to sum to 1 (should already be close)
    weights = w_prob / w_prob.sum()

    return GaussHermiteQuadrature(
        points=theta.astype(np.float64),
        weights=weights.astype(np.float64),
    )
