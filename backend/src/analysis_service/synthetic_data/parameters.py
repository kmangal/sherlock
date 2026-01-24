"""
IRT parameter configuration and joint sampling using Gaussian copula.

This module provides:
- Configuration dataclasses for specifying marginal distributions
- Correlation matrix building and validation
- Gaussian copula-based joint sampling of IRT parameters
- Config loading from YAML files using OmegaConf
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from omegaconf import OmegaConf
from scipy import stats

from analysis_service.synthetic_data.config import (
    CorrelationsConfig,
    DistributionConfig,
    GenerationConfig,
)
from analysis_service.synthetic_data.sampling import (
    Distribution,
    registry,
)

# =============================================================================
# Correlation Matrix Utilities
# =============================================================================


def build_correlation_matrix(
    correlations: CorrelationsConfig,
) -> NDArray[np.float64]:
    """Convert pairwise correlations to 2x2 correlation matrix.

    Order: [discrimination, intercept, correct_gap]

    Args:
        correlations: Pairwise correlation configuration

    Returns:
        3x3 correlation matrix

    Raises:
        ValueError: If correlations do not form a valid (PSD) correlation matrix
    """
    R = np.eye(3, dtype=np.float64)
    R[0, 1] = R[1, 0] = correlations.discrimination_intercept
    R[0, 2] = R[2, 0] = correlations.discrimination_correct_gap
    R[1, 2] = R[2, 1] = correlations.intercept_correct_gap

    if not _validate_psd(R):
        raise ValueError(
            f"Specified correlations do not form a valid correlation matrix. "
            f"The matrix must be positive semi-definite. Got:\n{R}"
        )
    return R


def _validate_psd(matrix: NDArray[np.float64]) -> bool:
    """Check if correlation matrix is positive semi-definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigenvalues >= -1e-10))


# =============================================================================
# Distribution Factory using sampling.py registry
# =============================================================================


def create_distribution(config: DistributionConfig) -> Distribution:
    """Create a Distribution from configuration using the sampling distributoin registry.

    Args:
        config: Distribution configuration

    Returns:
        Distribution object that supports .sample(), .inverse_cdf(), and .cdf()

    Raises:
        ValueError: If distribution type is unknown
    """
    return registry.get_sampler(
        name=config.distribution,
        params=config.params,
    )


# =============================================================================
# Joint Parameter Sampler
# =============================================================================


class SamplingDistributions(TypedDict):
    discrimination: Distribution
    intercept: Distribution
    correct_discrimination_gap: Distribution


@dataclass
class SampledParameters:
    discrimination: NDArray[np.float64]
    intercept: NDArray[np.float64]


class JointParameterSampler:
    """Sample IRT parameters using Gaussian copula for correlated parameters.

    The copula approach allows specifying:
    - Arbitrary marginal distributions for each parameter
    - Pairwise correlations between parameters
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.correlation_matrix = build_correlation_matrix(config.correlations)

        # Precompute Cholesky decomposition for correlated sampling
        self._cholesky = np.linalg.cholesky(self.correlation_matrix)

        # Create distributions for each parameter using the registry
        self._distributions: SamplingDistributions = {
            "discrimination": create_distribution(
                config.nrm_parameters.discrimination
            ),
            "intercept": create_distribution(config.nrm_parameters.intercept),
            "correct_discrimination_gap": create_distribution(
                config.nrm_parameters.correct_discrimination_gap
            ),
        }

    def sample(
        self,
        n_questions: int,
        n_choices: int,
        correct_answer_ix: NDArray[np.int64],
        rng: Generator,
    ) -> SampledParameters:
        """Sample item parameters with correlations.

        Algorithm (Gaussian copula):
        1. Draw independent standard normals Z ~ N(0, I), shape (n, 3)
        2. Correlate: Y = Z @ L.T where L is Cholesky of correlation matrix
        3. Transform to uniform: U = norm.cdf(Y)
        4. Apply inverse marginal CDF (ppf) for each parameter
        """

        # Step 1: Independent standard normals
        Z = rng.standard_normal((n_questions * n_choices, 3))

        # Step 2: Apply correlation via Cholesky
        Y = Z @ self._cholesky.T

        # Step 3: Transform to uniform [0, 1]
        U = stats.norm.cdf(Y)

        # Step 4: Apply inverse marginal CDFs
        sampled_discriminations = self._distributions[
            "discrimination"
        ].inverse_cdf(U[:, 0])
        sampled_discriminations = sampled_discriminations.reshape(
            (n_questions, n_choices)
        )

        sampled_intercepts = self._distributions["discrimination"].inverse_cdf(
            U[:, 1]
        )
        sampled_intercepts = sampled_intercepts.reshape(
            (n_questions, n_choices)
        )
        sampled_gap = self._distributions[
            "correct_discrimination_gap"
        ].inverse_cdf(U[:, 2])
        sampled_gap = sampled_gap.reshape((n_questions, n_choices))

        # Zero out all values that do not pertain to the correct answer
        rows = np.arange(sampled_gap.shape[0])
        vals = sampled_gap[rows, correct_answer_ix]
        sampled_gap[:] = 0
        sampled_gap[rows, correct_answer_ix] = vals

        assert sampled_discriminations.shape == sampled_gap.shape

        sampled_discriminations += sampled_gap

        result = SampledParameters(
            discrimination=sampled_discriminations,
            intercept=sampled_intercepts,
        )

        return result


# =============================================================================
# Config Loading
# =============================================================================


def load_config(yaml_path: Path) -> GenerationConfig:
    """Load and validate parameters from YAML.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Validated GenerationConfig

    Raises:
        ValueError: If correlations are invalid (not PSD)
        FileNotFoundError: If yaml_path doesn't exist
    """
    # Create schema from dataclass
    schema = OmegaConf.structured(GenerationConfig)

    if yaml_path is not None:
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        user_config = OmegaConf.load(yaml_path)
        config = OmegaConf.merge(schema, user_config)
    else:
        config = schema

    # Convert to typed dataclass
    result = OmegaConf.to_object(config)
    assert isinstance(result, GenerationConfig)

    return result
