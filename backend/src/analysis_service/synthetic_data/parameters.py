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
    """Convert pairwise correlations to 3x3 correlation matrix.

    Order: [difficulty, discrimination, guessing]

    Args:
        correlations: Pairwise correlation configuration

    Returns:
        3x3 correlation matrix

    Raises:
        ValueError: If correlations do not form a valid (PSD) correlation matrix
    """
    R = np.eye(3, dtype=np.float64)
    R[0, 1] = R[1, 0] = correlations.difficulty_discrimination
    R[0, 2] = R[2, 0] = correlations.difficulty_guessing
    R[1, 2] = R[2, 1] = correlations.discrimination_guessing

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


class CorrelatedParameters(TypedDict):
    difficulty: NDArray[np.float64]
    discrimination: NDArray[np.float64]
    guessing: NDArray[np.float64]


@dataclass
class SampledParameters:
    difficulty: NDArray[np.float64]
    discrimination: NDArray[np.float64]
    guessing: NDArray[np.float64]
    distractor_quality: NDArray[np.float64]


class JointParameterSampler:
    """Sample IRT parameters using Gaussian copula for correlated parameters.

    The copula approach allows specifying:
    - Arbitrary marginal distributions for each parameter
    - Pairwise correlations between parameters

    Difficulty, discrimination, and guessing are sampled jointly with correlations.
    Distractor quality is sampled independently (per-distractor, not per-question).
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.correlation_matrix = build_correlation_matrix(config.correlations)

        # Precompute Cholesky decomposition for correlated sampling
        self._cholesky = np.linalg.cholesky(self.correlation_matrix)

        # Create distributions for each parameter using the registry
        self._distributions = {
            "difficulty": create_distribution(
                config.irt_parameters.difficulty
            ),
            "discrimination": create_distribution(
                config.irt_parameters.discrimination
            ),
            "guessing": create_distribution(config.irt_parameters.guessing),
        }
        self._distractor_dist = create_distribution(
            config.irt_parameters.distractor_quality
        )

    def sample(
        self, n_questions: int, n_distractors: int, rng: Generator
    ) -> SampledParameters:
        """Sample IRT parameters for multiple questions.

        Args:
            n_questions: Number of questions to sample parameters for
            n_distractors: Number of distractors per question (n_choices - 1)
            rng: NumPy random generator

        Returns:
            Dictionary with keys:
                - "difficulty": shape (n_questions,)
                - "discrimination": shape (n_questions,)
                - "guessing": shape (n_questions,)
                - "distractor_quality": shape (n_questions, n_distractors)
        """
        correlated = self._sample_correlated(n_questions, rng)
        distractor_quality = self._sample_distractor_quality(
            n_questions, n_distractors, rng
        )

        return SampledParameters(
            difficulty=correlated["difficulty"],
            discrimination=correlated["discrimination"],
            guessing=correlated["guessing"],
            distractor_quality=distractor_quality,
        )

    def _sample_correlated(
        self, n_questions: int, rng: Generator
    ) -> CorrelatedParameters:
        """Sample difficulty, discrimination, guessing with correlations.

        Algorithm (Gaussian copula):
        1. Draw independent standard normals Z ~ N(0, I), shape (n, 3)
        2. Correlate: Y = Z @ L.T where L is Cholesky of correlation matrix
        3. Transform to uniform: U = norm.cdf(Y)
        4. Apply inverse marginal CDF (ppf) for each parameter
        """
        # Step 1: Independent standard normals
        Z = rng.standard_normal((n_questions, 3))

        # Step 2: Apply correlation via Cholesky
        Y = Z @ self._cholesky.T

        # Step 3: Transform to uniform [0, 1]
        U = stats.norm.cdf(Y)

        # Step 4: Apply inverse marginal CDFs
        result: CorrelatedParameters = {
            "difficulty": self._distributions["difficulty"].inverse_cdf(
                U[:, 0]
            ),
            "discrimination": self._distributions[
                "discrimination"
            ].inverse_cdf(U[:, 1]),
            "guessing": self._distributions["guessing"].inverse_cdf(U[:, 2]),
        }

        return result

    def _sample_distractor_quality(
        self, n_questions: int, n_distractors: int, rng: Generator
    ) -> NDArray[np.float64]:
        """Sample distractor quality independently for each question and distractor.

        Uses the Distribution's sample method for direct sampling (no copula needed).
        """
        # Sample flat array and reshape
        flat_samples = self._distractor_dist.sample(
            n_questions * n_distractors, rng
        )
        result: NDArray[np.float64] = flat_samples.reshape(
            n_questions, n_distractors
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

    # Validate correlation matrix is PSD
    build_correlation_matrix(result.correlations)

    return result
