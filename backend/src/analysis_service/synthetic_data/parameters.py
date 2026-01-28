"""
IRT parameter configuration and joint sampling using Gaussian copula.

This module provides:
- Configuration dataclasses for specifying marginal distributions
- Correlation matrix building and validation
- Gaussian copula-based joint sampling of IRT parameters
- Config loading from YAML files using OmegaConf
"""

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
from analysis_service.synthetic_data.data_models import SampledParameters
from analysis_service.synthetic_data.missingness import (
    MissingnessModel,
    get_missingness_model,
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

        # Create missingness model
        self._missingness_model: MissingnessModel = get_missingness_model(
            config.missing.model, config.missing.params
        )

    @staticmethod
    def _recenter_array(array: NDArray[np.float64]) -> NDArray[np.float64]:
        center = np.mean(array, axis=1, keepdims=True, dtype=np.float64)
        return array - center

    def _recenter_parameters(
        self,
        params: SampledParameters,
    ) -> None:
        """Recenters the discrimination and intercept parameters"""
        assert params.includes_missing_values
        params.discrimination = self._recenter_array(params.discrimination)
        params.intercept = self._recenter_array(params.intercept)

    def sample(
        self,
        n_questions: int,
        n_response_categories: int,
        correct_answer_ix: NDArray[np.int64],
        theta: NDArray[np.float64],
        rng: Generator,
    ) -> SampledParameters:
        """Sample item parameters with correlations.

        Algorithm (Gaussian copula):
        1. Draw independent standard normals Z ~ N(0, I), shape (n, 3)
        2. Correlate: Y = Z @ L.T where L is Cholesky of correlation matrix
        3. Transform to uniform: U = norm.cdf(Y)
        4. Apply inverse marginal CDF (ppf) for each parameter
        5. Apply missingness model to add missing category

        Args:
            n_questions: Number of questions.
            n_response_categories: Number of response categories (excluding missing).
            correct_answer_ix: Index of correct answer for each question.
            theta: Candidate abilities, shape (n_candidates,).
            rng: Random number generator.

        Returns:
            SampledParameters with missing category included.
        """

        # Step 1: Independent standard normals
        Z = rng.standard_normal((n_questions * n_response_categories, 3))

        # Step 2: Apply correlation via Cholesky
        Y = Z @ self._cholesky.T

        # Step 3: Transform to uniform [0, 1]
        U = stats.norm.cdf(Y)

        # Step 4: Apply inverse marginal CDFs
        sampled_discriminations = self._distributions[
            "discrimination"
        ].inverse_cdf(U[:, 0])
        sampled_discriminations = sampled_discriminations.reshape(
            (n_questions, n_response_categories)
        )

        sampled_intercepts = self._distributions["intercept"].inverse_cdf(
            U[:, 1]
        )
        sampled_intercepts = sampled_intercepts.reshape(
            (n_questions, n_response_categories)
        )
        sampled_gap = self._distributions[
            "correct_discrimination_gap"
        ].inverse_cdf(U[:, 2])
        sampled_gap = sampled_gap.reshape((n_questions, n_response_categories))

        # For each correct item, add the sampled_gap value to the sampled_discrimination
        # Leave all the remaining sampled_discrimination values as is
        rows = np.arange(sampled_gap.shape[0])
        vals = sampled_gap[rows, correct_answer_ix]
        sampled_gap[:] = 0
        sampled_gap[rows, correct_answer_ix] = vals

        assert sampled_discriminations.shape == sampled_gap.shape

        sampled_discriminations += sampled_gap

        # Step 5: Apply missingness model to add missing category
        raw_parameters = SampledParameters(
            discrimination=sampled_discriminations,
            intercept=sampled_intercepts,
            includes_missing_values=False,
        )

        parameters = self._missingness_model.generate_missing_params(
            raw_parameters, theta, rng
        )

        # Recenter parameters
        self._recenter_parameters(parameters)

        return parameters


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
