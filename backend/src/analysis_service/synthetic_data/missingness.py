"""
Missing answer mechanisms via item parameter generation.

This module generates item parameters for a "missing" response category.
Instead of applying missingness post-hoc, the missing category is modeled
as an additional response option in the NRM.

Supports:
- NoMissingness: ~0 probability missing category
- MCAR: Missing Completely At Random
- AbilityDependentMissingness: Lower ability -> higher missing rate
- PositionDependentMissingness: Later questions -> higher missing rate (fatigue)
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.optimize import brentq

from analysis_service.synthetic_data.data_models import SampledParameters

# Constant for near-zero probability missing category
NO_MISSING_INTERCEPT = -30.0

# Bounds for intercept search
INTERCEPT_SEARCH_MIN = -50.0
INTERCEPT_SEARCH_MAX = 50.0


def _compute_expected_missing_rate(
    intercept: float,
    discrimination: float,
    item_scores: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> float:
    """Compute expected missing rate for a single item across all candidates.

    Args:
        intercept: Missing category intercept (b_missing).
        discrimination: Missing category discrimination (a_missing).
        item_scores: Shape (n_candidates, n_categories) - scores for existing categories.
        theta: Shape (n_candidates,) - candidate abilities.

    Returns:
        Expected missing rate E[P(missing|theta)].
    """
    # Missing category score for each candidate
    missing_scores = (
        discrimination * theta + intercept
    )  # Shape: (n_candidates,)

    # For numerical stability, subtract max before exp
    all_scores = np.column_stack([item_scores, missing_scores])
    max_scores = np.max(all_scores, axis=1, keepdims=True)
    exp_scores = np.exp(all_scores - max_scores)

    # Missing probability for each candidate
    missing_probs = exp_scores[:, -1] / np.sum(exp_scores, axis=1)

    # Expected rate is mean across candidates
    return float(np.mean(missing_probs))


def _calibrate_intercept(
    target_rate: float,
    discrimination: float,
    item_scores: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> float:
    """Find intercept that achieves target missing rate in expectation.

    Uses Brent's method to solve:
        E[P(missing|theta)] = target_rate

    Args:
        target_rate: Desired expected missing rate.
        discrimination: Missing category discrimination.
        item_scores: Shape (n_candidates, n_categories) - scores for existing categories.
        theta: Shape (n_candidates,) - candidate abilities.

    Returns:
        Calibrated intercept value.
    """

    def objective(intercept: float) -> float:
        actual = _compute_expected_missing_rate(
            intercept, discrimination, item_scores, theta
        )
        return actual - target_rate

    try:
        result = brentq(objective, INTERCEPT_SEARCH_MIN, INTERCEPT_SEARCH_MAX)
        return float(result)
    except ValueError:
        # If root not found in range, use boundary that gets closest
        low_rate = _compute_expected_missing_rate(
            INTERCEPT_SEARCH_MIN, discrimination, item_scores, theta
        )
        high_rate = _compute_expected_missing_rate(
            INTERCEPT_SEARCH_MAX, discrimination, item_scores, theta
        )
        if abs(low_rate - target_rate) < abs(high_rate - target_rate):
            return INTERCEPT_SEARCH_MIN
        return INTERCEPT_SEARCH_MAX


class MissingnessModel(ABC):
    """Abstract base class for missingness models.

    All models append a missing category to the item parameters,
    setting discrimination and intercept to achieve the desired
    missing probability structure.
    """

    @abstractmethod
    def generate_missing_params(
        self,
        params: SampledParameters,
        theta: NDArray[np.float64],
        rng: Generator,
    ) -> SampledParameters:
        """Generate item parameters including missing category.

        Args:
            params: Raw sampled parameters without missing category.
            theta: Candidate abilities, shape (n_candidates,).
            rng: Random number generator.

        Returns:
            SampledParameters with missing category appended and
            includes_missing_values=True.
        """
        raise NotImplementedError


# =============================================================================
# Registry
# =============================================================================

MissingnessModelFactory = Callable[..., MissingnessModel]


class MissingnessRegistry:
    """Registry for missingness models."""

    def __init__(self) -> None:
        self._models: dict[str, MissingnessModelFactory] = {}

    def register(
        self, name: str
    ) -> Callable[[MissingnessModelFactory], MissingnessModelFactory]:
        """Decorator to register a missingness model factory."""

        def decorator(
            factory: MissingnessModelFactory,
        ) -> MissingnessModelFactory:
            self._models[name] = factory
            return factory

        return decorator

    def get_model(self, name: str, params: dict[str, Any]) -> MissingnessModel:
        """Get a missingness model instance.

        Args:
            name: Model name (e.g., "none", "mcar", "ability_dependent").
            params: Model-specific parameters.

        Returns:
            Configured MissingnessModel instance.

        Raises:
            ValueError: If model name is not registered.
        """
        if name not in self._models:
            available = ", ".join(sorted(self._models.keys()))
            raise ValueError(
                f"Unknown missingness model: {name}. Available: {available}"
            )
        return self._models[name](**params)


missingness_registry = MissingnessRegistry()


# =============================================================================
# Model Implementations
# =============================================================================


@dataclass
class NoMissingness(MissingnessModel):
    """No missing values - adds missing category with ~0 probability."""

    def generate_missing_params(
        self,
        params: SampledParameters,
        theta: NDArray[np.float64],
        rng: Generator,
    ) -> SampledParameters:
        assert not params.includes_missing_values
        n_questions = params.discrimination.shape[0]

        # Append missing category: discrimination=0, intercept=-30 (~0 probability)
        updated_discrimination = np.hstack(
            [params.discrimination, np.zeros((n_questions, 1))]
        )
        updated_intercept = np.hstack(
            [params.intercept, np.full((n_questions, 1), NO_MISSING_INTERCEPT)]
        )

        return SampledParameters(
            discrimination=updated_discrimination,
            intercept=updated_intercept,
            includes_missing_values=True,
        )


@missingness_registry.register("none")
def create_no_missingness() -> NoMissingness:
    """Factory for NoMissingness model."""
    return NoMissingness()


@dataclass
class MCARMissingness(MissingnessModel):
    """Missing Completely At Random (MCAR).

    Each response has an independent probability of being missing,
    regardless of ability, question, or response value.

    The intercept is calibrated so that E[P(missing|theta)] = rate,
    where the expectation is over the ability distribution.
    """

    rate: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.rate < 1.0):
            raise ValueError(f"rate must be in [0, 1), got {self.rate}")

    def generate_missing_params(
        self,
        params: SampledParameters,
        theta: NDArray[np.float64],
        rng: Generator,
    ) -> SampledParameters:
        assert not params.includes_missing_values
        n_questions = params.discrimination.shape[0]

        # Missing category: discrimination=0 (ability-independent)
        missing_discrimination = 0.0

        # Calibrate intercept for each question to achieve target rate
        # across the ability distribution
        missing_intercepts = np.zeros(n_questions, dtype=np.float64)
        for q in range(n_questions):
            # Compute scores for this question across all candidates
            # Shape: (n_candidates, n_categories)
            item_scores = (
                np.outer(theta, params.discrimination[q, :])
                + params.intercept[q, :]
            )
            missing_intercepts[q] = _calibrate_intercept(
                target_rate=self.rate,
                discrimination=missing_discrimination,
                item_scores=item_scores,
                theta=theta,
            )

        updated_discrimination = np.hstack(
            [params.discrimination, np.zeros((n_questions, 1))]
        )
        updated_intercept = np.hstack(
            [params.intercept, missing_intercepts.reshape(-1, 1)]
        )

        return SampledParameters(
            discrimination=updated_discrimination,
            intercept=updated_intercept,
            includes_missing_values=True,
        )


@missingness_registry.register("mcar")
def create_mcar_missingness(rate: float = 0.05) -> MCARMissingness:
    """Factory for MCAR missingness model."""
    return MCARMissingness(rate=rate)


@dataclass
class AbilityDependentMissingness(MissingnessModel):
    """Missingness that depends on candidate ability.

    Lower ability candidates are more likely to skip questions.
    Uses negative discrimination so P(missing) increases as ability decreases.

    The intercept is calibrated so that E[P(missing|theta)] = base_rate,
    where the expectation is over the ability distribution.
    """

    base_rate: float = 0.02
    ability_effect: float = 0.05
    ability_threshold: float = 0.0
    max_rate: float = 0.3

    def __post_init__(self) -> None:
        if not (0.0 <= self.base_rate < 1.0):
            raise ValueError(
                f"base_rate must be in [0, 1), got {self.base_rate}"
            )
        if self.ability_effect < 0:
            raise ValueError(
                f"ability_effect must be >= 0, got {self.ability_effect}"
            )
        if not (0.0 < self.max_rate <= 1.0):
            raise ValueError(
                f"max_rate must be in (0, 1], got {self.max_rate}"
            )

    def generate_missing_params(
        self,
        params: SampledParameters,
        theta: NDArray[np.float64],
        rng: Generator,
    ) -> SampledParameters:
        assert not params.includes_missing_values
        n_questions = params.discrimination.shape[0]

        # Negative discrimination: lower ability -> higher missing probability
        missing_discrimination = -self.ability_effect

        # Calibrate intercept for each question to achieve base_rate
        # across the ability distribution
        missing_intercepts = np.zeros(n_questions, dtype=np.float64)
        for q in range(n_questions):
            item_scores = (
                np.outer(theta, params.discrimination[q, :])
                + params.intercept[q, :]
            )
            missing_intercepts[q] = _calibrate_intercept(
                target_rate=self.base_rate,
                discrimination=missing_discrimination,
                item_scores=item_scores,
                theta=theta,
            )

        updated_discrimination = np.hstack(
            [
                params.discrimination,
                np.full((n_questions, 1), missing_discrimination),
            ]
        )
        updated_intercept = np.hstack(
            [params.intercept, missing_intercepts.reshape(-1, 1)]
        )

        return SampledParameters(
            discrimination=updated_discrimination,
            intercept=updated_intercept,
            includes_missing_values=True,
        )


@missingness_registry.register("ability_dependent")
def create_ability_dependent_missingness(
    base_rate: float = 0.02,
    ability_effect: float = 0.05,
    ability_threshold: float = 0.0,
    max_rate: float = 0.3,
) -> AbilityDependentMissingness:
    """Factory for ability-dependent missingness model."""
    return AbilityDependentMissingness(
        base_rate=base_rate,
        ability_effect=ability_effect,
        ability_threshold=ability_threshold,
        max_rate=max_rate,
    )


@dataclass
class PositionDependentMissingness(MissingnessModel):
    """Missingness that depends on question position (fatigue effect).

    Later questions are more likely to be skipped due to test fatigue.
    The missing rate increases linearly with question position.

    The intercept for each question is calibrated so that
    E[P(missing|theta)] = base_rate + position_effect * position,
    clamped to max_rate.
    """

    base_rate: float = 0.01
    position_effect: float = 0.002
    max_rate: float = 0.2

    def __post_init__(self) -> None:
        if not (0.0 <= self.base_rate < 1.0):
            raise ValueError(
                f"base_rate must be in [0, 1), got {self.base_rate}"
            )
        if self.position_effect < 0:
            raise ValueError(
                f"position_effect must be >= 0, got {self.position_effect}"
            )
        if not (0.0 < self.max_rate <= 1.0):
            raise ValueError(
                f"max_rate must be in (0, 1], got {self.max_rate}"
            )

    def generate_missing_params(
        self,
        params: SampledParameters,
        theta: NDArray[np.float64],
        rng: Generator,
    ) -> SampledParameters:
        assert not params.includes_missing_values
        n_questions = params.discrimination.shape[0]

        # Compute target missing rate for each position
        positions = np.arange(n_questions, dtype=np.float64)
        target_rates = self.base_rate + self.position_effect * positions
        target_rates = np.minimum(target_rates, self.max_rate)
        # Clamp to valid probability range
        target_rates = np.clip(target_rates, 0.001, 0.999)

        # Position-dependent missingness is ability-independent (discrimination=0)
        missing_discrimination = 0.0

        # Calibrate intercept for each question to achieve position-dependent rate
        missing_intercepts = np.zeros(n_questions, dtype=np.float64)
        for q in range(n_questions):
            item_scores = (
                np.outer(theta, params.discrimination[q, :])
                + params.intercept[q, :]
            )
            missing_intercepts[q] = _calibrate_intercept(
                target_rate=target_rates[q],
                discrimination=missing_discrimination,
                item_scores=item_scores,
                theta=theta,
            )

        updated_discrimination = np.hstack(
            [params.discrimination, np.zeros((n_questions, 1))]
        )
        updated_intercept = np.hstack(
            [params.intercept, missing_intercepts.reshape(-1, 1)]
        )

        return SampledParameters(
            discrimination=updated_discrimination,
            intercept=updated_intercept,
            includes_missing_values=True,
        )


@missingness_registry.register("position_dependent")
def create_position_dependent_missingness(
    base_rate: float = 0.01,
    position_effect: float = 0.002,
    max_rate: float = 0.2,
) -> PositionDependentMissingness:
    """Factory for position-dependent missingness model."""
    return PositionDependentMissingness(
        base_rate=base_rate,
        position_effect=position_effect,
        max_rate=max_rate,
    )


# =============================================================================
# Factory Function
# =============================================================================


def get_missingness_model(
    model_name: str, params: dict[str, Any] | None = None
) -> MissingnessModel:
    """Get a missingness model by name.

    Args:
        model_name: Name of the model ("none", "mcar", "ability_dependent",
            "position_dependent").
        params: Model-specific parameters.

    Returns:
        Configured MissingnessModel instance.
    """
    if params is None:
        params = {}
    return missingness_registry.get_model(model_name, params)
