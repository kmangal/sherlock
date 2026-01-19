"""
Missing answer mechanisms.

This module introduces missing responses (`*`) according to specified mechanisms.

Initial support is only for Missing Completely At Random (MCAR).
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.core.utils import get_rng


class MissingnessModel(ABC):
    """Abstract base class for missingness models."""

    @abstractmethod
    def apply(
        self,
        responses: NDArray[np.int8],
        rng: Generator | None = None,
    ) -> NDArray[np.int8]:
        """
        Apply missingness to responses.

        Args:
            responses: Array of shape (n_candidates, n_questions) with response indices.
            rng: Random number generator.

        Returns:
            Array with same shape, where `MISSING_VALUE` indicates missing responses.
        """
        raise NotImplementedError


class MCARMissingness(MissingnessModel):
    """
    Missing Completely At Random (MCAR) mechanism.

    Each response has an independent probability of being missing,
    regardless of ability, question, or response value.
    """

    def __init__(self, missing_rate: float) -> None:
        """
        Initialize MCAR missingness.

        Args:
            missing_rate: Probability that any given response is missing.
                Must be in [0, 1).
        """
        if not (0.0 <= missing_rate < 1.0):
            raise ValueError(
                f"missing_rate must be in [0, 1), got {missing_rate}"
            )
        self.missing_rate = missing_rate

    def apply(
        self,
        responses: NDArray[np.int8],
        rng: Generator | None = None,
    ) -> NDArray[np.int8]:
        """
        Apply MCAR missingness to responses.

        Args:
            responses: Array of shape (n_candidates, n_questions).
            rng: Random number generator.

        Returns:
            Copy of responses with `MISSING_VALUE` for missing values.
        """
        if rng is None:
            rng = get_rng()

        if self.missing_rate == 0.0:
            return responses.copy()

        result = responses.copy()
        mask = rng.random(responses.shape) < self.missing_rate
        result[mask] = MISSING_VALUE
        return result


class AbilityDependentMissingness(MissingnessModel):
    """
    Missingness that depends on candidate ability.

    Lower ability candidates are more likely to skip questions.
    This models students who give up on difficult questions.

    The probability of missing for a candidate with ability theta is:
        P(missing) = base_rate + ability_effect * (threshold - theta)
                     when theta < threshold
        P(missing) = base_rate otherwise

    This is clamped to [0, max_rate].
    """

    def __init__(
        self,
        abilities: NDArray[np.float64],
        base_rate: float = 0.02,
        ability_effect: float = 0.05,
        ability_threshold: float = 0.0,
        max_rate: float = 0.3,
    ) -> None:
        """
        Initialize ability-dependent missingness.

        Args:
            base_rate: Base missing rate for all candidates.
            ability_effect: How much each unit below threshold increases missing rate.
            ability_threshold: Ability level below which missingness increases.
            max_rate: Maximum missing rate for any candidate.
        """
        self.abilities = abilities
        self.base_rate = base_rate
        self.ability_effect = ability_effect
        self.ability_threshold = ability_threshold
        self.max_rate = max_rate

    def apply(
        self,
        responses: NDArray[np.int8],
        rng: Generator | None = None,
    ) -> NDArray[np.int8]:
        """
        Apply ability-dependent missingness.

        Args:
            responses: Array of shape (n_candidates, n_questions).
            rng: Random number generator.

        Returns:
            Copy of responses with `MISSING_VALUE` for missing values.
        """

        if rng is None:
            rng = get_rng()

        n_questions = responses.shape[1]
        result = responses.copy()

        for i, ability in enumerate(self.abilities):
            # Compute missing rate for this candidate
            if ability < self.ability_threshold:
                rate = self.base_rate + self.ability_effect * (
                    self.ability_threshold - ability
                )
            else:
                rate = self.base_rate

            rate = min(rate, self.max_rate)

            # Apply missingness
            mask = rng.random(n_questions) < rate
            result[i, mask] = MISSING_VALUE

        return result


class PositionDependentMissingness(MissingnessModel):
    """
    Missingness that depends on question position (fatigue effect).

    Later questions are more likely to be skipped due to test fatigue.
    The missing rate increases linearly with question position.
    """

    def __init__(
        self,
        base_rate: float = 0.01,
        position_effect: float = 0.002,
        max_rate: float = 0.2,
    ) -> None:
        """
        Initialize position-dependent missingness.

        Args:
            base_rate: Missing rate for the first question.
            position_effect: Increase in missing rate per question.
            max_rate: Maximum missing rate for any question.
        """
        self.base_rate = base_rate
        self.position_effect = position_effect
        self.max_rate = max_rate

    def apply(
        self,
        responses: NDArray[np.int8],
        rng: Generator | None = None,
    ) -> NDArray[np.int8]:
        """
        Apply position-dependent missingness.

        Args:
            responses: Array of shape (n_candidates, n_questions).
            rng: Random number generator.

        Returns:
            Copy of responses with `MISSING_VALUE` for missing values.
        """
        if rng is None:
            rng = get_rng()

        n_candidates, n_questions = responses.shape
        result = responses.copy()

        # Compute missing rate for each position
        positions = np.arange(n_questions)
        rates = self.base_rate + self.position_effect * positions
        rates = np.minimum(rates, self.max_rate)

        # Apply missingness
        for j, rate in enumerate(rates):
            mask = rng.random(n_candidates) < rate
            result[mask, j] = MISSING_VALUE

        return result


def apply_missingness(
    responses: NDArray[np.int8],
    missing_rate: float = 0.0,
    mechanism: Literal["mcar"] = "mcar",
    abilities: NDArray[np.float64] | None = None,
    rng: Generator | None = None,
) -> NDArray[np.int8]:
    """
    Convenience function to apply missingness with common defaults.

    Args:
        responses: Array of shape (n_candidates, n_questions).
        missing_rate: Overall target missing rate.
        mechanism: Missingness mechanism ("mcar" for now).
        abilities: Candidate abilities (for future mechanisms).
        rng: Random number generator.

    Returns:
        Responses with missing values marked as `MISSING_VALUE`.
    """
    if rng is None:
        rng = get_rng()

    if mechanism == "mcar":
        model = MCARMissingness(missing_rate)
        return model.apply(responses, rng)
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")
