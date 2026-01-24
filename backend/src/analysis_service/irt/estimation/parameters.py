"""
NRM item parameter representation.

The Nominal Response Model (Bock, 1972) parameterization:
    P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

Identification constraint: sum-to-zero (Σa_k = 0, Σb_k = 0).
Free parameters: (K-1) discriminations + (K-1) intercepts = 2(K-1) per item.
The last category (K-1) is derived from the constraint: a_{K-1} = -Σa_k for k<K-1.
"""

from typing import Self

import numpy as np
from numpy.typing import NDArray

# Exponent clipping bounds to prevent overflow
EXPONENT_CLIP_MIN = -30.0
EXPONENT_CLIP_MAX = 30.0


class NRMItemParameters:
    """
    Parameters for one item under the Nominal Response Model.

    Attributes:
        item_id: Unique identifier for the item.
        discriminations: Slope parameters (a_k) for each category.
            Tuple of length K where K = n_categories.
            Sum-to-zero identification: Σa_k = 0.
        intercepts: Intercept parameters (b_k) for each category.
            Tuple of length K where K = n_categories.
            Sum-to-zero identification: Σb_k = 0.
        correct_answer: Index of the correct answer (0-indexed), or None if unknown.
            When known, a soft penalty encourages a_correct > a_distractor.
    """

    def __init__(
        self,
        item_id: int,
        discriminations: tuple[float, ...],
        intercepts: tuple[float, ...],
        correct_answer: int | None = None,
    ) -> None:
        self.item_id = item_id

        self.discriminations = discriminations
        self.intercepts = intercepts
        self.correct_answer = correct_answer

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate parameters."""
        if len(self.discriminations) != len(self.intercepts):
            raise ValueError(
                f"discriminations and intercepts must have same length, "
                f"got {len(self.discriminations)} and {len(self.intercepts)}"
            )
        if len(self.discriminations) < 2:
            raise ValueError(
                f"Must have at least 2 categories, got {len(self.discriminations)}"
            )
        if self.correct_answer is not None:
            if not (0 <= self.correct_answer < len(self.discriminations)):
                raise ValueError(
                    f"correct_answer must be in [0, {len(self.discriminations)}), "
                    f"got {self.correct_answer}"
                )

    @property
    def has_correct_answer(self) -> bool:
        """Whether this item has a known correct answer."""
        return self.correct_answer is not None

    @property
    def n_categories(self) -> int:
        """Number of response categories."""
        return len(self.discriminations)

    def compute_probabilities(
        self, theta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute NRM probabilities for all categories at given theta values.

        P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

        Args:
            theta: Ability values, shape (n_theta,).

        Returns:
            Probabilities, shape (n_theta, n_categories).
        """
        discriminations = np.array(self.discriminations, dtype=np.float64)
        intercepts = np.array(self.intercepts, dtype=np.float64)

        # Compute logits: a_k * theta + b_k
        # Shape: (n_theta, n_categories)
        logits = np.outer(theta, discriminations) + intercepts[np.newaxis, :]

        # Clip for numerical stability
        logits = np.clip(logits, EXPONENT_CLIP_MIN, EXPONENT_CLIP_MAX)

        # Log-sum-exp trick for stability
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        probs: NDArray[np.float64] = exp_shifted / np.sum(
            exp_shifted, axis=1, keepdims=True
        )

        return probs

    def to_array(self) -> NDArray[np.float64]:
        """
        Flatten free parameters to 1D array for optimization.

        Sum-to-zero identification: first K-1 parameters are free,
        last is derived as negative sum of the others.
        Layout: [a_0, a_1, ..., a_{K-2}, b_0, b_1, ..., b_{K-2}]

        Returns:
            1D array of shape (2 * (K-1),).
        """
        # First K-1 categories are free (last is constrained by sum-to-zero)
        free_a = self.discriminations[:-1]
        free_b = self.intercepts[:-1]
        return np.array(free_a + free_b, dtype=np.float64)

    @classmethod
    def from_array(
        cls,
        item_id: int,
        arr: NDArray[np.float64],
        n_categories: int,
        correct_answer: int | None = None,
    ) -> Self:
        """
        Reconstruct parameters from flattened array.

        Uses sum-to-zero constraint: last category parameter = -sum(free params).

        Args:
            item_id: Item identifier.
            arr: 1D array of free parameters from to_array().
            n_categories: Number of response categories.
            correct_answer: Index of the correct answer, or None if unknown.

        Returns:
            NRMItemParameters instance with sum-to-zero constraint applied.
        """
        n_free = n_categories - 1

        # Extract free parameters
        free_a = arr[:n_free]
        free_b = arr[n_free:]

        # Compute last category from sum-to-zero constraint
        a_last = -np.sum(free_a)
        b_last = -np.sum(free_b)

        discriminations = tuple(float(a) for a in free_a) + (float(a_last),)
        intercepts = tuple(float(b) for b in free_b) + (float(b_last),)

        return cls(
            item_id=item_id,
            discriminations=discriminations,
            intercepts=intercepts,
            correct_answer=correct_answer,
        )

    @classmethod
    def n_free_parameters(cls, n_categories: int) -> int:
        """
        Number of free parameters per item.

        Args:
            n_categories: Number of response categories.

        Returns:
            2 * (K - 1) free parameters per item.
        """
        return 2 * (n_categories - 1)

    @classmethod
    def create_default(
        cls,
        item_id: int,
        n_categories: int,
        correct_answer: int | None = None,
    ) -> Self:
        """
        Create default (neutral) parameters.

        All categories have equal probability at θ=0.

        Args:
            item_id: Item identifier.
            n_categories: Number of response categories.
            correct_answer: Index of the correct answer, or None if unknown.

        Returns:
            NRMItemParameters with all parameters set to 0.
        """
        discriminations = tuple(0.0 for _ in range(n_categories))
        intercepts = tuple(0.0 for _ in range(n_categories))
        return cls(
            item_id=item_id,
            discriminations=discriminations,
            intercepts=intercepts,
            correct_answer=correct_answer,
        )
