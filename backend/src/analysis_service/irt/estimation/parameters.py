"""
NRM item parameter representation.

The Nominal Response Model (Bock, 1972) parameterization:
    P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

Missing responses are modeled as a separate category with its own probability.
With K response categories (0..K-1) plus missing (K), we have K+1 total categories.

Identification constraint: sum-to-zero (Σa_k = 0, Σb_k = 0) across all K+1 categories.
Free parameters: K discriminations + K intercepts = 2K per item.
The missing category (K) is derived: a_K = -Σa_k for k<K.
"""

from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator

# Exponent clipping bounds to prevent overflow
EXPONENT_CLIP_MIN = -30.0
EXPONENT_CLIP_MAX = 30.0


class NRMItemParameters(BaseModel):
    """
    Parameters for one item under the Nominal Response Model.

    Missing responses are modeled as an additional category (index K) with its
    own probability P(Y=K|θ). This allows the model to learn patterns in missingness.

    Attributes:
        item_id: Unique identifier for the item.
        discriminations: Slope parameters (a_k) for each category.
            Tuple of length K+1 where K = n_response_categories.
            Categories 0..K-1 are responses, category K is missing.
            Sum-to-zero identification: Σa_k = 0.
        intercepts: Intercept parameters (b_k) for each category.
            Tuple of length K+1 where K = n_response_categories.
            Sum-to-zero identification: Σb_k = 0.
        correct_answer: Index of the correct answer (0-indexed), or None if unknown.
            Must be in [0, K-1] (cannot be the missing category).
            When known, a soft penalty encourages a_correct > a_distractor.
    """

    item_id: int
    discriminations: tuple[float, ...]
    intercepts: tuple[float, ...]
    correct_answer: int | None = None

    @model_validator(mode="after")
    def _validate_parameters_have_same_length(self) -> "NRMItemParameters":
        if len(self.discriminations) != len(self.intercepts):
            raise ValueError(
                f"discriminations and intercepts must have same length, "
                f"got {len(self.discriminations)} and {len(self.intercepts)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_num_response_categories(self) -> "NRMItemParameters":
        # Need at least 2 response categories + 1 missing = 3 total
        if len(self.discriminations) < 3:
            raise ValueError(
                f"Must have at least 3 total categories (2 responses + missing), "
                f"got {len(self.discriminations)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_correct_answer(self) -> "NRMItemParameters":
        # correct_answer must be in [0, K-1] where K = n_response_categories
        # i.e., it cannot be the missing category
        if self.correct_answer is not None:
            if not (0 <= self.correct_answer < self.n_response_categories):
                raise ValueError(
                    f"correct_answer must be in [0, {self.n_response_categories}), "
                    f"got {self.correct_answer}"
                )
        return self

    @property
    def has_correct_answer(self) -> bool:
        """Whether this item has a known correct answer."""
        return self.correct_answer is not None

    @property
    def n_categories(self) -> int:
        """Total number of categories including missing."""
        return len(self.discriminations)

    @property
    def n_response_categories(self) -> int:
        """Number of response categories (excludes missing)."""
        return len(self.discriminations) - 1

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

        Sum-to-zero identification: first K response categories are free,
        missing category (K) is derived as negative sum of the others.
        Layout: [a_0, a_1, ..., a_{K-1}, b_0, b_1, ..., b_{K-1}]

        Returns:
            1D array of shape (2 * K,) where K = n_response_categories.
        """
        # Response categories (0..K-1) are free, missing (K) is constrained
        n_resp = self.n_response_categories
        free_a = self.discriminations[:n_resp]
        free_b = self.intercepts[:n_resp]
        return np.array(free_a + free_b, dtype=np.float64)

    @classmethod
    def from_array(
        cls,
        item_id: int,
        arr: NDArray[np.float64],
        n_response_categories: int,
        correct_answer: int | None = None,
    ) -> Self:
        """
        Reconstruct parameters from flattened array.

        Uses sum-to-zero constraint: missing category (K) parameter = -sum(free params).

        Args:
            item_id: Item identifier.
            arr: 1D array of free parameters from to_array().
            n_response_categories: Number of response categories (K, excludes missing).
            correct_answer: Index of the correct answer, or None if unknown.

        Returns:
            NRMItemParameters instance with sum-to-zero constraint applied.
            Has K+1 total categories (K response + 1 missing).
        """
        n_free = n_response_categories

        # Extract free parameters (response categories 0..K-1)
        free_a = arr[:n_free]
        free_b = arr[n_free:]

        # Compute missing category from sum-to-zero constraint
        a_missing = -np.sum(free_a)
        b_missing = -np.sum(free_b)

        discriminations = tuple(float(a) for a in free_a) + (float(a_missing),)
        intercepts = tuple(float(b) for b in free_b) + (float(b_missing),)

        return cls(
            item_id=item_id,
            discriminations=discriminations,
            intercepts=intercepts,
            correct_answer=correct_answer,
        )

    @classmethod
    def n_free_parameters(cls, n_response_categories: int) -> int:
        """
        Number of free parameters per item.

        Args:
            n_response_categories: Number of response categories (K, excludes missing).

        Returns:
            2 * K free parameters per item.
        """
        return 2 * n_response_categories

    @classmethod
    def create_default(
        cls,
        item_id: int,
        n_response_categories: int,
        correct_answer: int | None = None,
    ) -> Self:
        """
        Create default (neutral) parameters.

        All categories have equal probability at θ=0.

        Args:
            item_id: Item identifier.
            n_response_categories: Number of response categories (K, excludes missing).
            correct_answer: Index of the correct answer, or None if unknown.

        Returns:
            NRMItemParameters with all parameters set to 0.
            Has K+1 total categories (K response + 1 missing).
        """
        n_total = n_response_categories + 1
        discriminations = tuple(0.0 for _ in range(n_total))
        intercepts = tuple(0.0 for _ in range(n_total))
        return cls(
            item_id=item_id,
            discriminations=discriminations,
            intercepts=intercepts,
            correct_answer=correct_answer,
        )
