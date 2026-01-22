"""
NRM item parameter representation.

The Nominal Response Model (Bock, 1972) parameterization:
    P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

Identification constraint: a_0 = 0, b_0 = 0 (reference category).
Free parameters: (K-1) discriminations + (K-1) intercepts = 2(K-1) per item.
"""

import numpy as np
from numpy.typing import NDArray

from analysis_service.irt.estimation.parameters import ItemParameters


class NRMItemParameters(ItemParameters):
    """
    Parameters for one item under the Nominal Response Model.

    Attributes:
        item_id: Unique identifier for the item.
        discriminations: Slope parameters (a_k) for each category.
            Tuple of length K where K = n_categories.
            discriminations[0] = 0 (reference category constraint).
        intercepts: Intercept parameters (b_k) for each category.
            Tuple of length K where K = n_categories.
            intercepts[0] = 0 (reference category constraint).
    """

    def __init__(
        self,
        item_id: int,
        discriminations: tuple[float, ...],
        intercepts: tuple[float, ...],
    ) -> None:
        self.discriminations = discriminations
        self.intercepts = intercepts
        super().__init__(item_id)

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

    @property
    def n_categories(self) -> int:
        """Number of response categories."""
        return len(self.discriminations)

    def to_array(self) -> NDArray[np.float64]:
        """
        Flatten free parameters to 1D array for optimization.

        Only includes free parameters (excludes reference category k=0).
        Layout: [a_1, a_2, ..., a_{K-1}, b_1, b_2, ..., b_{K-1}]

        Returns:
            1D array of shape (2 * (K-1),).
        """
        # Skip reference category (index 0)
        free_a = self.discriminations[1:]
        free_b = self.intercepts[1:]
        return np.array(free_a + free_b, dtype=np.float64)

    @classmethod
    def from_array(
        cls, item_id: int, arr: NDArray[np.float64], n_categories: int
    ) -> "NRMItemParameters":
        """
        Reconstruct parameters from flattened array.

        Args:
            item_id: Item identifier.
            arr: 1D array of free parameters from to_array().
            n_categories: Number of response categories.

        Returns:
            NRMItemParameters instance with reference category fixed at 0.
        """
        n_free = n_categories - 1

        # Extract free parameters
        free_a = arr[:n_free]
        free_b = arr[n_free:]

        # Add reference category (a_0 = 0, b_0 = 0)
        discriminations = (0.0,) + tuple(float(a) for a in free_a)
        intercepts = (0.0,) + tuple(float(b) for b in free_b)

        return cls(
            item_id=item_id,
            discriminations=discriminations,
            intercepts=intercepts,
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
        cls, item_id: int, n_categories: int
    ) -> "NRMItemParameters":
        """
        Create default (neutral) parameters.

        All categories have equal probability at θ=0.

        Args:
            item_id: Item identifier.
            n_categories: Number of response categories.

        Returns:
            NRMItemParameters with all parameters set to 0.
        """
        discriminations = tuple(0.0 for _ in range(n_categories))
        intercepts = tuple(0.0 for _ in range(n_categories))
        return cls(
            item_id=item_id,
            discriminations=discriminations,
            intercepts=intercepts,
        )
