from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ItemParameters(ABC):
    """
    Abstract base class for IRT item parameters.

    Subclasses define model-specific parameters (NRM, 3PL, etc.).
    """

    def __init__(self, item_id: int) -> None:
        self._item_id = item_id

    @property
    def item_id(self) -> int:
        return self._item_id

    @abstractmethod
    def to_array(self) -> NDArray[np.float64]:
        """
        Flatten parameters to 1D array for optimization.

        Returns:
            1D array of parameter values.
        """
        ...

    @classmethod
    @abstractmethod
    def from_array(
        cls, item_id: int, arr: NDArray[np.float64], n_categories: int
    ) -> "ItemParameters":
        """
        Reconstruct parameters from flattened array.

        Args:
            item_id: Identifier for the item.
            arr: 1D array of parameter values from to_array().
            n_categories: Number of response categories.

        Returns:
            ItemParameters instance.
        """
        ...

    @classmethod
    @abstractmethod
    def n_free_parameters(cls, n_categories: int) -> int:
        """
        Number of free parameters per item.

        Args:
            n_categories: Number of response categories.

        Returns:
            Number of free parameters to estimate.
        """
        ...
