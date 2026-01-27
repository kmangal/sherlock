"""
Data models for IRT estimation input/output.

This module defines the data structures for:
- ResponseMatrix: Input data for IRT estimation
- FittedModel: Output from IRT estimation
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE


def response_code_to_category_index(
    responses: NDArray[np.int8],
    n_response_categories: int,
) -> NDArray[np.int64]:
    """
    Map response codes to category indices.

    Response codes: -1 (missing), 0, 1, ..., K-1
    Category indices: 0, 1, ..., K-1, K (where K = missing)

    Args:
        responses: Response codes, shape arbitrary.
        n_response_categories: Number of response categories (K), excluding missing.

    Returns:
        Category indices with same shape as responses.
        Missing values (-1) map to index K.
    """
    result = responses.astype(np.int64)
    result[responses == MISSING_VALUE] = n_response_categories
    return result


@dataclass(frozen=True)
class ResponseMatrix:
    """
    Response data for IRT estimation.

    Attributes:
        responses: Array of shape (n_candidates, n_items) containing response
            indices (0-indexed). Missing responses are indicated by MISSING_VALUE.
        n_categories: Number of response categories (same for all items).
    """

    responses: NDArray[np.int8]
    n_categories: int

    def __post_init__(self) -> None:
        """Validate response matrix."""
        if self.responses.ndim != 2:
            raise ValueError(
                f"responses must be 2D, got shape {self.responses.shape}"
            )
        if self.n_categories < 2:
            raise ValueError(
                f"n_categories must be >= 2, got {self.n_categories}"
            )
        # Validate response values are in valid range
        valid_mask = self.responses != MISSING_VALUE
        valid_responses = self.responses[valid_mask]
        if len(valid_responses) > 0:
            if valid_responses.min() < 0:
                raise ValueError(
                    f"Response values must be >= 0, got min {valid_responses.min()}"
                )
            if valid_responses.max() >= self.n_categories:
                raise ValueError(
                    f"Response values must be < n_categories ({self.n_categories}), "
                    f"got max {valid_responses.max()}"
                )

    @property
    def n_candidates(self) -> int:
        """Number of candidates (rows)."""
        return self.responses.shape[0]

    @property
    def n_items(self) -> int:
        """Number of items (columns)."""
        return self.responses.shape[1]

    @property
    def n_total_categories(self) -> int:
        """
        Total number of categories including missing.

        Returns K+1 where K = n_categories.
        """
        return self.n_categories + 1

    @property
    def missing_mask(self) -> NDArray[np.bool_]:
        """Boolean mask where True indicates missing response."""
        result: NDArray[np.bool_] = self.responses == MISSING_VALUE
        return result

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Boolean mask where True indicates valid (non-missing) response."""
        result: NDArray[np.bool_] = self.responses != MISSING_VALUE
        return result

    def item_response_counts(self, item_idx: int) -> NDArray[np.int64]:
        """
        Count responses for each category of an item (excluding missing).

        Args:
            item_idx: Index of the item.

        Returns:
            Array of shape (n_categories,) with counts per category.
        """
        item_responses = self.responses[:, item_idx]
        valid = item_responses[item_responses != MISSING_VALUE]
        counts = np.bincount(
            valid.astype(np.int64), minlength=self.n_categories
        )
        return counts.astype(np.int64)

    def item_category_counts(self, item_idx: int) -> NDArray[np.int64]:
        """
        Count responses for each category including missing.

        Args:
            item_idx: Index of the item.

        Returns:
            Array of shape (n_total_categories,) with counts.
            Last element is the count of missing responses.
        """
        item_responses = self.responses[:, item_idx]
        category_indices = response_code_to_category_index(
            item_responses, self.n_categories
        )
        counts = np.bincount(
            category_indices, minlength=self.n_total_categories
        )
        return counts.astype(np.int64)
