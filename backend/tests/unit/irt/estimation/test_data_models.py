"""
Tests for estimation data models.
"""

import numpy as np
import pytest

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.core.data_models import (
    ResponseMatrix,
    response_code_to_category_index,
)


class TestResponseMatrix:
    def test_basic_construction(self) -> None:
        """Should construct from valid responses."""
        responses = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int8)
        rm = ResponseMatrix(responses=responses, n_categories=3)

        assert rm.n_candidates == 3
        assert rm.n_items == 3
        assert rm.n_categories == 3

    def test_missing_values(self) -> None:
        """Should handle missing values correctly."""
        responses = np.array(
            [[0, MISSING_VALUE, 2], [1, 2, MISSING_VALUE]], dtype=np.int8
        )
        rm = ResponseMatrix(responses=responses, n_categories=3)

        expected_missing = np.array(
            [[False, True, False], [False, False, True]]
        )
        np.testing.assert_array_equal(rm.missing_mask, expected_missing)

        expected_valid = ~expected_missing
        np.testing.assert_array_equal(rm.valid_mask, expected_valid)

    def test_item_response_counts(self) -> None:
        """Should count responses correctly."""
        responses = np.array(
            [[0, 1], [0, 2], [1, MISSING_VALUE], [1, 0]], dtype=np.int8
        )
        rm = ResponseMatrix(responses=responses, n_categories=3)

        # Item 0: 2 zeros, 2 ones, 0 twos
        counts_0 = rm.item_response_counts(0)
        np.testing.assert_array_equal(counts_0, [2, 2, 0])

        # Item 1: 1 zero, 1 one, 1 two (one missing)
        counts_1 = rm.item_response_counts(1)
        np.testing.assert_array_equal(counts_1, [1, 1, 1])

    def test_validation_non_2d(self) -> None:
        """Should reject non-2D responses."""
        responses = np.array([0, 1, 2], dtype=np.int8)
        with pytest.raises(ValueError, match="must be 2D"):
            ResponseMatrix(responses=responses, n_categories=3)

    def test_validation_n_categories(self) -> None:
        """Should reject n_categories < 2."""
        responses = np.array([[0, 0], [0, 0]], dtype=np.int8)
        with pytest.raises(ValueError, match="n_categories must be >= 2"):
            ResponseMatrix(responses=responses, n_categories=1)

    def test_validation_negative_response(self) -> None:
        """Should reject negative responses (except MISSING_VALUE)."""
        responses = np.array([[0, -2], [1, 0]], dtype=np.int8)
        with pytest.raises(ValueError, match="must be >= 0"):
            ResponseMatrix(responses=responses, n_categories=3)

    def test_validation_response_too_large(self) -> None:
        """Should reject responses >= n_categories."""
        responses = np.array([[0, 3], [1, 0]], dtype=np.int8)
        with pytest.raises(ValueError, match="must be < n_categories"):
            ResponseMatrix(responses=responses, n_categories=3)

    def test_n_total_categories(self) -> None:
        """n_total_categories should return K+1."""
        responses = np.array([[0, 1], [1, 2]], dtype=np.int8)
        rm = ResponseMatrix(responses=responses, n_categories=3)

        # K=3 response categories + 1 missing = 4 total
        assert rm.n_total_categories == 4

    def test_item_category_counts(self) -> None:
        """item_category_counts should count K+1 categories including missing."""
        responses = np.array(
            [[0, 1], [0, 2], [1, MISSING_VALUE], [1, 0]], dtype=np.int8
        )
        rm = ResponseMatrix(responses=responses, n_categories=3)

        # Item 0: 2 zeros, 2 ones, 0 twos, 0 missing
        counts_0 = rm.item_category_counts(0)
        np.testing.assert_array_equal(counts_0, [2, 2, 0, 0])

        # Item 1: 1 zero, 1 one, 1 two, 1 missing
        counts_1 = rm.item_category_counts(1)
        np.testing.assert_array_equal(counts_1, [1, 1, 1, 1])


class TestResponseCodeToCategoryIndex:
    def test_valid_responses_unchanged(self) -> None:
        """Valid responses 0..K-1 should map to themselves."""
        responses = np.array([0, 1, 2, 0, 1], dtype=np.int8)
        result = response_code_to_category_index(
            responses, n_response_categories=3
        )
        np.testing.assert_array_equal(result, [0, 1, 2, 0, 1])

    def test_missing_maps_to_k(self) -> None:
        """Missing value (-1) should map to category K."""
        responses = np.array(
            [0, MISSING_VALUE, 2, MISSING_VALUE], dtype=np.int8
        )
        result = response_code_to_category_index(
            responses, n_response_categories=3
        )
        # K=3, so missing maps to 3
        np.testing.assert_array_equal(result, [0, 3, 2, 3])

    def test_2d_array(self) -> None:
        """Should work with 2D arrays."""
        responses = np.array(
            [[0, MISSING_VALUE], [MISSING_VALUE, 1]], dtype=np.int8
        )
        result = response_code_to_category_index(
            responses, n_response_categories=2
        )
        expected = np.array([[0, 2], [2, 1]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype(self) -> None:
        """Output should be int64."""
        responses = np.array([0, 1], dtype=np.int8)
        result = response_code_to_category_index(
            responses, n_response_categories=2
        )
        assert result.dtype == np.int64
