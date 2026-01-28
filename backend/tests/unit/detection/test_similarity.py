"""
Tests for similarity computation functions.
"""

import numpy as np

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.detection.similarity import (
    count_matching_responses,
    find_max_similarity,
    max_similarity_per_candidate,
    measure_observed_similarity,
)


class TestCountMatchingResponses:
    def test_all_matching(self) -> None:
        """All responses match."""
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        b = np.array([1, 2, 3, 4], dtype=np.int8)
        assert count_matching_responses(a, b) == 4

    def test_no_matching(self) -> None:
        """No responses match."""
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        b = np.array([4, 3, 2, 1], dtype=np.int8)
        assert count_matching_responses(a, b) == 0

    def test_partial_matching(self) -> None:
        """Some responses match."""
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        b = np.array([1, 2, 2, 3], dtype=np.int8)
        assert count_matching_responses(a, b) == 2

    def test_missing_in_first(self) -> None:
        """Missing values in first array are ignored."""
        a = np.array([MISSING_VALUE, 2, 3, 4], dtype=np.int8)
        b = np.array([1, 2, 3, 4], dtype=np.int8)
        # First position ignored, 3 matches
        assert count_matching_responses(a, b) == 3

    def test_missing_in_second(self) -> None:
        """Missing values in second array are ignored."""
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        b = np.array([1, MISSING_VALUE, 3, 4], dtype=np.int8)
        # Second position ignored, 3 matches
        assert count_matching_responses(a, b) == 3

    def test_missing_in_both(self) -> None:
        """Missing values in both arrays."""
        a = np.array([MISSING_VALUE, 2, 3, MISSING_VALUE], dtype=np.int8)
        b = np.array([1, MISSING_VALUE, 3, 4], dtype=np.int8)
        # Positions 0, 1, 3 ignored, only position 2 counts
        assert count_matching_responses(a, b) == 1

    def test_all_missing(self) -> None:
        """All values are missing."""
        a = np.array([MISSING_VALUE, MISSING_VALUE], dtype=np.int8)
        b = np.array([MISSING_VALUE, MISSING_VALUE], dtype=np.int8)
        assert count_matching_responses(a, b) == 0

    def test_empty_arrays(self) -> None:
        """Empty arrays return 0."""
        a = np.array([], dtype=np.int8)
        b = np.array([], dtype=np.int8)
        assert count_matching_responses(a, b) == 0


class TestFindMaxSimilarity:
    def test_simple_case(self) -> None:
        """Find max across simple dataset."""
        responses = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],  # Identical to row 0
                [4, 3, 2, 1],  # No matches with others
            ],
            dtype=np.int8,
        )
        assert find_max_similarity(responses) == 4

    def test_no_similarity(self) -> None:
        """No candidates share any responses."""
        responses = np.array(
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3],
            ],
            dtype=np.int8,
        )
        assert find_max_similarity(responses) == 0

    def test_single_candidate(self) -> None:
        """Single candidate has max similarity of 0."""
        responses = np.array([[1, 2, 3, 4]], dtype=np.int8)
        assert find_max_similarity(responses) == 0

    def test_with_missing_values(self) -> None:
        """Missing values are ignored in similarity computation."""
        responses = np.array(
            [
                [1, 2, MISSING_VALUE, 4],
                [1, 2, 3, 4],  # 3 matches (position 2 ignored)
                [1, 2, 3, MISSING_VALUE],  # 2 matches with row 0
            ],
            dtype=np.int8,
        )
        # Max is between rows 0 and 1: 3 matches
        assert find_max_similarity(responses) == 3


class TestMaxSimilarityPerCandidate:
    def test_simple_case(self) -> None:
        """Compute max similarity for each candidate."""
        responses = np.array(
            [
                [1, 2, 3, 4],  # 4 matches with row 1, 0 with row 2
                [1, 2, 3, 4],  # 4 matches with row 0, 0 with row 2
                [4, 3, 2, 1],  # 0 matches with rows 0 and 1
            ],
            dtype=np.int8,
        )
        result = max_similarity_per_candidate(responses)
        np.testing.assert_array_equal(result, [4, 4, 0])

    def test_partial_overlap(self) -> None:
        """Different overlap patterns."""
        responses = np.array(
            [
                [1, 1, 1, 1],  # 2 with row 1, 0 with row 2
                [1, 1, 2, 2],  # 2 with row 0, 2 with row 2
                [2, 2, 2, 2],  # 0 with row 0, 2 with row 1
            ],
            dtype=np.int8,
        )
        result = max_similarity_per_candidate(responses)
        np.testing.assert_array_equal(result, [2, 2, 2])

    def test_single_candidate(self) -> None:
        """Single candidate has max similarity of 0."""
        responses = np.array([[1, 2, 3, 4]], dtype=np.int8)
        result = max_similarity_per_candidate(responses)
        np.testing.assert_array_equal(result, [0])

    def test_memory_efficiency(self) -> None:
        """Verify memory-efficient computation for larger datasets."""
        n_candidates = 1000
        n_items = 50
        rng = np.random.default_rng(42)
        responses = rng.integers(
            1, 5, size=(n_candidates, n_items), dtype=np.int8
        )

        result = max_similarity_per_candidate(responses)

        assert result.shape == (n_candidates,)
        assert result.dtype == np.uint32
        # With 50 items and 4 options, we expect some overlap
        assert np.max(result) > 0


class TestMeasureObservedSimilarity:
    def test_symmetry(self) -> None:
        """Similarity matrix is symmetric."""
        responses = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 2, 3],
                [4, 3, 2, 1],
            ],
            dtype=np.int8,
        )
        result = measure_observed_similarity(responses)
        np.testing.assert_array_equal(result, result.T)

    def test_diagonal_is_zero(self) -> None:
        """Diagonal should be zero (not comparing with self in meaningful way)."""
        responses = np.array(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            dtype=np.int8,
        )
        result = measure_observed_similarity(responses)
        # Diagonal is 0 due to implementation (lower triangular + transpose)
        np.testing.assert_array_equal(np.diag(result), [0, 0])

    def test_correct_pairwise_values(self) -> None:
        """Verify pairwise similarity values are correct."""
        responses = np.array(
            [
                [1, 2, 3, 4],  # 2 matches with row 1, 0 with row 2
                [1, 2, 1, 1],  # 2 matches with row 0, 1 with row 2
                [4, 3, 2, 1],  # 0 matches with row 0, 1 with row 1
            ],
            dtype=np.int8,
        )
        result = measure_observed_similarity(responses)

        assert result[0, 1] == 2
        assert result[1, 0] == 2
        assert result[0, 2] == 0
        assert result[2, 0] == 0
        assert result[1, 2] == 1
        assert result[2, 1] == 1

    def test_with_missing_values(self) -> None:
        """Missing values are ignored in pairwise computation."""
        responses = np.array(
            [
                [1, MISSING_VALUE, 3, 4],
                [1, 2, 3, 4],
            ],
            dtype=np.int8,
        )
        result = measure_observed_similarity(responses)
        # Position 1 ignored, 3 matches at positions 0, 2, 3
        assert result[0, 1] == 3
        assert result[1, 0] == 3
