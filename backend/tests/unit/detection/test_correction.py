"""
Tests for multiple testing correction functions.
"""

import numpy as np

from analysis_service.detection.correction import benjamini_hochberg


class TestBenjaminiHochberg:
    def test_all_significant(self) -> None:
        """All p-values below threshold."""
        p_values = np.array([0.001, 0.002, 0.003])
        result = benjamini_hochberg(p_values, alpha=0.05)
        np.testing.assert_array_equal(result, [True, True, True])

    def test_none_significant(self) -> None:
        """No p-values below threshold."""
        p_values = np.array([0.5, 0.6, 0.7])
        result = benjamini_hochberg(p_values, alpha=0.05)
        np.testing.assert_array_equal(result, [False, False, False])

    def test_some_significant(self) -> None:
        """Some p-values significant."""
        # BH procedure: sort p-values, compare to (rank/n) * alpha
        # Sorted: 0.01, 0.04, 0.5
        # Critical: 0.0167, 0.0333, 0.05
        # 0.01 < 0.0167: significant
        # 0.04 > 0.0333: not significant
        p_values = np.array([0.04, 0.01, 0.5])
        result = benjamini_hochberg(p_values, alpha=0.05)
        # Only the smallest p-value (index 1) should be significant
        np.testing.assert_array_equal(result, [False, True, False])

    def test_order_preserved(self) -> None:
        """Result indices match original p-value positions."""
        p_values = np.array([0.5, 0.001, 0.3, 0.002])
        result = benjamini_hochberg(p_values, alpha=0.05)
        # Indices 1 and 3 have the smallest p-values
        assert result[1] is np.True_
        assert result[3] is np.True_
        assert result[0] is np.False_
        assert result[2] is np.False_

    def test_empty_array(self) -> None:
        """Empty array returns empty result."""
        p_values = np.array([])
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert len(result) == 0

    def test_single_element_significant(self) -> None:
        """Single significant p-value."""
        p_values = np.array([0.01])
        result = benjamini_hochberg(p_values, alpha=0.05)
        np.testing.assert_array_equal(result, [True])

    def test_single_element_not_significant(self) -> None:
        """Single non-significant p-value."""
        p_values = np.array([0.1])
        result = benjamini_hochberg(p_values, alpha=0.05)
        np.testing.assert_array_equal(result, [False])

    def test_step_up_property(self) -> None:
        """BH rejects all hypotheses with rank <= k where k is max significant rank."""
        # If sorted p-value at rank k passes, all ranks <= k should be rejected
        # Sorted: 0.005, 0.015, 0.03, 0.8
        # n=4, alpha=0.1
        # Critical: 0.025, 0.05, 0.075, 0.1
        # 0.005 < 0.025: pass
        # 0.015 < 0.05: pass
        # 0.03 < 0.075: pass
        # 0.8 > 0.1: fail
        # Max k = 3, so first 3 rejected
        p_values = np.array([0.8, 0.015, 0.005, 0.03])
        result = benjamini_hochberg(p_values, alpha=0.1)
        # Indices 1, 2, 3 should be rejected (they have the 3 smallest p-values)
        np.testing.assert_array_equal(result, [False, True, True, True])

    def test_ties_in_p_values(self) -> None:
        """Handles tied p-values correctly."""
        p_values = np.array([0.01, 0.01, 0.5])
        result = benjamini_hochberg(p_values, alpha=0.05)
        # Both 0.01 values should be significant
        assert result[0] is np.True_
        assert result[1] is np.True_
        assert result[2] is np.False_

    def test_stringent_alpha(self) -> None:
        """Very small alpha rejects fewer hypotheses."""
        p_values = np.array([0.001, 0.01, 0.05])
        result_strict = benjamini_hochberg(p_values, alpha=0.001)
        result_lenient = benjamini_hochberg(p_values, alpha=0.1)

        # Stricter alpha should reject fewer or equal
        assert np.sum(result_strict) <= np.sum(result_lenient)
