"""
Tests for NRM item parameters.

With K response categories (0..K-1) and 1 missing category (K), total K+1 categories.
Sum-to-zero identification: Σa_k = 0, Σb_k = 0 across all K+1 categories.
Free parameters: first K categories (0..K-1); missing (K) derived as negative sum.
"""

import numpy as np
import pytest

from analysis_service.irt.estimation.parameters import NRMItemParameters


class TestNRMItemParameters:
    def test_basic_construction(self) -> None:
        """Should construct with valid parameters (K+1 categories)."""
        # 3 total categories: 2 response + 1 missing
        params = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.5, -0.5),
            intercepts=(0.0, 0.2, -0.2),
        )

        assert params.item_id == 0
        assert params.n_categories == 3  # K+1 total
        assert params.n_response_categories == 2  # K response categories
        assert params.discriminations == (0.0, 0.5, -0.5)
        assert params.intercepts == (0.0, 0.2, -0.2)

    def test_to_array(self) -> None:
        """Should flatten to array correctly with sum-to-zero layout."""
        # K=3 response categories + missing = 4 total
        params = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.5, 1.0, -1.5),  # Sum = 0
            intercepts=(0.2, -0.3, 0.4, -0.3),  # Sum = 0
        )

        arr = params.to_array()

        # Should have 2 * K = 6 free parameters (K=3 response categories)
        assert len(arr) == 6
        # First 3 are free discriminations (response categories k=0,1,2)
        np.testing.assert_array_equal(arr[:3], [0.0, 0.5, 1.0])
        # Last 3 are free intercepts (response categories k=0,1,2)
        np.testing.assert_array_equal(arr[3:], [0.2, -0.3, 0.4])

    def test_from_array(self) -> None:
        """Should reconstruct from array with sum-to-zero constraint."""
        # K=3 response categories
        # Free params: a_0=0.5, a_1=1.0, a_2=1.5 -> a_missing = -(0.5+1.0+1.5) = -3.0
        # Free params: b_0=0.2, b_1=-0.3, b_2=0.4 -> b_missing = -(0.2-0.3+0.4) = -0.3
        arr = np.array([0.5, 1.0, 1.5, 0.2, -0.3, 0.4])

        params = NRMItemParameters.from_array(
            item_id=5, arr=arr, n_response_categories=3
        )

        assert params.item_id == 5
        assert params.n_categories == 4  # K+1 total
        assert params.n_response_categories == 3  # K response
        assert params.discriminations == (0.5, 1.0, 1.5, -3.0)
        assert params.intercepts == pytest.approx((0.2, -0.3, 0.4, -0.3))

    def test_sum_to_zero_constraint(self) -> None:
        """from_array should enforce sum-to-zero constraint."""
        # K=2 response categories -> 2 free a's, 2 free b's
        arr = np.array([1.0, -0.5, 0.3, -0.2])

        params = NRMItemParameters.from_array(
            item_id=0, arr=arr, n_response_categories=2
        )

        # Sum of discriminations should be 0 (across all K+1 = 3 categories)
        assert sum(params.discriminations) == pytest.approx(0.0)
        # Sum of intercepts should be 0
        assert sum(params.intercepts) == pytest.approx(0.0)

    def test_roundtrip(self) -> None:
        """to_array and from_array should be inverses for sum-to-zero params."""
        # K=2 response categories + missing = 3 total
        original = NRMItemParameters(
            item_id=3,
            discriminations=(0.5, 1.0, -1.5),  # Sum = 0
            intercepts=(-0.2, 0.3, -0.1),  # Sum = 0
        )

        arr = original.to_array()
        reconstructed = NRMItemParameters.from_array(
            item_id=3, arr=arr, n_response_categories=2
        )

        assert original.item_id == reconstructed.item_id
        np.testing.assert_allclose(
            original.discriminations, reconstructed.discriminations, rtol=1e-10
        )
        np.testing.assert_allclose(
            original.intercepts, reconstructed.intercepts, rtol=1e-10
        )

    def test_n_free_parameters(self) -> None:
        """Should report correct number of free parameters."""
        # n_free = 2 * K where K = n_response_categories
        assert (
            NRMItemParameters.n_free_parameters(n_response_categories=2) == 4
        )
        assert (
            NRMItemParameters.n_free_parameters(n_response_categories=3) == 6
        )
        assert (
            NRMItemParameters.n_free_parameters(n_response_categories=4) == 8
        )

    def test_create_default(self) -> None:
        """Should create neutral default parameters with K+1 categories."""
        # K=3 response categories -> 4 total
        params = NRMItemParameters.create_default(
            item_id=0, n_response_categories=3
        )

        assert params.item_id == 0
        assert params.n_categories == 4  # K+1 total
        assert params.n_response_categories == 3  # K response
        assert params.discriminations == (0.0, 0.0, 0.0, 0.0)
        assert params.intercepts == (0.0, 0.0, 0.0, 0.0)

    def test_validation_length_mismatch(self) -> None:
        """Should reject mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            NRMItemParameters(
                item_id=0,
                discriminations=(0.0, 0.5, 1.0),
                intercepts=(0.0, 0.2),  # Wrong length
            )

    def test_validation_too_few_categories(self) -> None:
        """Should reject less than 3 total categories (2 response + missing)."""
        with pytest.raises(ValueError, match="at least 3 total categories"):
            NRMItemParameters(
                item_id=0,
                discriminations=(0.0, 0.5),  # Only 2 total
                intercepts=(0.0, -0.5),
            )

    def test_correct_answer_validation(self) -> None:
        """correct_answer must be a response category (not missing)."""
        # K=2 response categories + missing = 3 total
        # correct_answer must be in [0, K-1] = [0, 1]
        with pytest.raises(ValueError, match="correct_answer must be in"):
            NRMItemParameters(
                item_id=0,
                discriminations=(0.0, 0.5, -0.5),
                intercepts=(0.0, 0.2, -0.2),
                correct_answer=2,  # This is the missing category
            )
