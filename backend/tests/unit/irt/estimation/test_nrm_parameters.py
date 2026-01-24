"""
Tests for NRM item parameters.

Sum-to-zero identification: Σa_k = 0, Σb_k = 0.
Free parameters: first K-1 categories; last derived as negative sum.
"""

import numpy as np
import pytest

from analysis_service.irt.estimation.parameters import NRMItemParameters


class TestNRMItemParameters:
    def test_basic_construction(self) -> None:
        """Should construct with valid parameters."""
        params = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.5, 1.0),
            intercepts=(0.0, 0.2, -0.3),
        )

        assert params.item_id == 0
        assert params.n_categories == 3
        assert params.discriminations == (0.0, 0.5, 1.0)
        assert params.intercepts == (0.0, 0.2, -0.3)

    def test_to_array(self) -> None:
        """Should flatten to array correctly with sum-to-zero layout."""
        params = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.5, 1.0, -1.5),  # Sum = 0
            intercepts=(0.2, -0.3, 0.4, -0.3),  # Sum = 0
        )

        arr = params.to_array()

        # Should have 2 * (K-1) = 6 free parameters
        assert len(arr) == 6
        # First 3 are free discriminations (k=0,1,2)
        np.testing.assert_array_equal(arr[:3], [0.0, 0.5, 1.0])
        # Last 3 are free intercepts (k=0,1,2)
        np.testing.assert_array_equal(arr[3:], [0.2, -0.3, 0.4])

    def test_from_array(self) -> None:
        """Should reconstruct from array with sum-to-zero constraint."""
        # Free params: a_0=0.5, a_1=1.0, a_2=1.5 -> a_3 = -(0.5+1.0+1.5) = -3.0
        # Free params: b_0=0.2, b_1=-0.3, b_2=0.4 -> b_3 = -(0.2-0.3+0.4) = -0.3
        arr = np.array([0.5, 1.0, 1.5, 0.2, -0.3, 0.4])

        params = NRMItemParameters.from_array(
            item_id=5, arr=arr, n_categories=4
        )

        assert params.item_id == 5
        assert params.discriminations == (0.5, 1.0, 1.5, -3.0)
        assert params.intercepts == pytest.approx((0.2, -0.3, 0.4, -0.3))

    def test_sum_to_zero_constraint(self) -> None:
        """from_array should enforce sum-to-zero constraint."""
        arr = np.array([1.0, -0.5, 0.3, -0.2])  # 2 free a's, 2 free b's

        params = NRMItemParameters.from_array(
            item_id=0, arr=arr, n_categories=3
        )

        # Sum of discriminations should be 0
        assert sum(params.discriminations) == pytest.approx(0.0)
        # Sum of intercepts should be 0
        assert sum(params.intercepts) == pytest.approx(0.0)

    def test_roundtrip(self) -> None:
        """to_array and from_array should be inverses for sum-to-zero params."""
        # Create params that satisfy sum-to-zero
        original = NRMItemParameters(
            item_id=3,
            discriminations=(0.5, 1.0, -1.5),  # Sum = 0
            intercepts=(-0.2, 0.3, -0.1),  # Sum = 0
        )

        arr = original.to_array()
        reconstructed = NRMItemParameters.from_array(
            item_id=3, arr=arr, n_categories=3
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
        assert NRMItemParameters.n_free_parameters(n_categories=3) == 4
        assert NRMItemParameters.n_free_parameters(n_categories=4) == 6
        assert NRMItemParameters.n_free_parameters(n_categories=5) == 8

    def test_create_default(self) -> None:
        """Should create neutral default parameters."""
        params = NRMItemParameters.create_default(item_id=0, n_categories=4)

        assert params.item_id == 0
        assert params.n_categories == 4
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
        """Should reject less than 2 categories."""
        with pytest.raises(ValueError, match="at least 2 categories"):
            NRMItemParameters(
                item_id=0,
                discriminations=(0.0,),
                intercepts=(0.0,),
            )
