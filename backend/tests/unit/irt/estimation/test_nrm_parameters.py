"""
Tests for NRM item parameters.
"""

import numpy as np
import pytest

from analysis_service.irt.estimation.nrm.parameters import NRMItemParameters


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
        """Should flatten to array correctly."""
        params = NRMItemParameters(
            item_id=0,
            discriminations=(0.0, 0.5, 1.0, 1.5),
            intercepts=(0.0, 0.2, -0.3, 0.4),
        )

        arr = params.to_array()

        # Should have 2 * (K-1) = 6 free parameters
        assert len(arr) == 6
        # First 3 are free discriminations (skip k=0)
        np.testing.assert_array_equal(arr[:3], [0.5, 1.0, 1.5])
        # Last 3 are free intercepts (skip k=0)
        np.testing.assert_array_equal(arr[3:], [0.2, -0.3, 0.4])

    def test_from_array(self) -> None:
        """Should reconstruct from array correctly."""
        arr = np.array([0.5, 1.0, 1.5, 0.2, -0.3, 0.4])

        params = NRMItemParameters.from_array(
            item_id=5, arr=arr, n_categories=4
        )

        assert params.item_id == 5
        assert params.discriminations == (0.0, 0.5, 1.0, 1.5)
        assert params.intercepts == (0.0, 0.2, -0.3, 0.4)

    def test_roundtrip(self) -> None:
        """to_array and from_array should be inverses."""
        original = NRMItemParameters(
            item_id=3,
            discriminations=(0.0, 0.5, 1.0),
            intercepts=(0.0, -0.2, 0.3),
        )

        arr = original.to_array()
        reconstructed = NRMItemParameters.from_array(
            item_id=3, arr=arr, n_categories=3
        )

        assert original.item_id == reconstructed.item_id
        assert original.discriminations == reconstructed.discriminations
        assert original.intercepts == reconstructed.intercepts

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
