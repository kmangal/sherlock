import numpy as np
import pytest

from analysis_service.synthetic_data.validation import (
    ValidationError,
    compute_missing_rate,
    validate_missing_rate,
)


def test_compute_missing_rate() -> None:
    answer_strings = ["AB*D", "A*CD", "ABCD"]
    rate = compute_missing_rate(answer_strings)
    # 2 missing out of 12 total
    assert np.allclose(rate, 2 / 12)


def test_validate_missing_rate_passes() -> None:
    answer_strings = ["A*CD"] * 100
    validate_missing_rate(answer_strings, expected_rate=0.25, tolerance=0.05)


def test_validate_missing_rate_fails() -> None:
    answer_strings = ["A*CD"] * 100  # 25% missing
    with pytest.raises(ValidationError):
        validate_missing_rate(
            answer_strings, expected_rate=0.10, tolerance=0.05
        )
