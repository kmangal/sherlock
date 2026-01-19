import numpy as np

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.core.utils import get_rng
from analysis_service.synthetic_data.missingness import (
    MCARMissingness,
    apply_missingness,
)

ATOL = 0.0
RTOL = 0.01
LARGE_SAMPLE_SIZE = 5000


def test_mcar_rate_recovery() -> None:
    """Test that MCAR produces the expected missing rate."""
    n_candidates, n_questions = LARGE_SAMPLE_SIZE, 100
    target_rate = 0.15
    rng = get_rng(42)

    responses = rng.integers(
        0, 4, size=(n_candidates, n_questions), dtype=np.int8
    )
    model = MCARMissingness(target_rate)
    result = model.apply(responses, rng)

    actual_rate = np.mean(result == MISSING_VALUE)
    assert np.allclose(actual_rate, target_rate, atol=ATOL, rtol=RTOL)


def test_mcar_zero_rate() -> None:
    """Test that zero missing rate produces no missing values."""
    rng = get_rng(42)
    responses = rng.integers(0, 4, size=(100, 50))
    model = MCARMissingness(0.0)
    result = model.apply(responses, rng)

    assert np.sum(result == MISSING_VALUE) == 0


def test_apply_missingness_convenience() -> None:
    rng = get_rng(42)
    responses = rng.integers(0, 4, size=(100, 50))

    result = apply_missingness(responses, missing_rate=0.1, rng=rng)

    assert result.shape == responses.shape
    assert np.any(result == MISSING_VALUE)
