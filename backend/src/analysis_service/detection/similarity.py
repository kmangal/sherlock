import numpy as np
from numba import njit  # type: ignore
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE


@njit  # type: ignore
def find_max_similarity(responses: NDArray[np.int8]) -> np.uint32:
    """Find maximum similarity without allocating full matrix. Memory-efficient."""
    n = responses.shape[0]
    max_sim = np.uint32(0)

    for i in range(n):
        for j in range(i):
            sim = hamming_distance_ignore_missing(
                responses[i, :], responses[j, :]
            )

            if sim > max_sim:
                max_sim = sim

    return max_sim


@njit  # type: ignore
def measure_observed_similarity(
    responses: NDArray[np.int8], upper_triangle_only: bool = False
) -> NDArray[np.uint32]:
    """
    Compute pairwise similarities while ignoring missing values (encoded as 0).

    Only counts similarities for questions where both candidates provided non-missing responses.
    """
    n = responses.shape[0]
    similarity = np.zeros(shape=(n, n), dtype=np.uint32)

    for i in range(n):
        for j in range(i):
            similarity[i, j] = hamming_distance_ignore_missing(
                responses[i, :], responses[j, :]
            )

    if upper_triangle_only:
        return similarity.astype(np.uint32)
    else:
        result = similarity + similarity.T
        return result.astype(np.uint32)


@njit  # type: ignore
def hamming_distance_ignore_missing(
    a: NDArray[np.int8], b: NDArray[np.int8]
) -> np.uint32:
    """
    Compute hamming distance while ignoring positions where either candidate has missing values.

    We only count positions where both candidates have non-missing responses
    (both != `MISSING_VALUE`) and those responses are identical.
    """
    # Create mask for valid positions (both candidates have non-missing responses)
    valid_mask = (a != MISSING_VALUE) & (b != MISSING_VALUE)

    # Among valid positions, count how many are identical
    matches = (a == b) & valid_mask
    return np.uint32(np.count_nonzero(matches))
