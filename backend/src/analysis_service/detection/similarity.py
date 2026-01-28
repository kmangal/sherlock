import numpy as np
from numba import njit, prange  # type: ignore
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE


@njit  # type: ignore
def count_matching_responses(
    a: NDArray[np.int8], b: NDArray[np.int8]
) -> np.uint32:
    """
    Count matching responses while ignoring positions where either candidate has missing values.

    We only count positions where both candidates have non-missing responses
    (both != `MISSING_VALUE`) and those responses are identical.
    """
    valid_mask = (a != MISSING_VALUE) & (b != MISSING_VALUE)
    matches = (a == b) & valid_mask
    return np.uint32(np.count_nonzero(matches))


@njit(parallel=True)  # type: ignore
def max_similarity_per_candidate(
    responses: NDArray[np.int8],
) -> NDArray[np.uint32]:
    """
    Compute the maximum similarity for each candidate across all other candidates.

    Returns shape (N,) instead of (N, N), providing O(N) memory instead of O(N²).
    """
    n = responses.shape[0]
    max_sim = np.zeros(n, dtype=np.uint32)

    for i in prange(n):
        for j in range(i):
            sim = count_matching_responses(responses[i, :], responses[j, :])
            if sim > max_sim[i]:
                max_sim[i] = sim
            if sim > max_sim[j]:
                max_sim[j] = sim

    return max_sim


@njit  # type: ignore
def find_max_similarity(responses: NDArray[np.int8]) -> np.uint32:
    """Find maximum similarity across all pairwise comparisons."""
    sims = max_similarity_per_candidate(responses)
    return np.uint32(np.max(sims))
