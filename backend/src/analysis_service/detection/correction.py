import numpy as np
from numpy.typing import NDArray


def benjamini_hochberg(
    p_values: NDArray[np.floating], alpha: float
) -> NDArray[np.bool_]:
    """
    Apply Benjamini-Hochberg procedure for FDR control.

    Args:
        p_values: Array of p-values, shape (N,)
        alpha: Target false discovery rate (e.g., 0.05)

    Returns:
        Boolean array indicating which hypotheses are rejected (True = significant)
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=np.bool_)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Compute BH critical values: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    critical_values = (ranks / n) * alpha

    # Find largest k where p_(k) <= (k/n) * alpha
    significant_mask = sorted_p_values <= critical_values

    # All tests with rank <= k are rejected
    if not np.any(significant_mask):
        return np.zeros(n, dtype=np.bool_)

    max_significant_rank = np.max(np.where(significant_mask)[0]) + 1
    rejected_sorted = np.arange(n) < max_significant_rank

    # Map back to original order
    rejected = np.zeros(n, dtype=np.bool_)
    rejected[sorted_indices] = rejected_sorted

    return rejected
