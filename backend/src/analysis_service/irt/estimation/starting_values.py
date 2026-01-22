"""
Starting value computation for IRT estimation.

This module provides data-driven initialization strategies for item parameters,
shared across different IRT models.
"""

import numpy as np
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.irt.estimation.data_models import ResponseMatrix


def compute_response_proportions(
    data: ResponseMatrix,
    item_idx: int,
    add_constant: float = 0.5,
) -> NDArray[np.float64]:
    """
    Compute response proportions for an item with additive smoothing.

    Args:
        data: Response matrix.
        item_idx: Index of the item.
        add_constant: Additive smoothing constant (Laplace smoothing).

    Returns:
        Array of shape (n_categories,) with smoothed proportions.
    """
    counts = data.item_response_counts(item_idx).astype(np.float64)
    total = counts.sum()

    if total == 0:
        # No valid responses - return uniform
        uniform: NDArray[np.float64] = (
            np.ones(data.n_categories, dtype=np.float64) / data.n_categories
        )
        return uniform

    # Additive smoothing
    smoothed: NDArray[np.float64] = (counts + add_constant) / (
        total + add_constant * data.n_categories
    )
    return smoothed


def compute_log_odds_relative_to_reference(
    proportions: NDArray[np.float64],
    reference_idx: int = 0,
) -> NDArray[np.float64]:
    """
    Compute log-odds relative to a reference category.

    log(p_k / p_ref) for each category k.

    Args:
        proportions: Response proportions, shape (n_categories,).
        reference_idx: Index of the reference category.

    Returns:
        Array of shape (n_categories,) with log-odds.
        Value at reference_idx is 0.
    """
    log_props = np.log(proportions + 1e-10)
    log_odds: NDArray[np.float64] = log_props - log_props[reference_idx]
    return log_odds


def compute_total_score(data: ResponseMatrix) -> NDArray[np.float64]:
    """
    Compute total score (number of valid responses) for each candidate.

    This can be used as a simple proxy for ability in initialization.

    Args:
        data: Response matrix.

    Returns:
        Array of shape (n_candidates,) with total scores.
    """
    valid = data.valid_mask
    total_scores: NDArray[np.float64] = valid.sum(axis=1).astype(np.float64)
    return total_scores


def compute_item_proportion_correct(
    data: ResponseMatrix,
    correct_answers: NDArray[np.int8] | None = None,
) -> NDArray[np.float64]:
    """
    Compute proportion correct for each item.

    If correct_answers is not provided, uses first category (index 0).
    This is useful for NRM where we don't have a correct answer.

    Args:
        data: Response matrix.
        correct_answers: Array of correct answer indices, shape (n_items,).
            If None, defaults to 0 for all items.

    Returns:
        Array of shape (n_items,) with proportion correct.
    """
    if correct_answers is None:
        correct_answers = np.zeros(data.n_items, dtype=np.int8)

    prop_correct = np.zeros(data.n_items, dtype=np.float64)

    for item_idx in range(data.n_items):
        item_responses = data.responses[:, item_idx]
        valid = item_responses != MISSING_VALUE

        if valid.sum() == 0:
            prop_correct[item_idx] = 0.0
        else:
            correct = item_responses[valid] == correct_answers[item_idx]
            prop_correct[item_idx] = correct.mean()

    return prop_correct
