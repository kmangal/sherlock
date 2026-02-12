"""
Diagnostic utilities for IRT model validation.

Provides functions to compare empirical response probabilities against
model-predicted probabilities.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.core.data_models import ResponseMatrix
from analysis_service.irt.estimation.data_models import IRTEstimationResult


@dataclass
class ResponseProbComparison:
    """Comparison of empirical vs model response probabilities.

    Stores category as int (0-based index, with n_categories representing missing).
    """

    item_id: NDArray[np.int64]
    category: NDArray[
        np.int64
    ]  # 0 to n_categories (inclusive, last is missing)
    empirical_prob: NDArray[np.float64]
    model_prob: NDArray[np.float64]
    difference: NDArray[np.float64]


def compute_response_prob_comparison(
    data: ResponseMatrix,
    model: IRTEstimationResult,
    abilities: NDArray[np.float64],
) -> ResponseProbComparison:
    """Compare empirical vs model response probabilities.

    Includes all response categories (0, 1, ..., n_categories-1) plus missing.
    Categories are stored as integers: 0 to n_categories-1 for valid responses,
    and n_categories for missing.

    Args:
        data: Response matrix with observed responses
        model: Fitted IRT model
        abilities: Estimated ability values for each candidate

    Returns:
        ResponseProbComparison with empirical and model probabilities per item/category
    """
    n_categories = data.n_categories
    n_candidates = data.n_candidates

    item_ids: list[int] = []
    categories: list[int] = []
    empirical_probs: list[float] = []
    model_probs: list[float] = []

    for item_idx, item_params in enumerate(model.item_parameters):
        item_responses = data.responses[:, item_idx]

        # Count each response category (0 to n_categories-1)
        for cat in range(n_categories):
            empirical_count = (item_responses == cat).sum()
            empirical_prob = empirical_count / n_candidates

            # Model: average P(cat|theta) across all candidates
            probs = item_params.compute_probabilities(abilities)
            model_prob = float(np.mean(probs[:, cat]))

            item_ids.append(item_idx)
            categories.append(cat)
            empirical_probs.append(empirical_prob)
            model_probs.append(model_prob)

        # Missing category (index n_categories in model)
        missing_count = (item_responses == MISSING_VALUE).sum()
        empirical_prob_missing = missing_count / n_candidates

        # Model probability for missing (last category in NRM)
        probs = item_params.compute_probabilities(abilities)
        model_prob_missing = float(np.mean(probs[:, n_categories]))

        item_ids.append(item_idx)
        categories.append(n_categories)  # missing category index
        empirical_probs.append(empirical_prob_missing)
        model_probs.append(model_prob_missing)

    empirical_arr = np.array(empirical_probs, dtype=np.float64)
    model_arr = np.array(model_probs, dtype=np.float64)

    return ResponseProbComparison(
        item_id=np.array(item_ids, dtype=np.int64),
        category=np.array(categories, dtype=np.int64),
        empirical_prob=empirical_arr,
        model_prob=model_arr,
        difference=empirical_arr - model_arr,
    )
