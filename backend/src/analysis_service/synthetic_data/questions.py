"""
Item parameter sampling factories for synthetic data generation.

This module provides factory functions for sampling IRT item parameters
from configured distributions. Works with the synthetic data generation
pipeline to create realistic exam item parameters.
"""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from analysis_service.core.utils import get_rng
from analysis_service.irt.estimation.parameters import (
    NRMItemParameters,
)
from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.parameters import (
    JointParameterSampler,
)


def sample_item_parameters(
    config: GenerationConfig,
    abilities: NDArray[np.float64],
    rng: Generator | None = None,
) -> list[NRMItemParameters]:
    """
    Sample item parameters for an exam.

    Parameters are sampled from a joint distribution specified by the config.
    Includes missing category parameters based on the missingness model.

    Args:
        config: Generation configuration with parameter distributions.
        abilities: Candidate abilities, shape (n_candidates,). Used for
            calibrating missingness parameters.
        rng: Random number generator.

    Returns:
        List of NRMItemParameters objects with sampled parameters
        (including missing category).
    """
    if rng is None:
        rng = get_rng(config.random_seed)

    # Validate abilities
    if abilities.size == 0:
        raise ValueError("abilities array must not be empty")
    if abilities.ndim != 1:
        raise ValueError(f"abilities must be 1D array, got {abilities.ndim}D")

    n_items = config.n_questions
    n_response_categories = config.n_response_categories

    # Determine correct answers
    correct_answers = rng.integers(0, n_response_categories, size=n_items)

    # Sample all parameters jointly (includes missingness)
    sampler = JointParameterSampler(config)
    sampled = sampler.sample(
        n_questions=n_items,
        n_response_categories=n_response_categories,
        correct_answer_ix=correct_answers,
        theta=abilities,
        rng=rng,
    )

    # Validate sampled parameters include missing category
    if not sampled.includes_missing_values:
        raise ValueError(
            "Sampled parameters must include missing category. "
            "Check missingness model configuration."
        )

    expected_n_categories = n_response_categories + 1  # +1 for missing
    if sampled.discrimination.shape[1] != expected_n_categories:
        raise ValueError(
            f"Expected {expected_n_categories} categories "
            f"(including missing), got {sampled.discrimination.shape[1]}"
        )

    # Build NRMItemParameters objects
    # Sampled parameters include missing category, use directly
    item_params = []
    for i in range(n_items):
        params = NRMItemParameters(
            item_id=i,
            discriminations=tuple(sampled.discrimination[i, :]),
            intercepts=tuple(sampled.intercept[i, :]),
            correct_answer=correct_answers[i],
        )
        item_params.append(params)

    return item_params
