"""
Response sampling for IRT models.

This module provides functions to sample responses given abilities and
item parameters. Works with any ItemParameters subclass (3PL, NRM, etc.).
"""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from analysis_service.core.utils import get_rng
from analysis_service.irt.estimation.parameters import NRMItemParameters


def sample_response(
    ability: float,
    item_params: NRMItemParameters,
    rng: Generator | None = None,
) -> int:
    """
    Sample a single response given ability and item parameters.

    Args:
        ability: Candidate's latent ability.
        item_params: Item parameters (any ItemParameters subclass).
        rng: Random number generator.

    Returns:
        Index of selected answer (0-based).
    """
    if rng is None:
        rng = get_rng()

    theta = np.array([ability], dtype=np.float64)
    probs = item_params.compute_probabilities(theta)[0]
    return int(rng.choice(item_params.n_categories, p=probs))


def sample_responses_batch(
    abilities: NDArray[np.float64],
    item_params_list: list[NRMItemParameters],
    rng: Generator | None = None,
) -> NDArray[np.int8]:
    """
    Sample responses for all candidates and items.

    Uses vectorized probability computation and sampling for efficiency.

    Args:
        abilities: Array of shape (n_candidates,) with ability values.
        item_params_list: List of ItemParameters objects (one per item).
        rng: Random number generator.

    Returns:
        Array of shape (n_candidates, n_items) with response indices.
    """
    if rng is None:
        rng = get_rng()

    n_candidates = len(abilities)
    n_items = len(item_params_list)

    responses = np.empty((n_candidates, n_items), dtype=np.int8)

    for j, item_params in enumerate(item_params_list):
        # Compute probabilities for all candidates at once
        probs = item_params.compute_probabilities(abilities)
        n_choices = item_params.n_categories

        # Vectorized sampling using cumulative probabilities
        cumprobs = np.cumsum(probs, axis=1)
        u = rng.random(n_candidates)

        # Find the choice index where cumulative probability exceeds u
        responses[:, j] = np.minimum(
            (cumprobs < u[:, np.newaxis]).sum(axis=1), n_choices - 1
        )

    return responses
