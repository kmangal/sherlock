"""
Response sampling for IRT models.

This module provides functions to sample responses given abilities and
item parameters. Works with any ItemParameters subclass (3PL, NRM, etc.).

Missing responses are modeled as a separate category. When sampled,
the missing category is returned as MISSING_VALUE (-1).
"""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.core.data_models import ResponseMatrix
from analysis_service.core.utils import get_rng
from analysis_service.irt.estimation.abilities import estimate_abilities_eap
from analysis_service.irt.estimation.config import EstimationConfig
from analysis_service.irt.estimation.data_models import IRTEstimationResult
from analysis_service.irt.estimation.parameters import NRMItemParameters


def sample_response(
    ability: float,
    item_params: NRMItemParameters,
    rng: Generator | None = None,
) -> int:
    """
    Sample a single response given ability and item parameters.

    Samples from all K+1 categories (K response + 1 missing).
    If the missing category is sampled, returns MISSING_VALUE (-1).

    Args:
        ability: Candidate's latent ability.
        item_params: Item parameters (K+1 categories including missing).
        rng: Random number generator.

    Returns:
        Index of selected answer (0-based), or MISSING_VALUE (-1) if missing.
    """
    if rng is None:
        rng = get_rng()

    theta = np.array([ability], dtype=np.float64)
    probs = item_params.compute_probabilities(theta)[0]
    sampled_category = int(rng.choice(item_params.n_categories, p=probs))

    # If missing category (K) was sampled, return MISSING_VALUE
    if sampled_category == item_params.n_response_categories:
        return MISSING_VALUE
    return sampled_category


def sample_responses_batch(
    abilities: NDArray[np.float64],
    item_params_list: list[NRMItemParameters],
    rng: Generator | None = None,
    allow_missing: bool = True,
) -> NDArray[np.int8]:
    """
    Sample responses for all candidates and items.

    Uses vectorized probability computation and sampling for efficiency.
    Samples from all K+1 categories (K response + 1 missing).
    If the missing category is sampled, returns MISSING_VALUE (-1).

    Args:
        abilities: Array of shape (n_candidates,) with ability values.
        item_params_list: List of ItemParameters objects (K+1 categories each).
        rng: Random number generator.

    Returns:
        Array of shape (n_candidates, n_items) with response indices.
        Missing responses are encoded as MISSING_VALUE (-1).
    """
    if rng is None:
        rng = get_rng()

    n_candidates = len(abilities)
    n_items = len(item_params_list)

    responses = np.empty((n_candidates, n_items), dtype=np.int8)

    for j, item_params in enumerate(item_params_list):
        # Compute probabilities for all candidates at once
        probs = item_params.compute_probabilities(abilities)
        n_total_categories = item_params.n_categories  # K+1
        n_response_categories = item_params.n_response_categories  # K

        if not allow_missing:
            # Sample only from response categories (exclude missing)
            probs = probs[:, :n_response_categories]
            probs = probs / probs.sum(axis=1, keepdims=True)
            n_choices = n_response_categories
        else:
            n_choices = n_total_categories

        # Vectorized sampling using cumulative probabilities
        cumprobs = np.cumsum(probs, axis=1)
        u = rng.random(n_candidates)

        # Find the choice index where cumulative probability exceeds u
        sampled = np.minimum(
            (cumprobs < u[:, np.newaxis]).sum(axis=1), n_choices - 1
        )

        # Map missing category (K) to MISSING_VALUE
        sampled_responses = sampled.astype(np.int8)
        sampled_responses[sampled == n_response_categories] = MISSING_VALUE
        responses[:, j] = sampled_responses

    return responses


def sample_synthetic_responses(
    data: ResponseMatrix,
    model: IRTEstimationResult,
    config: EstimationConfig | None = None,
    rng: Generator | None = None,
) -> ResponseMatrix:
    """
    Sample synthetic responses from a fitted IRT model.

    Args:
        data: Original response matrix (used to sample abilities).
        model: Fitted IRT model with item parameters.
        config: Estimation configuration.
        rng: Random number generator.

    Returns:
        ResponseMatrix with sampled responses (may include MISSING_VALUE).
    """
    item_params_list = model.item_parameters
    # Use n_response_categories (K) for the ResponseMatrix, not n_categories (K+1)
    n_response_categories = item_params_list[0].n_response_categories
    assert all(
        ip.n_response_categories == n_response_categories
        for ip in item_params_list
    )

    abilities = estimate_abilities_eap(data, model, config)
    sample = sample_responses_batch(abilities.eap, list(item_params_list), rng)
    return ResponseMatrix(responses=sample, n_categories=n_response_categories)
