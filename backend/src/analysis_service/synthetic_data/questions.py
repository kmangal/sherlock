"""
Item parameter sampling factories for synthetic data generation.

This module provides factory functions for sampling IRT item parameters
from configured distributions. Works with the synthetic data generation
pipeline to create realistic exam item parameters.
"""

from numpy.random import Generator

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
    rng: Generator | None = None,
) -> list[NRMItemParameters]:
    """
    Sample item parameters for an exam.

    Parameters are sampled from a joint distribution specified by the config

    Args:
        config: Generation configuration with parameter distributions.
        rng: Random number generator.

    Returns:
        List of NRMItemParameters objects with sampled parameters.
    """
    if rng is None:
        rng = get_rng(config.random_seed)

    n_items = config.n_questions
    n_choices = config.n_choices

    # Determine correct answers
    correct_answers = rng.integers(0, n_choices, size=n_items)

    # Sample all parameters jointly
    sampler = JointParameterSampler(config)
    sampled = sampler.sample(
        n_questions=n_items,
        n_choices=n_choices,
        correct_answer_ix=correct_answers,
        rng=rng,
    )

    # Build NRMItemParameters objects
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
