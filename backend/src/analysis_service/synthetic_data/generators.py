"""
Orchestration layer for synthetic exam data generation.

This module ties together abilities, item parameters, responses, and missingness
to generate complete exam response data.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analysis_service.core.utils import get_rng
from analysis_service.irt.estimation.parameters import NRMItemParameters
from analysis_service.irt.sampling import sample_responses_batch
from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.data_models import (
    GeneratedData,
)
from analysis_service.synthetic_data.questions import sample_item_parameters
from analysis_service.synthetic_data.sampling import draw_sample
from analysis_service.synthetic_data.utils import responses_to_string


def generate_exam_responses(
    config: GenerationConfig,
) -> GeneratedData:
    """
    Generate synthetic exam response data.

    This is the main entry point for the synthetic data generation pipeline.
    It orchestrates the full generation process:
        1. Sample candidate abilities
        2. Sample item parameters based on config.model_type
        3. Generate responses based on IRT model
        4. Apply missingness
        5. Convert to answer strings

    Args:
        config: Complete generation configuration.

    Returns:
        GeneratedData containing answer strings and all intermediate data.
    """
    rng = get_rng(config.random_seed)

    # Step 1: Sample candidate abilities
    abilities = draw_sample(
        n=config.n_candidates,
        distribution_name=config.ability.distribution,
        distribution_params=config.ability.params,
        rng=rng,
    )

    # Step 2: Sample item parameters (includes missing category)
    item_params = sample_item_parameters(
        config=config, abilities=abilities, rng=rng
    )

    # Step 3: Generate responses
    responses = sample_responses_batch(
        abilities=abilities,
        item_params_list=item_params,
        rng=rng,
    )

    # Step 4: Convert to answer strings
    candidate_ids = np.arange(config.n_candidates, dtype=np.int64)
    answer_strings = [
        responses_to_string(responses[i]) for i in range(config.n_candidates)
    ]

    return GeneratedData(
        candidate_ids=candidate_ids,
        answer_strings=answer_strings,
        abilities=abilities,
        item_params=item_params,
        responses=responses,
        config=config,
    )


def generate_from_item_params(
    item_params: list[NRMItemParameters],
    abilities: NDArray[np.float64],
    seed: int | None = None,
) -> tuple[NDArray[np.int8], list[str]]:
    """
    Generate responses given pre-specified item parameters and abilities.

    This is useful for controlled experiments where you want to fix
    the item parameters or abilities.

    Note: item_params should include the missing category. Missingness is
    modeled as a response category, not applied post-hoc.

    Args:
        item_params: List of NRMItemParameters objects (must include missing category).
        abilities: Array of candidate abilities.
        seed: Random seed.

    Returns:
        Tuple of (responses, answer_strings).
    """
    rng = get_rng(seed)

    # Validate item parameters include missing category
    for param in item_params:
        # NRMItemParameters.n_response_categories returns len(discriminations) - 1
        # So if discriminations has K+1 elements, n_response_categories = K
        # We need at least 2 response categories + 1 missing = 3 total
        if param.n_categories < 3:
            raise ValueError(
                f"Item {param.item_id} must have at least 3 categories "
                f"(2 response + 1 missing), got {param.n_categories}"
            )

    # Generate responses (missing responses are category n_response_categories)
    responses = sample_responses_batch(
        abilities=abilities,
        item_params_list=item_params,
        rng=rng,
    )

    # Convert to strings
    answer_strings = [
        responses_to_string(responses[i]) for i in range(len(abilities))
    ]

    return responses, answer_strings


def to_dataframe(data: GeneratedData) -> pd.DataFrame:
    """
    Convert GeneratedData to a pandas DataFrame.

    Args:
        data: Generated exam data.

    Returns:
        DataFrame with columns: candidate_id, answer_string.
    """
    return pd.DataFrame(
        {
            "candidate_id": data.candidate_ids,
            "answer_string": data.answer_strings,
        }
    )


def to_csv(data: GeneratedData, path: str) -> None:
    """
    Write GeneratedData to a CSV file.

    Args:
        data: Generated exam data.
        path: Output file path.
    """
    df = to_dataframe(data)
    df.to_csv(path, index=False)
