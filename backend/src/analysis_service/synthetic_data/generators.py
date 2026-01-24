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
from analysis_service.synthetic_data.missingness import apply_missingness
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

    # Step 2: Sample item parameters
    item_params = sample_item_parameters(config=config, rng=rng)

    # Step 3: Generate responses
    raw_responses = sample_responses_batch(
        abilities=abilities,
        item_params_list=item_params,
        rng=rng,
    )

    # Step 4: Apply missingness
    responses_with_missing = apply_missingness(
        responses=raw_responses,
        missing_rate=config.missing_rate,
        mechanism="mcar",
        rng=rng,
    )

    # Step 5: Convert to answer strings
    candidate_ids = np.arange(config.n_candidates, dtype=np.int64)
    answer_strings = [
        responses_to_string(responses_with_missing[i])
        for i in range(config.n_candidates)
    ]

    return GeneratedData(
        candidate_ids=candidate_ids,
        answer_strings=answer_strings,
        abilities=abilities,
        item_params=item_params,
        raw_responses=responses_with_missing,
        config=config,
    )


def generate_from_item_params(
    item_params: list[NRMItemParameters],
    abilities: NDArray[np.float64],
    missing_rate: float = 0.0,
    seed: int | None = None,
) -> tuple[NDArray[np.int8], list[str]]:
    """
    Generate responses given pre-specified item parameters and abilities.

    This is useful for controlled experiments where you want to fix
    the item parameters or abilities.

    Args:
        item_params: List of ItemParameters objects.
        abilities: Array of candidate abilities.
        missing_rate: Fraction of responses to mark as missing.
        seed: Random seed.

    Returns:
        Tuple of (raw_responses, answer_strings).
    """
    rng = get_rng(seed)

    # Generate responses
    raw_responses = sample_responses_batch(
        abilities=abilities,
        item_params_list=item_params,
        rng=rng,
    )

    # Apply missingness
    responses_with_missing = apply_missingness(
        responses=raw_responses,
        missing_rate=missing_rate,
        mechanism="mcar",
        rng=rng,
    )

    # Convert to strings
    answer_strings = [
        responses_to_string(responses_with_missing[i])
        for i in range(len(abilities))
    ]

    return responses_with_missing, answer_strings


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
