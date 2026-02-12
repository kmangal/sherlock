"""
Cheater injection utilities for evaluation.

Injects synthetic cheating patterns into exam data by copying responses
from source candidates to copier candidates.
"""

import numpy as np
from numpy.typing import NDArray

from analysis_service.evaluation.data_models import (
    CheaterConfig,
    CheaterPair,
    CheatingGroundTruth,
)
from analysis_service.synthetic_data.data_models import GeneratedData
from analysis_service.synthetic_data.generators import generate_exam_responses
from analysis_service.synthetic_data.presets import get_preset
from analysis_service.synthetic_data.utils import responses_to_string


def inject_cheaters(
    responses: NDArray[np.int8],
    config: CheaterConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int8], CheatingGroundTruth]:
    """
    Inject cheating patterns into a responses array.

    Selects source and copier candidates, then copies a specified number of
    items from each source to their copiers. Both sources and copiers are
    considered cheaters.

    Args:
        responses: Response matrix of shape (n_candidates, n_items).
        config: Configuration specifying cheating pattern.
        rng: Random number generator for reproducibility.

    Returns:
        Tuple of (modified responses array, CheatingGroundTruth).

    Raises:
        ValueError: If config requires more candidates than available,
            or if n_copied_items exceeds number of items.
    """
    n_candidates, n_items = responses.shape

    # Validate config
    required_candidates = config.total_cheaters
    if required_candidates > n_candidates:
        raise ValueError(
            f"Config requires {required_candidates} candidates "
            f"(sources + copiers), but only {n_candidates} available"
        )
    if config.n_copied_items > n_items:
        raise ValueError(
            f"n_copied_items ({config.n_copied_items}) exceeds "
            f"number of items ({n_items})"
        )

    # Select distinct candidates for sources and copiers
    all_cheater_indices = rng.choice(
        n_candidates, size=required_candidates, replace=False
    )

    # First n_sources are sources, rest are copiers
    source_indices = all_cheater_indices[: config.n_sources]
    copier_indices = all_cheater_indices[config.n_sources :]

    # Create modified responses (copy to avoid mutating original)
    new_responses = responses.copy()

    # Build cheater pairs and apply copying
    cheater_pairs: list[CheaterPair] = []

    for source_idx_position, source_idx in enumerate(source_indices):
        # Get copiers for this source
        copier_start = source_idx_position * config.n_copiers_per_source
        copier_end = copier_start + config.n_copiers_per_source

        for copier_idx in copier_indices[copier_start:copier_end]:
            # Select random items to copy
            items_to_copy = rng.choice(
                n_items, size=config.n_copied_items, replace=False
            )

            # Copy responses from source to copier
            new_responses[copier_idx, items_to_copy] = new_responses[
                source_idx, items_to_copy
            ]

            cheater_pairs.append(
                CheaterPair(
                    source_idx=int(source_idx),
                    copier_idx=int(copier_idx),
                    n_copied_items=config.n_copied_items,
                )
            )

    ground_truth = CheatingGroundTruth(cheater_pairs=tuple(cheater_pairs))

    return new_responses, ground_truth


def add_cheaters_to_preset(
    preset_name: str,
    cheater_config: CheaterConfig,
    data_seed: int,
    injection_seed: int,
) -> tuple[GeneratedData, CheatingGroundTruth]:
    """
    Generate exam data from a preset and inject cheaters.

    Convenience function that combines preset loading, data generation,
    and cheater injection.

    Args:
        preset_name: Name of the preset to use.
        cheater_config: Configuration for cheating pattern.
        data_seed: Random seed for data generation.
        injection_seed: Random seed for cheater injection.

    Returns:
        Tuple of (GeneratedData with cheaters, CheatingGroundTruth).
    """
    from analysis_service.synthetic_data.config import GenerationConfig

    # Load preset and override seed
    base_config = get_preset(preset_name)
    config = GenerationConfig(
        n_candidates=base_config.n_candidates,
        n_questions=base_config.n_questions,
        n_response_categories=base_config.n_response_categories,
        missing=base_config.missing,
        random_seed=data_seed,
        ability=base_config.ability,
        nrm_parameters=base_config.nrm_parameters,
        correlations=base_config.correlations,
    )

    # Generate base data
    data = generate_exam_responses(config)

    # Inject cheaters using a separate RNG stream
    rng = np.random.default_rng(injection_seed)
    new_responses, ground_truth = inject_cheaters(
        data.responses, cheater_config, rng
    )

    # Rebuild answer strings from modified responses
    new_answer_strings = [
        responses_to_string(new_responses[i])
        for i in range(config.n_candidates)
    ]

    new_data = GeneratedData(
        candidate_ids=data.candidate_ids,
        answer_strings=new_answer_strings,
        abilities=data.abilities,
        item_params=data.item_params,
        responses=new_responses,
        config=data.config,
    )

    return new_data, ground_truth
