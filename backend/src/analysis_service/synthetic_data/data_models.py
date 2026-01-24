"""
Data structures for synthetic exam data generation.

This module defines typed data structures for the synthetic data module.
It avoids embedding generation logic - only contracts are defined here.
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.irt.estimation.parameters import NRMItemParameters
from analysis_service.synthetic_data.config import GenerationConfig


class Candidate(BaseModel):
    """
    A candidate (test-taker) with a latent ability.

    Attributes:
        candidate_id: Unique identifier for the candidate.
        ability: Latent ability score (typically standardized, e.g., N(0,1)).
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: int
    ability: float


class GeneratedData(BaseModel):
    """
    Complete output from synthetic data generation.

    Contains both the final answer strings and intermediate data for validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Primary output
    candidate_ids: NDArray[np.int64]
    answer_strings: list[str]

    # Intermediate data (for validation and debugging)
    abilities: NDArray[np.float64]
    item_params: list[NRMItemParameters]
    raw_responses: NDArray[
        np.int8
    ]  # Shape: (n_candidates, n_items), MISSING_VALUE = missing

    # Generation metadata
    config: GenerationConfig

    @property
    def actual_missing_rate(self) -> float:
        """Calculate the actual missing rate from raw responses."""
        n_total = self.raw_responses.size
        n_missing = int(np.sum(self.raw_responses == MISSING_VALUE))
        return n_missing / n_total if n_total > 0 else 0.0


class ItemStatistics(BaseModel):
    proportion_correct: float = Field(..., ge=0.0, le=1.0)
    missing_rate: float = Field(..., ge=0.0, le=1.0)
