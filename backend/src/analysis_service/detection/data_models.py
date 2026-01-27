from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from analysis_service.core.data_models import ResponseMatrix


@dataclass(frozen=True)
class ExamDataset:
    candidate_ids: NDArray[np.str_]
    response_matrix: ResponseMatrix
    correct_answers: list[int] | None

    def __post_init__(self) -> None:
        n_responses = self.response_matrix.responses.shape[0]
        if self.candidate_ids.shape[0] != n_responses:
            raise ValueError(
                "# candidate IDs inconsistent with response matrix shape"
            )
        if self.correct_answers is not None and (
            len(self.correct_answers) != n_responses
        ):
            raise ValueError("# correct answers != # responses")
        if self.correct_answers is not None and (
            np.max(self.correct_answers) > self.response_matrix.n_categories
        ):
            raise ValueError(
                "Correct answer values not consistent with n categories"
            )


class SuspectGroup(BaseModel):
    candidate_ids: set[str]
    detection_threshold: float
    observed_similarity: int


# class CheatingDetectionResults(BaseModel):
#     similarity_threshold: int = Field(
#         description="Similarity threshold used for the analysis"
#     )
#     observed_distribution: dict[set[str], int] = Field(
#         description="Map from candidate IDs to # of observed similarities"
#     )
#     suspects: set[frozenset[str]] = Field(
#         description="set of groups of candidate IDs that are flagged as suspicious"
#     )
