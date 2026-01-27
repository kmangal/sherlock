import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from analysis_service.core.data_models import ResponseMatrix
from analysis_service.detection.data_models import (
    ExamDataset,
    SuspectGroup,
)
from analysis_service.detection.exceptions import ConverenceFailedError
from analysis_service.detection.similarity import (
    find_max_similarity,
    measure_observed_similarity,
)
from analysis_service.irt.estimation import NRMEstimator
from analysis_service.irt.sampling import sample_synthetic_responses

logger = logging.getLogger(__name__)


class DetectionPipeline(ABC):
    @staticmethod
    def _format_observed_similarity_matrix(
        observed_similarity: NDArray[np.uint32],
        candidate_ids: NDArray[np.str_],
    ) -> dict[set[str], int]:
        observed_distribution: dict[set[str], int] = {}
        for index, value in np.ndenumerate(observed_similarity):
            candidate_pair = {candidate_ids[index[0]], candidate_ids[index[1]]}
            observed_distribution[candidate_pair] = int(value)
        return observed_distribution

    @abstractmethod
    async def run(
        self,
        exam_dataset: ExamDataset,
        rng: np.random.Generator | None = None,
    ) -> list[SuspectGroup]: ...


class ThresholdDetectionPipeline(DetectionPipeline):
    """
    Detect cheaters based on a pre-specified threshold level.

    This helps avoid the computational cost of estimating an IRT model
    and is useful when the user has sufficiently strong priors on what
    the threshold level should be.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    async def run(
        self,
        exam_dataset: ExamDataset,
        rng: np.random.Generator | None = None,
    ) -> list[SuspectGroup]:
        responses = exam_dataset.response_matrix.responses
        candidate_ids = exam_dataset.candidate_ids

        observed_similarity = measure_observed_similarity(responses)

        flagged_candidate_ixs = np.argwhere(
            np.triu(observed_similarity > self.threshold, 1)
        )

        suspects: list[SuspectGroup] = []
        for flagged_pair_ixs in flagged_candidate_ixs:
            assert isinstance(flagged_pair_ixs, np.ndarray)
            suspect_ids = set(candidate_ids[flagged_pair_ixs])
            suspects.append(
                SuspectGroup(
                    candidate_ids=suspect_ids,
                    detection_threshold=self.threshold,
                    observed_similarity=int(
                        observed_similarity[
                            flagged_pair_ixs[0], flagged_pair_ixs[1]
                        ],
                    ),
                )
            )

        return suspects


class AutomaticDetectionPipeline(DetectionPipeline):
    """
    Use an IRT model to infer where the cheating threshold should be
    and then apply the model to set the threshold.
    """

    def __init__(
        self, significance_level: float, num_monte_carlo: int
    ) -> None:
        self.significance_level = significance_level
        self.num_monte_carlo = num_monte_carlo

    @staticmethod
    def _subset_short_list_response_matrix(
        response_matrix: ResponseMatrix,
        short_list_candidate_ixs: NDArray[np.intp],
    ) -> ResponseMatrix:
        responses = response_matrix.responses

        subset_responses = responses[
            short_list_candidate_ixs, short_list_candidate_ixs
        ]

        return ResponseMatrix(
            responses=subset_responses,
            n_categories=response_matrix.n_categories,
        )

    async def run(
        self,
        exam_dataset: ExamDataset,
        rng: np.random.Generator | None = None,
    ) -> list[SuspectGroup]:
        response_matrix = exam_dataset.response_matrix
        responses = response_matrix.responses
        candidate_ids = exam_dataset.candidate_ids
        correct_answers = exam_dataset.correct_answers

        # First find max in observed data
        logger.info("Finding maximum observed similarity")
        observed_similarity = measure_observed_similarity(responses)

        # Fit IRT model
        estimator = NRMEstimator(rng=rng)
        fitted_parameters = estimator.fit(
            data=response_matrix, correct_answers=correct_answers
        )

        if not fitted_parameters.converged:
            raise ConverenceFailedError

        # Get synthetic similarities for threshold determination
        logger.info(
            "Generating synthetic datasets for threshold determination"
        )
        synth_responses = sample_synthetic_responses(
            response_matrix, fitted_parameters, rng=rng
        )

        threshold = find_max_similarity(synth_responses.responses)
        logger.info(f"Threshold from synthetic data: {threshold}")

        flagged_candidate_ixs = np.argwhere(
            np.triu(observed_similarity > threshold, 1)
        )

        short_list_suspects: set[int] = set()
        for flagged_pair_ixs in flagged_candidate_ixs:
            short_list_suspects.update(flagged_pair_ixs)

        if len(short_list_suspects) == 0:
            return []

        short_list_candidate_ixs = np.array(
            sorted(short_list_suspects), dtype=np.intp
        )

        # Subset response matrix
        suspect_response_matrix = self._subset_short_list_response_matrix(
            response_matrix=response_matrix,
            short_list_candidate_ixs=short_list_candidate_ixs,
        )

        n_suspects = len(short_list_suspects)
        simulated_distributions = np.zeros(
            (n_suspects, n_suspects, self.num_monte_carlo), dtype=np.uint32
        )

        for iter in range(self.num_monte_carlo):
            logger.debug(f"Iteration #{iter}: Sampling synthetic responses")
            synth_response_matrix = sample_synthetic_responses(
                suspect_response_matrix, fitted_parameters, rng=rng
            )
            simulated_distributions[iter, :, :] = measure_observed_similarity(
                synth_response_matrix.responses
            )

        subset_observed_similarity = observed_similarity[
            short_list_candidate_ixs, short_list_candidate_ixs
        ]
        thresholds = np.quantile(
            simulated_distributions, q=(1 - self.significance_level), axis=2
        )

        flagged_candidate_ixs = np.argwhere(
            np.triu(observed_similarity > thresholds, 1)
        )

        suspects: list[SuspectGroup] = []
        for flagged_pair_ixs in flagged_candidate_ixs:
            assert isinstance(flagged_pair_ixs, np.ndarray)
            original_ixs = short_list_candidate_ixs[flagged_pair_ixs]
            suspect_ids = set(candidate_ids[original_ixs])

            suspects.append(
                SuspectGroup(
                    candidate_ids=suspect_ids,
                    detection_threshold=float(
                        thresholds[flagged_pair_ixs[0], flagged_pair_ixs[1]]
                    ),
                    observed_similarity=int(
                        subset_observed_similarity[
                            flagged_pair_ixs[0], flagged_pair_ixs[1]
                        ],
                    ),
                )
            )

        return suspects
