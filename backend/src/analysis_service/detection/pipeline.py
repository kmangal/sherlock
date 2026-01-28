import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from analysis_service.core.data_models import ExamDataset, ResponseMatrix
from analysis_service.detection.correction import benjamini_hochberg
from analysis_service.detection.data_models import Suspect
from analysis_service.detection.exceptions import ConvergenceFailedError
from analysis_service.detection.similarity import (
    find_max_similarity,
    max_similarity_per_candidate,
)
from analysis_service.irt.estimation import IRTEstimationResult, NRMEstimator
from analysis_service.irt.sampling import sample_synthetic_responses

logger = logging.getLogger(__name__)

DEFAULT_NUM_THRESHOLD_SAMPLES = 100


class DetectionPipeline(ABC):
    @abstractmethod
    async def run(
        self,
        exam_dataset: ExamDataset,
        rng: np.random.Generator | None = None,
    ) -> list[Suspect]: ...


class ThresholdDetectionPipeline(DetectionPipeline):
    """
    Detect cheaters based on a pre-specified threshold level.

    This helps avoid the computational cost of estimating an IRT model
    and is useful when the user has sufficiently strong priors on what
    the threshold level should be.
    """

    def __init__(self, threshold: int) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold

    async def run(
        self,
        exam_dataset: ExamDataset,
        rng: np.random.Generator | None = None,
    ) -> list[Suspect]:
        responses = exam_dataset.response_matrix.responses
        candidate_ids = exam_dataset.candidate_ids

        # Compute max similarity per candidate (memory-efficient)
        max_similarities = max_similarity_per_candidate(responses)

        # Flag candidates whose max similarity exceeds threshold
        flagged_ixs = np.where(max_similarities > self.threshold)[0]

        suspects: list[Suspect] = []
        for ix in flagged_ixs:
            suspects.append(
                Suspect(
                    candidate_id=candidate_ids[ix].item(),
                    detection_threshold=float(self.threshold),
                    observed_similarity=int(max_similarities[ix]),
                    p_value=None,
                )
            )

        return suspects


class AutomaticDetectionPipeline(DetectionPipeline):
    """
    Use an IRT model to infer where the cheating threshold should be
    and then apply the model to set the threshold.

    Statistical methodology:
    - Test statistic for candidate i: T_i = max_j(similarity(i, j))
    - Threshold calibrated from percentile of max(T_i) across K synthetic datasets
    - P-values computed per candidate via Monte Carlo
    - Multiple testing correction via Benjamini-Hochberg (FDR control)
    """

    def __init__(
        self,
        significance_level: float,
        num_monte_carlo: int,
        num_threshold_samples: int = DEFAULT_NUM_THRESHOLD_SAMPLES,
    ) -> None:
        if not (0 < significance_level < 1):
            raise ValueError("significance_level must be in (0, 1)")
        if num_monte_carlo <= 0:
            raise ValueError("num_monte_carlo must be positive")
        if num_threshold_samples <= 0:
            raise ValueError("num_threshold_samples must be positive")

        self.significance_level = significance_level
        self.num_monte_carlo = num_monte_carlo
        self.num_threshold_samples = num_threshold_samples

    def _calibrate_threshold(
        self,
        response_matrix: ResponseMatrix,
        fitted_parameters: IRTEstimationResult,
        rng: np.random.Generator | None,
    ) -> np.uint32:
        """
        Calibrate threshold using percentile of max similarity across K synthetic datasets.

        Returns the (1 - significance_level) percentile of max(max_similarity_per_candidate)
        across num_threshold_samples synthetic draws.
        """
        max_similarities_per_sample = np.zeros(
            self.num_threshold_samples, dtype=np.uint32
        )

        for k in range(self.num_threshold_samples):
            synth_responses = sample_synthetic_responses(
                response_matrix,
                fitted_parameters,
                rng=rng,
            )
            max_sim = find_max_similarity(synth_responses.responses)
            max_similarities_per_sample[k] = max_sim

        percentile = (1 - self.significance_level) * 100
        threshold = np.percentile(max_similarities_per_sample, percentile)
        return np.uint32(np.ceil(threshold))

    @staticmethod
    def _subset_response_matrix(
        response_matrix: ResponseMatrix,
        candidate_ixs: NDArray[np.intp],
    ) -> ResponseMatrix:
        subset_responses = response_matrix.responses[candidate_ixs, :]
        return ResponseMatrix(
            responses=subset_responses,
            n_categories=response_matrix.n_categories,
        )

    async def run(
        self,
        exam_dataset: ExamDataset,
        rng: np.random.Generator | None = None,
    ) -> list[Suspect]:
        response_matrix = exam_dataset.response_matrix
        responses = response_matrix.responses
        candidate_ids = exam_dataset.candidate_ids
        correct_answers = exam_dataset.correct_answers

        # Compute observed max similarity per candidate
        logger.info("Computing observed max similarity per candidate")
        observed_max_sim = max_similarity_per_candidate(responses)

        # Fit IRT model
        logger.info("Fitting IRT model")
        estimator = NRMEstimator(rng=rng)
        fitted_parameters = estimator.fit(
            data=response_matrix, correct_answers=correct_answers
        )

        if not fitted_parameters.converged:
            raise ConvergenceFailedError

        # Calibrate threshold from synthetic data
        logger.info(
            f"Calibrating threshold from {self.num_threshold_samples} synthetic samples"
        )
        threshold = self._calibrate_threshold(
            response_matrix, fitted_parameters, rng
        )
        logger.info(f"Calibrated threshold: {threshold}")

        # Short-list candidates exceeding threshold
        short_list_ixs = np.where(observed_max_sim > threshold)[0]
        n_short_list = len(short_list_ixs)

        if n_short_list == 0:
            logger.info("No candidates exceed threshold")
            return []

        logger.info(
            f"{n_short_list} candidates exceed threshold, computing p-values"
        )

        # Subset for Monte Carlo p-value computation
        suspect_response_matrix = self._subset_response_matrix(
            response_matrix, short_list_ixs
        )
        subset_observed_max_sim = observed_max_sim[short_list_ixs]

        # Monte Carlo p-value computation
        # Count how often synthetic max_sim >= observed max_sim for each candidate
        exceedance_counts = np.zeros(n_short_list, dtype=np.int32)

        for k in range(self.num_monte_carlo):
            logger.debug(
                f"Monte Carlo iteration {k + 1}/{self.num_monte_carlo}"
            )
            synth_responses = sample_synthetic_responses(
                suspect_response_matrix, fitted_parameters, rng=rng
            )
            synth_max_sim = max_similarity_per_candidate(
                synth_responses.responses
            )
            exceedance_counts += (
                synth_max_sim >= subset_observed_max_sim
            ).astype(np.int32)

        # P-values: proportion of synthetic samples where max_sim >= observed
        p_values = exceedance_counts / self.num_monte_carlo

        # Apply Benjamini-Hochberg correction
        significant = benjamini_hochberg(p_values, self.significance_level)
        significant_ixs = np.where(significant)[0]

        logger.info(
            f"{len(significant_ixs)} candidates significant after FDR correction"
        )

        # Build suspects list
        suspects: list[Suspect] = []
        for subset_ix in significant_ixs:
            original_ix = short_list_ixs[subset_ix]
            suspects.append(
                Suspect(
                    candidate_id=candidate_ids[original_ix].item(),
                    detection_threshold=float(threshold),
                    observed_similarity=int(
                        subset_observed_max_sim[subset_ix]
                    ),
                    p_value=float(p_values[subset_ix]),
                )
            )

        return suspects
