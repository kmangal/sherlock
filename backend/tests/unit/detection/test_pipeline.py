"""
Tests for detection pipelines.
"""

import numpy as np
import pytest

from analysis_service.core.data_models import ExamDataset, ResponseMatrix
from analysis_service.detection.pipeline import (
    AutomaticDetectionPipeline,
    ThresholdDetectionPipeline,
)


class TestThresholdDetectionPipelineValidation:
    def test_threshold_must_be_positive(self) -> None:
        """Threshold must be positive."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            ThresholdDetectionPipeline(threshold=0)

        with pytest.raises(ValueError, match="threshold must be positive"):
            ThresholdDetectionPipeline(threshold=-1)

    def test_valid_threshold(self) -> None:
        """Valid threshold is accepted."""
        pipeline = ThresholdDetectionPipeline(threshold=10)
        assert pipeline.threshold == 10


class TestAutomaticDetectionPipelineValidation:
    def test_significance_level_must_be_in_range(self) -> None:
        """Significance level must be in (0, 1)."""
        with pytest.raises(ValueError, match="significance_level must be in"):
            AutomaticDetectionPipeline(
                significance_level=0.0, num_monte_carlo=100
            )

        with pytest.raises(ValueError, match="significance_level must be in"):
            AutomaticDetectionPipeline(
                significance_level=1.0, num_monte_carlo=100
            )

        with pytest.raises(ValueError, match="significance_level must be in"):
            AutomaticDetectionPipeline(
                significance_level=-0.1, num_monte_carlo=100
            )

        with pytest.raises(ValueError, match="significance_level must be in"):
            AutomaticDetectionPipeline(
                significance_level=1.1, num_monte_carlo=100
            )

    def test_num_monte_carlo_must_be_positive(self) -> None:
        """num_monte_carlo must be positive."""
        with pytest.raises(
            ValueError, match="num_monte_carlo must be positive"
        ):
            AutomaticDetectionPipeline(
                significance_level=0.05, num_monte_carlo=0
            )

        with pytest.raises(
            ValueError, match="num_monte_carlo must be positive"
        ):
            AutomaticDetectionPipeline(
                significance_level=0.05, num_monte_carlo=-1
            )

    def test_num_threshold_samples_must_be_positive(self) -> None:
        """num_threshold_samples must be positive."""
        with pytest.raises(
            ValueError, match="num_threshold_samples must be positive"
        ):
            AutomaticDetectionPipeline(
                significance_level=0.05,
                num_monte_carlo=100,
                num_threshold_samples=0,
            )

    def test_valid_parameters(self) -> None:
        """Valid parameters are accepted."""
        pipeline = AutomaticDetectionPipeline(
            significance_level=0.05,
            num_monte_carlo=100,
            num_threshold_samples=50,
        )
        assert pipeline.significance_level == 0.05
        assert pipeline.num_monte_carlo == 100
        assert pipeline.num_threshold_samples == 50


class TestThresholdDetectionPipelineRun:
    @pytest.mark.asyncio
    async def test_flags_candidates_above_threshold(self) -> None:
        """Candidates with max similarity above threshold are flagged."""
        # Create data where candidates 0 and 1 are identical (high similarity)
        responses = np.array(
            [
                [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],  # Identical to row 1
                [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],  # Identical to row 0
                [4, 3, 2, 1, 4, 3, 2, 1, 4, 3],  # Different from 0 and 1
                [2, 3, 4, 1, 2, 3, 4, 1, 2, 3],  # Different from all
            ],
            dtype=np.int8,
        )
        candidate_ids = np.array(["A", "B", "C", "D"])

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=5
            ),
            correct_answers=None,
        )

        # Threshold of 5 should flag candidates 0 and 1 (similarity = 10)
        pipeline = ThresholdDetectionPipeline(threshold=5)
        suspects = await pipeline.run(exam_dataset)

        suspect_ids = {s.candidate_id for s in suspects}
        assert suspect_ids == {"A", "B"}

        for s in suspects:
            assert s.detection_threshold == 5.0
            assert s.observed_similarity == 10
            assert s.p_value is None

    @pytest.mark.asyncio
    async def test_no_flags_below_threshold(self) -> None:
        """No candidates flagged if all below threshold."""
        responses = np.array(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [2, 1, 4, 3],
            ],
            dtype=np.int8,
        )
        candidate_ids = np.array(["A", "B", "C"])

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=5
            ),
            correct_answers=None,
        )

        pipeline = ThresholdDetectionPipeline(threshold=10)
        suspects = await pipeline.run(exam_dataset)

        assert len(suspects) == 0

    @pytest.mark.asyncio
    async def test_single_candidate_no_flags(self) -> None:
        """Single candidate cannot be flagged (no pairs)."""
        responses = np.array([[1, 2, 3, 4]], dtype=np.int8)
        candidate_ids = np.array(["A"])

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=5
            ),
            correct_answers=None,
        )

        # With only one candidate, max similarity is 0 (no pairs to compare)
        # Any positive threshold should result in no flags
        pipeline = ThresholdDetectionPipeline(threshold=1)
        suspects = await pipeline.run(exam_dataset)

        assert len(suspects) == 0

    @pytest.mark.asyncio
    async def test_returns_correct_observed_similarity(self) -> None:
        """Each suspect has correct max observed similarity."""
        # Create specific similarity pattern
        responses = np.array(
            [
                [1, 1, 1, 1, 1],  # 5 matches with row 1
                [1, 1, 1, 1, 1],  # 5 matches with row 0
                [1, 1, 2, 2, 2],  # 2 matches with 0 and 1
            ],
            dtype=np.int8,
        )
        candidate_ids = np.array(["A", "B", "C"])

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=5
            ),
            correct_answers=None,
        )

        pipeline = ThresholdDetectionPipeline(threshold=3)
        suspects = await pipeline.run(exam_dataset)

        # Only A and B should be flagged (max similarity = 5)
        suspect_map = {s.candidate_id: s for s in suspects}
        assert set(suspect_map.keys()) == {"A", "B"}
        assert suspect_map["A"].observed_similarity == 5
        assert suspect_map["B"].observed_similarity == 5
