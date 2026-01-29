import numpy as np
import pytest
from pydantic import ValidationError

from analysis_service.api.schemas import (
    AnalysisRequest,
    EstimationConfigSchema,
    ExamDatasetSchema,
    JobStatus,
)


class TestExamDatasetSchema:
    def test_to_domain_basic(self) -> None:
        schema = ExamDatasetSchema(
            candidate_ids=["A", "B", "C"],
            response_matrix=[[0, 1, 2], [1, 0, 2], [2, 1, 0]],
            n_categories=3,
            correct_answers=None,
        )
        domain = schema.to_domain()
        assert domain.response_matrix.n_candidates == 3
        assert domain.response_matrix.n_items == 3
        assert domain.response_matrix.n_categories == 3
        assert domain.correct_answers is None
        np.testing.assert_array_equal(
            domain.candidate_ids, np.array(["A", "B", "C"])
        )

    def test_to_domain_with_correct_answers(self) -> None:
        schema = ExamDatasetSchema(
            candidate_ids=["A", "B"],
            response_matrix=[[0, 1], [1, 0]],
            n_categories=2,
            correct_answers=[0, 1],
        )
        domain = schema.to_domain()
        assert domain.correct_answers == [0, 1]

    def test_n_categories_minimum(self) -> None:
        with pytest.raises(ValidationError):
            ExamDatasetSchema(
                candidate_ids=["A"],
                response_matrix=[[0]],
                n_categories=1,
            )


class TestEstimationConfigSchema:
    def test_to_domain_defaults(self) -> None:
        schema = EstimationConfigSchema()
        config = schema.to_domain()
        assert config.convergence.max_em_iterations == 500

    def test_to_domain_override(self) -> None:
        schema = EstimationConfigSchema(max_em_iterations=100)
        config = schema.to_domain()
        assert config.convergence.max_em_iterations == 100


class TestAnalysisRequest:
    def test_defaults(self) -> None:
        req = AnalysisRequest(
            exam_dataset=ExamDatasetSchema(
                candidate_ids=["A"],
                response_matrix=[[0]],
                n_categories=2,
            ),
        )
        assert req.significance_level == 0.05
        assert req.num_monte_carlo == 100
        assert req.num_threshold_samples == 100
        assert req.threshold is None
        assert req.random_seed is None

    def test_significance_level_bounds(self) -> None:
        base = {
            "exam_dataset": {
                "candidate_ids": ["A"],
                "response_matrix": [[0]],
                "n_categories": 2,
            }
        }
        with pytest.raises(ValidationError):
            AnalysisRequest(**{**base, "significance_level": 0.0})
        with pytest.raises(ValidationError):
            AnalysisRequest(**{**base, "significance_level": 1.0})

    def test_num_monte_carlo_positive(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisRequest(
                exam_dataset=ExamDatasetSchema(
                    candidate_ids=["A"],
                    response_matrix=[[0]],
                    n_categories=2,
                ),
                num_monte_carlo=0,
            )

    def test_json_round_trip(self) -> None:
        req = AnalysisRequest(
            exam_dataset=ExamDatasetSchema(
                candidate_ids=["A", "B"],
                response_matrix=[[0, 1], [1, 0]],
                n_categories=2,
            ),
            threshold=5,
            random_seed=42,
        )
        json_str = req.model_dump_json()
        restored = AnalysisRequest.model_validate_json(json_str)
        assert restored.threshold == 5
        assert restored.random_seed == 42


class TestJobStatus:
    def test_values(self) -> None:
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
