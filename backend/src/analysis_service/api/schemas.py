from datetime import datetime
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from analysis_service.core.data_models import ExamDataset, ResponseMatrix
from analysis_service.detection.data_models import Suspect
from analysis_service.irt.estimation.config import EstimationConfig

# --- Enums ---


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisPhase(StrEnum):
    SIMILARITY = "similarity"
    IRT_FITTING = "irt_fitting"
    THRESHOLD_CALIBRATION = "threshold_calibration"
    MONTE_CARLO = "monte_carlo"


# --- Request schemas ---


class ExamDatasetSchema(BaseModel):
    candidate_ids: list[str]
    response_matrix: list[list[int]]
    n_categories: int = Field(ge=2)
    correct_answers: list[int] | None = None

    def to_domain(self) -> ExamDataset:
        responses = np.array(self.response_matrix, dtype=np.int8)
        candidate_ids = np.array(self.candidate_ids, dtype=np.str_)
        rm = ResponseMatrix(
            responses=responses, n_categories=self.n_categories
        )
        return ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=rm,
            correct_answers=self.correct_answers,
        )


class EstimationConfigSchema(BaseModel):
    max_em_iterations: int | None = None
    em_tolerance: float | None = None
    warmup_iterations: int | None = None

    def to_domain(self) -> EstimationConfig:
        from analysis_service.irt.estimation.config import ConvergenceConfig

        overrides: dict[str, int | float] = {}
        if self.max_em_iterations is not None:
            overrides["max_em_iterations"] = self.max_em_iterations
        if self.em_tolerance is not None:
            overrides["em_tolerance"] = self.em_tolerance
        if self.warmup_iterations is not None:
            overrides["warmup_iterations"] = self.warmup_iterations

        convergence = ConvergenceConfig(**overrides)  # type: ignore[arg-type]
        return EstimationConfig(convergence=convergence)


class AnalysisRequest(BaseModel):
    exam_dataset: ExamDatasetSchema
    threshold: int | None = None
    estimation_config: EstimationConfigSchema | None = None
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    num_monte_carlo: int = Field(default=100, gt=0)
    num_threshold_samples: int = Field(default=100, gt=0)
    random_seed: int | None = None


# --- Response schemas ---


class JobProgress(BaseModel):
    phase: AnalysisPhase
    current_step: int
    total_steps: int | None
    message: str


class AnalysisResultSchema(BaseModel):
    suspects: list[Suspect]
    model_version: str | None
    pipeline_type: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: JobProgress | None = None
    result: AnalysisResultSchema | None = None
    error: ErrorDetail | None = None
    created_at: datetime
    completed_at: datetime | None = None


class JobCreatedResponse(BaseModel):
    job_id: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
