import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from analysis_service.api.config import ApiSettings
from analysis_service.api.errors import (
    DataSizeExceededError,
    JobNotFoundError,
    TooManyJobsError,
)
from analysis_service.api.schemas import (
    AnalysisPhase,
    AnalysisRequest,
    AnalysisResultSchema,
    ErrorDetail,
    JobProgress,
    JobStatus,
    JobStatusResponse,
)
from analysis_service.detection.exceptions import ConvergenceFailedError
from analysis_service.detection.pipeline import (
    AutomaticDetectionPipeline,
    DetectionPipeline,
    ThresholdDetectionPipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    status: JobStatus
    created_at: datetime
    request: AnalysisRequest
    progress: JobProgress | None = None
    result: AnalysisResultSchema | None = None
    error: ErrorDetail | None = None
    completed_at: datetime | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class JobManager:
    def __init__(self, settings: ApiSettings) -> None:
        self._settings = settings
        self._jobs: dict[str, Job] = {}
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)

    def _validate_data_size(self, request: AnalysisRequest) -> None:
        ds = request.exam_dataset
        n_candidates = len(ds.candidate_ids)
        n_items = len(ds.response_matrix[0]) if ds.response_matrix else 0
        n_categories = ds.n_categories

        if n_candidates > self._settings.max_candidates:
            raise DataSizeExceededError(
                f"n_candidates={n_candidates} exceeds max={self._settings.max_candidates}"
            )
        if n_items > self._settings.max_items:
            raise DataSizeExceededError(
                f"n_items={n_items} exceeds max={self._settings.max_items}"
            )
        if n_categories > self._settings.max_categories:
            raise DataSizeExceededError(
                f"n_categories={n_categories} exceeds max={self._settings.max_categories}"
            )

    def submit(self, request: AnalysisRequest) -> str:
        self._validate_data_size(request)

        if self._semaphore._value == 0:  # noqa: SLF001
            raise TooManyJobsError

        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC),
            request=request,
        )
        self._jobs[job_id] = job
        job.task = asyncio.create_task(self._run_job(job))
        return job_id

    def get_status(self, job_id: str) -> JobStatusResponse:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.task and not job.task.done():
            job.task.cancel()
        del self._jobs[job_id]

    async def _run_job(self, job: Job) -> None:
        async with self._semaphore:
            job.status = JobStatus.RUNNING
            request = job.request

            def progress_cb(
                phase: str,
                current_step: int,
                total_steps: int | None,
                message: str,
            ) -> None:
                job.progress = JobProgress(
                    phase=AnalysisPhase(phase),
                    current_step=current_step,
                    total_steps=total_steps,
                    message=message,
                )

            try:
                rng = (
                    np.random.default_rng(request.random_seed)
                    if request.random_seed is not None
                    else None
                )
                exam_dataset = request.exam_dataset.to_domain()

                pipeline: DetectionPipeline
                if request.threshold is not None:
                    pipeline = ThresholdDetectionPipeline(
                        threshold=request.threshold
                    )
                    pipeline_type = "threshold"
                    model_version = None
                else:
                    pipeline = AutomaticDetectionPipeline(
                        significance_level=request.significance_level,
                        num_monte_carlo=request.num_monte_carlo,
                        num_threshold_samples=request.num_threshold_samples,
                    )
                    pipeline_type = "automatic"
                    model_version = None

                suspects = await pipeline.run(
                    exam_dataset, rng=rng, progress_callback=progress_cb
                )

                job.result = AnalysisResultSchema(
                    suspects=suspects,
                    model_version=model_version,
                    pipeline_type=pipeline_type,
                )
                job.status = JobStatus.COMPLETED

            except ConvergenceFailedError:
                job.status = JobStatus.FAILED
                job.error = ErrorDetail(
                    code="CONVERGENCE_FAILED",
                    message="IRT model failed to converge",
                )

            except Exception:
                logger.exception(f"Job {job.job_id} failed")
                job.status = JobStatus.FAILED
                job.error = ErrorDetail(
                    code="INTERNAL_ERROR",
                    message="Internal error during analysis",
                )

            finally:
                job.completed_at = datetime.now(UTC)

    async def cleanup_expired(self) -> None:
        now = datetime.now(UTC)
        expired = [
            jid
            for jid, job in self._jobs.items()
            if job.completed_at
            and (now - job.completed_at).total_seconds()
            > self._settings.job_ttl_seconds
        ]
        for jid in expired:
            del self._jobs[jid]
