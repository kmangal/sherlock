from fastapi import APIRouter, Depends, Response

from analysis_service.api.dependencies import get_job_manager, get_version
from analysis_service.api.jobs import JobManager
from analysis_service.api.schemas import (
    AnalysisRequest,
    HealthResponse,
    JobCreatedResponse,
    JobStatusResponse,
)

router = APIRouter(prefix="/api/v1")


@router.post("/analysis", status_code=202)
async def submit_analysis(
    request: AnalysisRequest,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobCreatedResponse:
    job_id = job_manager.submit(request)
    return JobCreatedResponse(job_id=job_id)


@router.get("/analysis/{job_id}")
async def get_analysis_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobStatusResponse:
    return job_manager.get_status(job_id)


@router.delete("/analysis/{job_id}", status_code=204)
async def cancel_analysis(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> Response:
    job_manager.cancel(job_id)
    return Response(status_code=204)


@router.get("/health")
async def health_check(
    version: str = Depends(get_version),
) -> HealthResponse:
    return HealthResponse(version=version)
