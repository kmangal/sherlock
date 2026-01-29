import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from analysis_service.api.schemas import ErrorDetail

logger = logging.getLogger(__name__)


class DataSizeExceededError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class JobNotFoundError(Exception):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}")


class TooManyJobsError(Exception):
    pass


def _get_request_id(request: Request) -> str | None:
    if hasattr(request.state, "request_id"):
        result: str = request.state.request_id
        return result
    return None


async def data_size_exceeded_handler(
    request: Request, exc: DataSizeExceededError
) -> JSONResponse:
    detail = ErrorDetail(
        code="DATA_SIZE_EXCEEDED",
        message=exc.message,
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=422, content=detail.model_dump())


async def job_not_found_handler(
    request: Request, exc: JobNotFoundError
) -> JSONResponse:
    detail = ErrorDetail(
        code="JOB_NOT_FOUND",
        message=str(exc),
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=404, content=detail.model_dump())


async def too_many_jobs_handler(
    request: Request, exc: TooManyJobsError
) -> JSONResponse:
    detail = ErrorDetail(
        code="TOO_MANY_JOBS",
        message="Too many concurrent jobs",
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=429, content=detail.model_dump())


async def validation_error_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    detail = ErrorDetail(
        code="VALIDATION_ERROR",
        message=str(exc),
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=422, content=detail.model_dump())


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception("Unhandled exception")
    detail = ErrorDetail(
        code="INTERNAL_ERROR",
        message="Internal server error",
        request_id=_get_request_id(request),
    )
    return JSONResponse(status_code=500, content=detail.model_dump())
