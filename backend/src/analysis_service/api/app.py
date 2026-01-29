import uuid
from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import FastAPI, Request, Response
from pydantic import ValidationError
from starlette.types import ExceptionHandler

from analysis_service.api.config import ApiSettings
from analysis_service.api.dependencies import get_settings, init_job_manager
from analysis_service.api.errors import (
    DataSizeExceededError,
    JobNotFoundError,
    TooManyJobsError,
    data_size_exceeded_handler,
    job_not_found_handler,
    too_many_jobs_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from analysis_service.api.routes import router


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(title="Sherlock Detection API")
    app.state.settings = settings
    init_job_manager(settings)

    # Exception handlers — cast needed because FastAPI expects
    # (Request, Exception) but our handlers use specific exc types.
    _eh = cast(ExceptionHandler, data_size_exceeded_handler)
    app.add_exception_handler(DataSizeExceededError, _eh)
    _eh = cast(ExceptionHandler, job_not_found_handler)
    app.add_exception_handler(JobNotFoundError, _eh)
    _eh = cast(ExceptionHandler, too_many_jobs_handler)
    app.add_exception_handler(TooManyJobsError, _eh)
    _eh = cast(ExceptionHandler, validation_error_handler)
    app.add_exception_handler(ValidationError, _eh)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Request-ID middleware
    @app.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(router)

    return app
