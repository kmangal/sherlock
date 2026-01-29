from functools import lru_cache

from analysis_service.api.config import ApiSettings
from analysis_service.api.jobs import JobManager


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()


_job_manager: JobManager | None = None


def init_job_manager(settings: ApiSettings) -> JobManager:
    global _job_manager  # noqa: PLW0603
    _job_manager = JobManager(settings)
    return _job_manager


def get_job_manager() -> JobManager:
    assert _job_manager is not None, "JobManager not initialized"
    return _job_manager


def get_version() -> str:
    from analysis_service.irt.estimation.config import _get_backend_version

    return _get_backend_version()
