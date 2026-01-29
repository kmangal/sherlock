from pydantic_settings import BaseSettings

SHERLOCK_ENV_PREFIX = "SHERLOCK_"


class ApiSettings(BaseSettings):
    model_config = {"env_prefix": SHERLOCK_ENV_PREFIX}

    max_candidates: int = 10000
    max_items: int = 500
    max_categories: int = 20
    host: str = "127.0.0.1"
    port: int = 8000
    max_concurrent_jobs: int = 4
    job_ttl_seconds: int = 3600
