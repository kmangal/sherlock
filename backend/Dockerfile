FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/

RUN uv sync --frozen --no-dev

ENV SHERLOCK_HOST=0.0.0.0

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "analysis_service.api"]
