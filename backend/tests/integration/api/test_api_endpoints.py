"""
Integration tests for the FastAPI detection pipeline server.

Uses httpx.AsyncClient + ASGITransport for in-process HTTP round-trips.
"""

import asyncio

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from analysis_service.api.app import create_app
from analysis_service.api.config import ApiSettings

SMALL_SETTINGS = ApiSettings(
    max_candidates=10000,
    max_items=500,
    max_categories=20,
    max_concurrent_jobs=4,
    job_ttl_seconds=3600,
)


def _make_client() -> AsyncClient:
    app = create_app(settings=SMALL_SETTINGS)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


def _small_dataset(
    n_candidates: int = 20,
    n_items: int = 10,
    n_categories: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    responses = rng.integers(
        0, n_categories, size=(n_candidates, n_items)
    ).tolist()
    candidate_ids = [f"C{i:04d}" for i in range(n_candidates)]
    return {
        "candidate_ids": candidate_ids,
        "response_matrix": responses,
        "n_categories": n_categories,
    }


def _dataset_with_cheaters(
    n_candidates: int = 50,
    n_items: int = 30,
    n_categories: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    responses = rng.integers(0, n_categories, size=(n_candidates, n_items))
    # Inject cheater: copy candidate 0's answers to candidate 1
    n_copy = int(n_items * 0.95)
    copy_items = rng.choice(n_items, size=n_copy, replace=False)
    responses[1, copy_items] = responses[0, copy_items]

    candidate_ids = [f"C{i:04d}" for i in range(n_candidates)]
    return {
        "candidate_ids": candidate_ids,
        "response_matrix": responses.tolist(),
        "n_categories": n_categories,
    }


async def _poll_until_done(
    client: AsyncClient,
    job_id: str,
    max_polls: int = 300,
    poll_interval: float = 0.5,
) -> dict[str, object]:
    for _ in range(max_polls):
        resp = await client.get(f"/api/v1/analysis/{job_id}")
        assert resp.status_code == 200
        data: dict[str, object] = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not complete in time")


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health(self) -> None:
        async with _make_client() as client:
            resp = await client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "version" in data


class TestThresholdPipelineE2E:
    @pytest.mark.asyncio
    async def test_submit_and_poll(self) -> None:
        async with _make_client() as client:
            payload = {
                "exam_dataset": _small_dataset(),
                "threshold": 5,
                "random_seed": 42,
            }
            resp = await client.post("/api/v1/analysis", json=payload)
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            result = await _poll_until_done(client, job_id)
            assert result["status"] == "completed"
            assert result["result"] is not None
            assert result["result"]["pipeline_type"] == "threshold"
            assert isinstance(result["result"]["suspects"], list)


class TestAutomaticPipelineE2E:
    @pytest.mark.asyncio
    async def test_submit_and_poll(self) -> None:
        async with _make_client() as client:
            payload = {
                "exam_dataset": _dataset_with_cheaters(),
                "significance_level": 0.05,
                "num_monte_carlo": 10,
                "num_threshold_samples": 10,
                "random_seed": 42,
            }
            resp = await client.post("/api/v1/analysis", json=payload)
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            result = await _poll_until_done(client, job_id)
            assert result["status"] == "completed"
            assert result["result"] is not None
            assert result["result"]["pipeline_type"] == "automatic"


class TestValidationRejection:
    @pytest.mark.asyncio
    async def test_oversized_dataset(self) -> None:
        settings = ApiSettings(max_candidates=5)
        app = create_app(settings=settings)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            payload = {
                "exam_dataset": _small_dataset(n_candidates=10),
                "threshold": 5,
            }
            resp = await client.post("/api/v1/analysis", json=payload)
            assert resp.status_code == 422
            data = resp.json()
            assert data["code"] == "DATA_SIZE_EXCEEDED"


class TestJobNotFound:
    @pytest.mark.asyncio
    async def test_nonexistent_job(self) -> None:
        async with _make_client() as client:
            resp = await client.get("/api/v1/analysis/nonexistent")
            assert resp.status_code == 404
            data = resp.json()
            assert data["code"] == "JOB_NOT_FOUND"


class TestProgressUpdates:
    @pytest.mark.asyncio
    async def test_progress_fields_present(self) -> None:
        async with _make_client() as client:
            payload = {
                "exam_dataset": _dataset_with_cheaters(),
                "significance_level": 0.05,
                "num_monte_carlo": 10,
                "num_threshold_samples": 10,
                "random_seed": 42,
            }
            resp = await client.post("/api/v1/analysis", json=payload)
            job_id = resp.json()["job_id"]

            # Poll a few times to capture progress
            for _ in range(300):
                resp = await client.get(f"/api/v1/analysis/{job_id}")
                data = resp.json()
                if data.get("progress") is not None:
                    assert "phase" in data["progress"]
                    assert "current_step" in data["progress"]
                    assert "message" in data["progress"]
                if data["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.2)

            # Progress may or may not be seen depending on timing,
            # but the job should complete
            assert data["status"] == "completed"


class TestCancelJob:
    @pytest.mark.asyncio
    async def test_delete_job(self) -> None:
        async with _make_client() as client:
            payload = {
                "exam_dataset": _dataset_with_cheaters(),
                "significance_level": 0.05,
                "num_monte_carlo": 100,
                "num_threshold_samples": 100,
                "random_seed": 42,
            }
            resp = await client.post("/api/v1/analysis", json=payload)
            job_id = resp.json()["job_id"]

            # Cancel it
            resp = await client.delete(f"/api/v1/analysis/{job_id}")
            assert resp.status_code == 204

            # Should be gone
            resp = await client.get(f"/api/v1/analysis/{job_id}")
            assert resp.status_code == 404
