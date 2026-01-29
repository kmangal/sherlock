# Sherlock

Sherlock identifies potential cheating on multiple choice exams using statistical analysis.

## How It Works

Sherlock looks for unusually high numbers of matching answers between test-takers and flags suspicious cases for investigation. While matching answers don't prove cheating occurred, they provide a focused starting point for further review.

Determining what counts as "unusually high" depends on the exam and test-takers. For instance, easy questions that everyone answers correctly create legitimate matches. The key question is: What's the likelihood that two candidates of similar ability would naturally answer the same questions identically?

To answer this, Sherlock builds a statistical model from the answer patterns, then simulates how often candidates would have similar answers based purely on their ability and question difficulty. Using these simulations, it calculates the probability of observing the actual degree of similarity and flags cases where this probability falls below a set threshold.

## Getting Started

See CONTRIBUTING.md for a guide to getting your local development environment set up.

### Running the Server

From the `backend/` directory, install dependencies and start the API server:

```bash
uv sync
uv run python -m analysis_service.api
```

The server starts at `http://127.0.0.1:8000` by default.

#### Docker

Build and run from the `backend/` directory:

```bash
docker build -t sherlock-backend backend/
docker run -p 8000:8000 sherlock-backend
```

Configure via environment variables prefixed with `SHERLOCK_`:

```bash
SHERLOCK_HOST=0.0.0.0 SHERLOCK_PORT=9000 uv run python -m analysis_service.api
```

| Variable | Default | Description |
|---|---|---|
| `SHERLOCK_HOST` | `127.0.0.1` | Server bind address |
| `SHERLOCK_PORT` | `8000` | Server port |
| `SHERLOCK_MAX_CANDIDATES` | `10000` | Max candidates per request |
| `SHERLOCK_MAX_ITEMS` | `500` | Max exam items per request |
| `SHERLOCK_MAX_CONCURRENT_JOBS` | `4` | Max parallel analysis jobs |
| `SHERLOCK_JOB_TTL_SECONDS` | `3600` | How long completed jobs are retained |

### API Overview

All endpoints live under `/api/v1`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/analysis` | Submit an analysis job |
| `GET` | `/api/v1/analysis/{job_id}` | Poll job status and results |
| `DELETE` | `/api/v1/analysis/{job_id}` | Cancel a job |

### Submitting an Analysis

Analysis is asynchronous: you submit a job, then poll for results.

**Request — `POST /api/v1/analysis`**

```json
{
  "exam_dataset": {
    "candidate_ids": ["C001", "C002", "C003"],
    "response_matrix": [
      [0, 1, 2, 3, 1],
      [0, 1, 2, 3, 1],
      [3, 2, 1, 0, 2]
    ],
    "n_categories": 4,
    "correct_answers": [0, 1, 2, 3, 2]
  },
  "significance_level": 0.05,
  "num_monte_carlo": 100,
  "random_seed": 42
}
```

- `response_matrix`: each row is a candidate's answers, each value is a 0-indexed category choice (use `-1` for missing).
- `n_categories`: number of answer choices per item (minimum 2).
- `correct_answers` (optional): the answer key, used to improve model estimation.
- `significance_level`: p-value cutoff for flagging suspects (default `0.05`).
- `num_monte_carlo`: number of Monte Carlo simulations (default `100`).
- `threshold` (optional): if provided, uses a fast fixed-threshold pipeline instead of the full IRT-based analysis.

The response returns a job ID:

```json
{ "job_id": "a1b2c3d4e5f6" }
```

### Reading Results

Poll `GET /api/v1/analysis/{job_id}` until `status` is `"completed"` or `"failed"`.

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "completed",
  "progress": null,
  "result": {
    "suspects": [
      {
        "candidate_id": "C002",
        "detection_threshold": 4.0,
        "observed_similarity": 5,
        "p_value": 0.03
      }
    ],
    "model_version": null,
    "pipeline_type": "automatic"
  },
  "error": null,
  "created_at": "2026-01-28T12:00:00+00:00",
  "completed_at": "2026-01-28T12:00:07+00:00"
}
```

- `status`: one of `"pending"`, `"running"`, `"completed"`, `"failed"`.
- `suspects`: candidates flagged for unusually high answer similarity.
  - `observed_similarity`: the candidate's maximum matching-answer count against any other candidate.
  - `detection_threshold`: the similarity count above which a candidate is flagged.
  - `p_value`: probability of observing this similarity by chance (null when using the threshold pipeline).
- `pipeline_type`: `"automatic"` (full IRT-based) or `"threshold"` (fixed cutoff).
- While a job is running, the `progress` field reports the current phase (`"similarity"`, `"irt_fitting"`, `"threshold_calibration"`, or `"monte_carlo"`) and step counts.