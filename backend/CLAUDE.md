# Project Overview

A Python backend service designed to detect cheating / copying on multiple choice exams.

# Methodology

1. Ingest exam data. For each candidate, there are two fields:
    candidate_id: str
    answer_string: str
2. Fit an IRT model to the data:
3. Generating synthetic datasets from the fit IRT model
4. Generate test statistics by comparing distribution of synthetic data to observed data
5. Flag candidates where test statistics exceed some critical value 

# Key design principles

* The statistical model layer must be decoupled from the service orchestration layer.
* <= 4GB RSS under max load
* All APIs must be batch-native, even if used in single-request contexts.
* Given identical inputs and model versions, outputs must be bit-stable
* The service must enforce hard limits on model dimensionality and inference steps
* Metrics must expose both system performance and model behavior
* Every prediction must be traceable to an immutable model version
* The same model code must support both offline calibration and online inference
* Provide a rich CLI experience, with generous logging and transparency

# Architecture

analysis_service
    - api: FastAPI endpoint
    - detection: Simulate the data generating process according to an IRT model and run statistical hypothesis tests comparing the observed data to the simulated data
    - evaluation: Measure key metrics, e.g. statistical power, false positive rate, etc.
    - irt: Define behavior of IRT models
    - synthetic_data: Generate synthetic exam data

# Development Workflow

**Do NOT worry about backwards compatibility. We should always write the best code possible, even if that means refactoring existing code.**

1. Make changes

2. Typecheck
uv run mypy

3. Run tests
uv run python -m pytest -s                       # Run tests
uv run python -m pytest --cov --cov-report=term  # Run tests with coverage
uv run python -m pytest --cov --cov-report=html  # Generate HTML coverage report (htmlcov/)

4. Lint
uv run ruff format src/                # Code formatting
uv run ruff check src/                 # Linting
uv run lint-imports                    # Import linting

# Style Guide

* Prefer strongly typed inputs and outputs as much as possible. Example:
    If the output keys are fixed and expected in other functions, then don't use a dict.
    ```python
    def my_function() -> dict[str, float]:
        val1 = 1.0
        val2 = 2.0
        return {"key1": val1, "key2": val2}
    
    def consuming_function(d: dict[str, float]) -> float:
        return d["key1"] + d["key2"]
    ```
    Instead make a type for this return object:
    ```python
    from dataclasses import dataclass

    @dataclass
    class Output:
        key1: float
        key2: flota
    
    def my_function() -> Output:
        return Output(key1=1.0, key2=2.0)
    
    def consuming_function(d: Output) -> float:
        return d.key1 + d.key2
    ```

* Avoid magic constants!

    Constants that represent fixed concepts should be placed in a constants.py file (e.g. anlaysis_service.core.constants).
    Otherwise, default paramter values should be declared at the top of the file in all caps, and the methods
    should have optional kwargs for overriding.

* As much as possible, rely on types to ensure logical consistency. But where necessary, functions should be written defensively: They should not assume that the inputs provided are correct, and should raise exceptions as soon as an inconsistency is detected.