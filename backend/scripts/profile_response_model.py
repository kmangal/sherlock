#!/usr/bin/env python
"""
Profile memory usage during IRT model estimation.

Uses memory-profiler to track memory allocations during model fitting.
Uses tracemalloc filtered to source code to identify top allocators.
Outputs JSON report with timing and memory statistics.
"""

import json
import logging
import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from memory_profiler import memory_usage  # type: ignore[import-untyped]
from numpy.typing import NDArray

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.core.data_models import ResponseMatrix
from analysis_service.irt.estimation.estimator import (
    IRTEstimationResult,
    NRMEstimator,
)
from analysis_service.synthetic_data.generators import generate_exam_responses
from analysis_service.synthetic_data.presets import (
    get_available_presets,
    get_preset,
)
from analysis_service.synthetic_data.utils import letter_to_index

BACKEND_DIR = Path(__file__).parent.parent.absolute()
SYNTHETIC_DATA_DIR = BACKEND_DIR / "data" / "synthetic"

DEFAULT_OUTPUT_DIR = BACKEND_DIR / "reports" / "estimator_memory_profile"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source code filter for tracemalloc
SOURCE_CODE_FILTER = "analysis_service"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scripts.profile_response_model")

app = typer.Typer()


@dataclass
class MemoryProfile:
    """Memory profiling results."""

    baseline_mib: float
    peak_mib: float
    final_mib: float
    memory_timeline: list[float]
    top_allocators: list[tuple[str, int]]


@dataclass
class ProfilingResult:
    """Complete profiling results."""

    # Metadata
    timestamp: str
    dataset: str

    # Data shape
    n_candidates: int
    n_items: int
    n_categories: int
    response_matrix_bytes: int

    # Timing
    fit_duration_seconds: float

    # Memory (in MiB from memory-profiler)
    baseline_mib: float
    peak_mib: float
    final_mib: float
    delta_mib: float

    # Top allocators from tracemalloc (source code only)
    top_allocators: list[tuple[str, int]]

    # Model fit
    n_iterations: int
    convergence_status: str
    log_likelihood: float


def parse_answer_string(answer_string: str) -> NDArray[np.int8]:
    """
    Parse an answer string into response indices.

    A-Z maps to 0-25, * maps to MISSING_VALUE (-1).
    """
    responses = []
    for char in answer_string:
        if char == "*":
            responses.append(MISSING_VALUE)
        else:
            responses.append(letter_to_index(char))
    return np.array(responses, dtype=np.int8)


def load_from_csv(path: Path) -> ResponseMatrix:
    """
    Load response data from a CSV file.

    CSV must have 'answer_string' column with A-Z or * characters.
    """
    df = pd.read_csv(path)
    if "answer_string" not in df.columns:
        raise ValueError(f"CSV must have 'answer_string' column: {path}")

    answer_strings = df["answer_string"].tolist()
    if not answer_strings:
        raise ValueError(f"CSV has no data rows: {path}")

    # Parse all answer strings
    response_arrays = [parse_answer_string(s) for s in answer_strings]
    responses = np.stack(response_arrays)

    # Determine n_categories from max valid response value
    valid_mask = responses != MISSING_VALUE
    if not valid_mask.any():
        raise ValueError("No valid responses found in data")
    max_response = responses[valid_mask].max()
    n_categories = int(max_response) + 1

    return ResponseMatrix(responses=responses, n_categories=n_categories)


def load_dataset(dataset: str) -> tuple[ResponseMatrix, str]:
    """
    Load dataset from various sources.

    Resolution order:
    1. If dataset is existing .csv file path -> load CSV
    2. If SYNTHETIC_DATA_DIR/{dataset}.csv exists -> load that CSV
    3. If dataset is in get_available_presets() -> generate from preset
    4. Error with available options

    Returns:
        Tuple of (ResponseMatrix, dataset_name).
    """
    dataset_path = Path(dataset)

    # Check if it's a direct CSV path
    if dataset_path.suffix == ".csv" and dataset_path.exists():
        logger.info("Loading from CSV file: %s", dataset_path)
        return load_from_csv(dataset_path), dataset_path.stem

    # Check synthetic data directory
    synthetic_path = SYNTHETIC_DATA_DIR / f"{dataset}.csv"
    if synthetic_path.exists():
        logger.info("Loading from synthetic data: %s", synthetic_path)
        return load_from_csv(synthetic_path), dataset

    # Check presets
    available_presets = get_available_presets()
    if dataset in available_presets:
        logger.info("Generating from preset: %s", dataset)
        config = get_preset(dataset)
        data = generate_exam_responses(config)

        # Convert to ResponseMatrix
        response_arrays = [parse_answer_string(s) for s in data.answer_strings]
        responses = np.stack(response_arrays)
        n_categories = config.n_response_categories

        return ResponseMatrix(
            responses=responses, n_categories=n_categories
        ), dataset

    # Error with available options
    options = []
    if available_presets:
        options.append(f"Presets: {', '.join(sorted(available_presets))}")

    # Check for existing CSVs
    if SYNTHETIC_DATA_DIR.exists():
        csv_files = list(SYNTHETIC_DATA_DIR.glob("*.csv"))
        if csv_files:
            csv_names = [f.stem for f in csv_files]
            options.append(f"Synthetic CSVs: {', '.join(sorted(csv_names))}")

    options_str = "\n  ".join(options) if options else "None available"
    raise typer.BadParameter(
        f"Dataset '{dataset}' not found.\n\nAvailable options:\n  {options_str}"
    )


def get_estimator() -> NRMEstimator:
    """Get estimator instance."""
    return NRMEstimator()


def get_source_allocators(n_top: int = 10) -> list[tuple[str, int]]:
    """
    Get top memory allocators from source code only.

    Filters tracemalloc results to only include allocations from
    analysis_service code, excluding library code.
    """
    snapshot = tracemalloc.take_snapshot()

    # Filter to only source code
    snapshot = snapshot.filter_traces(
        [tracemalloc.Filter(True, f"*{SOURCE_CODE_FILTER}*")]
    )

    stats = snapshot.statistics("lineno")
    return [(str(stat.traceback), stat.size) for stat in stats[:n_top]]


def profile_fit(
    estimator: NRMEstimator,
    data: ResponseMatrix,
) -> tuple[IRTEstimationResult, float, MemoryProfile]:
    """
    Profile estimator.fit() call.

    Uses memory-profiler for memory tracking and tracemalloc for
    identifying top allocators in source code.

    Returns:
        Tuple of (fitted_model, duration_seconds, memory_profile).
    """
    # Container for fitted model (to extract from closure)
    result_container: list[IRTEstimationResult] = []

    def fit_wrapper() -> None:
        result_container.append(estimator.fit(data))

    # Get baseline memory
    baseline_mem = memory_usage(max_usage=True)

    # Start tracemalloc for allocation tracking
    tracemalloc.start()

    # Profile the fit call
    start_time = time.perf_counter()
    mem_timeline = memory_usage((fit_wrapper,), interval=0.1, max_iterations=1)  # pyright: ignore[reportArgumentType]
    duration = time.perf_counter() - start_time

    # Get top allocators from source code
    top_allocators = get_source_allocators()

    tracemalloc.stop()

    # Extract memory stats
    peak_mem = max(mem_timeline) if mem_timeline else baseline_mem
    final_mem = mem_timeline[-1] if mem_timeline else baseline_mem

    profile = MemoryProfile(
        baseline_mib=float(baseline_mem),
        peak_mib=float(peak_mem),
        final_mib=float(final_mem),
        memory_timeline=list(mem_timeline),
        top_allocators=top_allocators,
    )

    return result_container[0], duration, profile


def build_profiling_result(
    dataset_name: str,
    data: ResponseMatrix,
    fitted_model: IRTEstimationResult,
    duration: float,
    profile: MemoryProfile,
) -> ProfilingResult:
    """Build complete profiling result from components."""
    response_matrix_bytes = data.responses.nbytes

    return ProfilingResult(
        timestamp=datetime.now(UTC).isoformat(),
        dataset=dataset_name,
        n_candidates=data.n_candidates,
        n_items=data.n_items,
        n_categories=data.n_categories,
        response_matrix_bytes=response_matrix_bytes,
        fit_duration_seconds=duration,
        baseline_mib=profile.baseline_mib,
        peak_mib=profile.peak_mib,
        final_mib=profile.final_mib,
        delta_mib=profile.peak_mib - profile.baseline_mib,
        top_allocators=profile.top_allocators,
        n_iterations=fitted_model.n_iterations,
        convergence_status=fitted_model.convergence_status,
        log_likelihood=fitted_model.log_likelihood,
    )


def format_bytes(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024**2:
        return f"{n_bytes / 1024:.2f} KB"
    elif n_bytes < 1024**3:
        return f"{n_bytes / 1024**2:.2f} MB"
    else:
        return f"{n_bytes / 1024**3:.2f} GB"


def format_mib(mib: float) -> str:
    """Format MiB as human-readable string."""
    if mib < 1:
        return f"{mib * 1024:.2f} KiB"
    elif mib < 1024:
        return f"{mib:.2f} MiB"
    else:
        return f"{mib / 1024:.2f} GiB"


def result_to_dict(result: ProfilingResult) -> dict[str, Any]:
    """Convert ProfilingResult to JSON-serializable dict."""
    return {
        "metadata": {
            "timestamp": result.timestamp,
            "dataset": result.dataset,
        },
        "data_shape": {
            "n_candidates": result.n_candidates,
            "n_items": result.n_items,
            "n_categories": result.n_categories,
            "response_matrix_bytes": result.response_matrix_bytes,
        },
        "timing": {
            "fit_duration_seconds": result.fit_duration_seconds,
        },
        "memory": {
            "baseline_mib": result.baseline_mib,
            "peak_mib": result.peak_mib,
            "final_mib": result.final_mib,
            "delta_mib": result.delta_mib,
            "top_allocators": [
                {
                    "location": Path(loc).relative_to(BACKEND_DIR).as_posix(),
                    "size_bytes": size,
                }
                for loc, size in result.top_allocators
            ],
        },
        "model_fit": {
            "n_iterations": result.n_iterations,
            "convergence_status": result.convergence_status,
            "log_likelihood": result.log_likelihood,
        },
    }


def write_json_report(result: ProfilingResult, path: Path) -> None:
    """Write profiling result as JSON."""
    data = result_to_dict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote JSON report: %s", path)


@app.command()
def main(
    dataset: str = typer.Argument(
        ...,
        help="Preset name or path to CSV file",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory for output reports",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Profile memory usage during IRT model estimation."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load dataset
    data, dataset_name = load_dataset(dataset)
    logger.info(
        "Loaded data: %d candidates, %d items, %d categories",
        data.n_candidates,
        data.n_items,
        data.n_categories,
    )

    # Get estimator
    est = get_estimator()

    # Profile fit
    logger.info("Starting fit...")
    fitted_model, duration, profile = profile_fit(est, data)
    logger.info("Fit completed in %.3fs", duration)

    # Build result
    result = build_profiling_result(
        dataset_name=dataset_name,
        data=data,
        fitted_model=fitted_model,
        duration=duration,
        profile=profile,
    )

    # Generate timestamp for filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"profile_{dataset_name}_{ts}"

    # Write report
    json_path = output_dir / f"{base_name}.json"
    write_json_report(result, json_path)

    # Print summary
    typer.echo("")
    typer.echo("=== Profiling Summary ===")
    typer.echo(f"Dataset: {dataset_name}")
    typer.echo(
        f"Shape: {data.n_candidates} x {data.n_items} "
        f"({data.n_categories} categories)"
    )
    typer.echo(f"Fit duration: {duration:.3f}s")
    typer.echo(f"Baseline memory: {format_mib(profile.baseline_mib)}")
    typer.echo(f"Peak memory: {format_mib(profile.peak_mib)}")
    typer.echo(f"Memory delta: {format_mib(result.delta_mib)}")
    typer.echo(
        f"Convergence: {fitted_model.convergence_status} "
        f"({fitted_model.n_iterations} iterations)"
    )
    typer.echo(f"Report: {json_path}")


if __name__ == "__main__":
    app()
