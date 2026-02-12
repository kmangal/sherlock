#!/usr/bin/env python
"""
Evaluate detection accuracy using synthetic data with injected cheaters.

Supports two input modes:
1. Preset - use synthetic data generation config
2. Data file - fit model to real exam data, use posterior abilities for synthetic generation
"""

import json
import logging
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from analysis_service.core.data import load_csv_to_response_matrix
from analysis_service.core.data_models import ExamDataset
from analysis_service.detection.pipeline import (
    AutomaticDetectionPipeline,
    DetectionPipeline,
    ThresholdDetectionPipeline,
)
from analysis_service.evaluation import (
    CheaterConfig,
    EvaluationRunResult,
    calculate_and_plot_confusion_matrix,
    calculate_confusion_matrix,
    fit_model_for_evaluation,
    run_evaluation,
    run_evaluation_from_fitted,
)
from analysis_service.synthetic_data.presets import get_available_presets

# Suppress verbose logging from third-party libs and analysis_service outside progress sections
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("analysis_service").setLevel(logging.WARNING)


MAX_VISIBLE_LOGS = 5


class _LogBuffer:
    """Ring buffer of recent log messages, renderable as dim Rich text."""

    def __init__(self, maxlen: int = MAX_VISIBLE_LOGS) -> None:
        self._messages: deque[str] = deque(maxlen=maxlen)

    def append(self, msg: str) -> None:
        self._messages.append(msg)

    def __rich__(self) -> Text:
        if not self._messages:
            return Text("")
        indented = "\n".join(f"  {m}" for m in self._messages)
        return Text(indented, style="dim")


class _BufferedLogHandler(logging.Handler):
    """Logging handler that appends log messages to a _LogBuffer."""

    def __init__(self, buffer: _LogBuffer) -> None:
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(record.getMessage())


@contextmanager
def _capture_logs() -> Iterator[_LogBuffer]:
    """Temporarily route analysis_service logs to a buffer for Live display."""
    logger = logging.getLogger("analysis_service")
    prev_level = logger.level
    prev_propagate = logger.propagate
    log_buffer = _LogBuffer()
    handler = _BufferedLogHandler(log_buffer)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield log_buffer
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate


BACKEND_DIR = Path(__file__).parent.parent.absolute()
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "reports" / "accuracy"

console = Console(force_terminal=True)
app = typer.Typer()


def load_exam_data_from_csv(path: Path) -> ExamDataset:
    """Load exam data from a CSV file into an ExamDataset.

    Expected CSV columns: candidate_id, answer_string.
    """
    candidate_ids, response_matrix = load_csv_to_response_matrix(path)

    return ExamDataset(
        candidate_ids=np.array(candidate_ids, dtype=np.str_),
        response_matrix=response_matrix,
        correct_answers=None,
    )


def build_pipeline(
    threshold: int | None,
    significance_level: float,
    num_monte_carlo: int,
    num_threshold_samples: int,
) -> DetectionPipeline:
    """Build detection pipeline based on CLI options."""
    if threshold is not None:
        return ThresholdDetectionPipeline(threshold=threshold)
    return AutomaticDetectionPipeline(
        significance_level=significance_level,
        num_monte_carlo=num_monte_carlo,
        num_threshold_samples=num_threshold_samples,
    )


def _run_preset_mode(
    preset_name: str,
    cheater_config: CheaterConfig,
    pipeline: DetectionPipeline,
    n_iterations: int,
    base_seed: int,
) -> Iterator[EvaluationRunResult]:
    """Return an iterator of evaluation results using preset data."""
    return run_evaluation(
        preset_name=preset_name,
        cheater_config=cheater_config,
        pipeline=pipeline,
        n_iterations=n_iterations,
        base_seed=base_seed,
    )


def _run_data_mode(
    data_path: Path,
    cheater_config: CheaterConfig,
    pipeline: DetectionPipeline,
    n_iterations: int,
    base_seed: int,
) -> Iterator[EvaluationRunResult]:
    """Load data, fit model, return an iterator of evaluation results."""
    with console.status("[bold]Loading exam data..."):
        exam_dataset = load_exam_data_from_csv(data_path)

    rm = exam_dataset.response_matrix
    console.print(
        f"[green]✓[/green] Loaded data: "
        f"{rm.n_candidates} candidates × {rm.n_items} items "
        f"({rm.n_categories} categories)"
    )

    root_seq = np.random.SeedSequence(base_seed)
    fit_seq, iter_seq = root_seq.spawn(2)

    fit_seed = int(fit_seq.generate_state(1)[0])
    with _capture_logs() as log_buffer:
        spinner = Spinner("dots", text="[bold]Fitting IRT model...")
        with Live(
            Group(spinner, log_buffer), console=console, refresh_per_second=10
        ):
            context = fit_model_for_evaluation(
                exam_dataset, base_seed=fit_seed
            )

    console.print("[green]✓[/green] Fitted IRT model")

    iter_seed = int(iter_seq.generate_state(1)[0])
    return run_evaluation_from_fitted(
        context=context,
        cheater_config=cheater_config,
        pipeline=pipeline,
        n_iterations=n_iterations,
        base_seed=iter_seed,
    )


def run_with_progress(
    iterator: Iterator[EvaluationRunResult],
    n_iterations: int,
) -> list[EvaluationRunResult]:
    """Consume an iterator of results while displaying a progress spinner.

    Logs from analysis_service are shown dimmed above the spinner,
    scrolling past without obscuring it.
    """
    results: list[EvaluationRunResult] = []
    with _capture_logs() as log_buffer:
        spinner = Spinner(
            "dots", text=f"[bold]Running iteration 1/{n_iterations}..."
        )
        with Live(
            Group(spinner, log_buffer),
            console=console,
            refresh_per_second=10,
        ):
            for result in iterator:
                results.append(result)
                completed = len(results)
                if completed < n_iterations:
                    spinner.update(
                        text=f"[bold]Running iteration {completed + 1}/{n_iterations}..."
                    )

    console.print(f"[green]✓[/green] Completed {n_iterations} iterations")
    return results


def save_results(
    results: list[EvaluationRunResult],
    output_dir: Path,
    params: dict[str, object],
) -> None:
    """Save evaluation results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled = calculate_confusion_matrix(results)

    # Per-run data
    runs_data = []
    for r in results:
        runs_data.append(
            {
                "run_index": r.run_index,
                "seed": r.seed,
                "detected_candidate_ids": list(r.detected_candidate_ids),
                "cheater_indices": sorted(r.ground_truth.cheater_indices),
            }
        )

    # summary.json: all input params + per-run data + pooled metrics
    summary = {
        "params": params,
        "pooled_metrics": {
            "recall": pooled.recall,
            "precision": pooled.precision,
            "f1_score": pooled.f1_score,
            "power": pooled.power,
            "false_positive_rate": pooled.false_positive_rate,
        },
        "pooled_confusion_matrix": {
            "true_positives": pooled.true_positives,
            "false_positives": pooled.false_positives,
            "true_negatives": pooled.true_negatives,
            "false_negatives": pooled.false_negatives,
        },
        "runs": runs_data,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # metrics.json: flat object with key metrics
    metrics = {
        "recall": pooled.recall,
        "precision": pooled.precision,
        "f1_score": pooled.f1_score,
        "power": pooled.power,
        "false_positive_rate": pooled.false_positive_rate,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion_matrix.png
    fig = calculate_and_plot_confusion_matrix(results)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


@app.command()
def main(
    preset: str | None = typer.Option(
        None,
        "-p",
        "--preset",
        help="Preset name for synthetic data generation",
    ),
    data: Path | None = typer.Option(
        None,
        "-d",
        "--data",
        help="Path to CSV with exam data (candidate_id, answer_string columns)",
    ),
    # CheaterConfig options
    n_sources: int = typer.Option(
        5,
        "--n-sources",
        help="Number of source candidates (people being copied from)",
    ),
    n_copiers_per_source: int = typer.Option(
        1,
        "--n-copiers-per-source",
        help="Number of copiers per source",
    ),
    n_copied_items: int = typer.Option(
        20,
        "--n-copied-items",
        help="Number of items copied per pair",
    ),
    # Pipeline options
    threshold: int | None = typer.Option(
        None,
        "--threshold",
        help="Use ThresholdDetectionPipeline with this threshold",
    ),
    significance_level: float = typer.Option(
        0.05,
        "--significance-level",
        help="Significance level for AutomaticDetectionPipeline",
    ),
    num_monte_carlo: int = typer.Option(
        100,
        "--num-monte-carlo",
        help="Number of Monte Carlo samples for p-value computation",
    ),
    num_threshold_samples: int = typer.Option(
        25,
        "--num-threshold-samples",
        help="Number of samples for threshold calibration",
    ),
    # Evaluation options
    n_iterations: int = typer.Option(
        10,
        "-n",
        "--n-iterations",
        help="Number of evaluation runs",
    ),
    seed: int = typer.Option(
        42,
        "-s",
        "--seed",
        help="Base random seed",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "-o",
        "--output-dir",
        help="Output directory for results",
    ),
) -> None:
    """
    Evaluate detection accuracy using synthetic data with injected cheaters.

    Use either --preset or --data (mutually exclusive, one required).
    """
    # Validate input mode
    if preset is None and data is None:
        console.print(
            "[red]Error: Must specify either --preset or --data[/red]"
        )
        raise typer.Exit(1)

    if preset is not None and data is not None:
        console.print(
            "[red]Error: Cannot specify both --preset and --data[/red]"
        )
        raise typer.Exit(1)

    # Validate preset if specified
    if preset is not None:
        available_presets = get_available_presets()
        if preset not in available_presets:
            console.print(
                f"[red]Unknown preset: {preset}[/red]\n"
                f"Available: {', '.join(sorted(available_presets))}"
            )
            raise typer.Exit(1)

    # Validate data file if specified
    if data is not None and not data.exists():
        console.print(f"[red]Data file not found: {data}[/red]")
        raise typer.Exit(1)

    # Build cheater config
    cheater_config = CheaterConfig(
        n_sources=n_sources,
        n_copiers_per_source=n_copiers_per_source,
        n_copied_items=n_copied_items,
    )

    # Build pipeline
    pipeline = build_pipeline(
        threshold=threshold,
        significance_level=significance_level,
        num_monte_carlo=num_monte_carlo,
        num_threshold_samples=num_threshold_samples,
    )

    # Determine input name for output directory
    input_name = preset if preset is not None else data.stem  # type: ignore[union-attr]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / input_name / timestamp

    # Display configuration
    pipeline_type = "Threshold" if threshold is not None else "Automatic"
    console.print(
        Panel(
            f"[bold]Accuracy Evaluation[/bold]\n\n"
            f"Input: [cyan]{preset or data}[/cyan]\n"
            f"Pipeline: [cyan]{pipeline_type}[/cyan]\n"
            f"Cheaters: [cyan]{cheater_config.total_cheaters} "
            f"({n_sources} sources x {n_copiers_per_source} copiers, "
            f"{n_copied_items} items)[/cyan]\n"
            f"Iterations: [cyan]{n_iterations}[/cyan]\n"
            f"Output: [cyan]{report_dir}[/cyan]",
            title="Configuration",
        )
    )

    # Build iterator based on mode
    if preset is not None:
        iterator = _run_preset_mode(
            preset_name=preset,
            cheater_config=cheater_config,
            pipeline=pipeline,
            n_iterations=n_iterations,
            base_seed=seed,
        )
    else:
        iterator = _run_data_mode(
            data_path=data,  # type: ignore[arg-type]
            cheater_config=cheater_config,
            pipeline=pipeline,
            n_iterations=n_iterations,
            base_seed=seed,
        )

    # Run with progress
    results = run_with_progress(iterator, n_iterations)

    # Capture all input params for replicability
    params: dict[str, object] = {
        "input_mode": "preset" if preset is not None else "data",
        "input_name": input_name,
        "n_sources": n_sources,
        "n_copiers_per_source": n_copiers_per_source,
        "n_copied_items": n_copied_items,
        "pipeline_type": pipeline_type,
        "threshold": threshold,
        "significance_level": significance_level,
        "num_monte_carlo": num_monte_carlo,
        "num_threshold_samples": num_threshold_samples,
        "n_iterations": n_iterations,
        "seed": seed,
    }

    # Save results
    console.print("[dim]Saving results...[/dim]")
    save_results(results, report_dir, params)

    # Display summary
    pooled = calculate_confusion_matrix(results)
    console.print()
    console.print(
        Panel(
            f"[bold green]Evaluation Complete[/bold green]\n\n"
            f"Pooled Metrics:\n"
            f"  Recall:    [cyan]{pooled.recall:.3f}[/cyan]\n"
            f"  Precision: [cyan]{pooled.precision:.3f}[/cyan]\n"
            f"  F1 Score:  [cyan]{pooled.f1_score:.3f}[/cyan]\n"
            f"  Power:     [cyan]{pooled.power:.3f}[/cyan]\n"
            f"  FPR:       [cyan]{pooled.false_positive_rate:.3f}[/cyan]\n\n"
            f"Output: [cyan]{report_dir}[/cyan]",
            title="Results",
        )
    )


if __name__ == "__main__":
    app()
