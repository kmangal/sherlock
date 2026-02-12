#!/usr/bin/env python
"""
Fit an IRT model to exam response data and save the fitted model.
"""

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

from analysis_service.core.data import load_csv_to_response_matrix
from analysis_service.irt import NRMEstimator
from analysis_service.irt.diagnostics import compute_response_prob_comparison
from analysis_service.irt.estimation import (
    IRTEstimationResult,
    estimate_abilities_eap,
)

BACKEND_DIR = Path(__file__).parent.parent.absolute()
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "fitted-models"

console = Console(force_terminal=True, legacy_windows=True)
app = typer.Typer()


def save_model(model: IRTEstimationResult, output_path: Path) -> None:
    """Save fitted model to json file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(model.model_dump_json(indent=4))


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="Path to CSV file with exam responses (columns: candidate_id, answer_string)",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "-o",
        "--output-dir",
        help="Output directory for fitted model",
    ),
    seed: int | None = typer.Option(
        None,
        "-s",
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Fit an IRT model to exam response data and save as JSON."""

    # Validate input
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        raise typer.Exit(1)
    if input_path.suffix != ".csv":
        console.print("[red]Only .csv files are supported[/red]")
        raise typer.Exit(1)

    # Load data
    console.print("[dim]Loading data...[/dim]")
    try:
        candidate_ids, data = load_csv_to_response_matrix(input_path)
    except ValueError as e:
        console.print(f"[red]Error loading CSV: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(
        Panel(
            f"[bold]Fit IRT Model[/bold]\n\n"
            f"Input: [cyan]{input_path}[/cyan]\n"
            f"Candidates: [cyan]{data.n_candidates}[/cyan]\n"
            f"Items: [cyan]{data.n_items}[/cyan]\n"
            f"Categories: [cyan]{data.n_categories}[/cyan]",
            title="Configuration",
        )
    )

    # Fit model
    console.print("[dim]Fitting IRT model...[/dim]")
    rng = np.random.default_rng(seed) if seed is not None else None
    estimator = NRMEstimator(rng=rng)
    model = estimator.fit(data)

    console.print(
        f"  {model.convergence_status.value} "
        f"({model.n_iterations} iterations, LL={model.log_likelihood:.2f})"
    )

    # Run diagnostics
    console.print("[dim]Running diagnostics...[/dim]")
    abilities = estimate_abilities_eap(data, model)
    prob_comparison = compute_response_prob_comparison(
        data, model, abilities.eap
    )
    mean_diff = float(np.mean(prob_comparison.difference))
    percentile_diffs: dict[int, float] = {}
    for p in [10, 25, 50, 75, 90]:
        percentile_diffs[p] = float(
            np.quantile(np.abs(prob_comparison.difference), q=p / 100)
        )

    max_abs_diff = float(np.max(np.abs(prob_comparison.difference)))

    console.print("Model Diagnostics:")
    console.print(f"  Mean diff = {mean_diff:.4f}")
    for p in [10, 25, 50, 75, 90]:
        console.print(f"  {p}th percentile |diff| = {percentile_diffs[p]:.4f}")
    console.print(f"  Max |diff| = {max_abs_diff:.4f}")

    # Save model
    output_path = output_dir / f"{input_path.stem}.json"
    save_model(model, output_path)

    console.print(
        Panel(
            f"[bold green]Model saved[/bold green]\n\n"
            f"Output: [cyan]{output_path}[/cyan]",
            title="Done",
        )
    )


if __name__ == "__main__":
    app()
