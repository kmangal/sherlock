#!/usr/bin/env python
"""
Model diagnostics for IRT model fitting validation.

Generates diagnostic plots and CSVs to validate that the IRT model
is fitting correctly and recovering true parameters.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.figure import Figure
from numpy.typing import NDArray
from rich.console import Console
from rich.panel import Panel
from statsmodels.nonparametric.smoothers_lowess import (  # type: ignore[import-untyped]
    lowess,
)

from analysis_service.core.constants import MISSING_CHAR, MISSING_VALUE
from analysis_service.core.data_models import ResponseMatrix
from analysis_service.core.utils import get_rng
from analysis_service.irt.estimation.abilities import (
    estimate_abilities_eap,
)
from analysis_service.irt.estimation.estimator import (
    IRTEstimationResult,
    NRMEstimator,
)
from analysis_service.irt.estimation.parameters import NRMItemParameters
from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.data_models import GeneratedData
from analysis_service.synthetic_data.generators import generate_exam_responses
from analysis_service.synthetic_data.presets import (
    get_available_presets,
    get_preset,
)
from analysis_service.synthetic_data.sampling import draw_sample
from analysis_service.synthetic_data.utils import (
    index_to_letter,
    letter_to_index,
)

# Suppress verbose logging from matplotlib and PIL
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

BACKEND_DIR = Path(__file__).parent.parent.absolute()
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "reports" / "model_diagnostics"

# Use ASCII-only console to avoid encoding issues on Windows
console = Console(force_terminal=True, legacy_windows=True)
app = typer.Typer()


# =============================================================================
# Data classes for diagnostic outputs
# =============================================================================


@dataclass
class ResponseProbComparison:
    """Comparison of empirical vs model response probabilities."""

    item_id: NDArray[np.int64]
    category: list[str]  # Response letters (A, B, C, ...) or MISSING_CHAR
    empirical_prob: NDArray[np.float64]
    model_prob: NDArray[np.float64]
    difference: NDArray[np.float64]


@dataclass
class ParameterRecovery:
    """Comparison of true vs fitted item parameters."""

    item_id: NDArray[np.int64]
    true_discrimination: NDArray[np.float64]
    fitted_discrimination: NDArray[np.float64]
    true_intercept: NDArray[np.float64]
    fitted_intercept: NDArray[np.float64]


# =============================================================================
# Data loading utilities
# =============================================================================


def parse_answer_string(answer_string: str) -> NDArray[np.int8]:
    """Parse an answer string into response indices."""
    responses = []
    for char in answer_string:
        if char == "*":
            responses.append(MISSING_VALUE)
        else:
            responses.append(letter_to_index(char))
    return np.array(responses, dtype=np.int8)


def generated_data_to_response_matrix(data: GeneratedData) -> ResponseMatrix:
    """Convert GeneratedData to ResponseMatrix."""
    response_arrays = [parse_answer_string(s) for s in data.answer_strings]
    responses = np.stack(response_arrays)
    n_categories = data.config.n_response_categories
    return ResponseMatrix(responses=responses, n_categories=n_categories)


def get_correct_answers(
    item_params: list[NRMItemParameters],
) -> list[int | None]:
    """Extract correct answers from item parameters."""
    return [p.correct_answer for p in item_params]


# =============================================================================
# Diagnostic computations
# =============================================================================


def compute_fraction_correct(
    responses: NDArray[np.int8],
    correct_answers: list[int | None],
) -> NDArray[np.float64]:
    """Compute fraction correct for each candidate."""
    n_candidates = responses.shape[0]
    n_items = responses.shape[1]

    correct_count = np.zeros(n_candidates, dtype=np.float64)
    valid_count = np.zeros(n_candidates, dtype=np.float64)

    for item_idx in range(n_items):
        correct = correct_answers[item_idx]
        if correct is None:
            continue

        item_responses = responses[:, item_idx]
        valid_mask = item_responses != MISSING_VALUE
        correct_mask = item_responses == correct

        correct_count += correct_mask.astype(np.float64)
        valid_count += valid_mask.astype(np.float64)

    # Avoid division by zero
    valid_count_safe: NDArray[np.float64] = np.maximum(valid_count, 1.0)
    result: NDArray[np.float64] = correct_count / valid_count_safe
    return result


def compute_fraction_missing(
    responses: NDArray[np.int8],
) -> NDArray[np.float64]:
    """Compute fraction missing for each candidate."""
    n_items = responses.shape[1]
    missing_count = np.sum(responses == MISSING_VALUE, axis=1)
    result: NDArray[np.float64] = missing_count.astype(np.float64) / n_items
    return result


def compute_lowess_fit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    frac: float = 0.3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute LOWESS smoothed curve."""
    # Sort by x for proper plotting
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Compute LOWESS
    smoothed = lowess(y_sorted, x_sorted, frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def compute_item_fraction_correct(
    responses: NDArray[np.int8],
    correct_answers: list[int | None],
) -> NDArray[np.float64]:
    """Compute fraction correct per item."""
    n_items = responses.shape[1]
    fractions = np.zeros(n_items, dtype=np.float64)

    for item_idx in range(n_items):
        correct = correct_answers[item_idx]
        if correct is None:
            fractions[item_idx] = np.nan
            continue

        item_responses = responses[:, item_idx]
        valid_mask = item_responses != MISSING_VALUE
        n_valid = valid_mask.sum()

        if n_valid == 0:
            fractions[item_idx] = np.nan
        else:
            n_correct = (item_responses == correct).sum()
            fractions[item_idx] = n_correct / n_valid

    return fractions


def compute_mean_intercept(item_params: NRMItemParameters) -> float:
    """Compute mean intercept across response categories (excluding missing)."""
    n_response_categories = item_params.n_response_categories
    intercepts = item_params.intercepts[:n_response_categories]
    return float(np.mean(intercepts))


def compute_response_prob_comparison(
    data: ResponseMatrix,
    model: IRTEstimationResult,
    abilities: NDArray[np.float64],
) -> ResponseProbComparison:
    """Compare empirical vs model response probabilities.

    Includes all response categories (A, B, C, ...) plus missing (*).
    Categories are represented as letters for readability.
    """
    n_categories = data.n_categories
    n_candidates = data.n_candidates

    item_ids: list[int] = []
    categories: list[str] = []
    empirical_probs: list[float] = []
    model_probs: list[float] = []

    for item_idx, item_params in enumerate(model.item_parameters):
        item_responses = data.responses[:, item_idx]

        # Count each response category (0 to n_categories-1) -> letters A, B, C, ...
        for cat in range(n_categories):
            cat_letter = index_to_letter(cat)
            empirical_count = (item_responses == cat).sum()
            empirical_prob = empirical_count / n_candidates

            # Model: average P(cat|theta) across all candidates
            probs = item_params.compute_probabilities(abilities)
            model_prob = float(np.mean(probs[:, cat]))

            item_ids.append(item_idx)
            categories.append(cat_letter)
            empirical_probs.append(empirical_prob)
            model_probs.append(model_prob)

        # Missing category (index n_categories in model, represented as MISSING_CHAR)
        missing_count = (item_responses == MISSING_VALUE).sum()
        empirical_prob_missing = missing_count / n_candidates

        # Model probability for missing (last category in NRM)
        probs = item_params.compute_probabilities(abilities)
        model_prob_missing = float(np.mean(probs[:, n_categories]))

        item_ids.append(item_idx)
        categories.append(MISSING_CHAR)
        empirical_probs.append(empirical_prob_missing)
        model_probs.append(model_prob_missing)

    empirical_arr = np.array(empirical_probs, dtype=np.float64)
    model_arr = np.array(model_probs, dtype=np.float64)

    return ResponseProbComparison(
        item_id=np.array(item_ids, dtype=np.int64),
        category=categories,
        empirical_prob=empirical_arr,
        model_prob=model_arr,
        difference=empirical_arr - model_arr,
    )


def compute_parameter_recovery(
    true_params: list[NRMItemParameters],
    fitted_params: tuple[NRMItemParameters, ...],
) -> ParameterRecovery:
    """Compare true vs fitted parameters."""
    n_items = len(true_params)

    item_ids = np.arange(n_items, dtype=np.int64)
    true_discrim = np.zeros(n_items, dtype=np.float64)
    fitted_discrim = np.zeros(n_items, dtype=np.float64)
    true_intercept = np.zeros(n_items, dtype=np.float64)
    fitted_intercept = np.zeros(n_items, dtype=np.float64)

    for i in range(n_items):
        true_p = true_params[i]
        fitted_p = fitted_params[i]

        # Mean discrimination across response categories (excluding missing)
        n_resp = true_p.n_response_categories
        true_discrim[i] = float(np.mean(true_p.discriminations[:n_resp]))
        fitted_discrim[i] = float(np.mean(fitted_p.discriminations[:n_resp]))

        # Mean intercept across response categories (excluding missing)
        true_intercept[i] = float(np.mean(true_p.intercepts[:n_resp]))
        fitted_intercept[i] = float(np.mean(fitted_p.intercepts[:n_resp]))

    return ParameterRecovery(
        item_id=item_ids,
        true_discrimination=true_discrim,
        fitted_discrimination=fitted_discrim,
        true_intercept=true_intercept,
        fitted_intercept=fitted_intercept,
    )


# =============================================================================
# Plotting utilities
# =============================================================================


def plot_scatter_with_lowess(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xlabel: str,
    ylabel: str,
    title: str,
) -> Figure:
    """Create scatter plot with LOWESS smoothing curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter points
    ax.scatter(x, y, alpha=0.4, s=20, color="steelblue")

    # LOWESS curve
    lowess_x, lowess_y = compute_lowess_fit(x, y)
    ax.plot(lowess_x, lowess_y, color="darkred", linewidth=2, label="LOWESS")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_histogram_comparison(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    label1: str,
    label2: str,
    xlabel: str,
    title: str,
) -> Figure:
    """Create overlaid histogram comparing two distributions."""
    fig, ax = plt.subplots(figsize=(8, 6))

    bins = list(
        np.linspace(
            min(float(data1.min()), float(data2.min())),
            max(float(data1.max()), float(data2.max())),
            50,
        )
    )

    ax.hist(data1, bins=bins, alpha=0.5, label=label1, density=True)
    ax.hist(data2, bins=bins, alpha=0.5, label=label2, density=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_difference_histogram(
    diffs: NDArray[np.float64],
    xlabel: str,
    title: str,
) -> Figure:
    """Create histogram of differences."""
    fig, ax = plt.subplots(figsize=(8, 6))

    mean_diff = float(np.mean(diffs))
    ax.hist(diffs, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
    ax.axvline(
        mean_diff,
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_diff:.4f}",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_param_recovery_scatter(
    true_vals: NDArray[np.float64],
    fitted_vals: NDArray[np.float64],
    param_name: str,
) -> Figure:
    """Create scatter plot for parameter recovery with identity line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter points
    ax.scatter(true_vals, fitted_vals, alpha=0.6, s=30, color="steelblue")

    # Identity line
    min_val = min(true_vals.min(), fitted_vals.min())
    max_val = max(true_vals.max(), fitted_vals.max())
    margin = (max_val - min_val) * 0.1
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, "r--", linewidth=2, label="Identity")

    # Compute correlation
    corr = np.corrcoef(true_vals, fitted_vals)[0, 1]

    ax.set_xlabel(f"True {param_name}")
    ax.set_ylabel(f"Fitted {param_name}")
    ax.set_title(f"Parameter Recovery: {param_name} (r = {corr:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# Output utilities
# =============================================================================


def save_response_prob_csv(
    comparison: ResponseProbComparison,
    path: Path,
) -> None:
    """Save response probability comparison to CSV."""
    df = pd.DataFrame(
        {
            "item_id": comparison.item_id,
            "category": comparison.category,
            "empirical_prob": comparison.empirical_prob,
            "model_prob": comparison.model_prob,
            "difference": comparison.difference,
        }
    )
    df.to_csv(path, index=False)


def save_param_recovery_csv(
    recovery: ParameterRecovery,
    path: Path,
) -> None:
    """Save parameter recovery comparison to CSV."""
    df = pd.DataFrame(
        {
            "item_id": recovery.item_id,
            "true_discrimination": recovery.true_discrimination,
            "fitted_discrimination": recovery.fitted_discrimination,
            "true_intercept": recovery.true_intercept,
            "fitted_intercept": recovery.fitted_intercept,
        }
    )
    df.to_csv(path, index=False)


def build_summary(
    preset: str,
    config: GenerationConfig,
    model: IRTEstimationResult,
    include_correct_answers: bool,
    prob_comparison: ResponseProbComparison,
    param_recovery: ParameterRecovery,
) -> dict[str, Any]:
    """Build summary dictionary for JSON output."""
    # Compute correlations for param recovery
    discrim_corr = float(
        np.corrcoef(
            param_recovery.true_discrimination,
            param_recovery.fitted_discrimination,
        )[0, 1]
    )
    intercept_corr = float(
        np.corrcoef(
            param_recovery.true_intercept,
            param_recovery.fitted_intercept,
        )[0, 1]
    )

    return {
        "metadata": {
            "preset": preset,
            "timestamp": datetime.now().isoformat(),
            "include_correct_answers": include_correct_answers,
        },
        "data_shape": {
            "n_candidates": config.n_candidates,
            "n_items": config.n_questions,
            "n_categories": config.n_response_categories,
        },
        "model_fit": {
            "n_iterations": model.n_iterations,
            "convergence_status": model.convergence_status.value,
            "log_likelihood": model.log_likelihood,
        },
        "response_prob_comparison": {
            "mean_difference": float(np.mean(prob_comparison.difference)),
            "std_difference": float(np.std(prob_comparison.difference)),
            "max_abs_difference": float(
                np.max(np.abs(prob_comparison.difference))
            ),
        },
        "parameter_recovery": {
            "discrimination_correlation": discrim_corr,
            "intercept_correlation": intercept_corr,
            "discrimination_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            param_recovery.true_discrimination
                            - param_recovery.fitted_discrimination
                        )
                        ** 2
                    )
                )
            ),
            "intercept_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            param_recovery.true_intercept
                            - param_recovery.fitted_intercept
                        )
                        ** 2
                    )
                )
            ),
        },
    }


# =============================================================================
# Main CLI
# =============================================================================


@app.command()
def main(
    preset: str = typer.Argument(
        ...,
        help="Preset name for synthetic data generation",
    ),
    include_correct_answers: bool = typer.Option(
        False,
        "-c",
        "--include-correct-answers",
        help="Include correct answers in model fitting (enables additional diagnostics)",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "-o",
        "--output-dir",
        help="Output directory for diagnostic reports",
    ),
    seed: int | None = typer.Option(
        None,
        "-s",
        "--seed",
        help="Random seed (overrides preset seed)",
    ),
) -> None:
    """
    Run model diagnostics for IRT model fitting validation.

    Generates diagnostic plots and CSVs comparing:
    - Empirical vs model response probabilities
    - True vs fitted item parameters
    - Ability estimates (posterior vs prior)
    - Theta vs fraction correct/missing (when correct answers provided)
    """
    # Validate preset
    available_presets = get_available_presets()
    if preset not in available_presets:
        console.print(
            f"[red]Unknown preset: {preset}[/red]\n"
            f"Available: {', '.join(sorted(available_presets))}"
        )
        raise typer.Exit(1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / preset / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Model Diagnostics[/bold]\n\n"
            f"Preset: [cyan]{preset}[/cyan]\n"
            f"Include correct answers: [cyan]{include_correct_answers}[/cyan]\n"
            f"Output: [cyan]{report_dir}[/cyan]",
            title="Configuration",
        )
    )

    # Step 1: Load preset and generate data
    console.print("[dim]Loading preset and generating data...[/dim]")
    config = get_preset(preset)
    if seed is not None:
        # Override seed if provided
        config = GenerationConfig(
            n_candidates=config.n_candidates,
            n_questions=config.n_questions,
            n_response_categories=config.n_response_categories,
            missing=config.missing,
            random_seed=seed,
            ability=config.ability,
            nrm_parameters=config.nrm_parameters,
            correlations=config.correlations,
        )
    generated = generate_exam_responses(config)
    data = generated_data_to_response_matrix(generated)

    console.print(
        f"  Data: {data.n_candidates} candidates x {data.n_items} items "
        f"({data.n_categories} categories)"
    )

    # Step 2: Fit model
    console.print("[dim]Fitting IRT model...[/dim]")
    correct_answers: list[int | None] | None = None
    if include_correct_answers:
        correct_answers = get_correct_answers(generated.item_params)

    estimator = NRMEstimator()
    model = estimator.fit(data, correct_answers)

    console.print(
        f"  Model: {model.convergence_status.value} "
        f"({model.n_iterations} iterations, LL={model.log_likelihood:.2f})"
    )

    # Step 3: Sample abilities
    console.print("[dim]Sampling abilities from posterior...[/dim]")
    rng = get_rng(config.random_seed)
    estimated_abilities = estimate_abilities_eap(data, model)

    # Step 4: Compute diagnostics
    console.print("[dim]Computing diagnostics...[/dim]")

    # Response probability comparison
    prob_comparison = compute_response_prob_comparison(
        data, model, estimated_abilities.eap
    )

    # Parameter recovery
    param_recovery = compute_parameter_recovery(
        generated.item_params, model.item_parameters
    )

    # Step 5: Generate outputs
    console.print("[dim]Generating outputs...[/dim]")

    # -- Response probability comparison CSV --
    save_response_prob_csv(
        prob_comparison, report_dir / "response_prob_comparison.csv"
    )

    # -- Response probability histogram --
    fig = plot_difference_histogram(
        prob_comparison.difference,
        "Empirical - Model Probability",
        "Response Probability Differences",
    )
    fig.savefig(report_dir / "response_prob_histogram.png", dpi=150)
    plt.close(fig)

    # -- Ability posterior vs prior --
    prior_abilities = draw_sample(
        n=data.n_candidates,
        distribution_name=config.ability.distribution,
        distribution_params=config.ability.params,
        rng=rng,
    )
    fig = plot_histogram_comparison(
        estimated_abilities.eap,
        prior_abilities,
        "Posterior",
        "Prior",
        "Ability (theta)",
        "Ability Distribution: Posterior vs Prior",
    )
    fig.savefig(report_dir / "ability_posterior_vs_prior.png", dpi=150)
    plt.close(fig)

    # -- Parameter recovery plots and CSV --
    save_param_recovery_csv(param_recovery, report_dir / "param_recovery.csv")

    fig = plot_param_recovery_scatter(
        param_recovery.true_discrimination,
        param_recovery.fitted_discrimination,
        "Discrimination",
    )
    fig.savefig(report_dir / "param_recovery_discrimination.png", dpi=150)
    plt.close(fig)

    fig = plot_param_recovery_scatter(
        param_recovery.true_intercept,
        param_recovery.fitted_intercept,
        "Intercept",
    )
    fig.savefig(report_dir / "param_recovery_intercept.png", dpi=150)
    plt.close(fig)

    # -- Diagnostics requiring correct answers --
    if include_correct_answers:
        correct_answers_list = get_correct_answers(generated.item_params)

        # Theta vs fraction correct
        fraction_correct = compute_fraction_correct(
            data.responses, correct_answers_list
        )
        fig = plot_scatter_with_lowess(
            estimated_abilities.eap,
            fraction_correct,
            "Estimated Ability (theta)",
            "Fraction Correct",
            "Ability vs Fraction Correct",
        )
        fig.savefig(report_dir / "theta_vs_fraction_correct.png", dpi=150)
        plt.close(fig)

        # Theta vs fraction missing
        fraction_missing = compute_fraction_missing(data.responses)
        fig = plot_scatter_with_lowess(
            estimated_abilities.eap,
            fraction_missing,
            "Estimated Ability (theta)",
            "Fraction Missing",
            "Ability vs Fraction Missing",
        )
        fig.savefig(report_dir / "theta_vs_fraction_missing.png", dpi=150)
        plt.close(fig)

        # Item intercept vs fraction correct
        item_fraction_correct = compute_item_fraction_correct(
            data.responses, correct_answers_list
        )
        item_mean_intercepts = np.array(
            [compute_mean_intercept(p) for p in model.item_parameters]
        )

        # Filter out NaN items (no correct answer known)
        valid_mask = ~np.isnan(item_fraction_correct)
        if valid_mask.sum() > 0:
            fig = plot_scatter_with_lowess(
                item_mean_intercepts[valid_mask],
                item_fraction_correct[valid_mask],
                "Mean Item Intercept",
                "Fraction Correct",
                "Item Intercept vs Fraction Correct",
            )
            fig.savefig(
                report_dir / "item_intercept_vs_fraction_correct.png", dpi=150
            )
            plt.close(fig)

    # -- Summary JSON --
    summary = build_summary(
        preset=preset,
        config=config,
        model=model,
        include_correct_answers=include_correct_answers,
        prob_comparison=prob_comparison,
        param_recovery=param_recovery,
    )
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    console.print()
    console.print(
        Panel(
            f"[bold green]Diagnostics Complete[/bold green]\n\n"
            f"Parameter Recovery:\n"
            f"  Discrimination r = [cyan]{summary['parameter_recovery']['discrimination_correlation']:.3f}[/cyan]\n"
            f"  Intercept r = [cyan]{summary['parameter_recovery']['intercept_correlation']:.3f}[/cyan]\n\n"
            f"Response Probability:\n"
            f"  Mean diff = [cyan]{summary['response_prob_comparison']['mean_difference']:.4f}[/cyan]\n"
            f"  Max |diff| = [cyan]{summary['response_prob_comparison']['max_abs_difference']:.4f}[/cyan]\n\n"
            f"Output: [cyan]{report_dir}[/cyan]",
            title="Results",
        )
    )


if __name__ == "__main__":
    app()
