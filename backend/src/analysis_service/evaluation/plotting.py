"""
Plotting utilities for evaluation results visualization.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure

from analysis_service.evaluation.data_models import (
    ConfusionMatrix,
    EvaluationRunResult,
)
from analysis_service.evaluation.metrics import calculate_confusion_matrix


def _format_percentage(num: float, decimals: int = 4) -> str:
    """This function shows significant zeros while avoiding trailing zeros"""
    if decimals < 1:
        raise ValueError("must specify at least 1 decimal")
    formatted = f"{num:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def calculate_and_plot_confusion_matrix(
    results: Sequence[EvaluationRunResult],
) -> Figure:
    """
    Create heatmap of pooled, row-normalized confusion matrix.

    Pools TP/FP/TN/FN across all runs, then normalizes each row
    by its row sum (true-class normalization).

    Args:
        results: Sequence of EvaluationRunResult from evaluation runs.

    Returns:
        matplotlib Figure with heatmap.
    """
    pooled = calculate_confusion_matrix(results)
    n_runs = len(results)
    return plot_confusion_matrix(pooled, n_runs)


def plot_confusion_matrix(
    matrix: ConfusionMatrix,
    n_runs: int,
) -> Figure:
    import matplotlib.pyplot as plt

    # Row 0: actual positive → [TP, FN]
    # Row 1: actual negative → [FP, TN]
    raw = np.array(
        [
            [matrix.true_positives, matrix.false_negatives],
            [matrix.false_positives, matrix.true_negatives],
        ],
        dtype=np.float64,
    )

    # Normalize each row by row sum
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    normalized = raw / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create color array: green for diagonal, red for off-diagonal
    colors = np.zeros((2, 2, 3))  # RGB colors

    # Green
    green = np.array([21, 214, 73]) / 255

    # Diagonal (good): white → green based on value
    colors[0, 0] = 1 - normalized[0, 0] * (1 - green)  # TP
    colors[1, 1] = 1 - normalized[1, 1] * (1 - green)  # TN

    # Red
    red = np.array([214, 82, 21]) / 255

    incorrect_value_01 = max(normalized[0, 1], 0.1 * (normalized[0, 1] > 0))
    incorrect_value_10 = max(normalized[1, 0], 0.1 * (normalized[1, 0] > 0))

    # Off-diagonal (bad): white → red based on value
    colors[0, 1] = 1 - incorrect_value_01 * (1 - red)  # FN
    colors[1, 0] = 1 - incorrect_value_10 * (1 - red)  # FP

    ax.imshow(colors)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Flagged Candidates", "Not Flagged Candidates"])
    ax.set_yticklabels(["Cheater", "Not Cheater"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{_format_percentage(100 * normalized[i, j])}",
                ha="center",
                va="center",
                color="black",
                fontsize=14,
            )

    ax.set_title(f"Confusion Matrix (n={n_runs} runs, row-normalized)")

    fig.tight_layout()
    return fig
