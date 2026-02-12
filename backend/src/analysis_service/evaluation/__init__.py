"""
Evaluation module for measuring detection pipeline performance.

This module provides tools for:
- Injecting synthetic cheaters into exam data
- Computing evaluation metrics (recall, precision, F1, etc.)
- Running evaluation experiments across multiple iterations
- Visualizing evaluation results
"""

from analysis_service.evaluation.data_models import (
    CheaterConfig,
    CheaterPair,
    CheatingGroundTruth,
    ConfusionMatrix,
    EvaluationRunResult,
    FittedModelContext,
)
from analysis_service.evaluation.injection import (
    add_cheaters_to_preset,
    inject_cheaters,
)
from analysis_service.evaluation.metrics import calculate_confusion_matrix
from analysis_service.evaluation.plotting import (
    calculate_and_plot_confusion_matrix,
)
from analysis_service.evaluation.runner import (
    fit_model_for_evaluation,
    run_evaluation,
    run_evaluation_from_data,
    run_evaluation_from_fitted,
)

__all__ = [
    # Data models
    "CheaterConfig",
    "CheaterPair",
    "CheatingGroundTruth",
    "ConfusionMatrix",
    "EvaluationRunResult",
    "FittedModelContext",
    # Injection
    "inject_cheaters",
    "add_cheaters_to_preset",
    # Metrics
    "calculate_confusion_matrix",
    # Runner
    "fit_model_for_evaluation",
    "run_evaluation",
    "run_evaluation_from_data",
    "run_evaluation_from_fitted",
    # Plotting
    "calculate_and_plot_confusion_matrix",
]
