"""
Metrics computation for evaluation.
"""

from collections.abc import Sequence

from analysis_service.evaluation.data_models import (
    ConfusionMatrix,
    EvaluationRunResult,
)


def calculate_confusion_matrix(
    results: Sequence[EvaluationRunResult],
) -> ConfusionMatrix:
    """Pool confusion matrices across multiple evaluation runs.

    Sums TP/FP/TN/FN across all runs to produce a single pooled
    confusion matrix. This is statistically correct for small sample
    sizes, unlike averaging per-run metric ratios.

    Args:
        results: Sequence of EvaluationRunResult from evaluation runs.

    Returns:
        A single ConfusionMatrix with summed counts.

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("Cannot pool empty results")

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for r in results:
        cm = r.confusion_matrix()
        tp += cm.true_positives
        fp += cm.false_positives
        tn += cm.true_negatives
        fn += cm.false_negatives

    return ConfusionMatrix(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )
