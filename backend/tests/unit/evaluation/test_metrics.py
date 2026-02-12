"""Tests for metrics computation."""

import pytest

from analysis_service.evaluation.data_models import (
    CheaterPair,
    CheatingGroundTruth,
    EvaluationRunResult,
)
from analysis_service.evaluation.metrics import calculate_confusion_matrix


def _make_result(
    detected: tuple[str, ...],
    cheater_pairs: tuple[CheaterPair, ...],
    all_ids: tuple[str, ...] | None = None,
    run_index: int = 0,
    seed: int = 42,
) -> EvaluationRunResult:
    if all_ids is None:
        all_ids = tuple(f"C{i}" for i in range(100))
    return EvaluationRunResult(
        run_index=run_index,
        ground_truth=CheatingGroundTruth(cheater_pairs=cheater_pairs),
        detected_candidate_ids=detected,
        all_candidate_ids=all_ids,
        seed=seed,
    )


class TestPoolConfusionMatrices:
    def test_single_run(self) -> None:
        pairs = (CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),)
        result = _make_result(detected=("C0", "C1"), cheater_pairs=pairs)
        pooled = calculate_confusion_matrix([result])
        assert pooled.true_positives == 2
        assert pooled.false_positives == 0
        assert pooled.false_negatives == 0
        assert pooled.true_negatives == 98

    def test_multiple_runs_sum(self) -> None:
        pairs = (CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),)
        r1 = _make_result(detected=("C0", "C1"), cheater_pairs=pairs, seed=1)
        r2 = _make_result(detected=("C0",), cheater_pairs=pairs, seed=2)
        r3 = _make_result(detected=(), cheater_pairs=pairs, seed=3)

        pooled = calculate_confusion_matrix([r1, r2, r3])
        # r1: TP=2, FP=0, FN=0, TN=98
        # r2: TP=1, FP=0, FN=1, TN=98
        # r3: TP=0, FP=0, FN=2, TN=98
        assert pooled.true_positives == 3
        assert pooled.false_positives == 0
        assert pooled.false_negatives == 3
        assert pooled.true_negatives == 294
        assert pooled.recall == 0.5

    def test_with_false_positives(self) -> None:
        pairs = (CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),)
        r1 = _make_result(
            detected=("C0", "C1", "C10"), cheater_pairs=pairs, seed=1
        )
        pooled = calculate_confusion_matrix([r1])
        assert pooled.true_positives == 2
        assert pooled.false_positives == 1
        assert pooled.false_negatives == 0
        assert pooled.true_negatives == 97

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot pool empty results"):
            calculate_confusion_matrix([])

    def test_pooled_metrics(self) -> None:
        """Pooled metrics should be computed from summed counts, not averaged ratios."""
        pairs = (CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),)
        all_ids = tuple(f"C{i}" for i in range(10))
        # Run 1: detect both cheaters + 1 FP
        r1 = _make_result(
            detected=("C0", "C1", "C5"),
            cheater_pairs=pairs,
            all_ids=all_ids,
            seed=1,
        )
        # Run 2: detect neither cheater
        r2 = _make_result(
            detected=(),
            cheater_pairs=pairs,
            all_ids=all_ids,
            seed=2,
        )

        pooled = calculate_confusion_matrix([r1, r2])
        # Summed: TP=2, FP=1, FN=2, TN=15
        assert pooled.true_positives == 2
        assert pooled.false_positives == 1
        assert pooled.false_negatives == 2
        assert pooled.true_negatives == 15
        assert pooled.recall == 0.5
        assert pooled.precision == 2 / 3
