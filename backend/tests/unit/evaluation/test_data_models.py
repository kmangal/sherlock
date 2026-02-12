"""Tests for evaluation data models."""

import pytest

from analysis_service.evaluation.data_models import (
    CheaterConfig,
    CheaterPair,
    CheatingGroundTruth,
    ConfusionMatrix,
    EvaluationRunResult,
)


class TestCheaterConfig:
    def test_valid_config(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=3,
            n_copied_items=10,
        )
        assert config.n_sources == 2
        assert config.n_copiers_per_source == 3
        assert config.n_copied_items == 10

    def test_total_cheaters(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=3,
            n_copied_items=10,
        )
        # 2 sources + 2*3 copiers = 8
        assert config.total_cheaters == 8

    def test_invalid_n_sources(self) -> None:
        with pytest.raises(ValueError, match="n_sources must be >= 1"):
            CheaterConfig(
                n_sources=0, n_copiers_per_source=1, n_copied_items=10
            )

    def test_invalid_n_copiers_per_source(self) -> None:
        with pytest.raises(
            ValueError, match="n_copiers_per_source must be >= 1"
        ):
            CheaterConfig(
                n_sources=1, n_copiers_per_source=0, n_copied_items=10
            )

    def test_invalid_n_copied_items(self) -> None:
        with pytest.raises(ValueError, match="n_copied_items must be >= 1"):
            CheaterConfig(
                n_sources=1, n_copiers_per_source=1, n_copied_items=0
            )


class TestCheatingGroundTruth:
    def test_cheater_indices(self) -> None:
        pairs = (
            CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),
            CheaterPair(source_idx=2, copier_idx=3, n_copied_items=10),
        )
        gt = CheatingGroundTruth(cheater_pairs=pairs)
        assert gt.cheater_indices == frozenset({0, 1, 2, 3})

    def test_source_indices(self) -> None:
        pairs = (
            CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),
            CheaterPair(source_idx=2, copier_idx=3, n_copied_items=10),
        )
        gt = CheatingGroundTruth(cheater_pairs=pairs)
        assert gt.source_indices == frozenset({0, 2})

    def test_copier_indices(self) -> None:
        pairs = (
            CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),
            CheaterPair(source_idx=2, copier_idx=3, n_copied_items=10),
        )
        gt = CheatingGroundTruth(cheater_pairs=pairs)
        assert gt.copier_indices == frozenset({1, 3})

    def test_one_to_many_copying(self) -> None:
        # One source with two copiers
        pairs = (
            CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),
            CheaterPair(source_idx=0, copier_idx=2, n_copied_items=10),
        )
        gt = CheatingGroundTruth(cheater_pairs=pairs)
        assert gt.source_indices == frozenset({0})
        assert gt.copier_indices == frozenset({1, 2})
        assert gt.cheater_indices == frozenset({0, 1, 2})


class TestConfusionMatrix:
    def test_valid_matrix(self) -> None:
        cm = ConfusionMatrix(
            true_positives=10,
            false_positives=2,
            true_negatives=80,
            false_negatives=5,
        )
        assert cm.true_positives == 10
        assert cm.false_positives == 2
        assert cm.true_negatives == 80
        assert cm.false_negatives == 5

    def test_invalid_negative_values(self) -> None:
        with pytest.raises(ValueError, match="true_positives must be >= 0"):
            ConfusionMatrix(
                true_positives=-1,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
            )

        with pytest.raises(ValueError, match="false_positives must be >= 0"):
            ConfusionMatrix(
                true_positives=0,
                false_positives=-1,
                true_negatives=0,
                false_negatives=0,
            )

    def test_perfect_detection(self) -> None:
        cm = ConfusionMatrix(
            true_positives=10,
            false_positives=0,
            true_negatives=90,
            false_negatives=0,
        )
        assert cm.recall == 1.0
        assert cm.precision == 1.0
        assert cm.f1_score == 1.0
        assert cm.power == 1.0
        assert cm.false_positive_rate == 0.0

    def test_no_detection(self) -> None:
        cm = ConfusionMatrix(
            true_positives=0,
            false_positives=0,
            true_negatives=90,
            false_negatives=10,
        )
        assert cm.recall == 0.0
        assert cm.precision == 0.0
        assert cm.f1_score == 0.0
        assert cm.power == 0.0
        assert cm.false_positive_rate == 0.0

    def test_all_false_positives(self) -> None:
        cm = ConfusionMatrix(
            true_positives=0,
            false_positives=10,
            true_negatives=80,
            false_negatives=0,
        )
        assert cm.precision == 0.0
        assert cm.false_positive_rate == 10 / 90

    def test_partial_detection(self) -> None:
        cm = ConfusionMatrix(
            true_positives=5,
            false_positives=2,
            true_negatives=88,
            false_negatives=5,
        )
        assert cm.recall == 0.5
        assert cm.precision == 5 / 7
        expected_f1 = 2 * 0.5 * (5 / 7) / (0.5 + 5 / 7)
        assert abs(cm.f1_score - expected_f1) < 1e-10

    def test_no_positives_in_ground_truth(self) -> None:
        cm = ConfusionMatrix(
            true_positives=0,
            false_positives=5,
            true_negatives=95,
            false_negatives=0,
        )
        assert cm.recall == 0.0
        assert cm.precision == 0.0

    def test_no_negatives_in_ground_truth(self) -> None:
        cm = ConfusionMatrix(
            true_positives=100,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
        )
        assert cm.false_positive_rate == 0.0


class TestEvaluationRunResult:
    def _make_result(
        self,
        detected: tuple[str, ...],
        cheater_pairs: tuple[CheaterPair, ...],
        all_ids: tuple[str, ...] | None = None,
    ) -> EvaluationRunResult:
        if all_ids is None:
            all_ids = tuple(f"C{i}" for i in range(100))
        return EvaluationRunResult(
            run_index=0,
            ground_truth=CheatingGroundTruth(cheater_pairs=cheater_pairs),
            detected_candidate_ids=detected,
            all_candidate_ids=all_ids,
            seed=42,
        )

    def test_confusion_matrix_perfect(self) -> None:
        pairs = (
            CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),
            CheaterPair(source_idx=2, copier_idx=3, n_copied_items=10),
        )
        result = self._make_result(
            detected=("C0", "C1", "C2", "C3"), cheater_pairs=pairs
        )
        cm = result.confusion_matrix()
        assert cm.true_positives == 4
        assert cm.false_positives == 0
        assert cm.false_negatives == 0
        assert cm.true_negatives == 96

    def test_confusion_matrix_no_detection(self) -> None:
        pairs = (CheaterPair(source_idx=0, copier_idx=1, n_copied_items=10),)
        result = self._make_result(detected=(), cheater_pairs=pairs)
        cm = result.confusion_matrix()
        assert cm.true_positives == 0
        assert cm.false_negatives == 2
        assert cm.true_negatives == 98
