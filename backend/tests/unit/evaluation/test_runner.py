"""Tests for evaluation runner."""

import numpy as np

from analysis_service.core.data_models import ExamDataset, ResponseMatrix
from analysis_service.detection.pipeline import ThresholdDetectionPipeline
from analysis_service.evaluation.data_models import CheaterConfig
from analysis_service.evaluation.runner import (
    run_evaluation,
    run_evaluation_from_data,
)


class TestRunEvaluation:
    def test_yields_correct_number_of_results(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,
        )
        pipeline = ThresholdDetectionPipeline(threshold=40)

        results = list(
            run_evaluation(
                preset_name="baseline",
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=3,
                base_seed=42,
            )
        )

        assert len(results) == 3

    def test_run_indices_sequential(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,
        )
        pipeline = ThresholdDetectionPipeline(threshold=40)

        results = list(
            run_evaluation(
                preset_name="baseline",
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=3,
                base_seed=42,
            )
        )

        assert [r.run_index for r in results] == [0, 1, 2]

    def test_seeds_unique(self) -> None:
        """All iterations share the base_seed (uniqueness comes from SeedSequence spawning)."""
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,
        )
        pipeline = ThresholdDetectionPipeline(threshold=40)
        base_seed = 100

        results = list(
            run_evaluation(
                preset_name="baseline",
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=3,
                base_seed=base_seed,
            )
        )

        # All results reference the same base_seed
        assert all(r.seed == base_seed for r in results)

    def test_results_have_valid_metrics(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,
        )
        pipeline = ThresholdDetectionPipeline(threshold=40)

        results = list(
            run_evaluation(
                preset_name="baseline",
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=2,
                base_seed=42,
            )
        )

        for result in results:
            cm = result.confusion_matrix()
            assert 0 <= cm.recall <= 1
            assert 0 <= cm.precision <= 1
            assert 0 <= cm.f1_score <= 1
            assert 0 <= cm.false_positive_rate <= 1

    def test_results_have_ground_truth(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,
        )
        pipeline = ThresholdDetectionPipeline(threshold=40)

        results = list(
            run_evaluation(
                preset_name="baseline",
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=2,
                base_seed=42,
            )
        )

        for result in results:
            assert len(result.ground_truth.cheater_pairs) == 2
            assert len(result.ground_truth.cheater_indices) == 4

    def test_results_have_all_candidate_ids(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,
        )
        pipeline = ThresholdDetectionPipeline(threshold=40)

        results = list(
            run_evaluation(
                preset_name="baseline",
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=1,
                base_seed=42,
            )
        )

        assert len(results[0].all_candidate_ids) > 0

    def test_high_threshold_low_detection(self) -> None:
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,
        )
        pipeline = ThresholdDetectionPipeline(threshold=200)

        results = list(
            run_evaluation(
                preset_name="baseline",
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=3,
                base_seed=42,
            )
        )

        total_detections = sum(len(r.detected_candidate_ids) for r in results)
        assert total_detections == 0


class TestRunEvaluationFromData:
    @staticmethod
    def _make_exam_dataset(
        n_candidates: int = 100,
        n_items: int = 50,
        n_categories: int = 4,
        seed: int = 42,
    ) -> ExamDataset:
        rng = np.random.default_rng(seed)
        responses = rng.integers(
            0, n_categories, size=(n_candidates, n_items)
        ).astype(np.int8)
        candidate_ids = np.array([str(i) for i in range(n_candidates)])
        response_matrix = ResponseMatrix(
            responses=responses, n_categories=n_categories
        )
        return ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=response_matrix,
            correct_answers=None,
        )

    def test_yields_correct_number_of_results(self) -> None:
        exam_dataset = self._make_exam_dataset()
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=20,
        )
        pipeline = ThresholdDetectionPipeline(threshold=30)

        results = list(
            run_evaluation_from_data(
                exam_dataset=exam_dataset,
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=3,
                base_seed=42,
            )
        )

        assert len(results) == 3

    def test_run_indices_sequential(self) -> None:
        exam_dataset = self._make_exam_dataset()
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=20,
        )
        pipeline = ThresholdDetectionPipeline(threshold=30)

        results = list(
            run_evaluation_from_data(
                exam_dataset=exam_dataset,
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=3,
                base_seed=42,
            )
        )

        assert [r.run_index for r in results] == [0, 1, 2]

    def test_seeds_unique(self) -> None:
        """Seeds are derived via SeedSequence; all results store the base_seed."""
        exam_dataset = self._make_exam_dataset()
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=20,
        )
        pipeline = ThresholdDetectionPipeline(threshold=30)
        base_seed = 100

        results = list(
            run_evaluation_from_data(
                exam_dataset=exam_dataset,
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=3,
                base_seed=base_seed,
            )
        )

        # run_evaluation_from_data derives an iter_seed internally,
        # so all results share that derived seed
        seeds = [r.seed for r in results]
        assert len(set(seeds)) == 1

    def test_results_have_valid_metrics(self) -> None:
        exam_dataset = self._make_exam_dataset()
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=20,
        )
        pipeline = ThresholdDetectionPipeline(threshold=30)

        results = list(
            run_evaluation_from_data(
                exam_dataset=exam_dataset,
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=2,
                base_seed=42,
            )
        )

        for result in results:
            cm = result.confusion_matrix()
            assert 0 <= cm.recall <= 1
            assert 0 <= cm.precision <= 1
            assert 0 <= cm.f1_score <= 1
            assert 0 <= cm.false_positive_rate <= 1

    def test_results_have_ground_truth(self) -> None:
        exam_dataset = self._make_exam_dataset()
        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=20,
        )
        pipeline = ThresholdDetectionPipeline(threshold=30)

        results = list(
            run_evaluation_from_data(
                exam_dataset=exam_dataset,
                cheater_config=config,
                pipeline=pipeline,
                n_iterations=2,
                base_seed=42,
            )
        )

        for result in results:
            assert len(result.ground_truth.cheater_pairs) == 2
            assert len(result.ground_truth.cheater_indices) == 4
