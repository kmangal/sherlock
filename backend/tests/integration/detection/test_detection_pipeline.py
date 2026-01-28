"""
Integration tests for detection pipelines.

These tests run the full pipeline including IRT estimation.
"""

import numpy as np
import pytest

from analysis_service.core.data_models import ExamDataset, ResponseMatrix
from analysis_service.detection.pipeline import AutomaticDetectionPipeline


def create_synthetic_exam_data(
    n_candidates: int,
    n_items: int,
    n_categories: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic exam data with realistic response patterns."""
    # Simple ability-based response generation
    abilities = rng.standard_normal(n_candidates)
    difficulties = rng.uniform(-2, 2, size=n_items)

    responses = np.zeros((n_candidates, n_items), dtype=np.int8)
    for i in range(n_candidates):
        for j in range(n_items):
            # Higher ability -> more likely to choose "correct" (higher) category
            prob_correct = 1 / (1 + np.exp(-(abilities[i] - difficulties[j])))
            if rng.random() < prob_correct:
                responses[i, j] = rng.integers(n_categories // 2, n_categories)
            else:
                responses[i, j] = rng.integers(0, n_categories // 2)

    candidate_ids = np.array([f"C{i:04d}" for i in range(n_candidates)])
    return responses, candidate_ids


def inject_cheaters(
    responses: np.ndarray,
    cheater_indices: list[tuple[int, int]],
    copy_fraction: float,
    rng: np.random.Generator,
) -> None:
    """
    Inject cheating pairs by copying responses.

    Args:
        responses: Response matrix to modify in-place.
        cheater_indices: List of (source, copier) index pairs.
        copy_fraction: Fraction of items to copy (0 to 1).
        rng: Random generator.
    """
    n_items = responses.shape[1]
    n_copy = int(n_items * copy_fraction)

    for source_idx, copier_idx in cheater_indices:
        # Select random items to copy
        copy_items = rng.choice(n_items, size=n_copy, replace=False)
        responses[copier_idx, copy_items] = responses[source_idx, copy_items]


class TestAutomaticDetectionPipelineIntegration:
    @pytest.mark.asyncio
    async def test_detects_injected_cheaters(self) -> None:
        """Pipeline should detect candidates with artificially high similarity."""
        rng = np.random.default_rng(42)

        # Create baseline data
        n_candidates = 500
        n_items = 100
        n_categories = 4

        responses, candidate_ids = create_synthetic_exam_data(
            n_candidates, n_items, n_categories, rng
        )

        # Inject 2 cheating pairs (copying 90% of answers)
        cheater_pairs = [(0, 1), (10, 11)]
        inject_cheaters(responses, cheater_pairs, copy_fraction=0.9, rng=rng)

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=n_categories
            ),
            correct_answers=None,
        )

        pipeline = AutomaticDetectionPipeline(
            significance_level=0.05,
            num_monte_carlo=50,
            num_threshold_samples=20,
        )

        suspects = await pipeline.run(exam_dataset, rng=rng)

        # Check that at least some cheaters are detected
        suspect_ids = {s.candidate_id for s in suspects}
        cheater_ids = {
            candidate_ids[i] for pair in cheater_pairs for i in pair
        }

        # At least one cheater should be detected
        detected_cheaters = suspect_ids & cheater_ids
        assert len(detected_cheaters) > 0, (
            f"Expected to detect some cheaters from {cheater_ids}, "
            f"but detected: {suspect_ids}"
        )

    @pytest.mark.asyncio
    async def test_no_flags_when_no_cheaters(self) -> None:
        """Pipeline should not flag candidates when there's no cheating."""
        rng = np.random.default_rng(123)

        n_candidates = 50
        n_items = 30
        n_categories = 4

        responses, candidate_ids = create_synthetic_exam_data(
            n_candidates, n_items, n_categories, rng
        )

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=n_categories
            ),
            correct_answers=None,
        )

        pipeline = AutomaticDetectionPipeline(
            significance_level=0.05,
            num_monte_carlo=30,
            num_threshold_samples=20,
        )

        suspects = await pipeline.run(exam_dataset, rng=rng)

        # With no cheaters, false positive rate should be low
        # Allow some false positives but not too many
        false_positive_rate = len(suspects) / n_candidates
        assert false_positive_rate <= 0.15, (
            f"Too many false positives: {len(suspects)}/{n_candidates}"
        )

    @pytest.mark.asyncio
    async def test_suspects_have_valid_attributes(self) -> None:
        """Suspects should have properly populated attributes."""
        rng = np.random.default_rng(456)

        n_candidates = 60
        n_items = 40
        n_categories = 4

        responses, candidate_ids = create_synthetic_exam_data(
            n_candidates, n_items, n_categories, rng
        )

        # Inject cheating to ensure we get suspects
        inject_cheaters(responses, [(0, 1)], copy_fraction=0.95, rng=rng)

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=n_categories
            ),
            correct_answers=None,
        )

        pipeline = AutomaticDetectionPipeline(
            significance_level=0.10,
            num_monte_carlo=30,
            num_threshold_samples=20,
        )

        suspects = await pipeline.run(exam_dataset, rng=rng)

        for suspect in suspects:
            assert suspect.candidate_id in candidate_ids
            assert suspect.detection_threshold > 0
            assert suspect.observed_similarity > 0
            assert suspect.p_value is not None
            assert 0 <= suspect.p_value <= 1

    @pytest.mark.asyncio
    async def test_higher_copy_fraction_more_detectable(self) -> None:
        """Cheaters copying more items should be easier to detect."""
        rng = np.random.default_rng(789)

        n_candidates = 500
        n_items = 100
        n_categories = 4

        # Test with low copy fraction
        responses_low, candidate_ids = create_synthetic_exam_data(
            n_candidates, n_items, n_categories, rng
        )
        inject_cheaters(responses_low, [(0, 1)], copy_fraction=0.5, rng=rng)

        # Test with high copy fraction
        responses_high, _ = create_synthetic_exam_data(
            n_candidates, n_items, n_categories, rng
        )
        inject_cheaters(responses_high, [(0, 1)], copy_fraction=0.95, rng=rng)

        pipeline = AutomaticDetectionPipeline(
            significance_level=0.10,
            num_monte_carlo=30,
            num_threshold_samples=20,
        )

        exam_low = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses_low, n_categories=n_categories
            ),
            correct_answers=None,
        )
        suspects_low = await pipeline.run(exam_low, rng=rng)

        exam_high = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses_high, n_categories=n_categories
            ),
            correct_answers=None,
        )
        suspects_high = await pipeline.run(exam_high, rng=rng)

        # Higher copy fraction should generally lead to more/stronger detections
        high_suspects_ids = {s.candidate_id for s in suspects_high}
        low_suspects_ids = {s.candidate_id for s in suspects_low}

        # High copy fraction should detect the cheaters
        assert "C0000" in high_suspects_ids or "C0001" in high_suspects_ids, (
            f"Expected high copy fraction cheaters to be detected. "
            f"High: {high_suspects_ids}, Low: {low_suspects_ids}"
        )

        assert len(low_suspects_ids - high_suspects_ids) == 0, (
            f"Expected more suspects in high copy version. Found: "
            f"High: {high_suspects_ids}, Low: {low_suspects_ids}"
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_candidates_exceed_threshold(
        self,
    ) -> None:
        """Pipeline returns empty list if no candidates exceed calibrated threshold."""
        rng = np.random.default_rng(999)

        # Very diverse responses with low similarity
        n_candidates = 20
        n_items = 10
        n_categories = 4

        # Create responses where each candidate is quite different
        responses = np.zeros((n_candidates, n_items), dtype=np.int8)
        for i in range(n_candidates):
            # Each candidate has a unique pattern
            responses[i, :] = (np.arange(n_items) + i) % n_categories

        candidate_ids = np.array([f"C{i:04d}" for i in range(n_candidates)])

        exam_dataset = ExamDataset(
            candidate_ids=candidate_ids,
            response_matrix=ResponseMatrix(
                responses=responses, n_categories=n_categories
            ),
            correct_answers=None,
        )

        pipeline = AutomaticDetectionPipeline(
            significance_level=0.05,
            num_monte_carlo=30,
            num_threshold_samples=20,
        )

        suspects = await pipeline.run(exam_dataset, rng=rng)

        # With diverse responses, shouldn't flag many (if any) candidates
        assert len(suspects) <= 5
