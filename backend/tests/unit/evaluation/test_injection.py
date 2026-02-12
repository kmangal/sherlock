"""Tests for cheater injection."""

import numpy as np
import pytest

from analysis_service.evaluation.data_models import CheaterConfig
from analysis_service.evaluation.injection import inject_cheaters
from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.data_models import GeneratedData
from analysis_service.synthetic_data.generators import generate_exam_responses


def _create_test_data(
    n_candidates: int = 100, n_questions: int = 50, seed: int = 42
) -> tuple[GeneratedData, GenerationConfig]:
    """Create test data for injection tests."""
    config = GenerationConfig(
        n_candidates=n_candidates,
        n_questions=n_questions,
        n_response_categories=4,
        random_seed=seed,
    )
    data = generate_exam_responses(config)
    return data, config


class TestInjectCheaters:
    def test_exact_copy_count(self) -> None:
        """Verify exact number of items are copied."""
        data, _ = _create_test_data()
        n_copied_items = 20

        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=n_copied_items,
        )

        rng = np.random.default_rng(123)
        new_responses, ground_truth = inject_cheaters(
            data.responses, config, rng
        )

        # Check each cheater pair has exactly n_copied_items matching
        for pair in ground_truth.cheater_pairs:
            source_responses = new_responses[pair.source_idx]
            copier_responses = new_responses[pair.copier_idx]
            n_matching = np.sum(source_responses == copier_responses)
            # At least n_copied_items should match (could be more by chance)
            assert n_matching >= n_copied_items

    def test_original_unchanged(self) -> None:
        """Verify original data is not mutated."""
        data, _ = _create_test_data()
        original_responses = data.responses.copy()

        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=20,
        )

        rng = np.random.default_rng(123)
        inject_cheaters(data.responses, config, rng)

        np.testing.assert_array_equal(data.responses, original_responses)

    def test_reproducibility(self) -> None:
        """Verify same seed produces same results."""
        data, _ = _create_test_data()

        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=20,
        )

        rng1 = np.random.default_rng(123)
        new_responses1, gt1 = inject_cheaters(data.responses, config, rng1)

        rng2 = np.random.default_rng(123)
        new_responses2, gt2 = inject_cheaters(data.responses, config, rng2)

        np.testing.assert_array_equal(new_responses1, new_responses2)
        assert gt1.cheater_pairs == gt2.cheater_pairs

    def test_one_to_many_copying(self) -> None:
        """Test that one source can have multiple copiers."""
        data, _ = _create_test_data()

        config = CheaterConfig(
            n_sources=1,
            n_copiers_per_source=3,
            n_copied_items=20,
        )

        rng = np.random.default_rng(123)
        _, ground_truth = inject_cheaters(data.responses, config, rng)

        # Should have 3 pairs, all with same source
        assert len(ground_truth.cheater_pairs) == 3
        source_idx = ground_truth.cheater_pairs[0].source_idx
        for pair in ground_truth.cheater_pairs:
            assert pair.source_idx == source_idx

    def test_validation_too_many_candidates(self) -> None:
        """Error when config requires more candidates than available."""
        data, _ = _create_test_data(n_candidates=10)

        config = CheaterConfig(
            n_sources=5,
            n_copiers_per_source=2,
            n_copied_items=10,
        )
        # Requires 5 + 10 = 15 candidates, but only 10 available

        rng = np.random.default_rng(123)
        with pytest.raises(ValueError, match="candidates"):
            inject_cheaters(data.responses, config, rng)

    def test_validation_too_many_items(self) -> None:
        """Error when n_copied_items exceeds item count."""
        data, _ = _create_test_data(n_questions=20)

        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=1,
            n_copied_items=30,  # More than 20 items
        )

        rng = np.random.default_rng(123)
        with pytest.raises(ValueError, match="n_copied_items"):
            inject_cheaters(data.responses, config, rng)

    def test_ground_truth_structure(self) -> None:
        """Verify ground truth has correct structure."""
        data, _ = _create_test_data()

        config = CheaterConfig(
            n_sources=2,
            n_copiers_per_source=2,
            n_copied_items=20,
        )

        rng = np.random.default_rng(123)
        _, ground_truth = inject_cheaters(data.responses, config, rng)

        # Should have 2*2 = 4 pairs
        assert len(ground_truth.cheater_pairs) == 4

        # All indices should be distinct across sources and copiers
        all_indices = list(ground_truth.cheater_indices)
        assert len(all_indices) == len(set(all_indices))
        assert (
            len(ground_truth.cheater_indices) == 2 + 4
        )  # 2 sources + 4 copiers
