import numpy as np
import pytest

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.synthetic_data.config import (
    DistributionConfig,
    GenerationConfig,
)
from analysis_service.synthetic_data.generators import generate_exam_responses
from analysis_service.synthetic_data.presets import get_preset
from analysis_service.synthetic_data.validation import (
    compute_accuracy_by_ability,
)

LARGE_SAMPLE_SIZE = 10000


class TestParameterRecovery:
    """
    Test that with large enough samples, we can recover input parameters
    within the acceptance criteria (1% epsilon).
    """

    @pytest.fixture
    def baseline_config(self) -> GenerationConfig:
        config = get_preset("baseline")
        config.n_candidates = LARGE_SAMPLE_SIZE
        return config

    @pytest.mark.parametrize("target_rate", [0.05, 0.10, 0.15, 0.20])
    def test_missing_rate_recovery(
        self, baseline_config: GenerationConfig, target_rate: float
    ) -> None:
        """Test that missing rate is recovered within 1%."""
        baseline_config.missing_rate = target_rate

        data = generate_exam_responses(baseline_config)
        actual_rate = data.actual_missing_rate

        assert abs(actual_rate - target_rate) < 0.01

    def test_ability_mean_recovery(
        self, baseline_config: GenerationConfig
    ) -> None:
        """Test that ability mean is recovered within tolerance."""
        target_mean = 0.5
        baseline_config.ability.params["mean"] = target_mean

        data = generate_exam_responses(baseline_config)
        actual_mean = data.abilities.mean()

        # Should be within 1 std error
        assert abs(actual_mean - target_mean) < 0.05

    def test_ability_std_recovery(
        self, baseline_config: GenerationConfig
    ) -> None:
        """Test that ability std is recovered within tolerance."""
        target_std = 1.5
        baseline_config.ability.params["std"] = target_std

        data = generate_exam_responses(baseline_config)
        actual_std = data.abilities.std()

        # Should be within 1 std error
        assert abs(actual_std - target_std) < 0.01

    def test_guessing_parameter_recovery(
        self, baseline_config: GenerationConfig
    ) -> None:
        """Test that guessing parameter affects low-ability performance."""
        # With high guessing, even low ability candidates should get ~guessing% correct
        baseline_config.irt_parameters.guessing = DistributionConfig(
            distribution="truncated_normal",
            params={"mean": 0.25, "std": 0.001, "lower": 0.24, "upper": 0.26},
        )

        # Make questions hard
        baseline_config.irt_parameters.difficulty.params["mean"] = 3.0

        data = generate_exam_responses(baseline_config)

        # Get lowest ability candidates
        low_ability_mask = data.abilities < np.percentile(data.abilities, 10)
        low_ability_responses = data.raw_responses[low_ability_mask]

        # Count correct answers
        correct_answers = np.array([q.correct_answer for q in data.questions])
        correct_count = 0
        total_count = 0
        for candidate_responses in low_ability_responses:
            for j, response in enumerate(candidate_responses):
                if response != MISSING_VALUE:
                    total_count += 1
                    if response == correct_answers[j]:
                        correct_count += 1

        actual_accuracy = correct_count / total_count if total_count > 0 else 0

        # Low ability accuracy should be close to guessing rate
        assert abs(actual_accuracy - 0.25) < 0.05

    def test_monotonicity_ability_performance(
        self, baseline_config: GenerationConfig
    ) -> None:
        """Test that higher ability leads to higher performance."""

        data = generate_exam_responses(baseline_config)
        bin_centers, accuracies = compute_accuracy_by_ability(data, n_bins=10)

        # Remove NaN values
        valid = ~np.isnan(accuracies)
        valid_centers = bin_centers[valid]
        valid_accuracies = accuracies[valid]

        # Check that correlation is strongly positive
        correlation = np.corrcoef(valid_centers, valid_accuracies)[0, 1]
        assert correlation > 0.9
