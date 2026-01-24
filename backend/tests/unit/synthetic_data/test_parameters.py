"""
Tests for NRM parameter sampling using Gaussian copula.

The NRM model uses:
- discrimination: slope parameters (a_k) per category
- intercept: intercept parameters (b_k) per category
- correct_discrimination_gap: ensures a_correct > max(a_distractor)
"""

import numpy as np
import pytest

from analysis_service.core.utils import get_rng
from analysis_service.synthetic_data.config import (
    CorrelationsConfig,
    DistributionConfig,
    GenerationConfig,
)
from analysis_service.synthetic_data.parameters import (
    JointParameterSampler,
    build_correlation_matrix,
    create_distribution,
)
from analysis_service.synthetic_data.presets import get_preset


class TestCorrelationMatrix:
    def test_build_correlation_matrix_zeros(self) -> None:
        correlations = CorrelationsConfig(
            discrimination_intercept=0.0,
            discrimination_correct_gap=0.0,
            intercept_correct_gap=0.0,
        )
        R = build_correlation_matrix(correlations)

        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_build_correlation_matrix_nonzero(self) -> None:
        correlations = CorrelationsConfig(
            discrimination_intercept=-0.3,
            discrimination_correct_gap=0.1,
            intercept_correct_gap=-0.2,
        )
        R = build_correlation_matrix(correlations)

        assert R.shape == (3, 3)
        assert R[0, 1] == -0.3
        assert R[0, 2] == 0.1
        assert R[1, 2] == -0.2
        # Check symmetry
        np.testing.assert_array_almost_equal(R, R.T)

    def test_invalid_correlations_raises(self) -> None:
        # Correlations that don't form a valid PSD matrix
        correlations = CorrelationsConfig(
            discrimination_intercept=0.9,
            discrimination_correct_gap=0.9,
            intercept_correct_gap=-0.9,
        )
        with pytest.raises(ValueError, match="positive semi-definite"):
            build_correlation_matrix(correlations)


class TestJointParameterSampler:
    @pytest.fixture
    def baseline_config(self) -> GenerationConfig:
        return get_preset("baseline")

    def test_sample_returns_correct_shapes(
        self, baseline_config: GenerationConfig
    ) -> None:
        sampler = JointParameterSampler(baseline_config)
        rng = get_rng(42)
        n_questions = 50
        n_choices = baseline_config.n_choices
        correct_answers = rng.integers(0, n_choices, size=n_questions)

        result = sampler.sample(
            n_questions=n_questions,
            n_choices=n_choices,
            correct_answer_ix=correct_answers,
            rng=rng,
        )

        assert result.discrimination.shape == (n_questions, n_choices)
        assert result.intercept.shape == (n_questions, n_choices)

    def test_sample_correct_answer_gets_discrimination_boost(
        self, baseline_config: GenerationConfig
    ) -> None:
        """Correct answer discrimination should be boosted by the gap parameter.

        The sampler adds a positive gap to the correct answer's base discrimination.
        This doesn't guarantee a_correct > max(distractor), but should result
        in correct answers tending to have higher discrimination on average.
        """
        sampler = JointParameterSampler(baseline_config)
        rng = get_rng(42)
        n_questions = 500
        n_choices = baseline_config.n_choices
        correct_answers = rng.integers(0, n_choices, size=n_questions)

        result = sampler.sample(
            n_questions=n_questions,
            n_choices=n_choices,
            correct_answer_ix=correct_answers,
            rng=rng,
        )

        # Collect correct and distractor discriminations
        correct_discs = []
        distractor_discs = []
        for i in range(n_questions):
            correct_idx = correct_answers[i]
            correct_discs.append(result.discrimination[i, correct_idx])
            for k in range(n_choices):
                if k != correct_idx:
                    distractor_discs.append(result.discrimination[i, k])

        # Correct answers should have higher mean discrimination
        mean_correct = np.mean(correct_discs)
        mean_distractor = np.mean(distractor_discs)
        assert mean_correct > mean_distractor, (
            f"Mean correct ({mean_correct:.3f}) should exceed "
            f"mean distractor ({mean_distractor:.3f})"
        )

    def test_sample_reproducible(
        self, baseline_config: GenerationConfig
    ) -> None:
        sampler = JointParameterSampler(baseline_config)
        n_questions = 50
        n_choices = baseline_config.n_choices

        rng1 = get_rng(42)
        correct1 = rng1.integers(0, n_choices, size=n_questions)
        result1 = sampler.sample(
            n_questions=n_questions,
            n_choices=n_choices,
            correct_answer_ix=correct1,
            rng=rng1,
        )

        rng2 = get_rng(42)
        correct2 = rng2.integers(0, n_choices, size=n_questions)
        result2 = sampler.sample(
            n_questions=n_questions,
            n_choices=n_choices,
            correct_answer_ix=correct2,
            rng=rng2,
        )

        np.testing.assert_array_equal(
            result1.discrimination, result2.discrimination
        )
        np.testing.assert_array_equal(result1.intercept, result2.intercept)

    def test_different_seeds_produce_different_params(
        self, baseline_config: GenerationConfig
    ) -> None:
        sampler = JointParameterSampler(baseline_config)
        n_questions = 50
        n_choices = baseline_config.n_choices

        rng1 = get_rng(42)
        correct1 = rng1.integers(0, n_choices, size=n_questions)
        result1 = sampler.sample(
            n_questions=n_questions,
            n_choices=n_choices,
            correct_answer_ix=correct1,
            rng=rng1,
        )

        rng2 = get_rng(99)
        correct2 = rng2.integers(0, n_choices, size=n_questions)
        result2 = sampler.sample(
            n_questions=n_questions,
            n_choices=n_choices,
            correct_answer_ix=correct2,
            rng=rng2,
        )

        # Should produce different results
        assert not np.allclose(result1.discrimination, result2.discrimination)
        assert not np.allclose(result1.intercept, result2.intercept)


class TestDistributionConfig:
    def test_create_normal_distribution(self) -> None:
        config = DistributionConfig(
            distribution="normal", params={"mean": 0.0, "std": 0.4}
        )
        dist = create_distribution(config)

        rng = get_rng(42)
        samples = dist.sample(1000, rng)
        assert abs(samples.mean()) < 0.1
        assert abs(samples.std() - 0.4) < 0.05

    def test_create_truncated_normal_distribution(self) -> None:
        config = DistributionConfig(
            distribution="truncated_normal",
            params={"mean": 1.0, "std": 0.3, "lower": 0.3, "upper": None},
        )
        dist = create_distribution(config)

        rng = get_rng(42)
        samples = dist.sample(1000, rng)
        assert samples.min() >= 0.3

    def test_create_uniform_distribution(self) -> None:
        config = DistributionConfig(
            distribution="uniform", params={"low": 2.0, "high": 3.0}
        )
        dist = create_distribution(config)

        rng = get_rng(42)
        samples = dist.sample(1000, rng)
        assert samples.min() >= 2.0
        assert samples.max() <= 3.0
