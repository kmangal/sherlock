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
            difficulty_discrimination=0.0,
            difficulty_guessing=0.0,
            discrimination_guessing=0.0,
        )
        R = build_correlation_matrix(correlations)

        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_build_correlation_matrix_nonzero(self) -> None:
        correlations = CorrelationsConfig(
            difficulty_discrimination=-0.3,
            difficulty_guessing=0.1,
            discrimination_guessing=-0.2,
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

        # If difficulty is positively correlated with discrimination
        # and it is positively correlated with guessing, then discrimination
        # must also be positively correlated with guessing
        correlations = CorrelationsConfig(
            difficulty_discrimination=0.9,
            difficulty_guessing=0.9,
            discrimination_guessing=-0.9,
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

        result = sampler.sample(n_questions=50, n_distractors=3, rng=rng)

        assert result.difficulty.shape == (50,)
        assert result.discrimination.shape == (50,)
        assert result.guessing.shape == (50,)
        assert result.distractor_quality.shape == (50, 3)

    def test_sample_respects_bounds(
        self, baseline_config: GenerationConfig
    ) -> None:
        sampler = JointParameterSampler(baseline_config)
        rng = get_rng(42)

        result = sampler.sample(n_questions=1000, n_distractors=3, rng=rng)

        discrimination_min = (
            baseline_config.irt_parameters.discrimination.params["lower"]
        )
        guessing_min = baseline_config.irt_parameters.guessing.params["lower"]
        guessing_max = baseline_config.irt_parameters.guessing.params["upper"]

        assert discrimination_min is not None
        assert guessing_min is not None
        assert guessing_max is not None

        assert result.discrimination.min() >= discrimination_min
        assert result.guessing.max() <= guessing_max
        assert result.guessing.min() >= guessing_min

    def test_sample_with_correlations(
        self, baseline_config: GenerationConfig
    ) -> None:
        # Update baseline configuration with correlated distributions
        baseline_config.correlations = CorrelationsConfig(
            difficulty_discrimination=-0.5,
            difficulty_guessing=0.0,
            discrimination_guessing=0.0,
        )

        sampler = JointParameterSampler(baseline_config)
        rng = get_rng(42)

        result = sampler.sample(n_questions=5000, n_distractors=3, rng=rng)

        # Check that difficulty and discrimination are negatively correlated
        corr = np.corrcoef(result.difficulty, result.discrimination)[0, 1]
        assert corr < -0.3  # Should be close to -0.5

    def test_sample_reproducible(
        self, baseline_config: GenerationConfig
    ) -> None:
        sampler = JointParameterSampler(baseline_config)

        result1 = sampler.sample(
            n_questions=50, n_distractors=3, rng=get_rng(42)
        )
        result2 = sampler.sample(
            n_questions=50, n_distractors=3, rng=get_rng(42)
        )

        np.testing.assert_array_equal(result1.difficulty, result2.difficulty)
        np.testing.assert_array_equal(
            result1.discrimination, result2.discrimination
        )
        np.testing.assert_array_equal(result1.guessing, result2.guessing)
        np.testing.assert_array_equal(
            result1.distractor_quality, result2.distractor_quality
        )


class TestDistributionConfig:
    def test_create_normal_distribution(self) -> None:
        config = DistributionConfig(
            distribution="normal", params={"mean": 0.0, "std": 0.4}
        )
        dist = create_distribution(config)

        # Check that we can sample from it
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
