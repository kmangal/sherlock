import numpy as np
import pytest
from scipy import stats

from analysis_service.core.utils import get_rng
from analysis_service.synthetic_data.sampling import (
    MixtureDistribution,
    SamplerRegistry,
    ScipyDistribution,
    draw_sample,
    registry,
)


class TestScipyDistribution:
    def test_sample_returns_correct_shape(self) -> None:
        dist = ScipyDistribution(stats.norm(loc=0, scale=1))
        rng = get_rng(42)
        samples = dist.sample(100, rng)

        assert samples.shape == (100,)
        assert samples.dtype == np.float64

    def test_sample_reproducible_with_same_seed(self) -> None:
        dist = ScipyDistribution(stats.norm(loc=0, scale=1))

        samples1 = dist.sample(50, get_rng(123))
        samples2 = dist.sample(50, get_rng(123))

        np.testing.assert_array_equal(samples1, samples2)

    def test_cdf_returns_valid_probability(self) -> None:
        dist = ScipyDistribution(stats.norm(loc=0, scale=1))

        assert dist.cdf(0.0) == pytest.approx(0.5)
        assert dist.cdf(-10.0) == pytest.approx(0.0, abs=1e-6)
        assert dist.cdf(10.0) == pytest.approx(1.0, abs=1e-6)

    def test_inverse_cdf_inverts_cdf(self) -> None:
        dist = ScipyDistribution(stats.norm(loc=0, scale=1))
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

        values = dist.inverse_cdf(quantiles)

        for q, v in zip(quantiles, values, strict=True):
            assert dist.cdf(v) == pytest.approx(q, abs=1e-6)


class TestMixtureDistribution:
    def test_weights_must_match_components(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            MixtureDistribution(
                components=[
                    ScipyDistribution(stats.norm(0, 1)),
                    ScipyDistribution(stats.norm(1, 1)),
                ],
                weights=[0.5],
            )

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            MixtureDistribution(
                components=[
                    ScipyDistribution(stats.norm(0, 1)),
                    ScipyDistribution(stats.norm(1, 1)),
                ],
                weights=[0.3, 0.3],
            )

    def test_weights_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MixtureDistribution(
                components=[
                    ScipyDistribution(stats.norm(0, 1)),
                    ScipyDistribution(stats.norm(1, 1)),
                ],
                weights=[-0.5, 1.5],
            )

    def test_sample_returns_correct_shape(self) -> None:
        dist = MixtureDistribution(
            components=[
                ScipyDistribution(stats.norm(-2, 0.5)),
                ScipyDistribution(stats.norm(2, 0.5)),
            ],
            weights=[0.5, 0.5],
        )
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)
        assert samples.dtype == np.float64

    def test_sample_reproducible_with_same_seed(self) -> None:
        dist = MixtureDistribution(
            components=[
                ScipyDistribution(stats.norm(-2, 0.5)),
                ScipyDistribution(stats.norm(2, 0.5)),
            ],
            weights=[0.5, 0.5],
        )

        samples1 = dist.sample(50, get_rng(123))
        samples2 = dist.sample(50, get_rng(123))

        np.testing.assert_array_equal(samples1, samples2)

    def test_cdf_is_weighted_sum_of_component_cdfs(self) -> None:
        comp1 = ScipyDistribution(stats.norm(0, 1))
        comp2 = ScipyDistribution(stats.norm(3, 1))
        mixture = MixtureDistribution(
            components=[comp1, comp2],
            weights=[0.7, 0.3],
        )

        x = 1.0
        expected = 0.7 * comp1.cdf(x) + 0.3 * comp2.cdf(x)

        assert mixture.cdf(x) == pytest.approx(expected)


class TestSamplerRegistry:
    def test_register_and_retrieve(self) -> None:
        test_registry = SamplerRegistry()

        def make_test_dist(*, loc: float = 0.0) -> ScipyDistribution:
            return ScipyDistribution(stats.norm(loc=loc, scale=1))

        test_registry.register("test_dist")(make_test_dist)
        dist = test_registry.get_sampler("test_dist", {"loc": 5.0})

        assert isinstance(dist, ScipyDistribution)
        assert dist.cdf(5.0) == pytest.approx(0.5)

    def test_get_unknown_sampler_raises(self) -> None:
        test_registry = SamplerRegistry()

        with pytest.raises(ValueError, match="not registered"):
            test_registry.get_sampler("unknown", {})


class TestDrawSample:
    def test_returns_correct_shape(self) -> None:
        samples = draw_sample(100, "normal", {}, get_rng(42))

        assert samples.shape == (100,)
        assert samples.dtype == np.float64

    def test_reproducible_with_same_seed(self) -> None:
        samples1 = draw_sample(50, "normal", {}, get_rng(123))
        samples2 = draw_sample(50, "normal", {}, get_rng(123))

        np.testing.assert_array_equal(samples1, samples2)

    def test_respects_distribution_params(self) -> None:
        samples = draw_sample(
            1000,
            "normal",
            {"mean": 100.0, "std": 0.1},
            get_rng(42),
        )

        assert np.mean(samples) == pytest.approx(100.0, abs=0.1)
        assert np.std(samples) == pytest.approx(0.1, abs=0.05)

    def test_unknown_distribution_raises(self) -> None:
        with pytest.raises(ValueError, match="not registered"):
            draw_sample(10, "nonexistent", {}, get_rng(42))


class TestRegisteredDistributions:
    """Smoke tests that all registered distributions work correctly."""

    def test_normal(self) -> None:
        dist = registry.get_sampler("normal", {"mean": 0.0, "std": 1.0})
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)

    def test_skew_normal(self) -> None:
        dist = registry.get_sampler(
            "skew_normal", {"a": 5.0, "loc": 0.0, "scale": 1.0}
        )
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)

    def test_bimodal(self) -> None:
        dist = registry.get_sampler(
            "bimodal",
            {
                "loc1": -1.0,
                "scale1": 0.5,
                "loc2": 1.0,
                "scale2": 0.5,
                "weight1": 0.6,
            },
        )
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)

    def test_uniform(self) -> None:
        dist = registry.get_sampler("uniform", {"low": 0.0, "high": 1.0})
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_truncated_normal(self) -> None:
        dist = registry.get_sampler(
            "truncated_normal",
            {"mean": 0.0, "std": 1.0, "lower": -2.0, "upper": 2.0},
        )
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)
        assert samples.min() >= -2.0
        assert samples.max() <= 2.0

    def test_student_t(self) -> None:
        dist = registry.get_sampler(
            "student_t", {"df": 4.0, "loc": 0.0, "scale": 1.0}
        )
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)

    def test_log_normal(self) -> None:
        dist = registry.get_sampler("log_normal", {"mean": 1.0, "std": 0.5})
        samples = dist.sample(100, get_rng(42))

        assert samples.shape == (100,)
        assert samples.min() > 0  # log-normal is always positive
