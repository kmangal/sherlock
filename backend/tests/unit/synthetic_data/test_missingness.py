"""Tests for missingness models."""

import numpy as np
import numpy.typing as npt
import pytest

from analysis_service.core.utils import get_rng
from analysis_service.synthetic_data.data_models import SampledParameters
from analysis_service.synthetic_data.missingness import (
    AbilityDependentMissingness,
    MCARMissingness,
    NoMissingness,
    PositionDependentMissingness,
    get_missingness_model,
    missingness_registry,
)


def _create_test_params(
    n_questions: int, n_categories: int
) -> SampledParameters:
    """Create test parameters with realistic values."""
    rng = get_rng(42)
    discrimination = rng.uniform(0.5, 1.5, size=(n_questions, n_categories))
    intercept = rng.uniform(-1.0, 1.0, size=(n_questions, n_categories))
    return SampledParameters(
        discrimination=discrimination.astype(np.float64),
        intercept=intercept.astype(np.float64),
        includes_missing_values=False,
    )


def _compute_missing_probability(
    discrimination: npt.NDArray[np.float64],
    intercept: npt.NDArray[np.float64],
    theta: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute probability of missing category at given theta(s).

    Args:
        discrimination: Shape (n_questions, n_categories).
        intercept: Shape (n_questions, n_categories).
        theta: Single value or array of abilities.

    Returns:
        If theta is scalar: shape (n_questions,) - missing prob for each question.
        If theta is array: shape (n_questions,) - mean missing prob across thetas.
    """
    theta = np.atleast_1d(theta)
    n_questions = discrimination.shape[0]
    probs = np.zeros(n_questions, dtype=np.float64)

    for q in range(n_questions):
        # Scores for this question across all thetas
        # Shape: (n_theta, n_categories)
        scores = np.outer(theta, discrimination[q, :]) + intercept[q, :]
        # Stable softmax
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        question_probs = exp_scores[:, -1] / np.sum(exp_scores, axis=1)
        probs[q] = np.mean(question_probs)

    return probs


class TestMissingnessRegistry:
    def test_registry_has_expected_models(self) -> None:
        """All expected models should be registered."""
        assert "none" in missingness_registry._models
        assert "mcar" in missingness_registry._models
        assert "ability_dependent" in missingness_registry._models
        assert "position_dependent" in missingness_registry._models

    def test_get_missingness_model_mcar(self) -> None:
        """get_missingness_model should return MCAR with default params."""
        model = get_missingness_model("mcar")
        assert isinstance(model, MCARMissingness)
        assert model.rate == 0.05

    def test_get_missingness_model_with_params(self) -> None:
        """get_missingness_model should accept custom params."""
        model = get_missingness_model("mcar", {"rate": 0.2})
        assert isinstance(model, MCARMissingness)
        assert model.rate == 0.2

    def test_get_missingness_model_unknown(self) -> None:
        """get_missingness_model should raise for unknown model."""
        with pytest.raises(ValueError, match="Unknown missingness model"):
            get_missingness_model("unknown_model")


class TestNoMissingness:
    def test_appends_missing_category(self) -> None:
        """NoMissingness should append a category with ~0 probability."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0, 1.0, -1.0])

        model = NoMissingness()
        result = model.generate_missing_params(params, theta, rng)

        # Should have 5 categories now (4 + missing)
        assert result.discrimination.shape == (10, 5)
        assert result.intercept.shape == (10, 5)
        assert result.includes_missing_values

    def test_missing_probability_near_zero(self) -> None:
        """Missing probability should be effectively zero."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0])

        model = NoMissingness()
        result = model.generate_missing_params(params, theta, rng)

        # Compute missing probability at theta=0
        probs = _compute_missing_probability(
            result.discrimination, result.intercept, 0.0
        )
        assert np.all(probs < 1e-10)


class TestMCARMissingness:
    def test_validation_rejects_invalid_rate(self) -> None:
        """Should reject rates outside [0, 1)."""
        with pytest.raises(ValueError, match="rate must be in"):
            MCARMissingness(rate=-0.1)
        with pytest.raises(ValueError, match="rate must be in"):
            MCARMissingness(rate=1.0)

    def test_appends_missing_category(self) -> None:
        """MCAR should append a missing category."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0, 1.0, -1.0])

        model = MCARMissingness(rate=0.1)
        result = model.generate_missing_params(params, theta, rng)

        assert result.discrimination.shape == (10, 5)
        assert result.includes_missing_values

    def test_missing_discrimination_is_zero(self) -> None:
        """MCAR missing category should have discrimination=0."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0])

        model = MCARMissingness(rate=0.1)
        result = model.generate_missing_params(params, theta, rng)

        # Missing category discrimination should be 0
        assert np.allclose(result.discrimination[:, -1], 0.0)

    def test_expected_missing_rate_matches_target(self) -> None:
        """Expected missing rate across abilities should match target."""
        n_questions = 20
        params = _create_test_params(n_questions=n_questions, n_categories=4)
        rng = get_rng(42)
        theta = np.linspace(-2, 2, 100).astype(np.float64)

        rate = 0.15
        model = MCARMissingness(rate=rate)
        result = model.generate_missing_params(params, theta, rng)

        # Expected rate across all abilities should match target for each question
        expected_rates = _compute_missing_probability(
            result.discrimination, result.intercept, theta
        )
        assert np.allclose(expected_rates, rate, atol=0.02)


class TestAbilityDependentMissingness:
    def test_validation(self) -> None:
        """Should validate parameters."""
        with pytest.raises(ValueError, match="base_rate"):
            AbilityDependentMissingness(base_rate=-0.1)
        with pytest.raises(ValueError, match="ability_effect"):
            AbilityDependentMissingness(ability_effect=-0.1)
        with pytest.raises(ValueError, match="max_rate"):
            AbilityDependentMissingness(max_rate=0.0)

    def test_appends_missing_category(self) -> None:
        """Should append missing category."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0, 1.0, -1.0])

        model = AbilityDependentMissingness()
        result = model.generate_missing_params(params, theta, rng)

        assert result.discrimination.shape == (10, 5)
        assert result.includes_missing_values

    def test_negative_discrimination(self) -> None:
        """Missing category should have negative discrimination."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0])

        model = AbilityDependentMissingness(ability_effect=0.1)
        result = model.generate_missing_params(params, theta, rng)

        # Negative discrimination means lower ability -> higher score
        assert np.all(result.discrimination[:, -1] < 0)

    def test_lower_ability_higher_missing(self) -> None:
        """Lower ability should have higher missing probability."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.linspace(-2, 2, 100).astype(np.float64)

        model = AbilityDependentMissingness(ability_effect=0.1)
        result = model.generate_missing_params(params, theta, rng)

        low_ability_probs = _compute_missing_probability(
            result.discrimination, result.intercept, -2.0
        )
        high_ability_probs = _compute_missing_probability(
            result.discrimination, result.intercept, 2.0
        )

        # Low ability should have higher missing probability
        assert np.all(low_ability_probs > high_ability_probs)


class TestPositionDependentMissingness:
    def test_validation(self) -> None:
        """Should validate parameters."""
        with pytest.raises(ValueError, match="base_rate"):
            PositionDependentMissingness(base_rate=-0.1)
        with pytest.raises(ValueError, match="position_effect"):
            PositionDependentMissingness(position_effect=-0.1)
        with pytest.raises(ValueError, match="max_rate"):
            PositionDependentMissingness(max_rate=0.0)

    def test_appends_missing_category(self) -> None:
        """Should append missing category."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0, 1.0, -1.0])

        model = PositionDependentMissingness()
        result = model.generate_missing_params(params, theta, rng)

        assert result.discrimination.shape == (10, 5)
        assert result.includes_missing_values

    def test_zero_discrimination(self) -> None:
        """Missing category should have discrimination=0."""
        params = _create_test_params(n_questions=10, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0])

        model = PositionDependentMissingness()
        result = model.generate_missing_params(params, theta, rng)

        assert np.allclose(result.discrimination[:, -1], 0.0)

    def test_later_questions_higher_missing(self) -> None:
        """Later questions should have higher missing probability."""
        params = _create_test_params(n_questions=50, n_categories=4)
        rng = get_rng(42)
        theta = np.array([0.0])

        model = PositionDependentMissingness(
            base_rate=0.01, position_effect=0.005, max_rate=0.5
        )
        result = model.generate_missing_params(params, theta, rng)

        probs = _compute_missing_probability(
            result.discrimination, result.intercept, 0.0
        )

        # Later questions should have higher missing probability
        assert probs[-1] > probs[0]
        # Should be monotonically increasing (until max_rate)
        for i in range(len(probs) - 1):
            assert (
                probs[i + 1] >= probs[i] - 0.001
            )  # Allow small numerical error
