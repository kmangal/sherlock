import numpy as np
import pytest

from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.generators import generate_exam_responses
from analysis_service.synthetic_data.presets import get_preset
from analysis_service.synthetic_data.validation import (
    compute_missing_rate,
    validate_generated_data,
)


@pytest.fixture
def baseline_config() -> GenerationConfig:
    return get_preset("baseline")


def test_generate_exam_responses_basic(
    baseline_config: GenerationConfig,
) -> None:
    data = generate_exam_responses(baseline_config)

    n_candidates = baseline_config.n_candidates
    n_items = baseline_config.n_questions

    assert len(data.answer_strings) == n_candidates
    assert all(len(s) == n_items for s in data.answer_strings)
    assert data.abilities.shape == (n_candidates,)
    assert len(data.item_params) == n_items


def test_generate_exam_responses_reproducible(
    baseline_config: GenerationConfig,
) -> None:
    data1 = generate_exam_responses(baseline_config)
    data2 = generate_exam_responses(baseline_config)

    assert data1.answer_strings == data2.answer_strings
    np.testing.assert_array_equal(data1.abilities, data2.abilities)


def test_generate_exam_responses_with_missingness(
    baseline_config: GenerationConfig,
) -> None:
    data = generate_exam_responses(baseline_config)
    missing_rate = baseline_config.missing_rate

    actual_rate = compute_missing_rate(data.answer_strings)
    assert abs(actual_rate - missing_rate) < 0.02


def test_generate_exam_responses_validation_passes(
    baseline_config: GenerationConfig,
) -> None:
    data = generate_exam_responses(baseline_config)
    # Should not raise
    validate_generated_data(data)
