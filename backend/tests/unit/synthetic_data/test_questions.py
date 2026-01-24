import pytest

from analysis_service.core.utils import get_rng
from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.presets import get_preset
from analysis_service.synthetic_data.questions import sample_item_parameters


@pytest.fixture
def baseline_config() -> GenerationConfig:
    return get_preset("baseline")


class TestSampleParams:
    def test_returns_correct_count(
        self, baseline_config: GenerationConfig
    ) -> None:
        params = sample_item_parameters(baseline_config, rng=get_rng(42))

        assert len(params) == baseline_config.n_questions

    def test_creates_valid_params(
        self, baseline_config: GenerationConfig
    ) -> None:
        params = sample_item_parameters(baseline_config, rng=get_rng(42))

        n_choices = baseline_config.n_choices

        for p in params:
            assert p.correct_answer is not None
            assert 0 <= p.correct_answer < n_choices

    def test_item_ids_are_sequential(
        self, baseline_config: GenerationConfig
    ) -> None:
        params = sample_item_parameters(baseline_config, rng=get_rng(42))

        for i, p in enumerate(params):
            assert p.item_id == i

    def test_reproducible_with_same_seed(
        self, baseline_config: GenerationConfig
    ) -> None:
        p1 = sample_item_parameters(baseline_config, rng=get_rng(42))
        p2 = sample_item_parameters(baseline_config, rng=get_rng(42))

        assert len(p1) == len(p2)
        for i in range(len(p1)):
            assert p1[i].discriminations == p2[i].discriminations
            assert p1[i].intercepts == p2[i].intercepts
            assert p1[i].correct_answer == p2[i].correct_answer

    def test_different_seeds_produce_different_params(
        self, baseline_config: GenerationConfig
    ) -> None:
        p1 = sample_item_parameters(baseline_config, rng=get_rng(42))
        p2 = sample_item_parameters(baseline_config, rng=get_rng(99))

        # At least some items should differ
        discrimination_differences = sum(
            1
            for a, b in zip(p1, p2, strict=True)
            if a.discriminations != b.discriminations
        )
        assert discrimination_differences > 0

        # At least some items should differ
        intercept_differences = sum(
            1
            for a, b in zip(p1, p2, strict=True)
            if a.intercepts != b.intercepts
        )
        assert intercept_differences > 0

        # At least some items should differ
        correct_answer_differences = sum(
            1
            for a, b in zip(p1, p2, strict=True)
            if a.correct_answer != b.correct_answer
        )
        assert correct_answer_differences > 0
