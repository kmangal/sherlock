import pytest

from analysis_service.core.utils import get_rng
from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.presets import get_preset
from analysis_service.synthetic_data.questions import sample_questions


@pytest.fixture
def baseline_config() -> GenerationConfig:
    return get_preset("baseline")


class TestSampleQuestions:
    def test_returns_correct_count(
        self, baseline_config: GenerationConfig
    ) -> None:
        questions = sample_questions(baseline_config, rng=get_rng(42))

        assert len(questions) == baseline_config.n_questions

    def test_creates_valid_questions(
        self, baseline_config: GenerationConfig
    ) -> None:
        questions = sample_questions(baseline_config, rng=get_rng(42))

        n_choices = baseline_config.n_choices
        n_distractors = n_choices - 1

        for q in questions:
            assert q.correct_answer is not None
            assert 0 <= q.correct_answer < n_choices
            assert q.discrimination > 0
            assert 0 <= q.guessing <= 1
            assert len(q.distractor_quality) == n_distractors
            assert all(dq >= 0 for dq in q.distractor_quality)

    def test_question_ids_are_sequential(
        self, baseline_config: GenerationConfig
    ) -> None:
        questions = sample_questions(baseline_config, rng=get_rng(42))

        for i, q in enumerate(questions):
            assert q.question_id == i

    def test_reproducible_with_same_seed(
        self, baseline_config: GenerationConfig
    ) -> None:
        q1 = sample_questions(baseline_config, rng=get_rng(42))
        q2 = sample_questions(baseline_config, rng=get_rng(42))

        assert len(q1) == len(q2)
        for i in range(len(q1)):
            assert q1[i].difficulty == q2[i].difficulty
            assert q1[i].discrimination == q2[i].discrimination
            assert q1[i].guessing == q2[i].guessing
            assert q1[i].correct_answer == q2[i].correct_answer
            assert q1[i].distractor_quality == q2[i].distractor_quality

    def test_different_seeds_produce_different_questions(
        self, baseline_config: GenerationConfig
    ) -> None:
        q1 = sample_questions(baseline_config, rng=get_rng(42))
        q2 = sample_questions(baseline_config, rng=get_rng(99))

        # At least some questions should differ
        differences = sum(
            1
            for a, b in zip(q1, q2, strict=True)
            if a.difficulty != b.difficulty
        )
        assert differences > 0
