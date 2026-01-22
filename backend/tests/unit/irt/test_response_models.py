import numpy as np

from analysis_service.core.data_models import Question
from analysis_service.core.utils import get_rng
from analysis_service.irt import (
    NominalResponseModel,
    ThreePLResponseModel,
    sample_response,
    sample_responses_batch,
)
from analysis_service.irt.estimation.nrm.parameters import NRMItemParameters


class TestResponseModels:
    def test_three_pl_probabilities_sum_to_one(self) -> None:
        model = ThreePLResponseModel()
        question = Question(
            question_id=0,
            difficulty=0.0,
            discrimination=1.0,
            guessing=0.2,
            correct_answer=0,
            distractor_quality=(0.5, 0.5, 0.5),
        )

        for ability in [-3, -1, 0, 1, 3]:
            probs = model.compute_probabilities(ability, question, 4)
            assert np.isclose(probs.sum(), 1.0)
            assert np.all(probs >= 0)
            assert np.all(probs <= 1)

    def test_three_pl_guessing_floor(self) -> None:
        """Test that very low ability still has guessing probability."""
        model = ThreePLResponseModel()
        guessing = 0.25
        question = Question(
            question_id=0,
            difficulty=0.0,
            discrimination=2.0,
            guessing=guessing,
            correct_answer=0,
            distractor_quality=(0.5, 0.5, 0.5),
        )

        # Very low ability
        probs = model.compute_probabilities(-10.0, question, 4)

        # Probability of correct should be close to guessing parameter
        assert probs[0] >= guessing * 0.9  # Allow small tolerance

    def test_three_pl_high_ability_correct(self) -> None:
        """Test that high ability candidates get correct answer with high probability."""
        model = ThreePLResponseModel()
        question = Question(
            question_id=0,
            difficulty=0.0,
            discrimination=1.5,
            guessing=0.2,
            correct_answer=1,
            distractor_quality=(0.5, 0.5, 0.5),
        )

        probs = model.compute_probabilities(3.0, question, 4)

        # High ability should have high probability of correct
        assert probs[1] > 0.9

    def test_nominal_response_model_valid_probs(self) -> None:
        item_params = [
            NRMItemParameters(
                item_id=0,
                discriminations=(0.5, 0.9, 0.4, 0.5),
                intercepts=(0.0, 0.3, 0.5, 0.8),
            ),
            NRMItemParameters(
                item_id=1,
                discriminations=(0.5, 0.9, 0.4, 0.5),
                intercepts=(0.0, 0.3, 0.5, 0.8),
            ),
        ]
        model = NominalResponseModel(item_params)
        question = Question(
            question_id=0,
            difficulty=0.0,
            discrimination=1.0,
            guessing=0.2,
            correct_answer=2,
            distractor_quality=(0.3, 0.6, 0.9),
        )

        for ability in [-2, 0, 2]:
            probs = model.compute_probabilities(ability, question, 4)
            assert np.isclose(probs.sum(), 1.0)

    def test_sample_response_returns_valid_index(self) -> None:
        question = Question(
            question_id=0,
            difficulty=0.0,
            discrimination=1.0,
            guessing=0.2,
            correct_answer=0,
            distractor_quality=(0.5, 0.5, 0.5),
        )
        rng = get_rng(42)

        for _ in range(100):
            response = sample_response(0.0, question, 4, rng=rng)
            assert 0 <= response < 4

    def test_sample_responses_batch_shape(self) -> None:
        abilities = np.array([0.0, 1.0, -1.0])
        questions = [
            Question(
                question_id=i,
                difficulty=0.0,
                discrimination=1.0,
                guessing=0.2,
                correct_answer=0,
                distractor_quality=(0.5, 0.5, 0.5),
            )
            for i in range(5)
        ]
        rng = get_rng(42)

        responses = sample_responses_batch(abilities, questions, 4, rng=rng)

        assert responses.shape == (3, 5)
