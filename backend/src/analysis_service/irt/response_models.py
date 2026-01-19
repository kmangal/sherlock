"""
Probabilistic response models for exam responses.

This module computes P(answer | ability, question) using IRT-inspired models.
It converts latent utilities into choice probabilities.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from analysis_service.core.data_models import Question
from analysis_service.core.utils import get_rng, softmax


class ResponseModel(ABC):
    """Abstract Base Class for response models that compute answer probabilities."""

    @abstractmethod
    def compute_probabilities(
        self,
        ability: float,
        question: Question,
        n_choices: int,
    ) -> NDArray[np.float64]:
        """
        Compute probability distribution over answer choices.

        Args:
            ability: Candidate's latent ability.
            question: Question with IRT parameters.
            n_choices: Number of answer choices.

        Returns:
            Array of shape (n_choices,) with probabilities summing to 1.
        """
        ...

    def compute_probabilities_batch(
        self,
        abilities: NDArray[np.float64],
        question: Question,
        n_choices: int,
    ) -> NDArray[np.float64]:
        """
        Compute probability distributions for multiple candidates.

        Args:
            abilities: Array of shape (n_candidates,) with ability values.
            question: Question with IRT parameters.
            n_choices: Number of answer choices.

        Returns:
            Array of shape (n_candidates, n_choices) with probabilities.
        """
        n_candidates = len(abilities)
        probs = np.empty((n_candidates, n_choices), dtype=np.float64)
        for i, ability in enumerate(abilities):
            probs[i] = self.compute_probabilities(ability, question, n_choices)
        return probs


class ThreePLResponseModel(ResponseModel):
    """
    3-Parameter Logistic (3PL) IRT model with distractor attractiveness.

    The 3PL model extends the 2PL model with a guessing parameter (c):
        P(correct | theta) = c + (1 - c) / (1 + exp(-a * (theta - b)))

    Where:
        - theta: candidate ability
        - a: discrimination parameter
        - b: difficulty parameter
        - c: guessing (pseudo-chance) parameter

    The guessing parameter represents the probability that even very low
    ability candidates will answer correctly (through guessing).

    For incorrect answers, the remaining probability (1 - P(correct)) is
    distributed among distractors based on their quality parameters.
    """

    def compute_probabilities(
        self,
        ability: float,
        question: Question,
        n_choices: int,
    ) -> NDArray[np.float64]:
        """
        Compute probability distribution over answer choices using 3PL model.

        Args:
            ability: Candidate's latent ability (theta).
            question: Question with difficulty (b), discrimination (a),
                guessing (c), and distractor quality parameters.
            n_choices: Number of answer choices.

        Returns:
            Array of shape (n_choices,) with probabilities summing to 1.
        """
        probs = self.compute_probabilities_batch(
            np.array([ability], dtype=np.float64), question, n_choices
        )
        result: NDArray[np.float64] = probs[0]
        return result

    def compute_probabilities_batch(
        self,
        abilities: NDArray[np.float64],
        question: Question,
        n_choices: int,
    ) -> NDArray[np.float64]:
        """
        Vectorized probability computation for all candidates.

        Args:
            abilities: Array of shape (n_candidates,) with ability values.
            question: Question with IRT parameters.
            n_choices: Number of answer choices.

        Returns:
            Array of shape (n_candidates, n_choices) with probabilities.
        """
        n_candidates = len(abilities)
        a = question.discrimination
        b = question.difficulty
        c = question.guessing
        correct = question.correct_answer
        distractor_quality = np.array(
            question.distractor_quality, dtype=np.float64
        )

        # Vectorized 3PL probability of correct answer
        exponent = -a * (abilities - b)
        exponent = np.clip(exponent, -30, 30)
        p_correct = c + (1 - c) / (1 + np.exp(exponent))

        # Initialize probabilities
        probs = np.zeros((n_candidates, n_choices), dtype=np.float64)
        probs[:, correct] = p_correct

        # Probability of incorrect answer
        p_wrong = 1.0 - p_correct

        # Compute distractor weights for all candidates
        # Temperature varies with ability gap
        ability_gaps = b - abilities
        temperatures = np.maximum(0.5, 1.0 + 0.2 * ability_gaps)

        # Distractor logits: quality / temperature for each candidate
        # Shape: (n_candidates, n_distractors)
        logits = (
            distractor_quality[np.newaxis, :] / temperatures[:, np.newaxis]
        )

        # Softmax along distractor axis
        distractor_weights = softmax(logits, axis=1)

        # Assign distractor probabilities to non-correct choices
        distractor_idx = 0
        for k in range(n_choices):
            if k != correct:
                probs[:, k] = p_wrong * distractor_weights[:, distractor_idx]
                distractor_idx += 1

        return probs


class NominalResponseModel(ResponseModel):
    """
    Nominal Response Model (Bock, 1972) with 3PL-style guessing.

    This model combines the 3PL guessing parameter with a multinomial
    logit structure for distractor selection. It provides more flexibility
    in modeling how candidates select among distractors.

    The probability of selecting choice k is:
        - For correct answer: P(correct) from 3PL model
        - For distractors: (1 - P(correct)) * softmax(distractor_logits)

    Where distractor logits depend on ability relative to difficulty
    and distractor quality.
    """

    def compute_probabilities(
        self,
        ability: float,
        question: Question,
        n_choices: int,
    ) -> NDArray[np.float64]:
        """
        Compute probability distribution over answer choices.

        Args:
            ability: Candidate's latent ability (theta).
            question: Question with difficulty (b), discrimination (a),
                guessing (c), and distractor quality parameters.
            n_choices: Number of answer choices.

        Returns:
            Array of shape (n_choices,) with probabilities summing to 1.
        """
        probs = self.compute_probabilities_batch(
            np.array([ability], dtype=np.float64), question, n_choices
        )
        result: NDArray[np.float64] = probs[0]
        return result

    def compute_probabilities_batch(
        self,
        abilities: NDArray[np.float64],
        question: Question,
        n_choices: int,
    ) -> NDArray[np.float64]:
        """
        Vectorized probability computation for all candidates.

        Args:
            abilities: Array of shape (n_candidates,) with ability values.
            question: Question with IRT parameters.
            n_choices: Number of answer choices.

        Returns:
            Array of shape (n_candidates, n_choices) with probabilities.
        """
        n_candidates = len(abilities)
        a = question.discrimination
        b = question.difficulty
        c = question.guessing
        correct = question.correct_answer
        distractor_quality = np.array(
            question.distractor_quality, dtype=np.float64
        )

        # Vectorized 3PL probability of correct answer
        exponent = -a * (abilities - b)
        exponent = np.clip(exponent, -30, 30)
        p_correct = c + (1 - c) / (1 + np.exp(exponent))

        # Initialize probabilities
        probs = np.zeros((n_candidates, n_choices), dtype=np.float64)
        probs[:, correct] = p_correct

        # Probability of incorrect answer
        p_wrong = 1.0 - p_correct

        # Compute distractor logits for all candidates
        # Effect scales with ability gap
        ability_gaps = b - abilities
        scaling = np.maximum(0.5, 1.0 + 0.3 * ability_gaps)

        # Distractor logits: quality * scaling for each candidate
        # Shape: (n_candidates, n_distractors)
        logits = distractor_quality[np.newaxis, :] * scaling[:, np.newaxis]

        # Softmax along distractor axis
        distractor_probs = softmax(logits, axis=1)

        # Assign distractor probabilities to non-correct choices
        distractor_idx = 0
        for k in range(n_choices):
            if k != correct:
                probs[:, k] = p_wrong * distractor_probs[:, distractor_idx]
                distractor_idx += 1

        return probs


def sample_response(
    ability: float,
    question: Question,
    n_choices: int,
    response_model: ResponseModel | None = None,
    rng: Generator | None = None,
) -> int:
    """
    Sample a single response given ability and question.

    Args:
        ability: Candidate's latent ability.
        question: Question with IRT parameters.
        n_choices: Number of answer choices.
        response_model: Model to compute probabilities. Defaults to ThreePLResponseModel.
        rng: Random number generator.

    Returns:
        Index of selected answer (0-based).
    """
    if rng is None:
        rng = get_rng()
    if response_model is None:
        response_model = ThreePLResponseModel()

    probs = response_model.compute_probabilities(ability, question, n_choices)
    return int(rng.choice(n_choices, p=probs))


def sample_responses_batch(
    abilities: NDArray[np.float64],
    questions: list[Question],
    n_choices: int,
    response_model: ResponseModel | None = None,
    rng: Generator | None = None,
) -> NDArray[np.int8]:
    """
    Sample responses for all candidates and questions.

    Uses vectorized probability computation and sampling for efficiency.

    Args:
        abilities: Array of shape (n_candidates,) with ability values.
        questions: List of Question objects.
        n_choices: Number of answer choices.
        response_model: Model to compute probabilities.
        rng: Random number generator.

    Returns:
        Array of shape (n_candidates, n_questions) with response indices.
    """
    if rng is None:
        rng = get_rng()
    if response_model is None:
        response_model = ThreePLResponseModel()

    n_candidates = len(abilities)
    n_questions = len(questions)

    responses = np.empty((n_candidates, n_questions), dtype=np.int8)

    for j, question in enumerate(questions):
        # Compute probabilities for all candidates at once
        probs = response_model.compute_probabilities_batch(
            abilities, question, n_choices
        )

        # Vectorized sampling using cumulative probabilities
        cumprobs = np.cumsum(probs, axis=1)
        u = rng.random(n_candidates)

        # Find the choice index where cumulative probability exceeds u
        # (cumprobs < u[:, np.newaxis]).sum(axis=1) counts how many cum probs are < u
        responses[:, j] = np.minimum(
            (cumprobs < u[:, np.newaxis]).sum(axis=1), n_choices - 1
        )

    return responses
