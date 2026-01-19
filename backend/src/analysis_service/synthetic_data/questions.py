"""
Question parameter sampling.

This module defines question-level parameters including difficulty,
discrimination, guessing, and distractor quality for the 3PL IRT model.
"""

from numpy.random import Generator

from analysis_service.core.data_models import Question
from analysis_service.core.utils import get_rng
from analysis_service.synthetic_data.config import GenerationConfig
from analysis_service.synthetic_data.parameters import (
    JointParameterSampler,
)


def sample_questions(
    config: GenerationConfig,
    rng: Generator | None = None,
) -> list[Question]:
    """
    Sample question parameters for an exam using the 3PL IRT model.

    Parameters are sampled from a joint distribution specified by the config:
    - Difficulty, discrimination, guessing are sampled jointly with correlations
    - Distractor quality is sampled independently per distractor

    Args:
        exam_spec: Exam specification (number of questions, choices, correct answers).
        params: Question parameter configuration. If None, uses defaults.
        rng: Random number generator.

    Returns:
        List of Question objects with sampled parameters.
    """
    if rng is None:
        rng = get_rng(config.random_seed)

    n_questions = config.n_questions
    n_choices = config.n_choices
    n_distractors = n_choices - 1

    # Sample all parameters jointly
    sampler = JointParameterSampler(config)
    sampled = sampler.sample(n_questions, n_distractors, rng)

    # Determine correct answers
    correct_answers = rng.integers(0, n_choices, size=n_questions).tolist()

    # Build Question objects
    questions = []
    for i in range(n_questions):
        q = Question(
            question_id=i,
            difficulty=float(sampled.difficulty[i]),
            discrimination=float(sampled.discrimination[i]),
            guessing=float(sampled.guessing[i]),
            correct_answer=correct_answers[i],
            distractor_quality=tuple(
                float(dq) for dq in sampled.distractor_quality[i]
            ),
        )
        questions.append(q)

    return questions
