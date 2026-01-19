"""
Core data models shared across analysis service modules.

This module defines foundational data structures used by both the IRT
statistical models and the synthetic data generation layer, enabling
clean decoupling between these components.
"""

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat


class Question(BaseModel):
    """
    A question with 3PL IRT parameters and distractor properties.

    Attributes:
        question_id: Unique identifier for the question.
        difficulty: Item difficulty parameter (higher = harder).
            On IRT scale, typically in range [-3, 3].
        discrimination: Item discrimination parameter (higher = better at
            distinguishing abilities). Typically in range [0.5, 2.5].
        guessing: Pseudo-guessing parameter (lower asymptote).
            Probability of correct response by very low ability candidates.
            Typically in range [0, 0.35] for 4-choice items.
        correct_answer: Index of the correct answer (0-indexed).
        distractor_quality: Quality of each distractor (non-correct answer).
            Higher values make distractors more attractive to low-ability candidates.
            Shape: (n_choices - 1,). Order corresponds to non-correct choices
            in alphabetical order.
    """

    model_config = ConfigDict(frozen=True)

    question_id: int
    difficulty: float
    discrimination: float = Field(gt=0)
    guessing: float = Field(ge=0, le=1, default=0.0)
    correct_answer: int | None = Field(default=None, ge=0)
    distractor_quality: tuple[NonNegativeFloat, ...]
