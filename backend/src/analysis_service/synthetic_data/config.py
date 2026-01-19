from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class DistributionConfig:
    """Configuration for a single parameter's marginal distribution.

    Attributes:
        distribution: Distribution type ("normal", "truncated_normal", "uniform", "log_normal")
        mean: Mean of the distribution
        std: Standard deviation of the distribution
        bounds: Optional (min, max) bounds. Use None for unbounded.
    """

    distribution: str = MISSING
    params: dict[str, float | None] = MISSING


def _default_difficulty() -> DistributionConfig:
    return DistributionConfig(
        distribution="normal", params={"mean": 0.0, "std": 1.0}
    )


def _default_discrimination() -> DistributionConfig:
    return DistributionConfig(
        distribution="normal",
        params={"mean": 0.0, "std": 1.0, "lower": 0.3, "upper": None},
    )


def _default_guessing() -> DistributionConfig:
    return DistributionConfig(
        distribution="normal",
        params={"mean": 0.2, "std": 0.05, "lower": 0.0, "upper": 0.35},
    )


def _default_distractor_quality() -> DistributionConfig:
    return DistributionConfig(
        distribution="truncated_normal",
        params={"mean": 0.5, "std": 0.2, "lower": 0.0, "upper": None},
    )


def _default_ability() -> DistributionConfig:
    return DistributionConfig(
        distribution="normal", params={"mean": 0.0, "std": 3.0}
    )


@dataclass
class IRTParametersConfig:
    """Configuration for all IRT parameter marginal distributions.

    Attributes:
        difficulty: Difficulty parameter distribution
        discrimination: Discrimination parameter distribution
        guessing: Guessing (pseudo-chance) parameter distribution
        distractor_quality: Distractor quality parameter distribution
    """

    difficulty: DistributionConfig = field(default_factory=_default_difficulty)
    discrimination: DistributionConfig = field(
        default_factory=_default_discrimination
    )
    guessing: DistributionConfig = field(default_factory=_default_guessing)
    distractor_quality: DistributionConfig = field(
        default_factory=_default_distractor_quality
    )


@dataclass
class CorrelationsConfig:
    """Pairwise correlations for 3x3 joint distribution.

    Only difficulty, discrimination, and guessing are sampled jointly.
    Distractor quality is sampled independently (per-distractor, not per-question).

    Attributes:
        difficulty_discrimination: Correlation between difficulty and discrimination
        difficulty_guessing: Correlation between difficulty and guessing
        discrimination_guessing: Correlation between discrimination and guessing
    """

    difficulty_discrimination: float = 0.0
    difficulty_guessing: float = 0.0
    discrimination_guessing: float = 0.0


@dataclass
class GenerationConfig:
    """Complete configuration for generating synthetic dataset."""

    n_candidates: int
    n_questions: int
    n_choices: int

    # Missingness
    missing_rate: float = MISSING

    # Reproducibility
    random_seed: int = MISSING

    ability: DistributionConfig = field(default_factory=_default_ability)
    irt_parameters: IRTParametersConfig = field(
        default_factory=IRTParametersConfig
    )
    correlations: CorrelationsConfig = field(
        default_factory=CorrelationsConfig
    )

    def __post_init__(self) -> None:
        if self.n_candidates <= 1:
            raise ValueError("Must have at least 2 candidates")
        if self.n_questions <= 0:
            raise ValueError("Must have at least 1 question")
        if self.n_choices < 2 or self.n_choices > 26:
            raise ValueError("Must have between 2 and 26 choices per question")
