from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class DistributionConfig:
    """Configuration for a single parameter's marginal distribution.

    Attributes:
        distribution: Distribution type ("normal", "truncated_normal", "uniform", "log_normal")
        params: Distribution parameters (mean, std, lower, upper, etc.)
    """

    distribution: str = MISSING
    params: dict[str, float | None] = MISSING


@dataclass
class CorrelationsConfig:
    """Configure correlations between NRM parameters"""

    discrimination_intercept: float
    intercept_correct_gap: float
    discrimination_correct_gap: float

    def __post_init__(self) -> None:
        check1 = abs(self.discrimination_correct_gap) > 1
        check2 = abs(self.discrimination_intercept) > 1
        check3 = abs(self.intercept_correct_gap) > 1

        if any([check1, check2, check3]):
            raise ValueError("correlations should have absolute value < 1")


def _default_discrimination() -> DistributionConfig:
    """Default distribution for NRM distractor discriminations (a_k for k != correct)."""
    return DistributionConfig(
        distribution="truncated_normal",
        params={"mean": 0.5, "std": 0.3, "lower": 0.0, "upper": 2.5},
    )


def _default_intercept() -> DistributionConfig:
    """Default distribution for NRM intercepts (c_k)."""
    return DistributionConfig(
        distribution="normal",
        params={"mean": 0.0, "std": 1.0},
    )


def _default_correct_dicrimination_gap() -> DistributionConfig:
    """Default distribution for the gap between a_correct and max(a_distractor).

    This is the delta parameter where a_correct = max(a_distractor) + delta.
    Should be positive to ensure correct answer dominates at high ability.
    """
    return DistributionConfig(
        distribution="truncated_normal",
        params={"mean": 0.5, "std": 0.3, "lower": 0.1, "upper": 2.0},
    )


def _default_ability() -> DistributionConfig:
    return DistributionConfig(
        distribution="normal", params={"mean": 0.0, "std": 3.0}
    )


def _default_correlations() -> CorrelationsConfig:
    return CorrelationsConfig(
        discrimination_intercept=0.0,
        discrimination_correct_gap=0.0,
        intercept_correct_gap=0.0,
    )


@dataclass
class NoMissingParams:
    """Parameters for no missingness model."""

    pass


@dataclass
class MCARParams:
    """Parameters for MCAR missingness model.

    Attributes:
        rate: Probability that any given response is missing. Must be in [0, 1).
    """

    rate: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.rate < 1.0):
            raise ValueError(f"rate must be in [0, 1), got {self.rate}")


@dataclass
class AbilityDependentParams:
    """Parameters for ability-dependent missingness model.

    Lower ability candidates are more likely to skip questions.

    Attributes:
        base_rate: Base missing rate for all candidates.
        ability_effect: How much each unit below threshold increases missing rate.
        ability_threshold: Ability level below which missingness increases.
        max_rate: Maximum missing rate for any candidate.
    """

    base_rate: float = 0.02
    ability_effect: float = 0.05
    ability_threshold: float = 0.0
    max_rate: float = 0.3

    def __post_init__(self) -> None:
        if not (0.0 <= self.base_rate < 1.0):
            raise ValueError(
                f"base_rate must be in [0, 1), got {self.base_rate}"
            )
        if self.ability_effect < 0:
            raise ValueError(
                f"ability_effect must be >= 0, got {self.ability_effect}"
            )
        if not (0.0 < self.max_rate <= 1.0):
            raise ValueError(
                f"max_rate must be in (0, 1], got {self.max_rate}"
            )


@dataclass
class PositionDependentParams:
    """Parameters for position-dependent missingness model.

    Later questions are more likely to be skipped due to test fatigue.

    Attributes:
        base_rate: Missing rate for the first question.
        position_effect: Increase in missing rate per question.
        max_rate: Maximum missing rate for any question.
    """

    base_rate: float = 0.01
    position_effect: float = 0.002
    max_rate: float = 0.2

    def __post_init__(self) -> None:
        if not (0.0 <= self.base_rate < 1.0):
            raise ValueError(
                f"base_rate must be in [0, 1), got {self.base_rate}"
            )
        if self.position_effect < 0:
            raise ValueError(
                f"position_effect must be >= 0, got {self.position_effect}"
            )
        if not (0.0 < self.max_rate <= 1.0):
            raise ValueError(
                f"max_rate must be in (0, 1], got {self.max_rate}"
            )


@dataclass
class MissingConfig:
    """Configuration for missing values.

    Attributes:
        model: Model type ("none", "mcar", "ability_dependent", "position_dependent").
        params: Model-specific parameters as dict for OmegaConf compatibility.
    """

    model: str = "mcar"
    params: dict[str, float] = field(default_factory=lambda: {"rate": 0.05})


def _default_missing() -> MissingConfig:
    return MissingConfig()


@dataclass
class NRMParametersConfig:
    """Configuration for NRM parameter distributions.

    The NRM model: P(Y=k|θ) = exp(a_k*θ + c_k) / Σ exp(a_m*θ + c_m)

    When correct answer is known, we enforce a_correct > max(a_distractor)
    by sampling: a_correct = max(a_distractor) + correct_gap

    Attributes:
        discrimination: Distribution for distractor discriminations (a_k, k != correct)
        intercept: Distribution for intercepts (c_k)
        correct_gap: Distribution for gap delta = a_correct - max(a_distractor)
    """

    discrimination: DistributionConfig = field(
        default_factory=_default_discrimination
    )
    intercept: DistributionConfig = field(default_factory=_default_intercept)
    correct_discrimination_gap: DistributionConfig = field(
        default_factory=_default_correct_dicrimination_gap
    )


@dataclass
class GenerationConfig:
    """Complete configuration for generating synthetic dataset."""

    n_candidates: int
    n_questions: int
    n_response_categories: int

    # Missingness
    missing: MissingConfig = field(default_factory=_default_missing)

    # Reproducibility
    random_seed: int = MISSING

    ability: DistributionConfig = field(default_factory=_default_ability)
    nrm_parameters: NRMParametersConfig = field(
        default_factory=NRMParametersConfig
    )

    correlations: CorrelationsConfig = field(
        default_factory=_default_correlations
    )

    def __post_init__(self) -> None:
        if self.n_candidates <= 1:
            raise ValueError("Must have at least 2 candidates")
        if self.n_questions <= 0:
            raise ValueError("Must have at least 1 question")
        if self.n_response_categories < 2 or self.n_response_categories > 26:
            raise ValueError("Must have between 2 and 26 choices per question")
