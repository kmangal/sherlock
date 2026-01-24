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
    n_choices: int

    # Missingness
    missing_rate: float = MISSING

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
        if self.n_choices < 2 or self.n_choices > 26:
            raise ValueError("Must have between 2 and 26 choices per question")
