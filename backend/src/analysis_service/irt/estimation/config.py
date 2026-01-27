"""
Configuration dataclasses for IRT model estimation.

This module defines the configuration parameters for:
- Quadrature settings (Gauss-Hermite integration)
- Convergence criteria for EM algorithm
- Overall estimation settings
"""

from dataclasses import dataclass, field

import toml

from analysis_service.core.paths import get_backend_root_dir

# Default parameter bounds
# Sum-to-zero identification allows negative discriminations
DEFAULT_DISCRIMINATION_BOUNDS = (-5.0, 5.0)
DEFAULT_INTERCEPT_BOUNDS = (-10.0, 10.0)

# Default penalty settings for correct answer constraint
DEFAULT_PENALTY_LAMBDA = 0.5
DEFAULT_PENALTY_MARGIN = 0.1

# Default convergence settings
DEFAULT_MAX_EM_ITERATIONS = 500
DEFAULT_EM_TOLERANCE = 1e-6
DEFAULT_PARAM_TOLERANCE = 1e-2
DEFAULT_MAX_LBFGS_ITERATIONS = 100
DEFAULT_LBFGS_TOLERANCE = 1e-6

# Default warmup settings (per UPDATES.md recommendation)
# Running 1-2 EM iterations with fixed discriminations reduces local maxima
DEFAULT_WARMUP_ITERATIONS = 2

# Default quadrature settings
DEFAULT_QUADRATURE_POINTS = 41


def _get_backend_version() -> str:
    root_dir = get_backend_root_dir()
    with open(root_dir / "pyproject.toml") as f:
        data = toml.load(f)

    # Try standard format first
    version = data.get("project", {}).get("version")

    if not version:
        raise ValueError("Version not found in pyproject.toml")

    assert isinstance(version, str)
    return version


@dataclass(frozen=True)
class QuadratureConfig:
    """
    Configuration for Gauss-Hermite quadrature.

    Attributes:
        n_points: Number of quadrature points. Standard in IRT software
            (IRTPRO, flexMIRT) is 41 points.
        mean: Mean of the ability distribution (typically 0).
        std: Standard deviation of the ability distribution (typically 1).
    """

    n_points: int = DEFAULT_QUADRATURE_POINTS
    mean: float = 0.0
    std: float = 1.0


@dataclass(frozen=True)
class ConvergenceConfig:
    """
    Configuration for EM algorithm convergence.

    Attributes:
        max_em_iterations: Maximum number of EM iterations.
        em_tolerance: Convergence tolerance for log-likelihood change.
            EM stops when |LL_new - LL_old| / |LL_old| < tolerance.
        max_lbfgs_iterations: Maximum iterations for L-BFGS-B in M-step.
        lbfgs_tolerance: Convergence tolerance for L-BFGS-B optimizer.
        warmup_iterations: Number of initial EM iterations with fixed
            discriminations. Per UPDATES.md, this reduces local maxima issues.
            Set to 0 to disable warmup.
    """

    max_em_iterations: int = DEFAULT_MAX_EM_ITERATIONS
    em_tolerance: float = DEFAULT_EM_TOLERANCE
    param_tolerance: float = DEFAULT_PARAM_TOLERANCE
    max_lbfgs_iterations: int = DEFAULT_MAX_LBFGS_ITERATIONS
    lbfgs_tolerance: float = DEFAULT_LBFGS_TOLERANCE
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS


@dataclass(frozen=True)
class PenaltyConfig:
    """
    Configuration for correct answer soft penalty.

    The penalty encourages a_correct > a_distractor without hard constraints.
    Uses squared hinge: λ * Σ_{j≠correct} max(0, a_j - a_correct + margin)²

    Attributes:
        lambda_penalty: Penalty weight. Higher values enforce constraint more strictly.
        margin: Minimum margin between correct and distractor discriminations.
    """

    lambda_penalty: float = DEFAULT_PENALTY_LAMBDA
    margin: float = DEFAULT_PENALTY_MARGIN


@dataclass(frozen=True)
class ParameterBounds:
    """
    Bounds for item parameters during optimization.

    Attributes:
        discrimination: (min, max) bounds for discrimination parameters.
        intercept: (min, max) bounds for intercept parameters.
    """

    discrimination: tuple[float, float] = DEFAULT_DISCRIMINATION_BOUNDS
    intercept: tuple[float, float] = DEFAULT_INTERCEPT_BOUNDS


@dataclass(frozen=True)
class EstimationConfig:
    """
    Master configuration for IRT model estimation.

    Attributes:
        quadrature: Settings for Gauss-Hermite quadrature.
        convergence: Convergence criteria for EM algorithm.
        bounds: Parameter bounds for optimization.
        penalty: Settings for correct answer soft penalty.
        model_version: Version string for reproducibility tracking.
    """

    quadrature: QuadratureConfig = QuadratureConfig()
    convergence: ConvergenceConfig = ConvergenceConfig()
    bounds: ParameterBounds = ParameterBounds()
    penalty: PenaltyConfig = PenaltyConfig()
    model_version: str = field(default_factory=_get_backend_version)


def default_config() -> EstimationConfig:
    """Create a default estimation configuration."""
    return EstimationConfig()
