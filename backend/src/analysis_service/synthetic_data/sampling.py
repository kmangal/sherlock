"""
Sampling from statistical distributions

This module contains utilities for sampling from various statistical distributions.
Any scipy.stats distribution can be used, plus custom mixture distributions.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy import stats

from analysis_service.core.utils import get_rng


class FrozenRV(Protocol):
    def cdf(self, x: Any) -> NDArray[np.floating[Any]]: ...
    def rvs(
        self, size: Any, random_state: Any
    ) -> NDArray[np.floating[Any]]: ...
    def ppf(self, q: Any) -> NDArray[np.floating[Any]]: ...


class Distribution(ABC):
    """Abstract base class for a statistical distributions."""

    @abstractmethod
    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]:
        """
        Sample n values.

        Args:
            n: Number of samples.
            rng: Random number generator.

        Returns:
            Array of shape (n,) with sampled values.
        """
        ...

    @abstractmethod
    def inverse_cdf(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the inverse CDF (quantile function).

        Args:
            q: Quantile values in [0, 1].

        Returns:
            Values x such that CDF(x) = q.
        """
        ...

    @abstractmethod
    def cdf(self, x: float) -> float:
        """
        Compute the cumulative distribution function at point x.

        Args:
            x: Point at which to evaluate the CDF.

        Returns:
            Probability P(X <= x).
        """
        ...


@dataclass
class ScipyDistribution(Distribution):
    """
    Wrapper for any scipy.stats distribution.

    Examples:
        >>> dist = ScipyDistribution(stats.norm(loc=0, scale=1))
        >>> dist = ScipyDistribution(stats.skewnorm(a=5, loc=0, scale=1))
        >>> dist = ScipyDistribution(stats.t(df=4, loc=0, scale=1))
    """

    dist: FrozenRV

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]:
        samples: NDArray[np.float64] = self.dist.rvs(
            size=n, random_state=rng
        ).astype(np.float64)
        return samples

    def inverse_cdf(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = self.dist.ppf(q).astype(np.float64)
        return result

    def cdf(self, x: float) -> float:
        return float(self.dist.cdf(x))


@dataclass
class MixtureDistribution(Distribution):
    """
    Mixture of multiple distributions with specified weights.

    Useful for bimodal, multimodal, or any mixture distribution.

    Examples:
        >>> # Bimodal: 60% low ability, 40% high ability
        >>> dist = MixtureDistribution(
        ...     components=[
        ...         ScipyDistribution(stats.norm(loc=-1, scale=0.5)),
        ...         ScipyDistribution(stats.norm(loc=1.5, scale=0.5)),
        ...     ],
        ...     weights=[0.6, 0.4],
        ... )
    """

    components: list[Distribution]
    weights: list[float]

    def __post_init__(self) -> None:
        if len(self.components) != len(self.weights):
            raise ValueError(
                "Number of components must match number of weights"
            )
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1, got {sum(self.weights)}")
        if any(w < 0 for w in self.weights):
            raise ValueError("Weights must be non-negative")

    def sample(self, n: int, rng: Generator) -> NDArray[np.float64]:
        # Determine component assignments
        assignments = rng.choice(len(self.components), size=n, p=self.weights)

        # Sample from each component
        values = np.empty(n, dtype=np.float64)
        for k, component in enumerate(self.components):
            mask = assignments == k
            count = int(mask.sum())
            if count > 0:
                values[mask] = component.sample(count, rng)

        return values

    def inverse_cdf(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute inverse CDF for mixture distribution using numerical root finding.

        For a mixture distribution, the CDF is:
            F(x) = sum_k w_k * F_k(x)

        We solve F(x) = q numerically using bisection.
        """
        from scipy.optimize import brentq

        q = np.atleast_1d(q)
        result = np.empty_like(q, dtype=np.float64)

        for i, qi in enumerate(q):
            # Define the CDF of the mixture
            def mixture_cdf(x: float) -> float:
                return sum(
                    w * float(comp.cdf(x))
                    for w, comp in zip(
                        self.weights, self.components, strict=True
                    )
                )

            # Use Brent's method to find x such that mixture_cdf(x) = qi
            # Search bounds: use a wide range, could be refined based on components
            try:
                _, finder_result = brentq(
                    lambda x, q=qi: mixture_cdf(x) - q, -100, 100
                )
                result[i] = finder_result.root
            except ValueError:
                # If brentq fails (e.g., qi is 0 or 1), use fallback
                if qi <= 0:
                    result[i] = -np.inf
                elif qi >= 1:
                    result[i] = np.inf
                else:
                    raise

        return result

    def cdf(self, x: float) -> float:
        """Compute CDF of the mixture distribution at point x."""
        return sum(
            w * float(comp.cdf(x))
            for w, comp in zip(self.weights, self.components, strict=True)
        )


####################################################################
# Registry
####################################################################


DistributionGenerator = Callable[..., Distribution]


class SamplerRegistry:
    def __init__(self) -> None:
        self._samplers: dict[str, DistributionGenerator] = {}

    def register(
        self, name: str
    ) -> Callable[[DistributionGenerator], DistributionGenerator]:
        def decorator(
            func: DistributionGenerator,
        ) -> DistributionGenerator:
            self._samplers[name] = func
            return func

        return decorator

    def get_sampler(
        self, name: str, params: dict[str, float | None]
    ) -> Distribution:
        if name not in self._samplers:
            raise ValueError(f"Sampler {name} not registered")
        return self._samplers[name](**params)


registry = SamplerRegistry()


@registry.register("normal")
def normal(*, mean: float = 0.0, std: float = 1.0) -> ScipyDistribution:
    """Normal distribution."""
    return ScipyDistribution(stats.norm(loc=mean, scale=std))


@registry.register("skew_normal")
def skew_normal(
    *, a: float, loc: float = 0.0, scale: float = 1.0
) -> ScipyDistribution:
    """
    Skew-normal distribution.

    Args:
        a: Shape parameter controlling skewness. a > 0 gives right skew.
        loc: Location parameter.
        scale: Scale parameter.
    """
    return ScipyDistribution(stats.skewnorm(a=a, loc=loc, scale=scale))


@registry.register("bimodal")
def bimodal(
    *,
    loc1: float,
    scale1: float,
    loc2: float,
    scale2: float,
    weight1: float,
) -> Distribution:
    """
    Bimodal distribution (mixture of two normals).

    Args:
        loc1: Mean of first component.
        scale1: Std dev of first component.
        loc2: Mean of second component.
        scale2: Std dev of second component.
        weight1: Weight of first component (weight2 = 1 - weight1).
    """
    return MixtureDistribution(
        components=[
            ScipyDistribution(stats.norm(loc=loc1, scale=scale1)),
            ScipyDistribution(stats.norm(loc=loc2, scale=scale2)),
        ],
        weights=[weight1, 1.0 - weight1],
    )


@registry.register("uniform")
def uniform(*, low: float = -1.0, high: float = 1.0) -> ScipyDistribution:
    """Uniform distribution."""
    return ScipyDistribution(stats.uniform(loc=low, scale=high - low))


@registry.register("truncated_normal")
def truncated_normal(
    *,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float | None = None,
    upper: float | None = None,
) -> ScipyDistribution:
    """
    Truncated normal distribution.

    Args:
        mean: Mean of the underlying normal distribution.
        std: Standard deviation of the underlying normal distribution.
        lower: Lower bound (None = unbounded).
        upper: Upper bound (None = unbounded).
    """
    # Convert bounds to standardized form for scipy.stats.truncnorm
    a_std = (lower - mean) / std if lower is not None else -np.inf
    b_std = (upper - mean) / std if upper is not None else np.inf
    return ScipyDistribution(
        stats.truncnorm(a_std, b_std, loc=mean, scale=std)
    )


@registry.register("student_t")
def student_t(
    *, df: float, loc: float = 0.0, scale: float = 1.0
) -> ScipyDistribution:
    """Student's t distribution (heavier tails than normal)."""
    return ScipyDistribution(stats.t(df=df, loc=loc, scale=scale))


@registry.register("log_normal")
def log_normal(*, mean: float, std: float) -> ScipyDistribution:
    """
    Log-normal distribution parameterized by mean and std of the log-normal.

    Args:
        mean: Mean of the log-normal distribution (not the underlying normal).
        std: Standard deviation of the log-normal distribution.
    """
    # Convert mean/std of log-normal to underlying normal params
    variance = std**2
    sigma = np.sqrt(np.log(1 + variance / (mean**2)))
    mu = np.log(mean) - sigma**2 / 2
    return ScipyDistribution(stats.lognorm(s=sigma, scale=np.exp(mu)))


def draw_sample(
    n: int,
    distribution_name: str = "normal",
    distribution_params: dict[str, float | None] | None = None,
    rng: Generator | None = None,
) -> NDArray[np.float64]:
    """
    Sample values from a distribution.

    Args:
        n: Number of values.
        distribution_name: Name of the distribution to sample from.
        distribution_params: Parameter values to pass to the distribution sampler.
        rng: Random number generator.

    Returns:
        Array of shape (n,) with sampled values.
    """
    if rng is None:
        rng = get_rng()

    if distribution_params is None:
        distribution_params = {}

    distribution = registry.get_sampler(distribution_name, distribution_params)

    return distribution.sample(n, rng)
