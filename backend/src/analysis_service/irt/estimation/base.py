"""
Abstract base classes for IRT estimation.

This module defines the extensible architecture for IRT model estimation:
- ItemParameters: Abstract base for model-specific parameters
- IRTEstimator: Abstract base with shared EM loop logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from analysis_service.irt.estimation.config import EstimationConfig
from analysis_service.irt.estimation.data_models import (
    ResponseMatrix,
)
from analysis_service.irt.estimation.parameters import ItemParameters
from analysis_service.irt.estimation.quadrature import (
    GaussHermiteQuadrature,
    get_quadrature,
)

P = TypeVar("P", bound=ItemParameters)


@dataclass
class EStepResult:
    """
    Results from the E-step of EM algorithm.

    Attributes:
        posteriors: Posterior weights, shape (n_candidates, n_quadrature_points).
            posteriors[i, q] = P(theta = theta_q | responses_i, params).
        log_likelihood: Marginal log-likelihood for current parameters.
    """

    posteriors: NDArray[np.float32]
    log_likelihood: float


@dataclass(frozen=True)
class FittedModel[P]:
    """
    Result of IRT model estimation.

    Attributes:
        item_parameters: Tuple of estimated item parameters, one per item.
        log_likelihood: Final marginal log-likelihood value.
        n_iterations: Number of EM iterations performed.
        convergence_status: Status indicating how estimation terminated.
        model_version: Version string for reproducibility tracking.
    """

    item_parameters: tuple[P, ...]
    log_likelihood: float
    n_iterations: int
    convergence_status: Literal["converged", "max_iterations", "failed"]
    model_version: str

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return len(self.item_parameters)

    @property
    def converged(self) -> bool:
        """Whether estimation converged successfully."""
        return self.convergence_status == "converged"


class IRTEstimator[P](ABC):
    """
    Abstract base class for IRT model estimators using MML-EM.

    Subclasses implement model-specific likelihood and M-step optimization.
    The EM loop and E-step are shared across all models.
    """

    def __init__(self, config: EstimationConfig | None = None):
        """
        Initialize estimator.

        Args:
            config: Estimation configuration. If None, uses defaults.
        """
        self.config = config or EstimationConfig()
        self._quadrature = get_quadrature(self.config.quadrature)

    @property
    def quadrature(self) -> GaussHermiteQuadrature:
        """Access quadrature points and weights."""
        return self._quadrature

    def fit(self, data: ResponseMatrix) -> FittedModel[P]:
        """
        Fit IRT model to response data using EM algorithm.

        Args:
            data: Response matrix with candidate responses.

        Returns:
            FittedModel with estimated parameters and fit statistics.
        """
        # Initialize parameters
        params = self._initialize(data)

        prev_ll = -np.inf
        convergence_status: Literal[
            "converged", "max_iterations", "failed"
        ] = "max_iterations"

        for iteration in range(self.config.convergence.max_em_iterations):
            # E-step: compute posteriors
            e_result = self._e_step(data, params)

            if self.config.verbose:
                print(
                    f"Iteration {iteration + 1}: "
                    f"LL = {e_result.log_likelihood:.4f}"
                )

            # Check for numerical issues
            if not np.isfinite(e_result.log_likelihood):
                convergence_status = "failed"
                break

            # Check convergence
            if self._check_convergence(e_result.log_likelihood, prev_ll):
                convergence_status = "converged"
                break

            prev_ll = e_result.log_likelihood

            # M-step: optimize parameters
            params = self._m_step(data, e_result.posteriors, params)

        return FittedModel[P](
            item_parameters=tuple(params),
            log_likelihood=e_result.log_likelihood,
            n_iterations=iteration + 1,
            convergence_status=convergence_status,
            model_version=self.config.model_version,
        )

    def _e_step(
        self,
        data: ResponseMatrix,
        params: list[P],
    ) -> EStepResult:
        """
        E-step: compute posterior distribution over abilities.

        For each candidate, compute:
            P(theta_q | responses) ∝ P(responses | theta_q) * P(theta_q)

        where P(theta_q) is the quadrature weight (prior).

        Args:
            data: Response matrix.
            params: Current item parameters.

        Returns:
            EStepResult with posteriors and marginal log-likelihood.
        """
        n_candidates = data.n_candidates
        n_quadrature = self._quadrature.n_points

        # Compute log-likelihood for each candidate at each quadrature point
        # Shape: (n_candidates, n_quadrature_points)
        log_lik = np.zeros((n_candidates, n_quadrature), dtype=np.float64)

        # Add log prior (quadrature weights)
        log_prior = np.log(self._quadrature.weights + 1e-300)
        log_lik += log_prior[np.newaxis, :]

        # Add log-likelihood contribution from each item
        for item_idx, item_params in enumerate(params):
            item_log_lik = self._compute_item_log_likelihood(
                data.responses[:, item_idx],
                item_params,
                self._quadrature.points,
                data.n_categories,
                data.missing_mask[:, item_idx],
            )
            log_lik += item_log_lik

        # Log-sum-exp for numerical stability
        max_log_lik = np.max(log_lik, axis=1, keepdims=True)
        log_lik_shifted = log_lik - max_log_lik

        # Posteriors (normalized)
        posteriors = np.exp(log_lik_shifted)
        row_sums = posteriors.sum(axis=1, keepdims=True)
        posteriors = posteriors / (row_sums + 1e-300)

        # Marginal log-likelihood
        # LL = sum over candidates of log(sum over theta of P(responses|theta) * P(theta))
        log_marginal = max_log_lik.squeeze() + np.log(
            row_sums.squeeze() + 1e-300
        )
        total_ll = float(np.sum(log_marginal))

        # Cast posteriors to float32 to reduce memory (per UPDATES.md)
        # Precision is sufficient for EM weights
        posteriors_f32: NDArray[np.float32] = posteriors.astype(np.float32)

        return EStepResult(posteriors=posteriors_f32, log_likelihood=total_ll)

    def _m_step(
        self,
        data: ResponseMatrix,
        posteriors: NDArray[np.float32],
        current_params: list[P],
    ) -> list[P]:
        """
        M-step: optimize item parameters given posteriors.

        Args:
            data: Response matrix.
            posteriors: Posterior weights from E-step.
            current_params: Current parameter estimates.

        Returns:
            Updated parameter estimates.
        """
        new_params = []

        for item_idx, current in enumerate(current_params):
            new_item_params = self._optimize_item(
                item_idx=item_idx,
                responses=data.responses[:, item_idx],
                missing_mask=data.missing_mask[:, item_idx],
                posteriors=posteriors,
                current=current,
                n_categories=data.n_categories,
            )
            new_params.append(new_item_params)

        return new_params

    def _check_convergence(self, current_ll: float, prev_ll: float) -> bool:
        """
        Check if EM has converged based on log-likelihood change.

        Args:
            current_ll: Current log-likelihood.
            prev_ll: Previous log-likelihood.

        Returns:
            True if converged.
        """
        if prev_ll == -np.inf:
            return False

        # Relative change in log-likelihood
        rel_change = abs(current_ll - prev_ll) / (abs(prev_ll) + 1e-10)
        return bool(rel_change < self.config.convergence.em_tolerance)

    @abstractmethod
    def _initialize(self, data: ResponseMatrix) -> list[P]:
        """
        Initialize item parameters.

        Args:
            data: Response matrix.

        Returns:
            List of initial parameter estimates.
        """
        ...

    @abstractmethod
    def _compute_item_log_likelihood(
        self,
        responses: NDArray[np.int8],
        params: P,
        theta: NDArray[np.float64],
        n_categories: int,
        missing_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """
        Compute log-likelihood contribution of one item.

        Args:
            responses: Responses to this item, shape (n_candidates,).
            params: Item parameters.
            theta: Quadrature points, shape (n_quadrature,).
            n_categories: Number of response categories.
            missing_mask: Boolean mask where True = missing, shape (n_candidates,).

        Returns:
            Log-likelihood matrix, shape (n_candidates, n_quadrature).
            Missing responses contribute 0 to log-likelihood.
        """
        ...

    @abstractmethod
    def _optimize_item(
        self,
        item_idx: int,
        responses: NDArray[np.int8],
        missing_mask: NDArray[np.bool_],
        posteriors: NDArray[np.float32],
        current: P,
        n_categories: int,
    ) -> P:
        """
        Optimize parameters for one item in M-step.

        Args:
            item_idx: Index of the item.
            responses: Responses to this item, shape (n_candidates,).
            missing_mask: Boolean mask where True = missing.
            posteriors: Posterior weights, shape (n_candidates, n_quadrature).
            current: Current parameter estimates.
            n_categories: Number of response categories.

        Returns:
            Optimized item parameters.
        """
        ...
