"""
NRM estimator using MML-EM algorithm.

Implements the Nominal Response Model (Bock, 1972) estimator
with Marginal Maximum Likelihood via EM.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from analysis_service.core.utils import get_rng
from analysis_service.irt.estimation.config import EstimationConfig
from analysis_service.irt.estimation.data_models import (
    ResponseMatrix,
)
from analysis_service.irt.estimation.enums import ConvergenceStatus
from analysis_service.irt.estimation.gradients import (
    nrm_neg_ell_gradient_with_penalty,
    nrm_neg_ell_with_penalty,
    nrm_negative_expected_log_likelihood,
    nrm_negative_expected_log_likelihood_gradient,
)
from analysis_service.irt.estimation.parameters import NRMItemParameters
from analysis_service.irt.estimation.quadrature import (
    GaussHermiteQuadrature,
    get_quadrature,
)

logger = logging.getLogger(__name__)


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
class IRTEstimationResult:
    """
    Result of IRT model estimation.

    Attributes:
        item_parameters: Tuple of estimated item parameters, one per item.
        log_likelihood: Final marginal log-likelihood value.
        n_iterations: Number of EM iterations performed.
        convergence_status: Status indicating how estimation terminated.
        model_version: Version string for reproducibility tracking.
    """

    item_parameters: tuple[NRMItemParameters, ...]
    log_likelihood: float
    n_iterations: int
    convergence_status: ConvergenceStatus
    model_version: str

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return len(self.item_parameters)

    @property
    def converged(self) -> bool:
        """Whether estimation converged successfully."""
        return self.convergence_status == ConvergenceStatus.CONVERGED


class NRMEstimator:
    """
    Nominal Response Model estimator using MML-EM.

    The NRM probability model:
        P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

    Identification: sum-to-zero (Σa_k = 0, Σb_k = 0).

    Uses L-BFGS-B for M-step optimization with analytical gradients.
    When correct answers are known, applies a soft penalty to encourage
    a_correct > a_distractor.
    """

    def __init__(
        self,
        config: EstimationConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Initialize NRM estimator."""
        self.config = config or EstimationConfig()
        if rng is None:
            self.rng = get_rng()
        else:
            self.rng = rng

        self._quadrature = get_quadrature(self.config.quadrature)

        self._warmup_phase = False
        self._correct_answers: list[int | None] = []

    @property
    def quadrature(self) -> GaussHermiteQuadrature:
        """Access quadrature points and weights."""
        return self._quadrature

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

        # Absolute change in log-likelihood
        abs_change = abs(current_ll - prev_ll)
        return bool(abs_change < self.config.convergence.em_tolerance)

    def _e_step(
        self,
        data: ResponseMatrix,
        params: list[NRMItemParameters],
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

    def fit(
        self,
        data: ResponseMatrix,
        correct_answers: Sequence[int | None] | None = None,
    ) -> IRTEstimationResult:
        """
        Fit NRM to response data using MML-EM with optional warmup.

        Runs 1-2 EM iterations with fixed discriminations before
        full optimization to reduce local maxima.

        When correct_answers are provided, applies a soft penalty to encourage
        a_correct > a_distractor during M-step optimization.

        Args:
            data: Response matrix with candidate responses.
            correct_answers: Optional sequence of correct answer indices
                (0-indexed) for each item. Use None for unknown.

        Returns:
            FittedModel with estimated parameters and fit statistics.
        """
        # Store correct answers
        if correct_answers is None:
            self._correct_answers = [None] * data.n_items
        else:
            self._correct_answers = list(correct_answers)

        # Initialize parameters
        params = self._initialize(data)

        warmup_iters = self.config.convergence.warmup_iterations
        total_iterations = 0

        # Warmup phase: optimize only intercepts
        if warmup_iters > 0:
            self._warmup_phase = True
            logger.debug(f"Running {warmup_iters} warmup iterations...")

            for iteration in range(warmup_iters):
                e_result = self._e_step(data, params)
                logger.debug(
                    f"Warmup {iteration + 1}: "
                    f"LL = {e_result.log_likelihood:.4f}"
                )

                # M-step with fixed discriminations
                params = self._m_step_warmup(data, e_result.posteriors, params)
                total_iterations += 1

            self._warmup_phase = False

        # Main EM loop
        prev_ll = -np.inf
        convergence_status = ConvergenceStatus.MAX_ITERATIONS

        for iteration in range(self.config.convergence.max_em_iterations):
            e_result = self._e_step(data, params)
            logger.debug(
                f"Iteration {iteration + 1}: "
                f"LL = {e_result.log_likelihood:.4f}"
            )

            if not np.isfinite(e_result.log_likelihood):
                convergence_status = ConvergenceStatus.FAILED
                break

            if self._check_convergence(e_result.log_likelihood, prev_ll):
                convergence_status = ConvergenceStatus.CONVERGED
                total_iterations += iteration + 1
                break

            prev_ll = e_result.log_likelihood
            params = self._m_step(data, e_result.posteriors, params)
        else:
            total_iterations += self.config.convergence.max_em_iterations

        return IRTEstimationResult(
            item_parameters=tuple(params),
            log_likelihood=e_result.log_likelihood,
            n_iterations=total_iterations,
            convergence_status=convergence_status,
            model_version=self.config.model_version,
        )

    def _m_step(
        self,
        data: ResponseMatrix,
        posteriors: NDArray[np.float32],
        current_params: list[NRMItemParameters],
    ) -> list[NRMItemParameters]:
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

    def _m_step_warmup(
        self,
        data: ResponseMatrix,
        posteriors: NDArray[np.float32],
        current_params: list[NRMItemParameters],
    ) -> list[NRMItemParameters]:
        """M-step during warmup: only optimize intercepts."""
        new_params = []

        for item_idx, current in enumerate(current_params):
            new_item_params = self._optimize_item(
                item_idx=item_idx,
                responses=data.responses[:, item_idx],
                missing_mask=data.missing_mask[:, item_idx],
                posteriors=posteriors,
                current=current,
                n_categories=data.n_categories,
                fix_discriminations=True,
            )
            new_params.append(new_item_params)

        return new_params

    def _initialize(self, data: ResponseMatrix) -> list[NRMItemParameters]:
        """
        Initialize parameters with sum-to-zero constraint.

        - Initialize intercepts from category marginal frequencies (centered to mean)
        - Initialize discriminations to zeros (satisfies sum-to-zero)

        Args:
            data: Response matrix.

        Returns:
            List of initial NRMItemParameters.
        """
        params = []

        for item_idx in range(data.n_items):
            # Get response counts for this item
            counts = data.item_response_counts(item_idx)
            total = counts.sum()

            correct_answer = self._correct_answers[item_idx]

            if total == 0:
                # No valid responses - use uniform defaults
                item_params = NRMItemParameters.create_default(
                    item_id=item_idx,
                    n_categories=data.n_categories,
                    correct_answer=correct_answer,
                )
            else:
                # Compute proportions with additive smoothing
                props = (counts + 0.5) / (total + 0.5 * data.n_categories)

                # Initialize intercepts from log-proportions centered to mean (sum-to-zero)
                log_props = np.log(props + 1e-10)
                log_props_centered = log_props - np.mean(log_props)
                intercepts = tuple(float(b) for b in log_props_centered)

                # Initialize discriminations, with sum to zero constraint
                d_sampled = self.rng.standard_normal(data.n_categories).clip(
                    -1, 1
                )
                d_centered = d_sampled - np.mean(d_sampled)
                discriminations = tuple(d_centered)

                item_params = NRMItemParameters(
                    item_id=item_idx,
                    discriminations=discriminations,
                    intercepts=intercepts,
                    correct_answer=correct_answer,
                )

            params.append(item_params)

        return params

    def _compute_item_log_likelihood(
        self,
        responses: NDArray[np.int8],
        params: NRMItemParameters,
        theta: NDArray[np.float64],
        missing_mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """
        Compute log-likelihood contribution of one item.

        Args:
            responses: Responses to this item, shape (n_candidates,).
            params: NRM item parameters.
            theta: Quadrature points, shape (n_quadrature,).
            n_categories: Number of response categories.
            missing_mask: Boolean mask where True = missing.

        Returns:
            Log-likelihood matrix, shape (n_candidates, n_quadrature).
        """
        n_candidates = len(responses)
        n_quad = len(theta)

        # Compute probabilities at each quadrature point using item's method
        # Shape: (n_quadrature, n_categories)
        probs = params.compute_probabilities(theta)

        # Log probabilities
        log_probs = np.log(probs + 1e-300)  # (n_quadrature, n_categories)

        # Initialize log-likelihood contribution (0 for missing)
        log_lik = np.zeros((n_candidates, n_quad), dtype=np.float64)

        # For valid responses, look up log probability of observed response
        valid_mask = ~missing_mask
        valid_responses = responses[valid_mask].astype(np.int64)

        # log_probs[q, r] for each valid candidate's response r
        # We need to index: log_probs[:, valid_responses] gives (n_quad, n_valid)
        # Transpose to get (n_valid, n_quad)
        log_lik[valid_mask, :] = log_probs[:, valid_responses].T

        return log_lik

    def _optimize_item(
        self,
        item_idx: int,
        responses: NDArray[np.int8],
        missing_mask: NDArray[np.bool_],
        posteriors: NDArray[np.float32],
        current: NRMItemParameters,
        n_categories: int,
        fix_discriminations: bool = False,
    ) -> NRMItemParameters:
        """
        Optimize parameters for one item using L-BFGS-B.

        When correct_answer is known (and not in warmup), applies a soft penalty
        to encourage a_correct > a_distractor.

        Args:
            item_idx: Index of the item.
            responses: Responses to this item, shape (n_candidates,).
            missing_mask: Boolean mask where True = missing.
            posteriors: Posterior weights, shape (n_candidates, n_quadrature).
            current: Current parameter estimates.
            n_categories: Number of response categories.
            fix_discriminations: If True, only optimize intercepts (for warmup).

        Returns:
            Optimized NRMItemParameters.
        """
        # Filter to valid responses only
        valid_mask = ~missing_mask
        valid_responses = responses[valid_mask]
        valid_posteriors = posteriors[valid_mask, :]

        if len(valid_responses) == 0:
            # No valid responses - return current parameters
            return current

        # Create response indicator matrix
        # Shape: (n_valid, n_categories)
        n_valid = len(valid_responses)
        response_indicators = np.zeros(
            (n_valid, n_categories), dtype=np.float64
        )
        response_indicators[
            np.arange(n_valid), valid_responses.astype(np.int64)
        ] = 1.0

        theta = self._quadrature.points
        n_free = n_categories - 1

        if fix_discriminations:
            # Warmup phase: only optimize intercepts, keep discriminations fixed
            return self._optimize_intercepts_only(
                item_idx,
                current,
                theta,
                valid_posteriors,
                response_indicators,
                n_categories,
            )

        correct_answer = self._correct_answers[item_idx]
        x0 = current.to_array()

        # Parameter bounds (allow negative discriminations for sum-to-zero)
        bounds_a = [self.config.bounds.discrimination] * n_free
        bounds_b = [self.config.bounds.intercept] * n_free
        bounds = bounds_a + bounds_b

        # Use penalty objective when correct answer is known
        if correct_answer is not None:
            lambda_penalty = self.config.penalty.lambda_penalty
            margin = self.config.penalty.margin

            result = minimize(
                fun=nrm_neg_ell_with_penalty,
                x0=x0,
                args=(
                    theta,
                    valid_posteriors,
                    response_indicators,
                    n_categories,
                    correct_answer,
                    lambda_penalty,
                    margin,
                ),
                method="L-BFGS-B",
                jac=nrm_neg_ell_gradient_with_penalty,
                bounds=bounds,
                options={
                    "maxiter": self.config.convergence.max_lbfgs_iterations,
                    "ftol": self.config.convergence.lbfgs_tolerance,
                },
            )
        else:
            # No correct answer - optimize without penalty
            result = minimize(
                fun=nrm_negative_expected_log_likelihood,
                x0=x0,
                args=(
                    theta,
                    valid_posteriors,
                    response_indicators,
                    n_categories,
                ),
                method="L-BFGS-B",
                jac=nrm_negative_expected_log_likelihood_gradient,
                bounds=bounds,
                options={
                    "maxiter": self.config.convergence.max_lbfgs_iterations,
                    "ftol": self.config.convergence.lbfgs_tolerance,
                },
            )

        # Reconstruct parameters from optimized array
        return NRMItemParameters.from_array(
            item_idx, result.x, n_categories, correct_answer=correct_answer
        )

    def _optimize_intercepts_only(
        self,
        item_idx: int,
        current: NRMItemParameters,
        theta: NDArray[np.float64],
        posteriors: NDArray[np.float32],
        response_indicators: NDArray[np.float64],
        n_categories: int,
    ) -> NRMItemParameters:
        """
        Optimize only intercepts, keeping discriminations fixed.

        Used during warmup phase to reduce local maxima issues.
        Uses sum-to-zero constraint for intercepts.
        """
        n_free = n_categories - 1
        # Fixed discriminations (first K-1 for sum-to-zero)
        fixed_a = np.array(current.discriminations[:-1], dtype=np.float64)

        def objective(b_free: NDArray[np.float64]) -> float:
            # Build full parameter array with fixed discriminations
            params = np.concatenate([fixed_a, b_free])
            return float(
                nrm_negative_expected_log_likelihood(
                    params,
                    theta,
                    posteriors,
                    response_indicators,
                    n_categories,
                )
            )

        def gradient(b_free: NDArray[np.float64]) -> NDArray[np.float64]:
            params = np.concatenate([fixed_a, b_free])
            full_grad = nrm_negative_expected_log_likelihood_gradient(
                params, theta, posteriors, response_indicators, n_categories
            )
            # Return only the intercept gradients
            intercept_grad: NDArray[np.float64] = full_grad[n_free:]
            return intercept_grad

        # Initial intercepts (first K-1 for sum-to-zero)
        b0 = np.array(current.intercepts[:-1], dtype=np.float64)

        # Optimize intercepts only
        bounds_b = [self.config.bounds.intercept] * n_free

        result = minimize(
            fun=objective,
            x0=b0,
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds_b,
            options={
                "maxiter": self.config.convergence.max_lbfgs_iterations,
                "ftol": self.config.convergence.lbfgs_tolerance,
            },
        )

        # Reconstruct with fixed discriminations and optimized intercepts (sum-to-zero)
        opt_b_free = result.x
        b_last = -np.sum(opt_b_free)
        new_intercepts = tuple(float(b) for b in opt_b_free) + (float(b_last),)

        return NRMItemParameters(
            item_id=item_idx,
            discriminations=current.discriminations,
            intercepts=new_intercepts,
            correct_answer=current.correct_answer,
        )
