"""
NRM estimator using MML-EM algorithm.

Implements the Nominal Response Model (Bock, 1972) estimator
with Marginal Maximum Likelihood via EM.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from analysis_service.irt.estimation.base import FittedModel, IRTEstimator
from analysis_service.irt.estimation.config import EstimationConfig
from analysis_service.irt.estimation.data_models import (
    ResponseMatrix,
)
from analysis_service.irt.estimation.nrm.constants import (
    DEFAULT_INITIAL_DISCRIMINATION,
)
from analysis_service.irt.estimation.nrm.gradients import (
    compute_nrm_probabilities,
    nrm_negative_expected_log_likelihood,
    nrm_negative_expected_log_likelihood_gradient,
)
from analysis_service.irt.estimation.nrm.parameters import NRMItemParameters


class NRMEstimator(IRTEstimator[NRMItemParameters]):
    """
    Nominal Response Model estimator using MML-EM.

    The NRM probability model:
        P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

    Identification: a_0 = 0, b_0 = 0 (reference category).

    Uses L-BFGS-B for M-step optimization with analytical gradients.
    """

    def __init__(self, config: EstimationConfig | None = None):
        """Initialize NRM estimator."""
        super().__init__(config)
        self._warmup_phase = False

    def fit(self, data: ResponseMatrix) -> FittedModel[NRMItemParameters]:
        """
        Fit NRM to response data using MML-EM with optional warmup.

        Runs 1-2 EM iterations with fixed discriminations before
        full optimization to reduce local maxima.

        Args:
            data: Response matrix with candidate responses.

        Returns:
            FittedModel with estimated parameters and fit statistics.
        """
        # Initialize parameters
        params = self._initialize(data)

        warmup_iters = self.config.convergence.warmup_iterations
        total_iterations = 0

        # Warmup phase: optimize only intercepts
        if warmup_iters > 0:
            self._warmup_phase = True
            if self.config.verbose:
                print(f"Running {warmup_iters} warmup iterations...")

            for iteration in range(warmup_iters):
                e_result = self._e_step(data, params)

                if self.config.verbose:
                    print(
                        f"Warmup {iteration + 1}: "
                        f"LL = {e_result.log_likelihood:.4f}"
                    )

                # M-step with fixed discriminations
                params = self._m_step_warmup(data, e_result.posteriors, params)
                total_iterations += 1

            self._warmup_phase = False

        # Main EM loop
        prev_ll = -np.inf
        convergence_status: Literal[
            "converged", "max_iterations", "failed"
        ] = "max_iterations"

        for iteration in range(self.config.convergence.max_em_iterations):
            e_result = self._e_step(data, params)

            if self.config.verbose:
                print(
                    f"Iteration {iteration + 1}: "
                    f"LL = {e_result.log_likelihood:.4f}"
                )

            if not np.isfinite(e_result.log_likelihood):
                convergence_status = "failed"
                break

            if self._check_convergence(e_result.log_likelihood, prev_ll):
                convergence_status = "converged"
                total_iterations += iteration + 1
                break

            prev_ll = e_result.log_likelihood
            params = self._m_step(data, e_result.posteriors, params)
        else:
            total_iterations += self.config.convergence.max_em_iterations

        return FittedModel(
            item_parameters=tuple(params),
            log_likelihood=e_result.log_likelihood,
            n_iterations=total_iterations,
            convergence_status=convergence_status,
            model_version=self.config.model_version,
        )

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
        - Initialize intercepts from category marginal frequencies (log-odds)
        - Initialize discriminations near 0.3-0.5 (reduces local maxima issues)

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

            if total == 0:
                # No valid responses - use uniform defaults
                item_params = NRMItemParameters.create_default(
                    item_id=item_idx, n_categories=data.n_categories
                )
            else:
                # Compute proportions with additive smoothing
                props = (counts + 0.5) / (total + 0.5 * data.n_categories)

                # Initialize intercepts from log-odds relative to reference category
                # This is the key initialization per UPDATES.md
                log_props = np.log(props + 1e-10)
                log_props_centered = log_props - log_props[0]
                intercepts = tuple(float(b) for b in log_props_centered)

                # Initialize discriminations near 0.3-0.5 (UPDATES.md recommendation)
                # Using a small constant avoids local maxima issues
                discriminations = tuple(
                    DEFAULT_INITIAL_DISCRIMINATION
                    for _ in range(data.n_categories)
                )

                # Enforce reference category constraint (a_0 = 0, b_0 = 0)
                discriminations = (0.0,) + discriminations[1:]
                intercepts = (0.0,) + intercepts[1:]

                item_params = NRMItemParameters(
                    item_id=item_idx,
                    discriminations=discriminations,
                    intercepts=intercepts,
                )

            params.append(item_params)

        return params

    def _compute_item_log_likelihood(
        self,
        responses: NDArray[np.int8],
        params: NRMItemParameters,
        theta: NDArray[np.float64],
        n_categories: int,
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
        nrm_params = params
        if not isinstance(nrm_params, NRMItemParameters):
            raise TypeError(f"Expected NRMItemParameters, got {type(params)}")

        n_candidates = len(responses)
        n_quad = len(theta)

        # Compute probabilities at each quadrature point
        # Shape: (n_quadrature, n_categories)
        discriminations = np.array(
            nrm_params.discriminations, dtype=np.float64
        )
        intercepts = np.array(nrm_params.intercepts, dtype=np.float64)
        probs = compute_nrm_probabilities(theta, discriminations, intercepts)

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
        nrm_current = current
        if not isinstance(nrm_current, NRMItemParameters):
            raise TypeError(f"Expected NRMItemParameters, got {type(current)}")

        # Filter to valid responses only
        valid_mask = ~missing_mask
        valid_responses = responses[valid_mask]
        valid_posteriors = posteriors[valid_mask, :]

        if len(valid_responses) == 0:
            # No valid responses - return current parameters
            return nrm_current

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
                nrm_current,
                theta,
                valid_posteriors,
                response_indicators,
                n_categories,
            )

        # Full optimization: optimize both discriminations and intercepts
        x0 = nrm_current.to_array()

        # Parameter bounds
        bounds_a = [self.config.bounds.discrimination] * n_free
        bounds_b = [self.config.bounds.intercept] * n_free
        bounds = bounds_a + bounds_b

        result = minimize(
            fun=nrm_negative_expected_log_likelihood,
            x0=x0,
            args=(theta, valid_posteriors, response_indicators, n_categories),
            method="L-BFGS-B",
            jac=nrm_negative_expected_log_likelihood_gradient,
            bounds=bounds,
            options={
                "maxiter": self.config.convergence.max_lbfgs_iterations,
                "ftol": self.config.convergence.lbfgs_tolerance,
            },
        )

        # Reconstruct parameters from optimized array
        return NRMItemParameters.from_array(item_idx, result.x, n_categories)

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
        """
        n_free = n_categories - 1
        fixed_a = np.array(current.discriminations[1:], dtype=np.float64)

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

        # Initial intercepts (free parameters only)
        b0 = np.array(current.intercepts[1:], dtype=np.float64)

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

        # Reconstruct with fixed discriminations and optimized intercepts
        new_intercepts = (0.0,) + tuple(float(b) for b in result.x)

        return NRMItemParameters(
            item_id=item_idx,
            discriminations=current.discriminations,
            intercepts=new_intercepts,
        )
