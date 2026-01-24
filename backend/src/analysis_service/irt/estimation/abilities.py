"""
Ability estimation for IRT models.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis_service.irt.estimation.config import EstimationConfig
from analysis_service.irt.estimation.data_models import (
    ResponseMatrix,
)
from analysis_service.irt.estimation.estimator import IRTEstimationResult
from analysis_service.irt.estimation.parameters import NRMItemParameters
from analysis_service.irt.estimation.quadrature import get_quadrature


@dataclass(frozen=True)
class AbilityEstimates:
    """
    Ability estimates for candidates.

    Attributes:
        eap: Expected A Posteriori (posterior mean) estimates, shape (n_candidates,).
        se: Standard errors (posterior standard deviation), shape (n_candidates,).
    """

    eap: NDArray[np.float64]
    se: NDArray[np.float64]

    @property
    def n_candidates(self) -> int:
        """Number of candidates."""
        return len(self.eap)


def estimate_abilities(
    data: ResponseMatrix,
    model: IRTEstimationResult,
    config: EstimationConfig | None = None,
) -> AbilityEstimates:
    """
    Estimate abilities using Expected A Posteriori (EAP) method.

    EAP estimates are the posterior mean of ability given the responses
    and estimated item parameters:
        θ_EAP = E[θ | responses] = Σ_q θ_q * P(θ_q | responses)

    Standard errors are the posterior standard deviation:
        SE = sqrt(E[θ² | responses] - (E[θ | responses])²)

    Args:
        data: Response matrix.
        model: Fitted IRT model with item parameters.
        config: Estimation configuration. Uses defaults if None.

    Returns:
        AbilityEstimates with EAP estimates and standard errors.
    """
    if config is None:
        config = EstimationConfig()

    quadrature = get_quadrature(config.quadrature)
    theta = quadrature.points
    weights = quadrature.weights

    n_candidates = data.n_candidates
    n_quad = len(theta)

    # Compute log-likelihood for each candidate at each quadrature point
    log_lik = np.zeros((n_candidates, n_quad), dtype=np.float64)

    # Add log prior (quadrature weights)
    log_prior = np.log(weights + 1e-300)
    log_lik += log_prior[np.newaxis, :]

    # Add log-likelihood from each item
    for item_idx, item_params in enumerate(model.item_parameters):
        item_log_lik = _compute_item_log_likelihood_for_abilities(
            responses=data.responses[:, item_idx],
            params=item_params,
            theta=theta,
            missing_mask=data.missing_mask[:, item_idx],
        )
        log_lik += item_log_lik

    # Compute posteriors using log-sum-exp
    max_log_lik = np.max(log_lik, axis=1, keepdims=True)
    log_lik_shifted = log_lik - max_log_lik
    posteriors = np.exp(log_lik_shifted)
    posteriors = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-300)

    # EAP: posterior mean
    eap = posteriors @ theta

    # SE: posterior standard deviation
    # E[θ²] - E[θ]²
    theta_squared = theta**2
    eap_squared = posteriors @ theta_squared
    variance = eap_squared - eap**2
    # Ensure non-negative (numerical precision)
    variance = np.maximum(variance, 0.0)
    se = np.sqrt(variance)

    return AbilityEstimates(eap=eap, se=se)


def _compute_item_log_likelihood_for_abilities(
    responses: NDArray[np.int8],
    params: NRMItemParameters,
    theta: NDArray[np.float64],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Compute log-likelihood contribution of one item for ability estimation.

    Uses the item parameters' compute_probabilities method, which is
    implemented by all ItemParameters subclasses (NRM, 3PL, etc.).

    Args:
        responses: Responses to this item, shape (n_candidates,).
        params: Item parameters (any subclass of ItemParameters).
        theta: Quadrature points, shape (n_quadrature,).
        n_categories: Number of response categories.
        missing_mask: Boolean mask where True = missing.

    Returns:
        Log-likelihood matrix, shape (n_candidates, n_quadrature).
    """
    n_candidates = len(responses)
    n_quad = len(theta)

    # Compute probabilities at each quadrature point using the item's method
    probs = params.compute_probabilities(theta)

    # Log probabilities
    log_probs = np.log(probs + 1e-300)

    # Initialize log-likelihood contribution (0 for missing)
    log_lik = np.zeros((n_candidates, n_quad), dtype=np.float64)

    # For valid responses, look up log probability
    valid_mask = ~missing_mask
    valid_responses = responses[valid_mask].astype(np.int64)

    log_lik[valid_mask, :] = log_probs[:, valid_responses].T

    return log_lik
