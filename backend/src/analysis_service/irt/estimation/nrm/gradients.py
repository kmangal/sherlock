"""
Analytical gradients for NRM estimation.

The NRM probability is:
    P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

For the expected complete-data log-likelihood (weighted by posteriors w_iq):
    Q = Σ_i Σ_q w_iq * log P(Y_i | θ_q)

Gradients:
    ∂Q / ∂a_k = Σ_i Σ_q w_iq * θ_q * (I[Y_i=k] - P(Y=k | θ_q))
    ∂Q / ∂b_k = Σ_i Σ_q w_iq * (I[Y_i=k] - P(Y=k | θ_q))
"""

import numpy as np
from numpy.typing import NDArray

# Exponent clipping bounds to prevent overflow
EXPONENT_CLIP_MIN = -30.0
EXPONENT_CLIP_MAX = 30.0


def compute_nrm_probabilities(
    theta: NDArray[np.float64],
    discriminations: NDArray[np.float64],
    intercepts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute NRM probabilities for all categories at given theta values.

    P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

    Args:
        theta: Ability values, shape (n_theta,).
        discriminations: Discrimination parameters, shape (n_categories,).
        intercepts: Intercept parameters, shape (n_categories,).

    Returns:
        Probabilities, shape (n_theta, n_categories).
    """
    # Compute logits: a_k * theta + b_k
    # Shape: (n_theta, n_categories)
    logits = np.outer(theta, discriminations) + intercepts[np.newaxis, :]

    # Clip for numerical stability
    logits = np.clip(logits, EXPONENT_CLIP_MIN, EXPONENT_CLIP_MAX)

    # Log-sum-exp trick for stability
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    exp_shifted = np.exp(shifted)
    probs: NDArray[np.float64] = exp_shifted / np.sum(
        exp_shifted, axis=1, keepdims=True
    )

    return probs


def nrm_negative_expected_log_likelihood(
    params: NDArray[np.float64],
    theta: NDArray[np.float64],
    posteriors: NDArray[np.float32],
    response_indicators: NDArray[np.float64],
    n_categories: int,
) -> float:
    """
    Compute negative expected complete-data log-likelihood for one item.

    This is the objective function for M-step optimization.

    Args:
        params: Free parameters [a_1, ..., a_{K-1}, b_1, ..., b_{K-1}].
        theta: Quadrature points, shape (n_quadrature,).
        posteriors: Posterior weights, shape (n_candidates, n_quadrature).
            Only includes candidates with valid responses.
        response_indicators: Response indicators, shape (n_candidates, n_categories).
            response_indicators[i, k] = 1 if candidate i chose category k.
        n_categories: Number of response categories.

    Returns:
        Negative expected log-likelihood (to minimize).
    """
    # Reconstruct full parameters with reference category
    n_free = n_categories - 1
    free_a = params[:n_free]
    free_b = params[n_free:]

    discriminations = np.zeros(n_categories, dtype=np.float64)
    intercepts = np.zeros(n_categories, dtype=np.float64)
    discriminations[1:] = free_a
    intercepts[1:] = free_b

    # Compute probabilities: shape (n_quadrature, n_categories)
    probs = compute_nrm_probabilities(theta, discriminations, intercepts)

    # Log probabilities (add small constant for stability)
    log_probs = np.log(probs + 1e-300)

    # Expected log-likelihood:
    # Σ_i Σ_q w_iq * Σ_k I[Y_i=k] * log P(k | θ_q)
    # = Σ_i Σ_q Σ_k w_iq * I[Y_i=k] * log P(k | θ_q)
    #
    # response_indicators: (n_candidates, n_categories)
    # posteriors: (n_candidates, n_quadrature)
    # log_probs: (n_quadrature, n_categories)
    #
    # For each candidate i and category k they chose:
    # contribution = Σ_q w_iq * log P(k | θ_q)
    #              = posteriors[i, :] @ log_probs[:, k]

    # Vectorized: (n_candidates, n_quadrature) @ (n_quadrature, n_categories)
    # = (n_candidates, n_categories)
    candidate_log_probs = posteriors @ log_probs

    # Element-wise multiply by response indicators and sum
    expected_ll = float(np.sum(response_indicators * candidate_log_probs))

    return -expected_ll


def nrm_negative_expected_log_likelihood_gradient(
    params: NDArray[np.float64],
    theta: NDArray[np.float64],
    posteriors: NDArray[np.float32],
    response_indicators: NDArray[np.float64],
    n_categories: int,
) -> NDArray[np.float64]:
    """
    Compute gradient of negative expected log-likelihood for one item.

    Gradients (for free parameters, k > 0):
        ∂Q / ∂a_k = Σ_i Σ_q w_iq * θ_q * (I[Y_i=k] - P(k | θ_q))
        ∂Q / ∂b_k = Σ_i Σ_q w_iq * (I[Y_i=k] - P(k | θ_q))

    Args:
        params: Free parameters [a_1, ..., a_{K-1}, b_1, ..., b_{K-1}].
        theta: Quadrature points, shape (n_quadrature,).
        posteriors: Posterior weights, shape (n_candidates, n_quadrature).
        response_indicators: Response indicators, shape (n_candidates, n_categories).
        n_categories: Number of response categories.

    Returns:
        Gradient of negative expected log-likelihood, shape (2 * (K-1),).
    """
    n_free = n_categories - 1
    free_a = params[:n_free]
    free_b = params[n_free:]

    # Reconstruct full parameters
    discriminations = np.zeros(n_categories, dtype=np.float64)
    intercepts = np.zeros(n_categories, dtype=np.float64)
    discriminations[1:] = free_a
    intercepts[1:] = free_b

    # Compute probabilities: (n_quadrature, n_categories)
    probs = compute_nrm_probabilities(theta, discriminations, intercepts)

    # Expected response indicators under posterior:
    # E[I[Y_i=k]] = Σ_q w_iq * I[Y_i=k] (this is just the observed indicator)
    # Expected probability:
    # E[P(k | θ)] = Σ_q w_iq * P(k | θ_q)
    #             = posteriors @ probs  -> (n_candidates, n_categories)

    expected_probs = posteriors @ probs  # (n_candidates, n_categories)

    # Difference: I[Y_i=k] - E[P(k | θ)]
    # But we need the weighted version:
    # Σ_i Σ_q w_iq * (I[Y_i=k] - P(k | θ_q))
    # = Σ_i (I[Y_i=k] * Σ_q w_iq - Σ_q w_iq * P(k | θ_q))
    # = Σ_i (I[Y_i=k] - expected_probs[i, k])  (since Σ_q w_iq = 1)

    diff = response_indicators - expected_probs  # (n_candidates, n_categories)

    # Gradient for intercepts (b_k):
    # ∂Q / ∂b_k = Σ_i (I[Y_i=k] - expected_probs[i, k])
    grad_b = np.sum(diff, axis=0)  # (n_categories,)

    # Gradient for discriminations (a_k):
    # ∂Q / ∂a_k = Σ_i Σ_q w_iq * θ_q * (I[Y_i=k] - P(k | θ_q))
    #
    # More efficient computation:
    # Let weighted_theta[i] = Σ_q w_iq * θ_q (posterior mean theta for candidate i)
    # Then for I[Y_i=k]:
    #   Σ_i I[Y_i=k] * weighted_theta[i]
    # But we also need: - Σ_i Σ_q w_iq * θ_q * P(k | θ_q)
    #                 = - Σ_i (Σ_q w_iq * θ_q * P(k | θ_q))

    # posteriors: (n_candidates, n_quadrature)
    # theta: (n_quadrature,)
    # probs: (n_quadrature, n_categories)

    # Term 1: Σ_i I[Y_i=k] * (Σ_q w_iq * θ_q)
    #       = Σ_i I[Y_i=k] * (posteriors[i] @ theta)
    weighted_theta = posteriors @ theta  # (n_candidates,)
    term1 = response_indicators.T @ weighted_theta  # (n_categories,)

    # Term 2: Σ_i Σ_q w_iq * θ_q * P(k | θ_q)
    #       = Σ_q θ_q * P(k | θ_q) * (Σ_i w_iq)
    # where Σ_i w_iq is the total posterior mass at quadrature point q
    total_posterior = np.sum(posteriors, axis=0)  # (n_quadrature,)
    # (n_quadrature,) * (n_quadrature, n_categories) summed over q
    term2 = (total_posterior * theta) @ probs  # (n_categories,)

    grad_a = term1 - term2  # (n_categories,)

    # Extract free parameter gradients (exclude reference category k=0)
    grad_free_a = grad_a[1:]
    grad_free_b = grad_b[1:]

    # Return negative gradient (we're minimizing)
    gradient = np.concatenate([grad_free_a, grad_free_b])
    neg_gradient: NDArray[np.float64] = -gradient
    return neg_gradient
