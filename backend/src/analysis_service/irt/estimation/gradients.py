"""
Analytical gradients for NRM estimation.

The NRM probability is:
    P(Y=k | θ) = exp(a_k * θ + b_k) / Σ_h exp(a_h * θ + b_h)

With K response categories (0..K-1) and 1 missing category (K), total K+1 categories.
Identification: sum-to-zero (Σa_k = 0, Σb_k = 0) across all K+1 categories.
Free parameters: first K categories (0..K-1); missing (K) derived as a_K = -Σa_k.

For the expected complete-data log-likelihood (weighted by posteriors w_iq):
    Q = Σ_i Σ_q w_iq * log P(Y_i | θ_q)

Gradients (for full parameters):
    ∂Q / ∂a_k = Σ_i Σ_q w_iq * θ_q * (I[Y_i=k] - P(Y=k | θ_q))
    ∂Q / ∂b_k = Σ_i Σ_q w_iq * (I[Y_i=k] - P(Y=k | θ_q))

Chain rule for sum-to-zero (free params k=0..K-1):
    ∂Q / ∂a_k^free = ∂Q / ∂a_k - ∂Q / ∂a_K
"""

import numpy as np
from numba import njit, prange  # type: ignore
from numpy.typing import NDArray

# Exponent clipping bounds to prevent overflow
EXPONENT_CLIP_MIN = -30.0
EXPONENT_CLIP_MAX = 30.0


@njit(parallel=True)  # type: ignore
def rowwise_max(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """numba compatible implementation of np.max(a, axis=1, keepdims=True)"""
    n, m = a.shape
    out = np.empty((n, 1))  # keepdims=True equivalent

    # Run each row in parallel
    for i in prange(n):
        mx = a[i, 0]
        for j in range(1, m):
            if a[i, j] > mx:
                mx = a[i, j]
        out[i, 0] = mx
    return out


@njit(parallel=True)  # type: ignore
def rowwise_sum(a: NDArray[np.float64]) -> NDArray[np.float64]:
    n, m = a.shape
    out = np.empty((n, 1), dtype=a.dtype)

    for i in prange(n):
        s = 0.0
        for j in range(m):
            s += a[i, j]
        out[i, 0] = s

    return out


@njit  # type: ignore
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
    max_logits = rowwise_max(logits)
    shifted = logits - max_logits
    exp_shifted = np.exp(shifted)
    probs: NDArray[np.float64] = exp_shifted / rowwise_sum(exp_shifted)

    return probs


@njit  # type: ignore
def nrm_negative_expected_log_likelihood(
    params: NDArray[np.float64],
    theta: NDArray[np.float64],
    posteriors: NDArray[np.float64],
    response_indicators: NDArray[np.float64],
    n_total_categories: int,
) -> float:
    """
    Compute negative expected complete-data log-likelihood for one item.

    This is the objective function for M-step optimization.

    Args:
        params: Free parameters [a_0, ..., a_{K-1}, b_0, ..., b_{K-1}].
            Missing category (K) derived from sum-to-zero constraint.
        theta: Quadrature points, shape (n_quadrature,).
        posteriors: Posterior weights, shape (n_candidates, n_quadrature).
        response_indicators: Response indicators, shape (n_candidates, n_total_categories).
            response_indicators[i, k] = 1 if candidate i chose category k.
            Category K is the missing category.
        n_total_categories: Total number of categories (K+1, includes missing).

    Returns:
        Negative expected log-likelihood (to minimize).
    """
    # Reconstruct full parameters with sum-to-zero constraint
    n_free = n_total_categories - 1
    free_a = params[:n_free]
    free_b = params[n_free:]

    discriminations = np.zeros(n_total_categories, dtype=np.float64)
    intercepts = np.zeros(n_total_categories, dtype=np.float64)
    discriminations[:-1] = free_a
    discriminations[-1] = -np.sum(free_a)
    intercepts[:-1] = free_b
    intercepts[-1] = -np.sum(free_b)

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


@njit  # type: ignore
def nrm_negative_expected_log_likelihood_gradient(
    params: NDArray[np.float64],
    theta: NDArray[np.float64],
    posteriors: NDArray[np.float64],
    response_indicators: NDArray[np.float64],
    n_total_categories: int,
) -> NDArray[np.float64]:
    """
    Compute gradient of negative expected log-likelihood for one item.

    Full gradients:
        ∂Q / ∂a_k = Σ_i Σ_q w_iq * θ_q * (I[Y_i=k] - P(k | θ_q))
        ∂Q / ∂b_k = Σ_i Σ_q w_iq * (I[Y_i=k] - P(k | θ_q))

    Chain rule for sum-to-zero (free params k=0..K-1):
        ∂Q / ∂a_k^free = ∂Q / ∂a_k - ∂Q / ∂a_K

    Args:
        params: Free parameters [a_0, ..., a_{K-1}, b_0, ..., b_{K-1}].
        theta: Quadrature points, shape (n_quadrature,).
        posteriors: Posterior weights, shape (n_candidates, n_quadrature).
        response_indicators: Response indicators, shape (n_candidates, n_total_categories).
        n_total_categories: Total number of categories (K+1, includes missing).

    Returns:
        Gradient of negative expected log-likelihood, shape (2 * K,).
    """
    n_free = n_total_categories - 1
    free_a = params[:n_free]
    free_b = params[n_free:]

    # Reconstruct full parameters with sum-to-zero constraint
    discriminations = np.zeros(n_total_categories, dtype=np.float64)
    intercepts = np.zeros(n_total_categories, dtype=np.float64)
    discriminations[:-1] = free_a
    discriminations[-1] = -np.sum(free_a)
    intercepts[:-1] = free_b
    intercepts[-1] = -np.sum(free_b)

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

    # Apply chain rule for sum-to-zero constraint:
    # ∂L / ∂a_k^free = ∂L / ∂a_k - ∂L / ∂a_{K-1}
    # because a_{K-1} = -Σa_k, so ∂a_{K-1}/∂a_k = -1
    grad_free_a = grad_a[:-1] - grad_a[-1]
    grad_free_b = grad_b[:-1] - grad_b[-1]

    # Return negative gradient (we're minimizing)
    gradient = np.concatenate((grad_free_a, grad_free_b))
    neg_gradient: NDArray[np.float64] = -gradient
    return neg_gradient


@njit  # type: ignore
def compute_correct_answer_penalty(
    discriminations: NDArray[np.float64],
    correct_answer: int,
    lambda_penalty: float,
    margin: float,
) -> float:
    """
    Compute squared hinge penalty encouraging a_correct > a_distractor.

    Penalty = λ * Σ_{j≠correct} max(0, a_j - a_correct + margin)²

    Args:
        discriminations: Full discrimination parameters, shape (n_categories,).
        correct_answer: Index of the correct answer.
        lambda_penalty: Penalty weight.
        margin: Minimum margin between correct and distractor discriminations.

    Returns:
        Penalty value (non-negative).
    """
    a_correct = discriminations[correct_answer]
    penalty = 0.0

    for j, a_j in enumerate(discriminations):
        if j != correct_answer:
            violation = a_j - a_correct + margin
            if violation > 0:
                penalty += violation * violation

    return lambda_penalty * penalty


@njit  # type: ignore
def compute_correct_answer_penalty_gradient(
    discriminations: NDArray[np.float64],
    correct_answer: int,
    lambda_penalty: float,
    margin: float,
) -> NDArray[np.float64]:
    """
    Compute gradient of squared hinge penalty w.r.t. full discriminations.

    Penalty = λ * Σ_{j≠correct} max(0, a_j - a_correct + margin)²

    ∂P/∂a_j = 2λ * max(0, a_j - a_correct + margin)  for j ≠ correct
    ∂P/∂a_correct = -2λ * Σ_{j≠correct} max(0, a_j - a_correct + margin)

    Args:
        discriminations: Full discrimination parameters, shape (n_categories,).
        correct_answer: Index of the correct answer.
        lambda_penalty: Penalty weight.
        margin: Minimum margin between correct and distractor discriminations.

    Returns:
        Gradient of penalty w.r.t. discriminations, shape (n_categories,).
    """
    n_categories = len(discriminations)
    a_correct = discriminations[correct_answer]
    grad = np.zeros(n_categories, dtype=np.float64)

    for j, a_j in enumerate(discriminations):
        if j != correct_answer:
            violation = a_j - a_correct + margin
            if violation > 0:
                # ∂P/∂a_j = 2λ * violation
                grad[j] = 2 * lambda_penalty * violation
                # ∂P/∂a_correct accumulates -2λ * violation
                grad[correct_answer] -= 2 * lambda_penalty * violation

    return grad


@njit  # type: ignore
def nrm_neg_ell_with_penalty(
    params: NDArray[np.float64],
    theta: NDArray[np.float64],
    posteriors: NDArray[np.float64],
    response_indicators: NDArray[np.float64],
    n_total_categories: int,
    correct_answer: int,
    lambda_penalty: float,
    margin: float,
) -> float:
    """
    Compute negative expected log-likelihood plus correct answer penalty.

    The penalty only applies to response categories (0..K-1), not the missing
    category (K).

    Args:
        params: Free parameters [a_0, ..., a_{K-1}, b_0, ..., b_{K-1}].
        theta: Quadrature points.
        posteriors: Posterior weights.
        response_indicators: Response indicators.
        n_total_categories: Total number of categories (K+1, includes missing).
        correct_answer: Index of correct answer (must be < K).
        lambda_penalty: Penalty weight.
        margin: Minimum margin for penalty.

    Returns:
        Negative expected log-likelihood plus penalty.
    """
    # Get base negative ELL
    neg_ell = nrm_negative_expected_log_likelihood(
        params, theta, posteriors, response_indicators, n_total_categories
    )

    # Reconstruct full discriminations for penalty
    n_free = n_total_categories - 1
    free_a = params[:n_free]
    discriminations = np.zeros(n_total_categories, dtype=np.float64)
    discriminations[:-1] = free_a
    discriminations[-1] = -np.sum(free_a)

    # Add penalty (excludes missing category)
    n_response_categories = n_total_categories - 1
    penalty = compute_correct_answer_penalty(
        discriminations[:n_response_categories],
        correct_answer,
        lambda_penalty,
        margin,
    )

    return float(neg_ell + penalty)


@njit  # type: ignore
def nrm_neg_ell_gradient_with_penalty(
    params: NDArray[np.float64],
    theta: NDArray[np.float64],
    posteriors: NDArray[np.float64],
    response_indicators: NDArray[np.float64],
    n_total_categories: int,
    correct_answer: int,
    lambda_penalty: float,
    margin: float,
) -> NDArray[np.float64]:
    """
    Compute gradient of negative ELL plus penalty w.r.t. free parameters.

    The penalty only applies to response categories (0..K-1), not the missing
    category (K).

    Args:
        params: Free parameters [a_0, ..., a_{K-1}, b_0, ..., b_{K-1}].
        theta: Quadrature points.
        posteriors: Posterior weights.
        response_indicators: Response indicators.
        n_total_categories: Total number of categories (K+1, includes missing).
        correct_answer: Index of correct answer (must be < K).
        lambda_penalty: Penalty weight.
        margin: Minimum margin for penalty.

    Returns:
        Gradient of negative ELL plus penalty, shape (2 * K,).
    """
    # Get base gradient
    grad = nrm_negative_expected_log_likelihood_gradient(
        params, theta, posteriors, response_indicators, n_total_categories
    )

    # Reconstruct full discriminations for penalty gradient
    n_free = n_total_categories - 1
    free_a = params[:n_free]
    discriminations = np.zeros(n_total_categories, dtype=np.float64)
    discriminations[:-1] = free_a
    discriminations[-1] = -np.sum(free_a)

    # Compute penalty gradient w.r.t. response category discriminations only
    # (excludes missing category)
    n_response_categories = n_total_categories - 1
    penalty_grad_response = compute_correct_answer_penalty_gradient(
        discriminations[:n_response_categories],
        correct_answer,
        lambda_penalty,
        margin,
    )

    # All response categories are free params, so no chain rule needed for them
    # Just add the penalty gradient directly
    grad[:n_response_categories] += penalty_grad_response

    assert isinstance(grad, np.ndarray)
    return grad
