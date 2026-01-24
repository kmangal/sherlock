"""
Integration test for NRM parameter recovery.

Generates synthetic data with known NRM parameters and verifies
the estimator can recover them within statistical tolerance.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from analysis_service.irt.estimation.config import EstimationConfig
from analysis_service.irt.estimation.data_models import ResponseMatrix
from analysis_service.irt.estimation.estimator import NRMEstimator
from analysis_service.irt.estimation.gradients import (
    compute_nrm_probabilities,
)
from analysis_service.irt.estimation.parameters import NRMItemParameters
from analysis_service.irt.estimation.quadrature import get_quadrature

# Test configuration
N_ITEMS = 50
N_CATEGORIES = 4
N_CANDIDATES = 5000
SEED = 42


def generate_true_parameters(
    n_items: int,
    n_categories: int,
    rng: np.random.Generator,
) -> list[NRMItemParameters]:
    """
    Generate realistic NRM parameters for testing.

    Args:
        n_items: Number of items.
        n_categories: Number of response categories.
        rng: Random number generator.

    Returns:
        List of NRMItemParameters with known true values.
    """
    params = []
    for item_idx in range(n_items):
        # discriminations: Uniform(-0.5, 1.5)
        a_vals = rng.uniform(-0.5, 1.5, size=n_categories - 1)
        a_k = -np.sum(a_vals)

        # intercepts: Uniform(-1.5, 1.5)
        b_vals = rng.uniform(-1.5, 1.5, size=n_categories - 1)
        b_k = -np.sum(b_vals)

        discriminations = (a_k,) + tuple(float(a) for a in a_vals)
        intercepts = (b_k,) + tuple(float(b) for b in b_vals)

        params.append(
            NRMItemParameters(
                item_id=item_idx,
                discriminations=discriminations,
                intercepts=intercepts,
            )
        )
    return params


def generate_nrm_responses(
    true_params: list[NRMItemParameters],
    abilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int8]:
    """
    Simulate response matrix from NRM parameters.

    Args:
        true_params: True item parameters.
        abilities: Candidate abilities, shape (n_candidates,).
        rng: Random number generator.

    Returns:
        Response matrix of shape (n_candidates, n_items).
    """
    n_candidates = len(abilities)
    n_items = len(true_params)
    n_categories = true_params[0].n_categories

    responses = np.zeros((n_candidates, n_items), dtype=np.int8)

    for item_idx, params in enumerate(true_params):
        discriminations = np.array(params.discriminations, dtype=np.float64)
        intercepts = np.array(params.intercepts, dtype=np.float64)

        # Compute probabilities for all candidates at once
        probs = compute_nrm_probabilities(
            abilities, discriminations, intercepts
        )

        # Sample responses for each candidate
        for i in range(n_candidates):
            responses[i, item_idx] = rng.choice(n_categories, p=probs[i])

    return responses


def extract_free_parameters(
    params: list[NRMItemParameters],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Extract free parameters (excluding reference category) from item parameters.

    Args:
        params: List of NRMItemParameters.

    Returns:
        Tuple of (discriminations, intercepts) arrays, each of shape (n_items * (K-1),).
    """
    all_a: list[float] = []
    all_b: list[float] = []

    # During estimation the last item is used to normalize so use it as the reference
    # category

    for item_params in params:
        a_mean = np.mean(item_params.discriminations)
        diff_a = np.array(item_params.discriminations) - a_mean
        for a_i in diff_a[:-1]:
            all_a.append(a_i)

        for b_i in item_params.intercepts[:-1]:
            all_b.append(b_i)

    return np.array(all_a, dtype=np.float64), np.array(all_b, dtype=np.float64)


def compute_rmse(
    true: NDArray[np.float64], estimated: NDArray[np.float64]
) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((true - estimated) ** 2)))


#######################################################################


def test_parameter_recovery() -> None:
    """
    Test that NRM estimator recovers known parameters.

    Procedure:
    1. Generate true NRM parameters
    2. Simulate candidate responses
    3. Fit NRM estimator
    4. Compare estimated vs true parameters

    Assertions:
    - Pearson correlation sufficiently high
    - RMSE sufficiently low
    """
    rng = np.random.default_rng(SEED)

    # Step 1: Generate true parameters
    true_params = generate_true_parameters(N_ITEMS, N_CATEGORIES, rng)

    # Step 2: Generate abilities and simulate responses
    abilities = rng.standard_normal(N_CANDIDATES)
    responses = generate_nrm_responses(true_params, abilities, rng)

    # Step 3: Fit model
    data = ResponseMatrix(responses=responses, n_categories=N_CATEGORIES)
    estimator = NRMEstimator(rng=rng)
    fitted = estimator.fit(data)

    assert fitted.convergence_status == "converged", (
        f"Model did not converge: {fitted.convergence_status}"
    )

    # Step 4: Extract and compare parameters
    true_a, true_b = extract_free_parameters(true_params)
    est_a, est_b = extract_free_parameters(list(fitted.item_parameters))

    # Compute metrics for discriminations
    corr_a, _ = pearsonr(true_a, est_a)
    rmse_a = compute_rmse(true_a, est_a)

    # Compute metrics for intercepts
    corr_b, _ = pearsonr(true_b, est_b)
    rmse_b = compute_rmse(true_b, est_b)

    # Print diagnostics
    print(f"\nDiscriminations: r={corr_a:.3f}, RMSE={rmse_a:.3f}")
    print(f"Intercepts: r={corr_b:.3f}, RMSE={rmse_b:.3f}")

    # Recovery thresholds
    min_correlation = 0.85
    max_rmse = 0.2

    # Assert recovery thresholds
    assert corr_a > min_correlation, (
        f"Discrimination correlation {corr_a:.3f} < {min_correlation}"
    )
    assert corr_b > min_correlation, (
        f"Intercept correlation {corr_b:.3f} < {min_correlation}"
    )
    assert rmse_a < max_rmse, f"Discrimination RMSE {rmse_a:.3f} > {max_rmse}"
    assert rmse_b < max_rmse, f"Intercept RMSE {rmse_b:.3f} > {max_rmse}"


def compute_empirical_probabilities(
    responses: NDArray[np.int8],
    n_categories: int,
) -> NDArray[np.float64]:
    """
    Compute empirical response probabilities for each item and category.

    Args:
        responses: Response matrix, shape (n_candidates, n_items).
        n_categories: Number of response categories.

    Returns:
        Empirical probabilities, shape (n_items, n_categories).
    """
    n_items = responses.shape[1]
    empirical = np.zeros((n_items, n_categories), dtype=np.float64)

    for item_idx in range(n_items):
        item_responses = responses[:, item_idx]
        counts = np.bincount(
            item_responses.astype(np.int64), minlength=n_categories
        )
        empirical[item_idx, :] = counts / counts.sum()

    return empirical


def compute_predicted_probabilities(
    item_params: tuple[NRMItemParameters, ...],
    config: EstimationConfig,
) -> NDArray[np.float64]:
    """
    Compute predicted marginal response probabilities P(Y_i = k).

    Marginalizes over ability distribution using quadrature:
        P(Y_i = k) = ∫ P(Y_i = k | θ) f(θ) dθ ≈ Σ_q P(Y_i = k | θ_q) * w_q

    Args:
        item_params: Fitted item parameters.
        config: Estimation config (for quadrature settings).

    Returns:
        Predicted probabilities, shape (n_items, n_categories).
    """
    quadrature = get_quadrature(config.quadrature)
    n_items = len(item_params)
    n_categories = item_params[0].n_categories

    predicted = np.zeros((n_items, n_categories), dtype=np.float64)

    for item_idx, params in enumerate(item_params):
        discriminations = np.array(params.discriminations, dtype=np.float64)
        intercepts = np.array(params.intercepts, dtype=np.float64)

        # P(Y=k | θ) for each quadrature point, shape (n_quad, n_categories)
        probs_given_theta = compute_nrm_probabilities(
            quadrature.points, discriminations, intercepts
        )

        # Marginalize: Σ_q P(Y=k | θ_q) * w_q
        # weights shape: (n_quad,), probs shape: (n_quad, n_categories)
        predicted[item_idx, :] = quadrature.weights @ probs_given_theta

    return predicted


def test_probability_recovery() -> None:
    """
    Test that fitted model predicts correct marginal response probabilities.

    This test uses unconstrained parameters (all a_i, b_i free) and validates
    that the fitted model's predicted P(Y_i = k) matches empirical frequencies.

    Procedure:
    1. Generate true NRM parameters (all parameters free)
    2. Simulate candidate responses
    3. Fit NRM estimator
    4. Compare predicted vs empirical P(Y_i = k)

    Assertions:
    - Pearson correlation sufficiently high between predicted and empirical probabilities
    - RMSE sufficiently low between predicted and empirical probabilities
    """
    rng = np.random.default_rng(SEED + 1)  # Different seed for independence

    # Step 1: Generate unconstrained true parameters
    true_params = generate_true_parameters(N_ITEMS, N_CATEGORIES, rng)

    # Step 2: Generate abilities and simulate responses
    abilities = rng.standard_normal(N_CANDIDATES)
    responses = generate_nrm_responses(true_params, abilities, rng)

    # Step 3: Fit model
    data = ResponseMatrix(responses=responses, n_categories=N_CATEGORIES)
    config = EstimationConfig()
    estimator = NRMEstimator(config, rng=rng)
    fitted = estimator.fit(data)

    assert fitted.convergence_status == "converged", (
        f"Model did not converge: {fitted.convergence_status}"
    )

    # Step 4: Compute and compare probabilities
    empirical = compute_empirical_probabilities(responses, N_CATEGORIES)
    predicted = compute_predicted_probabilities(fitted.item_parameters, config)

    # Flatten for correlation/RMSE computation
    empirical_flat = empirical.flatten()
    predicted_flat = predicted.flatten()

    corr, _ = pearsonr(empirical_flat, predicted_flat)
    rmse = compute_rmse(empirical_flat, predicted_flat)

    # Print diagnostics
    print(f"\nProbability recovery: r={corr:.4f}, RMSE={rmse:.4f}")

    # We get faster convergence with empirical probabilities
    min_corr = 0.99
    max_rmse = 0.005

    # Assertions - probabilities should match very closely
    assert corr > min_corr, f"Probability correlation {corr:.4f} < {min_corr}"
    assert rmse < max_rmse, f"Probability RMSE {rmse:.4f} > {max_rmse}"
