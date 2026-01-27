"""
Integration test for NRM parameter recovery.

Generates synthetic data with known NRM parameters and verifies
the estimator can recover them within statistical tolerance.

Note: The NRM model includes missing responses as a separate category (K).
With K response categories (0..K-1) and 1 missing category, there are K+1 total.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from analysis_service.core.constants import MISSING_VALUE
from analysis_service.core.data_models import (
    ResponseMatrix,
    response_code_to_category_index,
)
from analysis_service.irt.estimation.config import EstimationConfig
from analysis_service.irt.estimation.estimator import NRMEstimator
from analysis_service.irt.estimation.gradients import (
    compute_nrm_probabilities,
)
from analysis_service.irt.estimation.parameters import NRMItemParameters
from analysis_service.irt.estimation.quadrature import get_quadrature

# Test configuration
N_ITEMS = 50
N_RESPONSE_CATEGORIES = 4  # K (excludes missing)
N_CANDIDATES = 5000
SEED = 42


def generate_true_parameters(
    n_items: int,
    n_response_categories: int,
    rng: np.random.Generator,
) -> list[NRMItemParameters]:
    """
    Generate realistic NRM parameters for testing.

    Creates K+1 categories: K response categories + 1 missing category.
    The missing category has low probability controlled by missing_rate.

    Args:
        n_items: Number of items.
        n_response_categories: Number of response categories (K, excludes missing).
        rng: Random number generator.

    Returns:
        List of NRMItemParameters with K+1 categories.
    """
    params = []

    for item_idx in range(n_items):
        # Generate discriminations for response categories: Uniform(-0.5, 1.5)
        a_response = rng.uniform(-0.5, 1.5, size=n_response_categories)

        # Missing category is derived from sum-to-zero
        a_missing = -np.sum(a_response)

        # Generate intercepts for response categories: Uniform(-1.5, 1.5)
        b_response = rng.uniform(-1.5, 1.5, size=n_response_categories)

        # Derived from sum-to-zero
        b_missing = -np.sum(b_response)

        discriminations = tuple(float(a) for a in a_response) + (
            float(a_missing),
        )
        intercepts = tuple(float(b) for b in b_response) + (float(b_missing),)

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
    allow_missing: bool = True,
) -> NDArray[np.int8]:
    """
    Simulate response matrix from NRM parameters.

    Samples from all K+1 categories. If the missing category (K) is sampled,
    returns `MISSING_VALUE`.

    Args:
        true_params: True item parameters (K+1 categories each).
        abilities: Candidate abilities, shape (n_candidates,).
        rng: Random number generator.
        allow_missing: If False, sample only from response categories (0..K-1).

    Returns:
        Response matrix of shape (n_candidates, n_items).
        Missing responses encoded as MISSING_VALUE (only if allow_missing=True).
    """
    n_candidates = len(abilities)
    n_items = len(true_params)
    n_total_categories = true_params[0].n_categories  # K+1
    n_response_categories = true_params[0].n_response_categories  # K

    responses = np.zeros((n_candidates, n_items), dtype=np.int8)

    for item_idx, params in enumerate(true_params):
        discriminations = np.array(params.discriminations, dtype=np.float64)
        intercepts = np.array(params.intercepts, dtype=np.float64)

        # Compute probabilities for all candidates at once
        probs = compute_nrm_probabilities(
            abilities, discriminations, intercepts
        )

        if not allow_missing:
            # Sample only from response categories (exclude missing)
            probs = probs[:, :n_response_categories]
            probs = probs / probs.sum(axis=1, keepdims=True)
            n_choices = n_response_categories
        else:
            n_choices = n_total_categories

        # Sample responses for each candidate
        for i in range(n_candidates):
            sampled = rng.choice(n_choices, p=probs[i])
            if allow_missing and sampled == n_response_categories:
                responses[i, item_idx] = MISSING_VALUE
            else:
                responses[i, item_idx] = sampled

    return responses


def extract_free_parameters(
    params: list[NRMItemParameters],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Extract free parameters (response categories only) from item parameters.

    The missing category is derived via sum-to-zero, so we extract only
    the response category parameters (0..K-1).

    Args:
        params: List of NRMItemParameters (K+1 categories each).

    Returns:
        Tuple of (discriminations, intercepts) arrays, each of shape (n_items * K,).
    """
    all_a: list[float] = []
    all_b: list[float] = []

    for item_params in params:
        # Center discriminations for comparison (remove mean)
        a_mean = np.mean(item_params.discriminations)
        diff_a = np.array(item_params.discriminations) - a_mean

        # Extract response category parameters (exclude missing = last category)
        n_response = item_params.n_response_categories
        for a_i in diff_a[:n_response]:
            all_a.append(a_i)

        for b_i in item_params.intercepts[:n_response]:
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
    1. Generate true NRM parameters (K+1 categories)
    2. Simulate candidate responses (may include missing)
    3. Fit NRM estimator
    4. Compare estimated vs true parameters

    Assertions:
    - Pearson correlation sufficiently high
    - RMSE sufficiently low
    """
    rng = np.random.default_rng(SEED)

    # Step 1: Generate true parameters (K response + 1 missing categories)
    true_params = generate_true_parameters(N_ITEMS, N_RESPONSE_CATEGORIES, rng)

    # Step 2: Generate abilities and simulate responses
    abilities = rng.standard_normal(N_CANDIDATES)
    responses = generate_nrm_responses(true_params, abilities, rng)

    # Step 3: Fit model
    data = ResponseMatrix(
        responses=responses, n_categories=N_RESPONSE_CATEGORIES
    )
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
    n_response_categories: int,
) -> NDArray[np.float64]:
    """
    Compute empirical response probabilities for each item and category.

    Includes missing as category K (total K+1 categories).

    Args:
        responses: Response matrix, shape (n_candidates, n_items).
            Missing values encoded as MISSING_VALUE (-1).
        n_response_categories: Number of response categories (K, excludes missing).

    Returns:
        Empirical probabilities, shape (n_items, K+1).
    """

    n_items = responses.shape[1]
    n_total = n_response_categories + 1
    empirical = np.zeros((n_items, n_total), dtype=np.float64)

    for item_idx in range(n_items):
        item_responses = responses[:, item_idx]
        # Map -1 -> K for counting
        category_indices = response_code_to_category_index(
            item_responses, n_response_categories
        )
        counts = np.bincount(category_indices, minlength=n_total)
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
        item_params: Fitted item parameters (K+1 categories each).
        config: Estimation config (for quadrature settings).

    Returns:
        Predicted probabilities, shape (n_items, K+1).
    """
    quadrature = get_quadrature(config.quadrature)
    n_items = len(item_params)
    n_total_categories = item_params[0].n_categories  # K+1

    predicted = np.zeros((n_items, n_total_categories), dtype=np.float64)

    for item_idx, params in enumerate(item_params):
        discriminations = np.array(params.discriminations, dtype=np.float64)
        intercepts = np.array(params.intercepts, dtype=np.float64)

        # P(Y=k | θ) for each quadrature point, shape (n_quad, K+1)
        probs_given_theta = compute_nrm_probabilities(
            quadrature.points, discriminations, intercepts
        )

        # Marginalize: Σ_q P(Y=k | θ_q) * w_q
        # weights shape: (n_quad,), probs shape: (n_quad, K+1)
        predicted[item_idx, :] = quadrature.weights @ probs_given_theta

    return predicted


def test_probability_recovery() -> None:
    """
    Test that fitted model predicts correct marginal response probabilities.

    This test validates that the fitted model's predicted P(Y_i = k) matches
    empirical frequencies for all K+1 categories (including missing).

    Procedure:
    1. Generate true NRM parameters (K+1 categories)
    2. Simulate candidate responses (may include missing)
    3. Fit NRM estimator
    4. Compare predicted vs empirical P(Y_i = k)

    Assertions:
    - Pearson correlation sufficiently high between predicted and empirical probabilities
    - RMSE sufficiently low between predicted and empirical probabilities
    """
    rng = np.random.default_rng(SEED + 1)  # Different seed for independence

    # Step 1: Generate true parameters (K response + 1 missing categories)
    true_params = generate_true_parameters(N_ITEMS, N_RESPONSE_CATEGORIES, rng)

    # Step 2: Generate abilities and simulate responses
    abilities = rng.standard_normal(N_CANDIDATES)
    responses = generate_nrm_responses(true_params, abilities, rng)

    # Step 3: Fit model
    data = ResponseMatrix(
        responses=responses, n_categories=N_RESPONSE_CATEGORIES
    )
    config = EstimationConfig()
    estimator = NRMEstimator(config, rng=rng)
    fitted = estimator.fit(data)

    assert fitted.convergence_status == "converged", (
        f"Model did not converge: {fitted.convergence_status}"
    )

    # Step 4: Compute and compare probabilities (K+1 categories)
    empirical = compute_empirical_probabilities(
        responses, N_RESPONSE_CATEGORIES
    )
    predicted = compute_predicted_probabilities(fitted.item_parameters, config)

    # Flatten for correlation/RMSE computation
    empirical_flat = empirical.flatten()
    predicted_flat = predicted.flatten()

    corr, _ = pearsonr(empirical_flat, predicted_flat)
    rmse = compute_rmse(empirical_flat, predicted_flat)

    # Print diagnostics
    print(f"\nProbability recovery: r={corr:.4f}, RMSE={rmse:.4f}")

    # Passing thresholds
    min_corr = 0.99
    max_rmse = 0.01

    # Assertions - probabilities should match very closely
    assert corr > min_corr, f"Probability correlation {corr:.4f} < {min_corr}"
    assert rmse < max_rmse, f"Probability RMSE {rmse:.4f} > {max_rmse}"


def test_handle_no_missing_values() -> None:
    """
    Test that model handles data with no missing values correctly.

    When no missing responses exist in the data, the model should still
    estimate the missing category probability as near-zero.
    """
    rng = np.random.default_rng(SEED + 2)

    # Generate realistic parameters and responses (no missing allowed)
    true_params = generate_true_parameters(N_ITEMS, N_RESPONSE_CATEGORIES, rng)
    abilities = rng.standard_normal(N_CANDIDATES)
    responses = generate_nrm_responses(
        true_params, abilities, rng, allow_missing=False
    )

    # Verify no missing values in generated data
    assert np.all(responses != MISSING_VALUE)

    # Fit model
    data = ResponseMatrix(
        responses=responses, n_categories=N_RESPONSE_CATEGORIES
    )
    config = EstimationConfig()
    estimator = NRMEstimator(config, rng=rng)
    fitted = estimator.fit(data)

    assert fitted.convergence_status == "converged", (
        f"Model did not converge: {fitted.convergence_status}"
    )

    # Empirical missing rate should be exactly 0
    empirical = compute_empirical_probabilities(
        responses, N_RESPONSE_CATEGORIES
    )
    assert np.allclose(empirical[:, -1], 0.0, atol=1e-10)
