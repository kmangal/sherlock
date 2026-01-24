"""
Validation and sanity checks for generated data.

This module validates answer strings, checks invariants, and provides
statistical validation of generated data.
"""

import numpy as np
from numpy.typing import NDArray

from analysis_service.synthetic_data.data_models import (
    GeneratedData,
    ItemStatistics,
)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def compute_missing_rate(
    answer_strings: list[str],
    missing_char: str = "*",
) -> float:
    """
    Compute the actual missing rate from answer strings.

    Args:
        answer_strings: List of answer strings.
        missing_char: Character representing missing answers.

    Returns:
        Fraction of responses that are missing.
    """
    if not answer_strings:
        return 0.0

    total = sum(len(s) for s in answer_strings)
    missing = sum(s.count(missing_char) for s in answer_strings)
    return missing / total if total > 0 else 0.0


def validate_missing_rate(
    answer_strings: list[str],
    expected_rate: float,
    tolerance: float = 0.01,
    missing_char: str = "*",
) -> None:
    """
    Validate that the missing rate is within tolerance of expected.

    Args:
        answer_strings: List of answer strings.
        expected_rate: Expected missing rate.
        tolerance: Allowed deviation from expected rate.
        missing_char: Character representing missing answers.

    Raises:
        ValidationError: If actual rate deviates too much from expected.
    """
    actual_rate = compute_missing_rate(answer_strings, missing_char)
    if abs(actual_rate - expected_rate) > tolerance:
        raise ValidationError(
            f"Missing rate {actual_rate:.4f} differs from expected "
            f"{expected_rate:.4f} by more than tolerance {tolerance}"
        )


def compute_accuracy_by_ability(
    data: GeneratedData,
    n_bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute accuracy (fraction correct) binned by ability.

    This is useful for validating that the IRT model is working correctly:
    higher ability should lead to higher accuracy.

    Args:
        data: Generated exam data.
        n_bins: Number of ability bins.

    Returns:
        Tuple of (bin_centers, accuracies) arrays.
    """
    abilities = data.abilities
    responses = data.raw_responses
    item_params = data.item_params

    correct_answers = np.array(
        [p.correct_answer for p in item_params], dtype=np.int64
    )
    assert not np.any(np.isnan(correct_answers)), (
        "All items require a correct answer"
    )

    # Create bins
    bin_edges = np.linspace(abilities.min(), abilities.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    accuracies = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (abilities >= bin_edges[i]) & (abilities < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (abilities >= bin_edges[i]) & (
                abilities <= bin_edges[i + 1]
            )

        if mask.sum() == 0:
            accuracies[i] = np.nan
            continue

        # Get responses for candidates in this bin
        bin_responses = responses[mask]

        # Count correct (excluding missing)
        n_correct = 0
        n_total = 0
        for candidate_responses in bin_responses:
            for j, response in enumerate(candidate_responses):
                if response != -1:  # Not missing
                    n_total += 1
                    if response == correct_answers[j]:
                        n_correct += 1

        accuracies[i] = n_correct / n_total if n_total > 0 else np.nan

    return bin_centers, accuracies


def validate_monotonicity(
    data: GeneratedData,
    n_bins: int = 10,
    min_correlation: float = 0.8,
) -> None:
    """
    Validate that accuracy increases monotonically with ability.

    Uses Spearman correlation to check for monotonic relationship.

    Args:
        data: Generated exam data.
        n_bins: Number of ability bins.
        min_correlation: Minimum acceptable correlation.

    Raises:
        ValidationError: If monotonicity is not satisfied.
    """
    bin_centers, accuracies = compute_accuracy_by_ability(data, n_bins)

    # Remove NaN values
    valid_mask = ~np.isnan(accuracies)
    if valid_mask.sum() < 3:
        raise ValidationError(
            "Not enough valid bins to validate monotonicity "
            f"(need at least 3, got {valid_mask.sum()})"
        )

    valid_centers = bin_centers[valid_mask]
    valid_accuracies = accuracies[valid_mask]

    # Compute Spearman rank correlation
    # Since we want monotonicity, we use ranks
    center_ranks = np.argsort(np.argsort(valid_centers))
    accuracy_ranks = np.argsort(np.argsort(valid_accuracies))

    n = len(center_ranks)
    d_squared = np.sum((center_ranks - accuracy_ranks) ** 2)
    spearman = 1 - (6 * d_squared) / (n * (n**2 - 1))

    if spearman < min_correlation:
        raise ValidationError(
            f"Ability-accuracy correlation {spearman:.3f} is below "
            f"minimum {min_correlation}. Accuracy should increase with ability."
        )


def validate_generated_data(
    data: GeneratedData,
    check_missing_rate: bool = True,
    check_monotonicity: bool = True,
    missing_rate_tolerance: float = 0.02,
    min_monotonicity_correlation: float = 0.7,
) -> None:
    """
    Run all validation checks on generated data.

    Args:
        data: Generated exam data.
        check_missing_rate: Whether to validate missing rate.
        check_monotonicity: Whether to validate ability-accuracy monotonicity.
        missing_rate_tolerance: Tolerance for missing rate validation.
        min_monotonicity_correlation: Minimum correlation for monotonicity.

    Raises:
        ValidationError: If any validation fails.
    """
    # Validate missing rate
    if check_missing_rate and data.config.missing_rate > 0:
        validate_missing_rate(
            data.answer_strings,
            data.config.missing_rate,
            tolerance=missing_rate_tolerance,
        )

    # Validate monotonicity (only if enough data)
    if check_monotonicity and data.config.n_candidates >= 100:
        validate_monotonicity(
            data,
            min_correlation=min_monotonicity_correlation,
        )


def compute_item_statistics(
    data: GeneratedData,
) -> list[ItemStatistics]:
    """
    Compute item-level statistics for validation.

    Args:
        data: Generated exam data.

    Returns:
        ItemStatistics with:
            - proportion_correct: Proportion correct for each item
            - missing_rates: Missing rate for each item
    """
    responses = data.raw_responses
    item_params = data.item_params

    correct_answers = np.array(
        [p.correct_answer for p in item_params], dtype=np.int64
    )

    assert not np.any(np.isnan(correct_answers)), (
        "All items require correct answers"
    )

    n_questions = data.config.n_questions

    statistics: list[ItemStatistics] = []

    for j in range(n_questions):
        item_responses = responses[:, j]
        non_missing = item_responses != -1
        n_responded = non_missing.sum()

        if n_responded > 0:
            correct = item_responses[non_missing] == correct_answers[j]
            prop_correct = correct.sum() / n_responded
        else:
            prop_correct = np.nan

        missing_rate = 1 - (n_responded / len(item_responses))

        statistics.append(
            ItemStatistics(
                proportion_correct=float(prop_correct),
                missing_rate=float(missing_rate),
            )
        )

    assert len(statistics) == n_questions, (
        f"Have {len(statistics)} but expected {n_questions}"
    )
    return statistics
