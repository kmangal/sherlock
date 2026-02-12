"""
Data models for evaluation of detection pipelines.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from analysis_service.irt.estimation.parameters import NRMItemParameters


class CheaterPair(BaseModel):
    """A single source-copier relationship."""

    source_idx: int
    copier_idx: int
    n_copied_items: int
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True)
class CheaterConfig:
    """Configuration for injecting cheaters into exam data.

    Attributes:
        n_sources: Number of source candidates (people being copied from).
        n_copiers_per_source: Number of copiers per source (supports 1→many copying).
        n_copied_items: Exact number of items copied from source to copier.
    """

    n_sources: int
    n_copiers_per_source: int
    n_copied_items: int

    def __post_init__(self) -> None:
        if self.n_sources < 1:
            raise ValueError(f"n_sources must be >= 1, got {self.n_sources}")
        if self.n_copiers_per_source < 1:
            raise ValueError(
                f"n_copiers_per_source must be >= 1, got {self.n_copiers_per_source}"
            )
        if self.n_copied_items < 1:
            raise ValueError(
                f"n_copied_items must be >= 1, got {self.n_copied_items}"
            )

    @property
    def total_cheaters(self) -> int:
        """Total number of cheaters (sources + copiers)."""
        return self.n_sources + self.n_sources * self.n_copiers_per_source


@dataclass(frozen=True)
class CheatingGroundTruth:
    """Ground truth about which candidates are cheaters.

    Both sources and copiers are considered cheaters.
    """

    cheater_pairs: tuple[CheaterPair, ...]

    @property
    def cheater_indices(self) -> frozenset[int]:
        """All cheater indices (both sources and copiers)."""
        indices: set[int] = set()
        for pair in self.cheater_pairs:
            indices.add(pair.source_idx)
            indices.add(pair.copier_idx)
        return frozenset(indices)

    @property
    def source_indices(self) -> frozenset[int]:
        """Source candidate indices."""
        return frozenset(pair.source_idx for pair in self.cheater_pairs)

    @property
    def copier_indices(self) -> frozenset[int]:
        """Copier candidate indices."""
        return frozenset(pair.copier_idx for pair in self.cheater_pairs)


@dataclass(frozen=True)
class ConfusionMatrix:
    """Confusion matrix for binary classification."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def __post_init__(self) -> None:
        if self.true_positives < 0:
            raise ValueError(
                f"true_positives must be >= 0, got {self.true_positives}"
            )
        if self.false_positives < 0:
            raise ValueError(
                f"false_positives must be >= 0, got {self.false_positives}"
            )
        if self.true_negatives < 0:
            raise ValueError(
                f"true_negatives must be >= 0, got {self.true_negatives}"
            )
        if self.false_negatives < 0:
            raise ValueError(
                f"false_negatives must be >= 0, got {self.false_negatives}"
            )

    @property
    def recall(self) -> float:
        """Recall (sensitivity, true positive rate).

        Returns 0.0 if there are no actual positives.
        """
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision (positive predictive value).

        Returns 0.0 if there are no predicted positives.
        """
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall).

        Returns 0.0 if precision + recall = 0.
        """
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def power(self) -> float:
        """Statistical power (same as recall/sensitivity)."""
        return self.recall

    @property
    def false_positive_rate(self) -> float:
        """False positive rate (1 - specificity).

        Returns 0.0 if there are no actual negatives.
        """
        total = self.false_positives + self.true_negatives
        return self.false_positives / total if total > 0 else 0.0


@dataclass(frozen=True)
class FittedModelContext:
    """Pre-fitted IRT model context for evaluation.

    Contains the fitted model parameters needed to generate
    synthetic data for evaluation iterations.
    """

    abilities: NDArray[np.float64]
    item_params: tuple[NRMItemParameters, ...]
    n_categories: int


class EvaluationRunResult(BaseModel):
    """Result from a single evaluation run."""

    run_index: int
    ground_truth: CheatingGroundTruth
    detected_candidate_ids: tuple[str, ...]
    all_candidate_ids: tuple[str, ...]
    seed: int
    model_config = ConfigDict(frozen=True)

    def confusion_matrix(self) -> ConfusionMatrix:
        """Compute confusion matrix from ground truth and detected candidates."""
        all_ids = set(self.all_candidate_ids)
        detected_ids = set(self.detected_candidate_ids)
        cheater_ids = {
            self.all_candidate_ids[idx]
            for idx in self.ground_truth.cheater_indices
        }

        true_positives = len(detected_ids & cheater_ids)
        false_positives = len(detected_ids - cheater_ids)
        false_negatives = len(cheater_ids - detected_ids)
        true_negatives = len(all_ids - cheater_ids - detected_ids)

        return ConfusionMatrix(
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
        )
