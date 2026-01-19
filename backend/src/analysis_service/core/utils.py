"""
Core utility functions shared across analysis service modules.

This module provides foundational utilities used by both the IRT
statistical models and the synthetic data generation layer.
"""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def get_rng(seed: int | None = None) -> Generator:
    """
    Create a numpy random Generator with optional seed.

    Args:
        seed: Random seed for reproducibility. If None, uses entropy.

    Returns:
        A numpy random Generator instance.
    """
    return np.random.default_rng(seed)


def softmax(
    logits: NDArray[np.floating], axis: int = -1
) -> NDArray[np.float64]:
    """
    Compute softmax probabilities from logits.

    Numerically stable implementation.

    Args:
        logits: Array of logits.
        axis: Axis along which to compute softmax.

    Returns:
        Array of probabilities that sum to 1 along the specified axis.
    """
    # Subtract max for numerical stability
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    result: NDArray[np.float64] = exp_logits / np.sum(
        exp_logits, axis=axis, keepdims=True
    )
    return result
