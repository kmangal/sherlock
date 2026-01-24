"""
IRT (Item Response Theory) module.

This module provides:
- Item parameter classes (3PL, NRM) with compute_probabilities methods
- Sampling functions for generating responses
- Estimation infrastructure for fitting IRT models to data
- Ability estimation
"""

from analysis_service.irt.estimation.estimator import NRMEstimator
from analysis_service.irt.estimation.parameters import (
    NRMItemParameters,
)
from analysis_service.irt.sampling import (
    sample_response,
    sample_responses_batch,
)

__all__ = [
    "NRMEstimator",
    "NRMItemParameters",
    "sample_response",
    "sample_responses_batch",
]
