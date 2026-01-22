"""
IRT model estimation module.

This module provides infrastructure for estimating Item Response Theory models
using Marginal Maximum Likelihood via the EM algorithm.

Key components:
- EstimationConfig: Configuration for estimation
- ResponseMatrix: Input data representation
- FittedModel: Output from estimation
- IRTEstimator: Abstract base class for estimators
- NRMEstimator: Nominal Response Model estimator
- estimate_abilities: EAP ability estimation
"""
