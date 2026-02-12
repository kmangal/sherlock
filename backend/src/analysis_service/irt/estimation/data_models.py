from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from analysis_service.irt.estimation.enums import ConvergenceStatus
from analysis_service.irt.estimation.parameters import NRMItemParameters


@dataclass
class EStepResult:
    """
    Results from the E-step of EM algorithm.

    Attributes:
        posteriors: Posterior weights, shape (n_candidates, n_quadrature_points).
            posteriors[i, q] = P(theta = theta_q | responses_i, params).
        log_likelihood: Marginal log-likelihood for current parameters.
    """

    posteriors: NDArray[np.float64]
    log_likelihood: float


class IRTEstimationResult(BaseModel):
    """
    Result of IRT model estimation.

    Attributes:
        item_parameters: Tuple of estimated item parameters, one per item.
        log_likelihood: Final marginal log-likelihood value.
        n_iterations: Number of EM iterations performed.
        convergence_status: Status indicating how estimation terminated.
        model_version: Version string for reproducibility tracking.
    """

    model_config = ConfigDict(frozen=True)

    item_parameters: tuple[NRMItemParameters, ...]
    log_likelihood: float
    n_iterations: int
    convergence_status: ConvergenceStatus
    model_version: str

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return len(self.item_parameters)

    @property
    def converged(self) -> bool:
        """Whether estimation converged successfully."""
        return self.convergence_status == ConvergenceStatus.CONVERGED
