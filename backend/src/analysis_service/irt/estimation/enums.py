from enum import Enum


class ConvergenceStatus(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"
