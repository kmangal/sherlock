from collections.abc import Callable

from analysis_service.api.schemas import AnalysisPhase

ProgressCallback = Callable[[AnalysisPhase, int, int | None, str], None]
