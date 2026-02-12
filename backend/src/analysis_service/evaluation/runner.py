"""
Evaluation runner for detection pipeline performance measurement.
"""

from collections.abc import Iterator

import numpy as np

from analysis_service.core.data_models import ExamDataset, ResponseMatrix
from analysis_service.detection.pipeline import DetectionPipeline
from analysis_service.evaluation.data_models import (
    CheaterConfig,
    EvaluationRunResult,
    FittedModelContext,
)
from analysis_service.evaluation.injection import (
    add_cheaters_to_preset,
    inject_cheaters,
)
from analysis_service.irt.estimation import (
    NRMEstimator,
    estimate_abilities_eap,
)
from analysis_service.irt.sampling import sample_responses_batch
from analysis_service.synthetic_data.data_models import GeneratedData


def _build_exam_dataset(data: GeneratedData) -> ExamDataset:
    """Convert GeneratedData to ExamDataset for pipeline consumption."""
    candidate_ids = np.array([str(cid) for cid in data.candidate_ids])
    response_matrix = ResponseMatrix(
        responses=data.responses,
        n_categories=data.config.n_response_categories,
    )

    return ExamDataset(
        candidate_ids=candidate_ids,
        response_matrix=response_matrix,
        correct_answers=None,
    )


def run_evaluation(
    preset_name: str,
    cheater_config: CheaterConfig,
    pipeline: DetectionPipeline,
    n_iterations: int,
    base_seed: int,
) -> Iterator[EvaluationRunResult]:
    """
    Run evaluation across multiple iterations.

    For each iteration:
    1. Generate data from preset with independent seed
    2. Inject cheaters according to config
    3. Run detection pipeline
    4. Yield result with detected and all candidate IDs

    Uses SeedSequence to derive cryptographically independent seeds per
    iteration and per phase (data generation, injection, detection).

    Args:
        preset_name: Name of the preset to use for data generation.
        cheater_config: Configuration for cheating pattern.
        pipeline: Detection pipeline to evaluate.
        n_iterations: Number of evaluation iterations.
        base_seed: Base random seed.

    Yields:
        EvaluationRunResult for each iteration.
    """
    root_seq = np.random.SeedSequence(base_seed)
    iter_seqs = root_seq.spawn(n_iterations)

    for i in range(n_iterations):
        data_seq, inject_seq, detect_seq = iter_seqs[i].spawn(3)
        data_seed = int(data_seq.generate_state(1)[0])
        inject_seed = int(inject_seq.generate_state(1)[0])

        # Generate data with cheaters
        data, ground_truth = add_cheaters_to_preset(
            preset_name=preset_name,
            cheater_config=cheater_config,
            data_seed=data_seed,
            injection_seed=inject_seed,
        )

        # Build ExamDataset
        exam_dataset = _build_exam_dataset(data)
        candidate_ids = tuple(str(cid) for cid in data.candidate_ids)

        # Run pipeline
        rng = np.random.default_rng(detect_seq)
        suspects = pipeline.run(exam_dataset, rng=rng)

        # Build result
        detected_ids = tuple(s.candidate_id for s in suspects)

        yield EvaluationRunResult(
            run_index=i,
            ground_truth=ground_truth,
            detected_candidate_ids=detected_ids,
            all_candidate_ids=candidate_ids,
            seed=base_seed,
        )


def fit_model_for_evaluation(
    exam_dataset: ExamDataset,
    base_seed: int,
) -> FittedModelContext:
    """Fit an IRT model to exam data for use in evaluation.

    Args:
        exam_dataset: Real exam data to fit the model to.
        base_seed: Random seed for model fitting.

    Returns:
        FittedModelContext with abilities, item params, and n_categories.
    """
    rng_fit = np.random.default_rng(base_seed)
    estimator = NRMEstimator(rng=rng_fit)
    fitted_model = estimator.fit(
        data=exam_dataset.response_matrix,
        correct_answers=exam_dataset.correct_answers,
    )

    ability_estimates = estimate_abilities_eap(
        data=exam_dataset.response_matrix,
        model=fitted_model,
    )

    return FittedModelContext(
        abilities=ability_estimates.eap,
        item_params=fitted_model.item_parameters,
        n_categories=exam_dataset.response_matrix.n_categories,
    )


def run_evaluation_from_fitted(
    context: FittedModelContext,
    cheater_config: CheaterConfig,
    pipeline: DetectionPipeline,
    n_iterations: int,
    base_seed: int,
) -> Iterator[EvaluationRunResult]:
    """Run evaluation iterations using a pre-fitted IRT model.

    Uses SeedSequence to derive cryptographically independent seeds per
    iteration and per phase (sampling, injection, detection).

    Args:
        context: Pre-fitted model context from fit_model_for_evaluation.
        cheater_config: Configuration for cheating pattern.
        pipeline: Detection pipeline to evaluate.
        n_iterations: Number of evaluation iterations.
        base_seed: Base random seed.

    Yields:
        EvaluationRunResult for each iteration.
    """
    abilities = context.abilities
    item_params = list(context.item_params)
    n_categories = context.n_categories
    n_candidates = len(abilities)

    root_seq = np.random.SeedSequence(base_seed)
    iter_seqs = root_seq.spawn(n_iterations)

    for i in range(n_iterations):
        sample_seq, inject_seq, detect_seq = iter_seqs[i].spawn(3)

        rng = np.random.default_rng(sample_seq)
        responses = sample_responses_batch(
            abilities=abilities,
            item_params_list=item_params,
            rng=rng,
        )

        rng_inject = np.random.default_rng(inject_seq)
        modified_responses, ground_truth = inject_cheaters(
            responses, cheater_config, rng_inject
        )

        candidate_ids = tuple(str(j) for j in range(n_candidates))
        response_matrix = ResponseMatrix(
            responses=modified_responses,
            n_categories=n_categories,
        )
        exam_dataset = ExamDataset(
            candidate_ids=np.array(candidate_ids),
            response_matrix=response_matrix,
            correct_answers=None,
        )

        rng_pipeline = np.random.default_rng(detect_seq)
        suspects = pipeline.run(exam_dataset, rng=rng_pipeline)

        detected_ids = tuple(s.candidate_id for s in suspects)

        yield EvaluationRunResult(
            run_index=i,
            ground_truth=ground_truth,
            detected_candidate_ids=detected_ids,
            all_candidate_ids=candidate_ids,
            seed=base_seed,
        )


def run_evaluation_from_data(
    exam_dataset: ExamDataset,
    cheater_config: CheaterConfig,
    pipeline: DetectionPipeline,
    n_iterations: int,
    base_seed: int,
) -> Iterator[EvaluationRunResult]:
    """Run evaluation using a fitted IRT model from real exam data.

    Convenience function that fits the model and runs iterations in one call.
    Uses SeedSequence to derive independent seeds for fitting vs iterations.

    Args:
        exam_dataset: Real exam data to fit the model to.
        cheater_config: Configuration for cheating pattern.
        pipeline: Detection pipeline to evaluate.
        n_iterations: Number of evaluation iterations.
        base_seed: Base random seed.

    Yields:
        EvaluationRunResult for each iteration.
    """
    root_seq = np.random.SeedSequence(base_seed)
    fit_seq, iter_seq = root_seq.spawn(2)

    fit_seed = int(fit_seq.generate_state(1)[0])
    context = fit_model_for_evaluation(exam_dataset, fit_seed)

    iter_seed = int(iter_seq.generate_state(1)[0])
    yield from run_evaluation_from_fitted(
        context=context,
        cheater_config=cheater_config,
        pipeline=pipeline,
        n_iterations=n_iterations,
        base_seed=iter_seed,
    )
