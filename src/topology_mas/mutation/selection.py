"""Versioned plausibility policy and deterministic mutation selection."""

from __future__ import annotations

from typing import Literal

from topology_mas.mutation.schemas import (
    CandidateEvaluation,
    MutationPipelineConfig,
    PlausibilityOracleResult,
)

SELECTION_POLICY_VERSION = "plausibility-tiered-v2"
SelectionTier = Literal["preferred", "coverage_fallback"]


def passes_core_plausibility(
    plausibility: PlausibilityOracleResult,
    config: MutationPipelineConfig,
) -> bool:
    """Require plausibility and coherence, while treating subtlety as a ranking signal."""

    required_dimensions = (
        plausibility.local_error_plausibility,
        plausibility.global_coherence,
        plausibility.minimality,
    )
    return bool(
        plausibility.model_plausible
        and plausibility.overall_score >= config.plausibility_threshold
        and min(required_dimensions) >= config.minimum_dimension_score
    )


def is_preferred_candidate(
    evaluation: CandidateEvaluation,
    config: MutationPipelineConfig,
) -> bool:
    plausibility = evaluation.plausibility
    return bool(
        evaluation.objective.passed
        and plausibility is not None
        and passes_core_plausibility(plausibility, config)
        and plausibility.subtlety >= config.preferred_subtlety_score
    )


def is_coverage_candidate(
    evaluation: CandidateEvaluation,
    config: MutationPipelineConfig,
) -> bool:
    plausibility = evaluation.plausibility
    return bool(
        evaluation.objective.passed
        and plausibility is not None
        and passes_core_plausibility(plausibility, config)
    )


def select_candidate_evaluation(
    evaluations: tuple[CandidateEvaluation, ...] | list[CandidateEvaluation],
    config: MutationPipelineConfig,
) -> tuple[CandidateEvaluation, SelectionTier] | None:
    """Prefer subtle candidates, but do not discard an otherwise valid task for subtlety alone."""

    preferred = [
        evaluation
        for evaluation in evaluations
        if is_preferred_candidate(evaluation, config)
    ]
    if preferred:
        pool = preferred
        tier: SelectionTier = "preferred"
    else:
        pool = [
            evaluation
            for evaluation in evaluations
            if is_coverage_candidate(evaluation, config)
        ]
        tier = "coverage_fallback"
    if not pool:
        return None
    selected = sorted(
        pool,
        key=lambda evaluation: (
            -evaluation.plausibility.overall_score,  # type: ignore[union-attr]
            -evaluation.plausibility.subtlety,  # type: ignore[union-attr]
            -evaluation.plausibility.minimality,  # type: ignore[union-attr]
            evaluation.candidate.candidate_id,
        ),
    )[0]
    return selected, tier
