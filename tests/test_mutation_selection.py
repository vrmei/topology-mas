from topology_mas.mutation.schemas import (
    ArithmeticStep,
    CandidateEvaluation,
    MutationCandidate,
    MutationPipelineConfig,
    ObjectiveOracleResult,
    PlausibilityOracleResult,
)
from topology_mas.mutation.selection import select_candidate_evaluation


def _evaluation(
    candidate_id: str,
    *,
    overall: float,
    subtlety: float,
    stored_plausible: bool = False,
) -> CandidateEvaluation:
    candidate = MutationCandidate(
        candidate_id=candidate_id,
        mutation_type="arithmetic_result",
        mutated_step_id="s1",
        steps=(
            ArithmeticStep(
                step_id="s1",
                expression="2+2",
                claimed_result="5",
                explanation="A single addition slip.",
                is_mutated=True,
                depends_on=(),
            ),
        ),
        final_answer="5",
        full_response="Two plus two is five.\n#### 5",
    )
    return CandidateEvaluation(
        candidate=candidate,
        objective=ObjectiveOracleResult(passed=True),
        plausibility=PlausibilityOracleResult(
            model_plausible=True,
            plausible=stored_plausible,
            local_error_plausibility=0.9,
            global_coherence=1.0,
            subtlety=subtlety,
            minimality=1.0,
            overall_score=overall,
        ),
    )


def test_selection_recovers_cached_candidate_rejected_only_for_subtlety() -> None:
    evaluation = _evaluation("c01", overall=0.8, subtlety=0.3)

    selected = select_candidate_evaluation(
        (evaluation,),
        MutationPipelineConfig(candidate_count=1),
    )

    assert selected is not None
    assert selected[0].candidate.candidate_id == "c01"
    assert selected[1] == "coverage_fallback"


def test_selection_prefers_high_subtlety_tier_before_coverage_fallback() -> None:
    higher_overall_fallback = _evaluation("c01", overall=0.9, subtlety=0.4)
    preferred = _evaluation("c02", overall=0.75, subtlety=0.7)

    selected = select_candidate_evaluation(
        (higher_overall_fallback, preferred),
        MutationPipelineConfig(candidate_count=2),
    )

    assert selected is not None
    assert selected[0].candidate.candidate_id == "c02"
    assert selected[1] == "preferred"
