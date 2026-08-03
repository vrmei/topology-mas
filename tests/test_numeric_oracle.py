from topology_mas.mutation.numeric_oracle import (
    NumericMutationOracle,
    SafeArithmeticEvaluator,
)
from topology_mas.mutation.schemas import ArithmeticStep, MutationCandidate


def make_valid_candidate() -> MutationCandidate:
    return MutationCandidate(
        candidate_id="c01",
        mutation_type="arithmetic_result",
        mutated_step_id="s1",
        steps=(
            ArithmeticStep(
                step_id="s1",
                expression="6 * 8",
                claimed_result="42",
                explanation="Six groups of eight are treated as forty-two.",
                is_mutated=True,
            ),
            ArithmeticStep(
                step_id="s2",
                expression="42 + 2",
                claimed_result="44",
                explanation="Add the remaining two items.",
            ),
        ),
        final_answer="44",
        full_response="Six groups are 42, then two more make 44.\n#### 44",
    )


def test_safe_arithmetic_evaluator() -> None:
    evaluator = SafeArithmeticEvaluator()

    assert str(evaluator.evaluate("(3 + 5) * 2 / 4")) == "4"


def test_objective_oracle_accepts_one_propagated_error() -> None:
    result = NumericMutationOracle().verify(make_valid_candidate(), reference_answer="50")

    assert result.passed is True
    assert [check.matches for check in result.step_checks] == [False, True]
    assert result.parsed_final_answer == "44"


def test_objective_oracle_rejects_unpropagated_error() -> None:
    payload = make_valid_candidate().model_dump()
    payload["steps"][1]["expression"] = "48 + 2"
    payload["steps"][1]["claimed_result"] = "50"
    payload["final_answer"] = "50"
    payload["full_response"] = "The answer is fifty.\n#### 50"
    candidate = MutationCandidate.model_validate(payload)

    result = NumericMutationOracle().verify(candidate, reference_answer="52")

    assert result.passed is False
    assert "does not propagate" in " ".join(result.reasons)


def test_objective_oracle_rejects_second_arithmetic_mismatch() -> None:
    payload = make_valid_candidate().model_dump()
    payload["steps"][1]["claimed_result"] = "45"
    payload["final_answer"] = "45"
    payload["full_response"] = "The answer is forty-five.\n#### 45"
    candidate = MutationCandidate.model_validate(payload)

    result = NumericMutationOracle().verify(candidate, reference_answer="50")

    assert result.passed is False
    assert "mismatches" in " ".join(result.reasons)


def test_objective_oracle_rejects_correct_final_answer() -> None:
    result = NumericMutationOracle().verify(make_valid_candidate(), reference_answer="44")

    assert result.passed is False
    assert "not wrong" in " ".join(result.reasons)


def test_candidate_normalizes_gsm8k_final_marker() -> None:
    payload = make_valid_candidate().model_dump()
    payload["final_answer"] = "#### 44"

    candidate = MutationCandidate.model_validate(payload)

    assert candidate.final_answer == "44"
