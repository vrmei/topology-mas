import pytest
from pydantic import ValidationError

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
                depends_on=(),
            ),
            ArithmeticStep(
                step_id="s2",
                expression="s1 + 2",
                claimed_result="44",
                explanation="Add the remaining two items.",
                depends_on=("s1",),
            ),
        ),
        final_answer="44",
        full_response="Six groups are 42, then two more make 44.\n#### 44",
    )


def test_safe_arithmetic_evaluator() -> None:
    evaluator = SafeArithmeticEvaluator()

    assert str(evaluator.evaluate("(3 + 5) * 2 / 4")) == "4"
    assert str(
        evaluator.evaluate("s1 + 2", variables={"s1": evaluator.evaluate("42")})
    ) == "44"

    with pytest.raises(ValueError, match="undeclared variable"):
        evaluator.evaluate("s1 + 2")


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
    assert "do not exactly match" in " ".join(result.reasons)


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


def test_objective_oracle_rejects_disconnected_intermediate_step() -> None:
    payload = make_valid_candidate().model_dump()
    steps = list(payload["steps"])
    steps.insert(
        1,
        {
            "step_id": "unused",
            "expression": "10 + 5",
            "claimed_result": "15",
            "explanation": "A disconnected calculation.",
            "is_mutated": False,
            "depends_on": (),
        },
    )
    payload["steps"] = steps
    candidate = MutationCandidate.model_validate(payload)

    result = NumericMutationOracle().verify(candidate, reference_answer="50")

    assert result.passed is False
    assert "do not contribute" in " ".join(result.reasons)


def test_candidate_schema_requires_two_to_six_steps() -> None:
    payload = make_valid_candidate().model_dump()
    payload["steps"] = payload["steps"][:1]

    with pytest.raises(ValidationError):
        MutationCandidate.model_validate(payload)


def test_candidate_schema_rejects_non_arithmetic_result_mutation() -> None:
    payload = make_valid_candidate().model_dump()
    payload["mutation_type"] = "sign_error"

    with pytest.raises(ValidationError):
        MutationCandidate.model_validate(payload)


def test_candidate_schema_requires_dependencies_to_point_backward() -> None:
    payload = make_valid_candidate().model_dump()
    payload["steps"][0]["depends_on"] = ("s2",)

    with pytest.raises(ValidationError, match="earlier steps"):
        MutationCandidate.model_validate(payload)


def test_objective_oracle_accepts_a_branching_dependency_dag() -> None:
    candidate = MutationCandidate(
        candidate_id="branching",
        mutation_type="arithmetic_result",
        mutated_step_id="s1",
        steps=(
            ArithmeticStep(
                step_id="s1",
                expression="6 * 8",
                claimed_result="42",
                explanation="A multiplication slip.",
                is_mutated=True,
                depends_on=(),
            ),
            ArithmeticStep(
                step_id="s2",
                expression="s1 + 2",
                claimed_result="44",
                explanation="Continue the first branch.",
                depends_on=("s1",),
            ),
            ArithmeticStep(
                step_id="s3",
                expression="3 * 2",
                claimed_result="6",
                explanation="Compute an independent branch.",
                depends_on=(),
            ),
            ArithmeticStep(
                step_id="s4",
                expression="s2 + s3",
                claimed_result="50",
                explanation="Merge both branches.",
                depends_on=("s2", "s3"),
            ),
        ),
        final_answer="50",
        full_response="The merged result is 50.\n#### 50",
    )

    result = NumericMutationOracle().verify(candidate, reference_answer="56")

    assert result.passed is True


def test_objective_oracle_accepts_mutation_in_final_step() -> None:
    payload = make_valid_candidate().model_dump()
    payload["steps"][0]["claimed_result"] = "48"
    payload["steps"][0]["is_mutated"] = False
    payload["steps"][1]["expression"] = "s1 + 2"
    payload["steps"][1]["claimed_result"] = "49"
    payload["steps"][1]["is_mutated"] = True
    payload["mutated_step_id"] = "s2"
    payload["final_answer"] = "49"
    payload["full_response"] = "The final addition is 49.\n#### 49"
    candidate = MutationCandidate.model_validate(payload)

    result = NumericMutationOracle().verify(candidate, reference_answer="50")

    assert result.passed is True
