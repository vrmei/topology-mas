import json
from pathlib import Path

import pytest

from topology_mas.execution import (
    ExecutionInputError,
    RoundZeroCache,
    RoundZeroCacheConfig,
    RoundZeroGenerator,
    TextGenerationRequest,
    TextGenerationResult,
    load_adversarial_answer_index,
    load_round_zero_collection,
    load_selected_adversarial_answers,
)
from topology_mas.models import TaskInstance
from topology_mas.mutation.schemas import (
    ArithmeticStep,
    CandidateEvaluation,
    MutationCandidate,
    MutationPipelineConfig,
    MutationRunResult,
    ObjectiveOracleResult,
    PlausibilityOracleResult,
)
from topology_mas.mutation.storage import MutationArtifactStore


class FakeGenerator:
    def generate(self, _: TextGenerationRequest) -> TextGenerationResult:
        return TextGenerationResult(
            raw_text="Independent answer.\nFINAL_ANSWER: 42",
            model_name="fake-model",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5,
        )


def task() -> TaskInstance:
    return TaskInstance(
        task_id="task-1",
        dataset="synthetic",
        split="test",
        prompt="What is 40 + 2?",
        reference_answer="42",
        oracle_type="numeric",
    )


def write_round_zero(root: Path) -> None:
    RoundZeroGenerator(
        FakeGenerator(),
        config=RoundZeroCacheConfig(
            replica_count=3,
            seeds=(0,),
            requested_model="fake-model",
            expected_returned_model="fake-model",
        ),
        cache=RoundZeroCache(root),
    ).generate((task(),))


def selected_mutation_result(*, selected: bool = True) -> MutationRunResult:
    candidate = MutationCandidate(
        candidate_id="c01",
        mutation_type="arithmetic_result",
        mutated_step_id="s1",
        steps=(
            ArithmeticStep(
                step_id="s1",
                expression="40 + 2",
                claimed_result="41",
                explanation="A local addition slip.",
                is_mutated=True,
                depends_on=(),
            ),
        ),
        final_answer="41",
        full_response="A plausible local arithmetic slip.\n#### 41",
    )
    evaluation = CandidateEvaluation(
        candidate=candidate,
        objective=ObjectiveOracleResult(passed=True),
        plausibility=PlausibilityOracleResult(
            model_plausible=True,
            plausible=True,
            local_error_plausibility=0.9,
            global_coherence=0.9,
            subtlety=0.8,
            minimality=1.0,
            overall_score=0.9,
            returned_model="deepseek-v4-flash",
        ),
    )
    return MutationRunResult(
        task_id="task-1",
        config=MutationPipelineConfig(candidate_count=1),
        generator_request=(),
        generator_response={},
        evaluations=(evaluation,),
        selected_candidate_id="c01" if selected else None,
    )


def write_mutation_batch(root: Path, *, selected: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_manifest.json").write_text(
        json.dumps({"task_ids": ["task-1"]}),
        encoding="utf-8",
    )
    MutationArtifactStore(root / "tasks").save(
        task(), selected_mutation_result(selected=selected)
    )


def test_round_zero_loader_requires_every_manifest_record(tmp_path: Path) -> None:
    write_round_zero(tmp_path)

    manifest, records = load_round_zero_collection(tmp_path)

    assert manifest.intended_record_count == 3
    assert [record.replica_slot for record in records] == [0, 1, 2]

    (tmp_path / "records" / "task-1" / "seed_0" / "replica_2.json").unlink()
    with pytest.raises(ExecutionInputError, match="incomplete"):
        load_round_zero_collection(tmp_path)


def test_mutation_loader_converts_only_the_eligible_selected_candidate(
    tmp_path: Path,
) -> None:
    write_mutation_batch(tmp_path)

    answers = load_selected_adversarial_answers(tmp_path)

    answer = answers["task-1"]
    assert answer.accepted
    assert answer.target_answer == "41"
    assert answer.rationale.endswith("#### 41")
    assert answer.metadata["candidate_id"] == "c01"


def test_mutation_loader_rejects_a_task_without_selection(tmp_path: Path) -> None:
    write_mutation_batch(tmp_path, selected=False)

    with pytest.raises(ExecutionInputError, match="no eligible selected mutation"):
        load_selected_adversarial_answers(tmp_path)


def test_adversarial_answer_index_loader_accepts_audited_jsonl(tmp_path: Path) -> None:
    result = selected_mutation_result()
    from topology_mas.mutation.pipeline import MutationPipeline

    answer = MutationPipeline.to_adversarial_answer(result)
    path = tmp_path / "selected_adversarial_answers.jsonl"
    path.write_text(answer.model_dump_json() + "\n", encoding="utf-8")

    loaded = load_adversarial_answer_index(path)

    assert loaded == {"task-1": answer}


def test_adversarial_answer_index_loader_rejects_duplicates(tmp_path: Path) -> None:
    result = selected_mutation_result()
    from topology_mas.mutation.pipeline import MutationPipeline

    answer = MutationPipeline.to_adversarial_answer(result)
    path = tmp_path / "selected_adversarial_answers.jsonl"
    path.write_text(
        answer.model_dump_json() + "\n" + answer.model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ExecutionInputError, match="duplicate"):
        load_adversarial_answer_index(path)
