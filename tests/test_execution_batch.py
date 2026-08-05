import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from topology_mas.execution import (
    BatchDisposition,
    BatchExecutionConfig,
    BatchExecutionConflictError,
    BatchExecutionRunner,
    SynchronousExecutionEngine,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.execution.answers import classify_numeric_answer, parse_numeric_answer
from topology_mas.execution.prompts import PROMPT_VERSION, build_node_messages
from topology_mas.execution.round_zero import RoundZeroRecord
from topology_mas.execution.seeding import round_zero_replica_seed
from topology_mas.models import (
    AdversarialAnswer,
    DirectedEdge,
    GraphSpec,
    OracleStatus,
    TaskInstance,
)


class CountingGenerator:
    def __init__(self, response: Callable[[TextGenerationRequest], str]) -> None:
        self.requests: list[TextGenerationRequest] = []
        self._response = response

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        self.requests.append(request)
        return TextGenerationResult(
            raw_text=self._response(request),
            model_name="fake-model",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
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


def graphs() -> tuple[GraphSpec, ...]:
    return (
        GraphSpec(
            graph_id="chain",
            node_count=3,
            edges=(DirectedEdge(source=0, target=1), DirectedEdge(source=1, target=2)),
            readout_node=2,
            max_rounds=2,
        ),
        GraphSpec(
            graph_id="star",
            node_count=3,
            edges=(DirectedEdge(source=0, target=2), DirectedEdge(source=1, target=2)),
            readout_node=2,
            max_rounds=2,
        ),
    )


def round_zero_records() -> tuple[RoundZeroRecord, ...]:
    prompt_messages = tuple(
        message.model_dump()
        for message in build_node_messages(
            task(), previous_output=None, incoming_messages=()
        )
    )
    records: list[RoundZeroRecord] = []
    for experiment_seed in (0, 1):
        for replica_slot, answer in enumerate(("42", "41", "40")):
            raw_output = f"Cached independent solution.\nFINAL_ANSWER: {answer}"
            parsed = parse_numeric_answer(raw_output)
            records.append(
                RoundZeroRecord(
                    record_id=f"record-{experiment_seed}-{replica_slot}",
                    request_fingerprint=hashlib.sha256(
                        f"{experiment_seed}\0{replica_slot}".encode()
                    ).hexdigest(),
                    task_id="task-1",
                    replica_slot=replica_slot,
                    experiment_seed=experiment_seed,
                    generation_seed=round_zero_replica_seed(
                        experiment_seed=experiment_seed,
                        task_id="task-1",
                        replica_slot=replica_slot,
                    ),
                    prompt_version=PROMPT_VERSION,
                    prompt_messages=prompt_messages,
                    raw_output=raw_output,
                    parsed_answer=parsed,
                    answer_state=classify_numeric_answer(
                        parsed,
                        reference_answer="42",
                        target_answer=None,
                    ),
                    is_correct=answer == "42",
                    requested_model="fake-model",
                    returned_model="fake-model",
                    input_tokens=10,
                    output_tokens=5,
                )
            )
    return tuple(records)


def adversarial_answer() -> AdversarialAnswer:
    return AdversarialAnswer(
        task_id="task-1",
        target_answer="41",
        rationale="Plausible wrong solution.\n#### 41",
        mutation_type="arithmetic_result",
        oracle_status=OracleStatus.PASSED,
        plausibility_score=0.9,
    )


def runner(tmp_path: Path, generator: CountingGenerator) -> BatchExecutionRunner:
    return BatchExecutionRunner(
        SynchronousExecutionEngine(generator),
        config=BatchExecutionConfig(
            experiment_seeds=(0, 1),
            assignment_seeds=(10, 11),
            requested_model="fake-model",
            expected_returned_model="fake-model",
        ),
        output_dir=tmp_path,
    )


def test_batch_builds_complete_paired_matrix_and_resumes(tmp_path: Path) -> None:
    generator = CountingGenerator(lambda _: "Updated solution.\nFINAL_ANSWER: 42")
    batch = runner(tmp_path, generator)
    run_kwargs = {
        "tasks": (task(),),
        "graphs": graphs(),
        "round_zero_records": round_zero_records(),
        "adversarial_answers": {"task-1": adversarial_answer()},
    }

    outcomes, summary = batch.run(**run_kwargs)

    assert len(outcomes) == 24
    assert summary.expected_runs == summary.completed_runs == 24
    assert summary.clean_runs == 8
    assert summary.attack_runs == 16
    assert summary.generated_runs == 24
    assert summary.cached_runs == 0
    assert summary.new_model_calls == summary.trace_model_calls == 72
    assert len(generator.requests) == 72
    assert len(list((tmp_path / "traces").glob("*.json"))) == 24
    assert all(outcome.disposition is BatchDisposition.GENERATED for outcome in outcomes)

    resumed_outcomes, resumed_summary = batch.run(**run_kwargs)

    assert len(generator.requests) == 72
    assert resumed_summary.generated_runs == 0
    assert resumed_summary.cached_runs == 24
    assert resumed_summary.new_model_calls == 0
    assert resumed_summary.trace_model_calls == 72
    assert all(
        outcome.disposition is BatchDisposition.CACHED for outcome in resumed_outcomes
    )


def test_batch_regenerates_only_a_missing_atomic_trace(tmp_path: Path) -> None:
    generator = CountingGenerator(lambda _: "Updated solution.\nFINAL_ANSWER: 42")
    batch = runner(tmp_path, generator)
    run_kwargs = {
        "tasks": (task(),),
        "graphs": graphs(),
        "round_zero_records": round_zero_records(),
        "adversarial_answers": {"task-1": adversarial_answer()},
    }
    outcomes, _ = batch.run(**run_kwargs)
    first_path = Path(outcomes[0].trace_path)
    first_calls = outcomes[0].model_calls
    first_path.unlink()

    _, resumed = batch.run(**run_kwargs)

    assert resumed.generated_runs == 1
    assert resumed.cached_runs == 23
    assert resumed.new_model_calls == first_calls
    assert len(generator.requests) == 72 + first_calls


def test_batch_rejects_a_tampered_cached_trace(tmp_path: Path) -> None:
    generator = CountingGenerator(lambda _: "Updated solution.\nFINAL_ANSWER: 42")
    batch = runner(tmp_path, generator)
    run_kwargs = {
        "tasks": (task(),),
        "graphs": graphs(),
        "round_zero_records": round_zero_records(),
        "adversarial_answers": {"task-1": adversarial_answer()},
    }
    outcomes, _ = batch.run(**run_kwargs)
    path = Path(outcomes[0].trace_path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["trace"]["final_raw_output"] = "tampered"
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(BatchExecutionConflictError, match="fingerprint"):
        batch.run(**run_kwargs)


def test_batch_fails_preflight_before_calls_when_round_zero_is_missing(
    tmp_path: Path,
) -> None:
    generator = CountingGenerator(lambda _: "FINAL_ANSWER: 42")
    incomplete = round_zero_records()[:-1]

    with pytest.raises(ValueError, match="round-zero records are missing"):
        runner(tmp_path, generator).run(
            tasks=(task(),),
            graphs=graphs(),
            round_zero_records=incomplete,
            adversarial_answers={"task-1": adversarial_answer()},
        )

    assert generator.requests == []


def test_batch_rejects_an_attack_target_equal_to_the_reference(tmp_path: Path) -> None:
    generator = CountingGenerator(lambda _: "FINAL_ANSWER: 42")
    invalid = adversarial_answer().model_copy(update={"target_answer": "42"})

    with pytest.raises(ValueError, match="equals the reference"):
        runner(tmp_path, generator).run(
            tasks=(task(),),
            graphs=graphs(),
            round_zero_records=round_zero_records(),
            adversarial_answers={"task-1": invalid},
        )

    assert generator.requests == []
