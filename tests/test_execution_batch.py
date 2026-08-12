import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from topology_mas.analysis import analyze_batch, load_complete_batch, write_analysis
from topology_mas.execution import (
    BatchDisposition,
    BatchExecutionConfig,
    BatchExecutionConflictError,
    BatchExecutionRunner,
    ExecutionSettings,
    StateConsistentReplayGenerator,
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
        for message in build_node_messages(task(), previous_output=None, incoming_messages=())
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
        max_workers=4,
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
    assert (tmp_path / "inputs" / "tasks.jsonl").exists()
    assert (tmp_path / "inputs" / "graphs.jsonl").exists()
    assert (
        len(
            (tmp_path / "inputs" / "round_zero_index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 6
    )
    assert (
        len(
            (tmp_path / "inputs" / "adversarial_answers.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 1
    )
    assert all(outcome.disposition is BatchDisposition.GENERATED for outcome in outcomes)

    resumed_outcomes, resumed_summary = batch.run(**run_kwargs)

    assert len(generator.requests) == 72
    assert resumed_summary.generated_runs == 0
    assert resumed_summary.cached_runs == 24
    assert resumed_summary.new_model_calls == 0
    assert resumed_summary.trace_model_calls == 72
    assert all(outcome.disposition is BatchDisposition.CACHED for outcome in resumed_outcomes)


def test_independent_round_zero_is_generated_per_graph_and_condition(
    tmp_path: Path,
) -> None:
    generator = CountingGenerator(lambda _: "Independent solution.\nFINAL_ANSWER: 42")
    batch = BatchExecutionRunner(
        SynchronousExecutionEngine(
            generator,
            settings=ExecutionSettings(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                initial_state_policy="independent_per_run",
            ),
        ),
        config=BatchExecutionConfig(
            experiment_seeds=(0,),
            assignment_seeds=(0,),
            initial_state_policy="independent_per_run",
            requested_model="fake-model",
            expected_returned_model="fake-model",
        ),
        output_dir=tmp_path,
        max_workers=4,
    )

    outcomes, summary = batch.run(
        tasks=(task(),),
        graphs=graphs(),
        adversarial_answers={"task-1": adversarial_answer()},
    )

    assert len(outcomes) == 6
    assert summary.clean_runs == 2
    assert summary.attack_runs == 4
    assert summary.state_replay_cache_hits == 0
    assert summary.trace_backend_calls == summary.trace_model_calls
    stored = [
        json.loads(Path(outcome.trace_path).read_text(encoding="utf-8"))["trace"]
        for outcome in outcomes
    ]
    normal_round_zero = [
        turn
        for trace in stored
        for turn in trace["turns"]
        if turn["round_index"] == 0 and not turn["metadata"].get("attack_replay")
    ]
    assert normal_round_zero
    assert all(turn["metadata"]["generator_called"] for turn in normal_round_zero)
    assert not any(
        turn["metadata"].get("round_zero_cache_replay") for turn in normal_round_zero
    )
    round_zero_seeds = [turn["generation_seed"] for turn in normal_round_zero]
    assert len(round_zero_seeds) == len(set(round_zero_seeds))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["initial_state_policy"] == "independent_per_run"
    assert manifest["round_zero_fingerprint"] is None
    assert (tmp_path / "inputs" / "round_zero_index.jsonl").read_text() == ""


def test_batch_reports_logical_calls_separately_from_state_replay_calls(
    tmp_path: Path,
) -> None:
    backend = CountingGenerator(lambda _: "Updated solution.\nFINAL_ANSWER: 42")
    replay = StateConsistentReplayGenerator(
        backend,
        cache_dir=tmp_path / "state-replay",
        requested_model="fake-model",
        expected_returned_model="fake-model",
        model_fingerprint="a" * 64,
        namespace="batch-test-v1",
    )
    batch = BatchExecutionRunner(
        SynchronousExecutionEngine(
            replay,
            settings=ExecutionSettings(
                state_transition_policy="state-consistent-replay-v1"
            ),
        ),
        config=BatchExecutionConfig(
            experiment_seeds=(0, 1),
            assignment_seeds=(10, 11),
            requested_model="fake-model",
            expected_returned_model="fake-model",
            state_replay_model_fingerprint="a" * 64,
            state_replay_namespace="batch-test-v1",
        ),
        output_dir=tmp_path / "batch",
        max_workers=4,
    )

    _, summary = batch.run(
        tasks=(task(),),
        graphs=graphs(),
        round_zero_records=round_zero_records(),
        adversarial_answers={"task-1": adversarial_answer()},
    )

    assert summary.trace_model_calls == 72
    assert 0 < summary.trace_backend_calls < summary.trace_model_calls
    assert summary.new_model_calls == summary.trace_backend_calls == len(backend.requests)
    assert summary.state_replay_cache_hits == (
        summary.trace_model_calls - summary.trace_backend_calls
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


def test_batch_rejects_a_tampered_input_snapshot(tmp_path: Path) -> None:
    generator = CountingGenerator(lambda _: "Updated solution.\nFINAL_ANSWER: 42")
    batch = runner(tmp_path, generator)
    run_kwargs = {
        "tasks": (task(),),
        "graphs": graphs(),
        "round_zero_records": round_zero_records(),
        "adversarial_answers": {"task-1": adversarial_answer()},
    }
    batch.run(**run_kwargs)
    graph_path = tmp_path / "inputs" / "graphs.jsonl"
    graph_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(BatchExecutionConflictError, match="graph snapshot"):
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


def test_analysis_uses_strict_pairs_and_exports_classical_initial_states(
    tmp_path: Path,
) -> None:
    def response(request: TextGenerationRequest) -> str:
        combined = "\n".join(message.content for message in request.messages)
        if "Plausible wrong solution." in combined:
            return "Plausible wrong solution.\nFINAL_ANSWER: 41"
        return "Updated solution.\nFINAL_ANSWER: 42"

    batch_dir = tmp_path / "batch"
    batch = runner(batch_dir, CountingGenerator(response))
    batch.run(
        tasks=(task(),),
        graphs=graphs(),
        round_zero_records=round_zero_records(),
        adversarial_answers={"task-1": adversarial_answer()},
    )

    result = analyze_batch(load_complete_batch(batch_dir))

    assert result.manifest.analyzed_runs == 24
    assert result.manifest.paired_attacks == 16
    assert len(result.classical_initial_states) == 8
    assert len(result.graph_metrics) == 2
    assert any(
        pair.observed_target_count_by_round[0] > 0 and pair.induced_target_count_by_round[0] == 0
        for pair in result.paired_attacks
    )
    for metric in result.graph_metrics:
        assert metric.utility == 1.0
        assert metric.r_mean == 0.0
        assert metric.r_worst == 0.0
        assert metric.d_mean == 1.0
        assert metric.d_max == 1.0
        assert metric.final_target_match_rate == 1.0
        assert metric.induced_readout_target_rate == 1.0
        assert metric.correct_to_target_flip_rate == 1.0

    output_dir = tmp_path / "analysis"
    write_analysis(output_dir, result)
    assert (output_dir / "graph_metrics.csv").exists()
    assert len((output_dir / "paired_attacks.jsonl").read_text(encoding="utf-8").splitlines()) == 16


def test_analysis_rejects_an_incomplete_trace_set(tmp_path: Path) -> None:
    generator = CountingGenerator(lambda _: "FINAL_ANSWER: 42")
    batch_dir = tmp_path / "batch"
    outcomes, _ = runner(batch_dir, generator).run(
        tasks=(task(),),
        graphs=graphs(),
        round_zero_records=round_zero_records(),
        adversarial_answers={"task-1": adversarial_answer()},
    )
    Path(outcomes[0].trace_path).unlink()

    with pytest.raises(BatchExecutionConflictError, match="trace set"):
        load_complete_batch(batch_dir)
