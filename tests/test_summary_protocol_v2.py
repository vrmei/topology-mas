import json

import pytest

from topology_mas.execution.batch import BatchExecutionStore, ExecutionRunSpec
from topology_mas.execution.engine import ExecutionInterruptedError, SynchronousExecutionEngine
from topology_mas.execution.schemas import (
    ExecutionSettings,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.execution.summary_protocol_v2 import (
    SUMMARY_PROTOCOL_V2,
    SolveThenSummarizeGeneratorV2,
    SummaryProtocolV2Cache,
    SummaryProtocolV2Error,
    freeze_attack_public_summary_v2,
    parse_summary_envelope_v2,
    summary_protocol_v2,
    validate_public_summary_v2,
)
from topology_mas.models import (
    AdversarialAnswer,
    DirectedEdge,
    GraphSpec,
    OracleStatus,
    RunCondition,
    TaskInstance,
)


def count_words(text: str) -> int:
    return len(text.split())


def task() -> TaskInstance:
    return TaskInstance(
        task_id="aime-test",
        dataset="aime",
        split="test",
        prompt="Find the requested integer.",
        reference_answer="42",
        oracle_type="aime_integer",
    )


def request() -> TextGenerationRequest:
    protocol = summary_protocol_v2(count_words)
    return TextGenerationRequest(
        request_id="request-v2",
        messages=protocol.build_messages(task(), previous_output=None, incoming_messages=()),
        seed=7,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_output_tokens=16384,
    )


class ScriptedBackend:
    def __init__(self, summary_outputs: list[str] | None = None) -> None:
        self.requests: list[TextGenerationRequest] = []
        self.summary_outputs = summary_outputs or [
            "SOLUTION_SUMMARY:\nUse the decisive identity.\n"
            "FINAL_ANSWER: \\boxed{042}"
        ]
        self.summary_index = 0

    def generate(self, generation_request: TextGenerationRequest) -> TextGenerationResult:
        self.requests.append(generation_request)
        is_summary = "summary-v2" in generation_request.request_id
        if is_summary:
            index = min(self.summary_index, len(self.summary_outputs) - 1)
            raw = self.summary_outputs[index]
            self.summary_index += 1
        else:
            raw = "Complete private reasoning.\nFINAL_ANSWER: \\boxed{042}"
        return TextGenerationResult(
            raw_text=raw,
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=count_words(raw),
            latency_ms=1.0,
        )


def test_v2_uses_separate_calls_and_python_assembles_the_envelope(tmp_path) -> None:
    backend = ScriptedBackend()
    generator = SolveThenSummarizeGeneratorV2(
        backend,
        cache=SummaryProtocolV2Cache(tmp_path),
        token_counter=count_words,
    )

    result = generator.generate(request())
    envelope = parse_summary_envelope_v2(result.raw_text)

    assert len(backend.requests) == 2
    assert "XML" not in backend.requests[0].messages[-1].content
    assert "<FULL_SOLUTION>" not in envelope.full_solution
    assert envelope.full_solution.endswith("FINAL_ANSWER: \\boxed{042}")
    assert envelope.public_summary.startswith("SOLUTION_SUMMARY:")
    assert result.metadata["generation_pipeline"] == SUMMARY_PROTOCOL_V2
    assert result.metadata["backend_call_count"] == 2
    assert result.metadata["summary_validation_passed"] is True


def test_failed_summary_is_not_retried_and_does_not_repeat_the_solve(tmp_path) -> None:
    backend = ScriptedBackend(
        ["I ignored the required structure. FINAL_ANSWER: \\boxed{042}"]
    )
    generator = SolveThenSummarizeGeneratorV2(
        backend,
        cache=SummaryProtocolV2Cache(tmp_path),
        token_counter=count_words,
    )

    with pytest.raises(SummaryProtocolV2Error):
        generator.generate(request())

    solve_requests = [row for row in backend.requests if "solve-v2" in row.request_id]
    summary_requests = [row for row in backend.requests if "summary-v2" in row.request_id]
    assert len(solve_requests) == 1
    assert len(summary_requests) == 1


def test_validated_summary_cache_reuses_only_the_transform(tmp_path) -> None:
    backend = ScriptedBackend()
    cache = SummaryProtocolV2Cache(tmp_path)
    first = SolveThenSummarizeGeneratorV2(
        backend, cache=cache, token_counter=count_words
    ).generate(request())
    call_count = len(backend.requests)
    second = SolveThenSummarizeGeneratorV2(
        backend, cache=cache, token_counter=count_words
    ).generate(request())

    assert len(backend.requests) == call_count
    assert first.raw_text == second.raw_text
    assert second.metadata["full_cache_hit"] is True
    assert second.metadata["summary_cache_hit"] is True
    assert second.metadata["backend_call_count"] == 0


def test_length_stopped_full_is_preserved_as_unparsed_and_not_resolved(tmp_path) -> None:
    class LengthBackend(ScriptedBackend):
        def generate(self, generation_request: TextGenerationRequest) -> TextGenerationResult:
            self.requests.append(generation_request)
            if "summary-v2" in generation_request.request_id:
                raw = "SOLUTION_SUMMARY:\nThe draft was incomplete.\nFINAL_ANSWER: UNPARSED"
                finish = "stop"
            else:
                raw = "Truncated work with an intermediate \\boxed{123}"
                finish = "length"
            return TextGenerationResult(
                raw_text=raw,
                model_name="Qwen/Qwen3-4B-Instruct-2507",
                finish_reason=finish,
                input_tokens=10,
                output_tokens=count_words(raw),
            )

    backend = LengthBackend()
    protocol = summary_protocol_v2(count_words)
    result = SolveThenSummarizeGeneratorV2(
        backend,
        cache=SummaryProtocolV2Cache(tmp_path),
        token_counter=count_words,
    ).generate(request())

    assert len(backend.requests) == 2
    assert protocol.parse_answer(result.raw_text, finish_reason=result.finish_reason) is None
    assert result.finish_reason == "length"
    assert result.metadata["full_parsed_answer"] is None
    assert "EXTRACTED_FULL_ANSWER: UNPARSED" in backend.requests[1].messages[-1].content


def test_unparsed_full_cannot_gain_an_answer_in_the_summary() -> None:
    with pytest.raises(ValueError, match="invented"):
        validate_public_summary_v2(
            "SOLUTION_SUMMARY:\nA guessed result.\nFINAL_ANSWER: \\boxed{123}",
            full_answer=None,
            finish_reason="stop",
            token_counter=count_words,
        )


def test_fixed_attack_uses_the_same_strict_v2_summary_interface() -> None:
    attack = AdversarialAnswer(
        task_id="aime-test",
        target_answer="41",
        rationale="Plausible but wrong derivation.\nFINAL_ANSWER: \\boxed{041}",
        mutation_type="test",
        oracle_status=OracleStatus.PASSED,
    )
    frozen = freeze_attack_public_summary_v2(
        attack,
        public_summary=(
            "SOLUTION_SUMMARY:\nPlausible wrong key step.\n"
            "FINAL_ANSWER: \\boxed{041}"
        ),
        token_counter=count_words,
    )
    protocol = summary_protocol_v2(count_words)
    envelope = parse_summary_envelope_v2(protocol.adversarial_output(frozen))

    assert envelope.full_solution.startswith("Plausible but wrong")
    assert protocol.public_message(protocol.adversarial_output(frozen)).startswith(
        "SOLUTION_SUMMARY:"
    )
    assert frozen.metadata["summary_protocol"] == SUMMARY_PROTOCOL_V2


def test_all_failed_summaries_raise_with_complete_attempts(tmp_path) -> None:
    backend = ScriptedBackend(["bad summary"])
    generator = SolveThenSummarizeGeneratorV2(
        backend,
        cache=SummaryProtocolV2Cache(tmp_path),
        token_counter=count_words,
    )

    with pytest.raises(SummaryProtocolV2Error) as captured:
        generator.generate(request())

    payload = captured.value.to_failure_payload()
    assert payload["full_completion"]["raw_text"].startswith("Complete private")
    assert len(payload["summary_attempts"]) == 1
    assert all(item["raw_text"] == "bad summary" for item in payload["summary_attempts"])
    assert len([row for row in backend.requests if "solve-v2" in row.request_id]) == 1


def test_engine_failure_retains_partial_trace_and_failed_completion(tmp_path) -> None:
    class RoundFailureBackend(ScriptedBackend):
        def generate(self, generation_request: TextGenerationRequest) -> TextGenerationResult:
            self.requests.append(generation_request)
            if "-t1-n1-summary-v2" in generation_request.request_id:
                raw = "bad summary"
            elif "summary-v2" in generation_request.request_id:
                raw = "SOLUTION_SUMMARY:\nFaithful.\nFINAL_ANSWER: \\boxed{042}"
            else:
                raw = (
                    f"Full solution for {generation_request.request_id}.\n"
                    "FINAL_ANSWER: \\boxed{042}"
                )
            return TextGenerationResult(
                raw_text=raw,
                model_name="Qwen/Qwen3-4B-Instruct-2507",
                finish_reason="stop",
                input_tokens=10,
                output_tokens=count_words(raw),
            )

    engine = SynchronousExecutionEngine(
        SolveThenSummarizeGeneratorV2(
            RoundFailureBackend(),
            cache=SummaryProtocolV2Cache(tmp_path / "cache"),
            token_counter=count_words,
        ),
        settings=ExecutionSettings(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_output_tokens=16384,
            initial_state_policy="independent_per_run",
            generation_pipeline=SUMMARY_PROTOCOL_V2,
        ),
        protocol=summary_protocol_v2(count_words),
    )
    graph = GraphSpec(
        graph_id="chain",
        node_count=2,
        edges=(DirectedEdge(source=0, target=1),),
        readout_node=1,
        max_rounds=1,
    )
    with pytest.raises(ExecutionInterruptedError) as captured:
        engine.run(graph=graph, task=task(), condition=RunCondition.CLEAN, seed=0)

    spec = ExecutionRunSpec(
        run_spec_id="failed-run",
        task_id=task().task_id,
        graph_id=graph.graph_id,
        experiment_seed=0,
        assignment_seed=0,
        condition=RunCondition.CLEAN,
    )
    path = BatchExecutionStore(tmp_path / "batch").save_failure(spec, captured.value)
    saved = json.loads(path.read_text(encoding="utf-8"))
    payload = saved["failure_payload"]
    assert payload["round_index"] == 1
    assert payload["node_id"] == 1
    assert len(payload["partial_turns"]) == 2
    assert len(payload["partial_messages"]) == 1
    assert len(payload["cause"]["summary_attempts"]) == 1
    assert payload["cause"]["full_completion"]["raw_text"].startswith("Full solution")
