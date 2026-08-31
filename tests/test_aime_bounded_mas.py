from collections.abc import Callable

from topology_mas.execution import (
    AIME_BOUNDED_PROTOCOL,
    BatchExecutionConfig,
    BatchExecutionRunner,
    ExecutionSettings,
    SynchronousExecutionEngine,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.execution.aime import AIME_BOUNDED_PROMPT_VERSION
from topology_mas.models import DirectedEdge, GraphSpec, RunCondition, TaskInstance


class CapturingGenerator:
    def __init__(self, response: Callable[[TextGenerationRequest], str]) -> None:
        self.requests: list[TextGenerationRequest] = []
        self._response = response

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        self.requests.append(request)
        return TextGenerationResult(
            raw_text=self._response(request),
            model_name="fake-qwen",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=40,
            latency_ms=1.0,
        )


def aime_task() -> TaskInstance:
    return TaskInstance(
        task_id="2026_AIME_I_P01",
        dataset="aime",
        split="test",
        prompt="Find the requested integer.",
        reference_answer="42",
        oracle_type="aime_integer",
    )


def test_aime_bounded_protocol_is_anonymous_and_candidate_free() -> None:
    graph = GraphSpec(
        graph_id="aime-fan-in",
        node_count=3,
        edges=(
            DirectedEdge(source=0, target=2),
            DirectedEdge(source=1, target=2),
        ),
        readout_node=2,
        max_rounds=1,
    )

    def response(request: TextGenerationRequest) -> str:
        answer = "041" if request.request_id.endswith("-t0-n0") else "042"
        return f"SOLUTION_SUMMARY:\nCompact audit.\nFINAL_ANSWER: \\boxed{{{answer}}}"

    generator = CapturingGenerator(response)
    trace = SynchronousExecutionEngine(
        generator,
        settings=ExecutionSettings(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_output_tokens=1024,
            initial_state_policy="independent_per_run",
        ),
        protocol=AIME_BOUNDED_PROTOCOL,
    ).run(
        graph=graph,
        task=aime_task(),
        condition=RunCondition.CLEAN,
        seed=0,
    )

    assert trace.prompt_version == AIME_BOUNDED_PROMPT_VERSION
    assert trace.final_parsed_answer == "42"
    assert trace.final_answer_state.value == "correct"
    assert all(request.max_output_tokens == 1024 for request in generator.requests)
    readout_update = next(
        request
        for request in generator.requests
        if request.request_id.endswith("-t1-n2")
    )
    visible = "\n".join(message.content for message in readout_update.messages)
    assert visible.count("<peer_message>") == 2
    assert "node 0" not in visible.lower()
    assert "node 1" not in visible.lower()
    assert "target_error" not in visible.lower()
    assert "YOUR_PREVIOUS_MESSAGE" in visible
    assert "preferably 512--768 tokens" in visible


def test_aime_protocol_rejects_gsm8k_oracle_type() -> None:
    incompatible = aime_task().model_copy(update={"oracle_type": "numeric"})
    generator = CapturingGenerator(
        lambda _: "SOLUTION_SUMMARY:\nWork.\nFINAL_ANSWER: \\boxed{042}"
    )
    graph = GraphSpec(
        graph_id="two-node",
        node_count=2,
        edges=(DirectedEdge(source=0, target=1),),
        readout_node=1,
        max_rounds=1,
    )

    try:
        SynchronousExecutionEngine(
            generator,
            protocol=AIME_BOUNDED_PROTOCOL,
        ).run(
            graph=graph,
            task=incompatible,
            condition=RunCondition.CLEAN,
            seed=0,
        )
    except ValueError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("AIME protocol accepted an incompatible oracle type")


def test_aime_protocol_never_scores_a_length_truncated_completion() -> None:
    assert (
        AIME_BOUNDED_PROTOCOL.parse_answer(
            "SOLUTION_SUMMARY:\npartial \\boxed{123}",
            finish_reason="length",
        )
        is None
    )


def test_clean_aime_batch_runs_independent_round_zero_end_to_end(tmp_path) -> None:
    generator = CapturingGenerator(
        lambda _: "SOLUTION_SUMMARY:\nAuditable work.\nFINAL_ANSWER: \\boxed{042}"
    )
    graph = GraphSpec(
        graph_id="aime-two-node",
        node_count=2,
        edges=(DirectedEdge(source=0, target=1),),
        readout_node=1,
        max_rounds=1,
    )
    runner = BatchExecutionRunner(
        SynchronousExecutionEngine(
            generator,
            settings=ExecutionSettings(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_output_tokens=1024,
                initial_state_policy="independent_per_run",
            ),
            protocol=AIME_BOUNDED_PROTOCOL,
        ),
        config=BatchExecutionConfig(
            experiment_seeds=(0,),
            assignment_seeds=(0,),
            include_attacks=False,
            initial_state_policy="independent_per_run",
            requested_model="fake-qwen",
            expected_returned_model="fake-qwen",
        ),
        output_dir=tmp_path,
        max_workers=1,
    )

    outcomes, summary = runner.run(tasks=(aime_task(),), graphs=(graph,))

    assert len(outcomes) == 1
    assert summary.expected_runs == summary.completed_runs == 1
    assert summary.clean_runs == 1
    assert summary.attack_runs == 0
    # Both nodes independently solve Round 0; only the readout can still affect
    # the endpoint in Round 1 under causal-cone pruning.
    assert summary.trace_model_calls == 3
    assert len(generator.requests) == 3
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert AIME_BOUNDED_PROMPT_VERSION in manifest
    assert '"initial_state_policy": "independent_per_run"' in manifest
