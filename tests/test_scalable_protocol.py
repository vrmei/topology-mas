import pytest

from topology_mas.execution.engine import SynchronousExecutionEngine
from topology_mas.execution.scalable_protocol import (
    DualChannelValidationError,
    SinglePassDualChannelGenerator,
    parse_dual_channel_output,
    scalable_gsm8k_protocol,
)
from topology_mas.execution.schemas import (
    ExecutionSettings,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.models import DirectedEdge, GraphSpec, RunCondition, TaskInstance


def count_words(text: str) -> int:
    return len(text.split())


def dual(answer: int, *, full: str = "careful derivation", summary: str = "key step") -> str:
    return (
        f"<FULL_SOLUTION>{full}\nFINAL_ANSWER: {answer}</FULL_SOLUTION>"
        f"<PUBLIC_SUMMARY>{summary}\nFINAL_ANSWER: {answer}</PUBLIC_SUMMARY>"
    )


class FixedBackend:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        self.requests.append(request)
        return TextGenerationResult(
            raw_text=self.output,
            model_name="model",
            finish_reason="stop",
            input_tokens=20,
            output_tokens=30,
            latency_ms=1.0,
        )


def request() -> TextGenerationRequest:
    from topology_mas.execution.schemas import ChatMessage

    return TextGenerationRequest(
        request_id="request-1",
        messages=(ChatMessage(role="user", content="problem"),),
        seed=1,
        temperature=0.7,
        max_output_tokens=100,
    )


def test_single_pass_generator_calls_backend_once_and_audits_both_channels() -> None:
    backend = FixedBackend(dual(42))
    generator = SinglePassDualChannelGenerator(
        backend,
        answer_parser=lambda text: "42" if "FINAL_ANSWER: 42" in text else None,
        token_counter=count_words,
        max_public_tokens=20,
    )

    result = generator.generate(request())

    assert len(backend.requests) == 1
    assert result.raw_text == backend.output
    assert result.metadata["backend_call_count"] == 1
    assert result.metadata["summary_validation_passed"] is True
    assert result.metadata["summary_answer_matches_raw"] is True
    assert result.metadata["summary_mode"] == "single_pass"
    assert result.metadata["public_output_tokens"] == 4


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (
            "<FULL_SOLUTION>FINAL_ANSWER: 42</FULL_SOLUTION>"
            "<PUBLIC_SUMMARY>FINAL_ANSWER: 41</PUBLIC_SUMMARY>",
            "differs",
        ),
        (
            "<FULL_SOLUTION>no answer</FULL_SOLUTION>"
            "<PUBLIC_SUMMARY>FINAL_ANSWER: 42</PUBLIC_SUMMARY>",
            "invented",
        ),
        ("FINAL_ANSWER: 42", "tags"),
    ],
)
def test_invalid_dual_channel_never_enters_execution(output: str, reason: str) -> None:
    backend = FixedBackend(output)
    generator = SinglePassDualChannelGenerator(
        backend,
        answer_parser=lambda text: (
            "42"
            if "FINAL_ANSWER: 42" in text
            else "41"
            if "FINAL_ANSWER: 41" in text
            else None
        ),
        token_counter=count_words,
        max_public_tokens=20,
    )

    with pytest.raises(DualChannelValidationError, match=reason):
        generator.generate(request())
    assert len(backend.requests) == 1


def test_engine_keeps_full_local_state_and_broadcasts_only_validated_summary() -> None:
    backend = FixedBackend(dual(42, full="PRIVATE UNIQUE WORK", summary="PUBLIC KEY STEP"))
    generator = SinglePassDualChannelGenerator(
        backend,
        answer_parser=lambda text: "42" if "FINAL_ANSWER: 42" in text else None,
        token_counter=count_words,
        max_public_tokens=20,
    )
    protocol = scalable_gsm8k_protocol(count_words)
    graph = GraphSpec(
        graph_id="chain",
        node_count=2,
        edges=(DirectedEdge(source=0, target=1),),
        readout_node=1,
        max_rounds=1,
    )
    task = TaskInstance(
        task_id="task-1",
        dataset="synthetic",
        split="test",
        prompt="What is 40 + 2?",
        reference_answer="42",
        oracle_type="numeric",
    )
    trace = SynchronousExecutionEngine(
        generator,
        settings=ExecutionSettings(
            temperature=0.7,
            max_output_tokens=100,
            initial_state_policy="independent_per_run",
        ),
        protocol=protocol,
    ).run(graph=graph, task=task, condition=RunCondition.CLEAN, seed=0)

    assert trace.total_model_calls == 3
    assert trace.total_backend_calls == 3
    assert "PRIVATE UNIQUE WORK" in trace.turns[0].raw_output
    assert "PRIVATE UNIQUE WORK" not in trace.messages[0].raw_text
    assert trace.messages[0].raw_text.startswith("PUBLIC KEY STEP")
    assert trace.messages[0].metadata["summary_validation_passed"] is True
    round_one_prompt = next(
        turn for turn in trace.turns if turn.node_id == 1 and turn.round_index == 1
    ).prompt_messages[-1]["content"]
    assert "<peer_public_summary>\nPUBLIC KEY STEP" in round_one_prompt
    assert "YOUR_PREVIOUS_FULL_SOLUTION:\nPRIVATE UNIQUE WORK" in round_one_prompt


def test_parser_rejects_text_outside_the_two_channels() -> None:
    with pytest.raises(DualChannelValidationError):
        parse_dual_channel_output(f"preface\n{dual(42)}")


def test_pool_mode_retains_invalid_raw_completion_with_explicit_failure() -> None:
    backend = FixedBackend("malformed but generated response")
    generator = SinglePassDualChannelGenerator(
        backend,
        answer_parser=lambda _: None,
        token_counter=count_words,
        strict_validation=False,
    )

    output = generator.generate(request())

    assert output.raw_text == "malformed but generated response"
    assert output.metadata["summary_validation_passed"] is False
    assert output.metadata["summary_mode"] == "single_pass_invalid_retained"
    assert "tags" in output.metadata["summary_validation_error"]
