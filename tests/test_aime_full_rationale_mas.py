import hashlib

from topology_mas.execution import (
    AIME_FULL_RATIONALE_PROTOCOL,
    ExecutionSettings,
    SynchronousExecutionEngine,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.execution.aime import AIME_FULL_RATIONALE_PROMPT_VERSION
from topology_mas.models import DirectedEdge, GraphSpec, RunCondition, TaskInstance


class FullResponseGenerator:
    def __init__(self) -> None:
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        self.requests.append(request)
        raw = f"FULL RAW RESPONSE {request.request_id}\nFINAL_ANSWER: \\boxed{{042}}"
        return TextGenerationResult(
            raw_text=raw,
            model_name="fake-qwen",
            finish_reason="stop",
            input_tokens=100 + len(self.requests),
            output_tokens=20 + len(self.requests),
            latency_ms=1.0,
        )


def test_full_rationale_protocol_is_one_stage_and_broadcasts_verbatim() -> None:
    graph = GraphSpec(
        graph_id="aime-chain",
        node_count=3,
        edges=(
            DirectedEdge(source=0, target=1),
            DirectedEdge(source=1, target=2),
        ),
        readout_node=2,
        max_rounds=2,
    )
    task = TaskInstance(
        task_id="2026_AIME_I_P01",
        dataset="aime",
        split="test",
        prompt="Find the requested integer.",
        reference_answer="42",
        oracle_type="aime_integer",
    )
    backend = FullResponseGenerator()
    trace = SynchronousExecutionEngine(
        backend,
        settings=ExecutionSettings(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_output_tokens=16384,
            initial_state_policy="independent_per_run",
        ),
        protocol=AIME_FULL_RATIONALE_PROTOCOL,
    ).run(graph=graph, task=task, condition=RunCondition.CLEAN, seed=0)

    assert trace.prompt_version == AIME_FULL_RATIONALE_PROMPT_VERSION
    assert trace.total_model_calls == len(trace.turns) == len(backend.requests) == 6
    assert trace.total_backend_calls == 6
    assert all(message.raw_text == trace.turns[
        next(
            index
            for index, turn in enumerate(trace.turns)
            if turn.round_index == message.round_index and turn.node_id == message.sender
        )
    ].raw_output for message in trace.messages)
    assert all(message.metadata["public_message_equals_raw_output"] for message in trace.messages)

    round_one_receiver = next(
        turn for turn in trace.turns if turn.round_index == 1 and turn.node_id == 1
    )
    visible_round_one = "\n".join(
        message["content"] for message in round_one_receiver.prompt_messages
    )
    prior_self = next(turn for turn in trace.turns if turn.round_index == 0 and turn.node_id == 1)
    incoming_zero = next(
        message for message in trace.messages if message.round_index == 0 and message.sender == 0
    )
    assert prior_self.raw_output in visible_round_one
    assert incoming_zero.raw_text in visible_round_one
    assert "summary" not in visible_round_one.lower()

    audit = round_one_receiver.metadata["communication_audit"]
    assert audit["receiver_id"] == 1
    assert audit["round"] == 1
    assert audit["own_previous_response_tokens"] == prior_self.output_tokens
    assert audit["own_previous_response_sha256"] == hashlib.sha256(
        prior_self.raw_output.encode("utf-8")
    ).hexdigest()
    assert audit["incoming_responses"] == [
        {
            "sender_id": 0,
            "message_id": incoming_zero.message_id,
            "tokens": incoming_zero.output_tokens,
            "chars": len(incoming_zero.raw_text),
            "sha256": hashlib.sha256(incoming_zero.raw_text.encode("utf-8")).hexdigest(),
        }
    ]
    assert audit["total_prompt_tokens"] == round_one_receiver.input_tokens
    assert audit["generated_tokens"] == round_one_receiver.output_tokens
    assert audit["stop_reason"] == "stop"
    assert audit["context_overflow"] is False
    assert audit["context_truncation"] is False
    assert audit["summarization"] is False
    assert audit["message_compression"] is False

    round_two_readout = next(
        turn for turn in trace.turns if turn.round_index == 2 and turn.node_id == 2
    )
    visible_round_two = "\n".join(
        message["content"] for message in round_two_readout.prompt_messages
    )
    prior_readout = next(
        turn for turn in trace.turns if turn.round_index == 1 and turn.node_id == 2
    )
    incoming_one = next(
        message for message in trace.messages if message.round_index == 1 and message.sender == 1
    )
    assert prior_readout.raw_output in visible_round_two
    assert incoming_one.raw_text in visible_round_two


def test_full_rationale_protocol_rejects_length_truncated_answer() -> None:
    assert AIME_FULL_RATIONALE_PROTOCOL.parse_answer(
        "partial work with \\boxed{042}", finish_reason="length"
    ) is None
