import pytest

from topology_mas.execution.engine import SynchronousExecutionEngine
from topology_mas.execution.scalable_protocol import (
    SCALABLE_PUBLIC_SUMMARY_MAX_TOKENS,
    DualChannelValidationError,
    SinglePassDualChannelGenerator,
    freeze_attack_public_summary,
    parse_dual_channel_output,
    scalable_gsm8k_protocol,
)
from topology_mas.execution.schemas import (
    ExecutionSettings,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.models import (
    AdversarialAnswer,
    AttackMode,
    DirectedEdge,
    GraphSpec,
    MessageRecord,
    MessageType,
    NodeSourceType,
    OracleStatus,
    RunCondition,
    TaskInstance,
)


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
    assert trace.messages[0].message_type is MessageType.SUMMARY
    assert trace.messages[0].summary_source_response_id == trace.turns[0].response_id
    assert trace.turns[0].source_type is NodeSourceType.NATURAL
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


def test_protocol_rejects_any_peer_message_that_is_not_a_summary() -> None:
    protocol = scalable_gsm8k_protocol(count_words)
    legacy_message = MessageRecord(
        message_id="message",
        run_id="run",
        task_id="task-1",
        graph_id="graph",
        round_index=0,
        sender=0,
        recipients=(1,),
        raw_text="hidden full rationale",
    )

    with pytest.raises(ValueError, match="only summary"):
        protocol.build_messages(
            TaskInstance(
                task_id="task-1",
                dataset="synthetic",
                split="test",
                prompt="What is 40 + 2?",
                reference_answer="42",
                oracle_type="numeric",
            ),
            previous_output=dual(42),
            incoming_messages=(legacy_message,),
        )


def test_summary_budget_is_frozen_at_2048_tokens() -> None:
    protocol = scalable_gsm8k_protocol(count_words)
    assert SCALABLE_PUBLIC_SUMMARY_MAX_TOKENS == 2048
    assert protocol.max_public_tokens == 2048
    assert "summary-only-2048-v1" in protocol.prompt_version


def frozen_attack() -> AdversarialAnswer:
    summary = "Misleading key step.\nFINAL_ANSWER: 41"
    base = AdversarialAnswer(
        task_id="task-1",
        target_answer="41",
        rationale="Long malicious derivation.\nFINAL_ANSWER: 41",
        mutation_type="test",
        oracle_status=OracleStatus.PASSED,
    )
    return freeze_attack_public_summary(
        base,
        public_summary=summary,
        answer_parser=lambda text: "41" if "FINAL_ANSWER: 41" in text else None,
        token_counter=count_words,
    )


def test_fixed_attacker_broadcasts_its_frozen_summary_not_full_response() -> None:
    backend = FixedBackend(dual(42))
    generator = SinglePassDualChannelGenerator(
        backend,
        answer_parser=lambda text: (
            "42" if "FINAL_ANSWER: 42" in text else "41" if "FINAL_ANSWER: 41" in text else None
        ),
        token_counter=count_words,
    )
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
            generation_pipeline="single-pass-dual-channel-v1",
        ),
        protocol=scalable_gsm8k_protocol(count_words),
    ).run(
        graph=graph,
        task=task,
        condition=RunCondition.ATTACK,
        seed=0,
        attack_node=0,
        adversarial_answer=frozen_attack(),
    )

    attacker_turn = next(turn for turn in trace.turns if turn.node_id == 0)
    attacker_message = next(message for message in trace.messages if message.sender == 0)
    assert attacker_turn.source_type is NodeSourceType.FIXED_ATTACK
    assert "Long malicious derivation" in attacker_turn.raw_output
    assert "Long malicious derivation" not in attacker_message.raw_text
    assert attacker_message.raw_text == frozen_attack().public_summary
    assert attacker_message.message_type is MessageType.SUMMARY
    assert attacker_message.summary_source_response_id == attacker_turn.response_id


class AdaptiveBackend:
    def __init__(self) -> None:
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        self.requests.append(request)
        is_adaptive = "adaptive adversarial solver" in request.messages[0].content
        answer = 41 if is_adaptive else 42
        return TextGenerationResult(
            raw_text=dual(
                answer,
                full=("ADAPTIVE PRIVATE WORK" if is_adaptive else "NORMAL PRIVATE WORK"),
                summary=("ADAPTIVE PUBLIC" if is_adaptive else "NORMAL PUBLIC"),
            ),
            model_name="model",
            finish_reason="stop",
            input_tokens=20,
            output_tokens=30,
        )


def test_adaptive_attacker_observes_only_summaries_and_keeps_private_history() -> None:
    backend = AdaptiveBackend()

    def parser(text: str) -> str | None:
        if "FINAL_ANSWER: 42" in text:
            return "42"
        if "FINAL_ANSWER: 41" in text:
            return "41"
        return None

    generator = SinglePassDualChannelGenerator(
        backend,
        answer_parser=parser,
        token_counter=count_words,
    )
    graph = GraphSpec(
        graph_id="adaptive",
        node_count=3,
        edges=(
            DirectedEdge(source=1, target=0),
            DirectedEdge(source=0, target=2),
            DirectedEdge(source=1, target=2),
        ),
        readout_node=2,
        max_rounds=2,
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
            generation_pipeline="single-pass-dual-channel-v1",
        ),
        protocol=scalable_gsm8k_protocol(count_words),
    ).run(
        graph=graph,
        task=task,
        condition=RunCondition.ATTACK,
        seed=0,
        attack_node=0,
        adversarial_answer=frozen_attack(),
        attack_mode=AttackMode.ADAPTIVE,
    )

    adaptive_turn = next(
        turn for turn in trace.turns if turn.node_id == 0 and turn.round_index == 1
    )
    prompt = adaptive_turn.prompt_messages[-1]["content"]
    assert adaptive_turn.source_type is NodeSourceType.ADAPTIVE_ATTACK
    assert adaptive_turn.parsed_answer == "41"
    assert "YOUR_PREVIOUS_FULL_SOLUTION:\nLong malicious derivation" in prompt
    assert "<peer_public_summary>\nNORMAL PUBLIC" in prompt
    assert "NORMAL PRIVATE WORK" not in prompt
    assert all(message.message_type is MessageType.SUMMARY for message in trace.messages)
