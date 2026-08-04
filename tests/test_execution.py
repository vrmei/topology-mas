from collections.abc import Callable

import pytest

from topology_mas.execution import (
    SynchronousExecutionEngine,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.execution.answers import parse_numeric_answer
from topology_mas.models import (
    AdversarialAnswer,
    AnswerState,
    DirectedEdge,
    GraphSpec,
    OracleStatus,
    RunCondition,
    TaskInstance,
)


class CapturingGenerator:
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


def chain() -> GraphSpec:
    return GraphSpec(
        graph_id="chain",
        node_count=3,
        edges=(DirectedEdge(source=0, target=1), DirectedEdge(source=1, target=2)),
        readout_node=2,
        max_rounds=2,
    )


def target_error() -> AdversarialAnswer:
    return AdversarialAnswer(
        task_id="task-1",
        target_answer="41",
        rationale="A plausible but wrong calculation.\n#### 41",
        mutation_type="arithmetic_result",
        oracle_status=OracleStatus.PASSED,
        plausibility_score=0.9,
    )


def request_for(
    requests: list[TextGenerationRequest], *, round_index: int, node_id: int
) -> TextGenerationRequest:
    suffix = f"-t{round_index}-n{node_id}"
    return next(request for request in requests if request.request_id.endswith(suffix))


def user_content(request: TextGenerationRequest) -> str:
    return next(message.content for message in request.messages if message.role == "user")


def test_numeric_parser_requires_an_explicit_marker() -> None:
    assert parse_numeric_answer("work 40 + 2 = 42") is None
    assert parse_numeric_answer("work\nFINAL_ANSWER: 42") == "42"
    assert parse_numeric_answer("work\n#### 1,234") == "1234"


def test_chain_is_synchronous_one_hop_and_causally_pruned() -> None:
    generator = CapturingGenerator(lambda _: "Independent solution.\nFINAL_ANSWER: 42")

    trace = SynchronousExecutionEngine(generator).run(
        graph=chain(),
        task=task(),
        condition=RunCondition.CLEAN,
        seed=7,
    )

    assert trace.total_model_calls == 6
    assert [(turn.round_index, turn.node_id) for turn in trace.turns] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]
    node_2_round_1 = request_for(generator.requests, round_index=1, node_id=2)
    content = user_content(node_2_round_1)
    assert content.count("PEER_MESSAGE_") == 1
    assert "n0" not in content
    assert trace.final_answer_state is AnswerState.CORRECT
    assert trace.total_input_tokens == 60
    assert trace.total_output_tokens == 30


def test_peer_labels_are_anonymous_and_sorted_by_sender() -> None:
    graph = GraphSpec(
        graph_id="fan-in",
        node_count=3,
        edges=(DirectedEdge(source=1, target=2), DirectedEdge(source=0, target=2)),
        readout_node=2,
        max_rounds=1,
    )

    def response(request: TextGenerationRequest) -> str:
        if request.request_id.endswith("-n0"):
            return "from zero\nFINAL_ANSWER: 42"
        if request.request_id.endswith("-n1"):
            return "from one\nFINAL_ANSWER: 42"
        return "readout\nFINAL_ANSWER: 42"

    generator = CapturingGenerator(response)
    SynchronousExecutionEngine(generator).run(
        graph=graph,
        task=task(),
        condition=RunCondition.CLEAN,
        seed=0,
    )

    prompt = user_content(request_for(generator.requests, round_index=1, node_id=2))
    assert prompt.index("from zero") < prompt.index("from one")
    assert "sender" not in prompt.lower()
    assert "node 0" not in prompt.lower()


def test_target_error_replay_replaces_attacker_calls_and_propagates() -> None:
    def follow_target_when_seen(request: TextGenerationRequest) -> str:
        content = user_content(request)
        if "#### 41" in content or "FINAL_ANSWER: 41" in content:
            return "I accept that calculation.\nFINAL_ANSWER: 41"
        return "My calculation is correct.\nFINAL_ANSWER: 42"

    generator = CapturingGenerator(follow_target_when_seen)
    trace = SynchronousExecutionEngine(generator).run(
        graph=chain(),
        task=task(),
        condition=RunCondition.ATTACK,
        attack_node=0,
        adversarial_answer=target_error(),
        seed=0,
    )

    assert trace.total_model_calls == 5
    attacker_turn = next(turn for turn in trace.turns if turn.node_id == 0)
    assert attacker_turn.metadata["generator_called"] is False
    assert attacker_turn.answer_state is AnswerState.TARGET_ERROR
    assert trace.final_answer_state is AnswerState.TARGET_ERROR
    assert trace.final_parsed_answer == "41"


def test_attack_requires_an_accepted_task_matching_target_error() -> None:
    generator = CapturingGenerator(lambda _: "FINAL_ANSWER: 42")
    wrong_task = target_error().model_copy(update={"task_id": "other"})

    with pytest.raises(ValueError, match="different task"):
        SynchronousExecutionEngine(generator).run(
            graph=chain(),
            task=task(),
            condition=RunCondition.ATTACK,
            attack_node=0,
            adversarial_answer=wrong_task,
            seed=0,
        )


def test_generation_seed_is_stable_per_node_round() -> None:
    first_generator = CapturingGenerator(lambda _: "FINAL_ANSWER: 42")
    second_generator = CapturingGenerator(lambda _: "FINAL_ANSWER: 42")

    SynchronousExecutionEngine(first_generator).run(
        graph=chain(), task=task(), condition=RunCondition.CLEAN, seed=11
    )
    SynchronousExecutionEngine(second_generator).run(
        graph=chain(), task=task(), condition=RunCondition.CLEAN, seed=11
    )

    assert [request.seed for request in first_generator.requests] == [
        request.seed for request in second_generator.requests
    ]


def test_round_zero_prompt_and_seed_are_paired_across_graphs() -> None:
    alternative = GraphSpec(
        graph_id="star",
        node_count=3,
        edges=(DirectedEdge(source=0, target=2), DirectedEdge(source=1, target=2)),
        readout_node=2,
        max_rounds=2,
    )
    chain_generator = CapturingGenerator(lambda _: "FINAL_ANSWER: 42")
    star_generator = CapturingGenerator(lambda _: "FINAL_ANSWER: 42")

    SynchronousExecutionEngine(chain_generator).run(
        graph=chain(), task=task(), condition=RunCondition.CLEAN, seed=5
    )
    SynchronousExecutionEngine(star_generator).run(
        graph=alternative, task=task(), condition=RunCondition.CLEAN, seed=5
    )

    for node_id in range(3):
        chain_request = request_for(chain_generator.requests, round_index=0, node_id=node_id)
        star_request = request_for(star_generator.requests, round_index=0, node_id=node_id)
        assert chain_request.seed == star_request.seed
        assert chain_request.messages == star_request.messages
