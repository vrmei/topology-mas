import hashlib
import re
from collections.abc import Callable

import pytest

from topology_mas.execution import (
    ExecutionSettings,
    InitialStateAssignment,
    SynchronousExecutionEngine,
    TextGenerationRequest,
    TextGenerationResult,
    relabel_assignment,
)
from topology_mas.execution.answers import parse_numeric_answer
from topology_mas.execution.prompts import PROMPT_VERSION, build_node_messages
from topology_mas.execution.round_zero import RoundZeroRecord
from topology_mas.execution.seeding import round_zero_replica_seed
from topology_mas.models import (
    AdversarialAnswer,
    AnswerState,
    DirectedEdge,
    GraphSpec,
    OracleStatus,
    RunCondition,
    TaskInstance,
)
from topology_mas.topology import relabel_graph


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


def initial_records(
    *,
    node_answers: tuple[str, ...],
    experiment_seed: int = 0,
) -> tuple[RoundZeroRecord, ...]:
    prompt_messages = tuple(
        message.model_dump()
        for message in build_node_messages(task(), previous_output=None, incoming_messages=())
    )
    records = []
    for replica_slot, answer in enumerate(node_answers):
        raw_output = f"Independent cached solution.\nFINAL_ANSWER: {answer}"
        fingerprint = hashlib.sha256(
            f"task-1\0{replica_slot}\0{experiment_seed}".encode()
        ).hexdigest()
        records.append(
            RoundZeroRecord(
                record_id=f"record-{replica_slot}",
                request_fingerprint=fingerprint,
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
                parsed_answer=answer,
                answer_state=(AnswerState.CORRECT if answer == "42" else AnswerState.OTHER_ERROR),
                is_correct=answer == "42",
                requested_model="cached-model",
                returned_model="cached-model",
                finish_reason="stop",
                input_tokens=10,
                output_tokens=5,
            )
        )
    return tuple(records)


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
    assert content.count("<peer_message>") == 1
    assert "n0" not in content
    assert trace.final_answer_state is AnswerState.CORRECT
    assert trace.total_input_tokens == 60
    assert trace.total_output_tokens == 30


def test_peer_labels_are_anonymous_and_do_not_expose_structural_ids() -> None:
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
    assert "from zero" in prompt
    assert "from one" in prompt
    assert "PEER_MESSAGE_1" not in prompt
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
    assert trace.target_answer == "41"
    assert trace.adversarial_answer_fingerprint is not None


def test_graph_depth_horizon_ends_a_shallow_graph_early_without_changing_fixed_default() -> None:
    shallow = GraphSpec(
        graph_id="shallow",
        node_count=3,
        edges=(DirectedEdge(source=0, target=2), DirectedEdge(source=1, target=2)),
        readout_node=2,
        max_rounds=2,
    )
    fixed = SynchronousExecutionEngine(
        CapturingGenerator(lambda _: "FINAL_ANSWER: 42")
    ).run(graph=shallow, task=task(), condition=RunCondition.CLEAN, seed=0)
    depth = SynchronousExecutionEngine(
        CapturingGenerator(lambda _: "FINAL_ANSWER: 42"),
        settings=ExecutionSettings(horizon_policy="graph_depth"),
    ).run(graph=shallow, task=task(), condition=RunCondition.CLEAN, seed=0)

    assert fixed.schedule.effective_horizon == 2
    assert depth.schedule.effective_horizon == 1
    assert fixed.total_model_calls == 7
    assert depth.total_model_calls == 4
    assert max(turn.round_index for turn in depth.turns) == 1
    assert fixed.run_id != depth.run_id


def test_graph_depth_horizon_keeps_persistent_attack_replay_within_the_causal_cone() -> None:
    generator = CapturingGenerator(lambda _: "FINAL_ANSWER: 42")
    trace = SynchronousExecutionEngine(
        generator,
        settings=ExecutionSettings(horizon_policy="graph_depth"),
    ).run(
        graph=GraphSpec(
            graph_id="depth-three",
            node_count=4,
            edges=(
                DirectedEdge(source=0, target=1),
                DirectedEdge(source=1, target=2),
                DirectedEdge(source=2, target=3),
            ),
            readout_node=3,
            max_rounds=3,
        ),
        task=task(),
        condition=RunCondition.ATTACK,
        attack_node=2,
        adversarial_answer=target_error(),
        seed=0,
    )

    attacker_turns = [turn for turn in trace.turns if turn.node_id == 2]
    assert [turn.round_index for turn in attacker_turns] == [0, 1, 2]
    assert all(turn.metadata["attack_replay"] is True for turn in attacker_turns)


def test_attack_content_changes_run_identity() -> None:
    first = SynchronousExecutionEngine(CapturingGenerator(lambda _: "FINAL_ANSWER: 42")).run(
        graph=chain(),
        task=task(),
        condition=RunCondition.ATTACK,
        attack_node=0,
        adversarial_answer=target_error(),
        seed=0,
    )
    second = SynchronousExecutionEngine(CapturingGenerator(lambda _: "FINAL_ANSWER: 42")).run(
        graph=chain(),
        task=task(),
        condition=RunCondition.ATTACK,
        attack_node=0,
        adversarial_answer=target_error().model_copy(
            update={"rationale": "A different wrong rationale.\n#### 41"}
        ),
        seed=0,
    )

    assert first.run_id != second.run_id
    assert first.adversarial_answer_fingerprint != second.adversarial_answer_fingerprint


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


def test_cached_round_zero_is_assigned_without_runtime_inference_calls() -> None:
    generator = CapturingGenerator(lambda _: "Updated.\nFINAL_ANSWER: 42")
    assignment = InitialStateAssignment(
        assignment_id="identity",
        node_count=3,
        assignment_seed=0,
        structural_node_to_replica=(0, 1, 2),
    )

    trace = SynchronousExecutionEngine(generator).run(
        graph=chain(),
        task=task(),
        condition=RunCondition.CLEAN,
        seed=0,
        round_zero_records=initial_records(node_answers=("42", "41", "40")),
        initial_assignment=assignment,
    )

    assert trace.total_model_calls == 3
    assert trace.initial_assignment_seed == 0
    assert trace.structural_node_to_replica == (0, 1, 2)
    assert trace.execution_settings.neighbor_message_order == "content_hash"
    round_zero_turns = [turn for turn in trace.turns if turn.round_index == 0]
    assert [turn.parsed_answer for turn in round_zero_turns] == ["42", "41", "40"]
    assert all(turn.metadata["generator_called"] is False for turn in round_zero_turns)
    assert all(turn.metadata["round_zero_cache_replay"] is True for turn in round_zero_turns)
    assert trace.total_input_tokens == 30
    assert trace.total_output_tokens == 15


def test_isomorphic_relabeling_is_equivariant_with_matched_initial_states() -> None:
    graph_a = GraphSpec(
        graph_id="iso-a",
        node_count=4,
        edges=(
            DirectedEdge(source=0, target=1),
            DirectedEdge(source=1, target=3),
            DirectedEdge(source=2, target=3),
        ),
        readout_node=3,
        max_rounds=2,
    )
    old_to_new = (2, 1, 0, 3)
    graph_b = relabel_graph(
        graph_a,
        old_node_to_new_node=old_to_new,
        graph_id="iso-b",
    )
    assignment_a = InitialStateAssignment(
        assignment_id="iso-identity",
        node_count=4,
        assignment_seed=0,
        structural_node_to_replica=(0, 1, 2, 3),
    )
    assignment_b = relabel_assignment(
        assignment_a,
        old_node_to_new_node=old_to_new,
    )
    records = initial_records(node_answers=("1", "2", "3", "4"))

    def symmetric_update(request: TextGenerationRequest) -> str:
        visible_answers = [
            int(value) for value in re.findall(r"FINAL_ANSWER:\s*(\d+)", user_content(request))
        ]
        return f"Symmetric minimum update.\nFINAL_ANSWER: {min(visible_answers)}"

    trace_a = SynchronousExecutionEngine(CapturingGenerator(symmetric_update)).run(
        graph=graph_a,
        task=task(),
        condition=RunCondition.CLEAN,
        seed=0,
        round_zero_records=records,
        initial_assignment=assignment_a,
    )
    trace_b = SynchronousExecutionEngine(CapturingGenerator(symmetric_update)).run(
        graph=graph_b,
        task=task(),
        condition=RunCondition.CLEAN,
        seed=0,
        round_zero_records=records,
        initial_assignment=assignment_b,
    )

    states_a = {
        (turn.round_index, old_to_new[turn.node_id]): (
            turn.raw_output,
            turn.generation_seed,
            turn.metadata["stochastic_stream_slot"],
        )
        for turn in trace_a.turns
    }
    states_b = {
        (turn.round_index, turn.node_id): (
            turn.raw_output,
            turn.generation_seed,
            turn.metadata["stochastic_stream_slot"],
        )
        for turn in trace_b.turns
    }
    assert states_a == states_b
    assert trace_a.final_parsed_answer == trace_b.final_parsed_answer == "1"
