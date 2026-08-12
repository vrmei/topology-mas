"""Strictly synchronous, round-aware execution over a fixed directed graph."""

from __future__ import annotations

from topology_mas.execution.answers import classify_numeric_answer, parse_numeric_answer
from topology_mas.execution.assignments import InitialStateAssignment
from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.prompts import PROMPT_VERSION, build_node_messages
from topology_mas.execution.round_zero import RoundZeroRecord
from topology_mas.execution.schemas import (
    ChatMessage,
    ExecutionSettings,
    RunTrace,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.execution.seeding import (
    anonymous_message_order_key,
    independent_run_round_seed,
    runtime_replica_round_seed,
    stable_fingerprint,
    stable_id,
)
from topology_mas.models import (
    AdversarialAnswer,
    GraphSpec,
    MessageRecord,
    NodeTurnRecord,
    RunCondition,
    TaskInstance,
)
from topology_mas.topology.graph_ops import build_causal_schedule, graph_depth_to_readout


class SynchronousExecutionEngine:
    def __init__(
        self,
        generator: TextGenerator,
        *,
        settings: ExecutionSettings | None = None,
    ) -> None:
        self._generator = generator
        self.settings = settings or ExecutionSettings()

    def run(
        self,
        *,
        graph: GraphSpec,
        task: TaskInstance,
        condition: RunCondition,
        seed: int,
        attack_node: int | None = None,
        adversarial_answer: AdversarialAnswer | None = None,
        round_zero_records: tuple[RoundZeroRecord, ...] | None = None,
        initial_assignment: InitialStateAssignment | None = None,
    ) -> RunTrace:
        self._validate_run(
            graph=graph,
            task=task,
            condition=condition,
            attack_node=attack_node,
            adversarial_answer=adversarial_answer,
            round_zero_records=round_zero_records,
            initial_assignment=initial_assignment,
        )
        effective_horizon = (
            graph.max_rounds
            if self.settings.horizon_policy == "fixed"
            else graph_depth_to_readout(graph)
        )
        schedule = build_causal_schedule(graph, effective_horizon=effective_horizon)
        assigned_initial = self._assigned_initial_records(
            graph=graph,
            task=task,
            seed=seed,
            round_zero_records=round_zero_records,
            initial_assignment=initial_assignment,
        )
        initial_identity = tuple(
            (node_id, record.request_fingerprint)
            for node_id, record in sorted(assigned_initial.items())
        )
        adversarial_answer_fingerprint = (
            stable_fingerprint(adversarial_answer.model_dump_json())
            if adversarial_answer is not None
            else None
        )
        run_id = stable_id(
            "run",
            graph.graph_id,
            task.task_id,
            condition.value,
            attack_node,
            seed,
            PROMPT_VERSION,
            self.settings.model_dump_json(),
            initial_assignment.assignment_id if initial_assignment else None,
            initial_identity,
            adversarial_answer_fingerprint,
        )
        target_answer = adversarial_answer.target_answer if adversarial_answer else None

        turns: list[NodeTurnRecord] = []
        messages: list[MessageRecord] = []
        messages_for_round: dict[int, dict[int, list[MessageRecord]]] = {}
        previous_outputs: dict[int, str] = {}
        model_calls = 0
        backend_calls = 0
        state_replay_cache_hits = 0
        known_input_tokens = 0
        known_output_tokens = 0
        input_tokens_complete = True
        output_tokens_complete = True

        for round_index, active_nodes in enumerate(schedule.active_nodes_by_round):
            round_outputs: dict[int, NodeTurnRecord] = {}
            deliveries = messages_for_round.get(round_index, {})
            for node_id in active_nodes:
                incoming = tuple(
                    sorted(
                        deliveries.get(node_id, ()),
                        key=lambda item: anonymous_message_order_key(
                            order_seed=self.settings.message_order_seed,
                            task_id=task.task_id,
                            round_index=round_index,
                            raw_text=item.raw_text,
                        ),
                    )
                )
                previous = previous_outputs.get(node_id)
                expected_prompt_messages = build_node_messages(
                    task,
                    previous_output=previous,
                    incoming_messages=incoming,
                )
                stochastic_stream_slot = (
                    initial_assignment.replica_for_node(node_id)
                    if initial_assignment is not None
                    else node_id
                )
                generation_seed = (
                    independent_run_round_seed(
                        run_id=run_id,
                        node_id=node_id,
                        round_index=round_index,
                    )
                    if self.settings.initial_state_policy == "independent_per_run"
                    else runtime_replica_round_seed(
                        experiment_seed=seed,
                        task_id=task.task_id,
                        replica_slot=stochastic_stream_slot,
                        round_index=round_index,
                    )
                )
                is_attacker = condition is RunCondition.ATTACK and node_id == attack_node
                if is_attacker:
                    generator_called = False
                    assert adversarial_answer is not None
                    completion = TextGenerationResult(
                        raw_text=adversarial_answer.rationale,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                        metadata={"generator_called": False, "attack_replay": True},
                    )
                    prompt_messages = expected_prompt_messages
                elif round_index == 0 and assigned_initial:
                    generator_called = False
                    cached = assigned_initial[node_id]
                    prompt_messages = tuple(
                        ChatMessage.model_validate(message) for message in cached.prompt_messages
                    )
                    if prompt_messages != expected_prompt_messages:
                        raise ValueError("cached round-zero prompt differs from execution prompt")
                    generation_seed = cached.generation_seed
                    completion = TextGenerationResult(
                        raw_text=cached.raw_output,
                        model_name=cached.returned_model,
                        finish_reason=cached.finish_reason,
                        input_tokens=cached.input_tokens,
                        output_tokens=cached.output_tokens,
                        latency_ms=cached.latency_ms,
                        metadata={
                            "generator_called": False,
                            "round_zero_cache_replay": True,
                            "round_zero_record_id": cached.record_id,
                            "replica_slot": cached.replica_slot,
                        },
                    )
                else:
                    generator_called = True
                    prompt_messages = expected_prompt_messages
                    request = TextGenerationRequest(
                        request_id=f"{run_id}-t{round_index}-n{node_id}",
                        messages=prompt_messages,
                        seed=generation_seed,
                        temperature=self.settings.temperature,
                        top_p=self.settings.top_p,
                        top_k=self.settings.top_k,
                        min_p=self.settings.min_p,
                        presence_penalty=self.settings.presence_penalty,
                        max_output_tokens=self.settings.max_output_tokens,
                    )
                    completion = self._generator.generate(request)
                    model_calls += 1
                    backend_called = completion.metadata.get("backend_called", True)
                    cache_hit = completion.metadata.get("state_replay_cache_hit", False)
                    if not isinstance(backend_called, bool) or not isinstance(cache_hit, bool):
                        raise ValueError("generator cache metadata must contain booleans")
                    backend_calls += int(backend_called)
                    state_replay_cache_hits += int(cache_hit)

                parsed = parse_numeric_answer(completion.raw_text)
                state = classify_numeric_answer(
                    parsed,
                    reference_answer=task.reference_answer,
                    target_answer=target_answer,
                )
                turn = NodeTurnRecord(
                    run_id=run_id,
                    task_id=task.task_id,
                    graph_id=graph.graph_id,
                    condition=condition,
                    attack_node=attack_node,
                    seed=seed,
                    round_index=round_index,
                    node_id=node_id,
                    incoming_message_ids=tuple(message.message_id for message in incoming),
                    previous_raw_output=previous,
                    prompt_messages=tuple(message.model_dump() for message in prompt_messages),
                    generation_seed=generation_seed,
                    raw_output=completion.raw_text,
                    parsed_answer=parsed,
                    answer_state=state,
                    is_correct=state.value == "correct",
                    matches_target_error=state.value == "target_error",
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    latency_ms=completion.latency_ms,
                    model_name=completion.model_name,
                    finish_reason=completion.finish_reason,
                    metadata={
                        **completion.metadata,
                        "generator_called": generator_called,
                        "prompt_version": PROMPT_VERSION,
                        "stochastic_stream_slot": stochastic_stream_slot,
                    },
                )
                turns.append(turn)
                round_outputs[node_id] = turn
                previous_outputs[node_id] = completion.raw_text

                if generator_called:
                    if completion.input_tokens is None:
                        input_tokens_complete = False
                    else:
                        known_input_tokens += completion.input_tokens
                    if completion.output_tokens is None:
                        output_tokens_complete = False
                    else:
                        known_output_tokens += completion.output_tokens

            if round_index == effective_horizon:
                continue
            recipients_by_sender: dict[int, list[int]] = {}
            for edge in schedule.active_edges_by_round[round_index]:
                recipients_by_sender.setdefault(edge.source, []).append(edge.target)
            for sender, recipients in sorted(recipients_by_sender.items()):
                source_turn = round_outputs[sender]
                message = MessageRecord(
                    message_id=f"{run_id}-m{round_index}-n{sender}",
                    run_id=run_id,
                    task_id=task.task_id,
                    graph_id=graph.graph_id,
                    round_index=round_index,
                    sender=sender,
                    recipients=tuple(sorted(recipients)),
                    raw_text=source_turn.raw_output,
                    parsed_answer=source_turn.parsed_answer,
                    answer_state=source_turn.answer_state,
                    output_tokens=source_turn.output_tokens,
                    metadata={"broadcast_copy": True},
                )
                messages.append(message)
                next_round = messages_for_round.setdefault(round_index + 1, {})
                for recipient in message.recipients:
                    next_round.setdefault(recipient, []).append(message)

        final_turn = next(
            turn
            for turn in reversed(turns)
            if turn.node_id == graph.readout_node and turn.round_index == effective_horizon
        )
        return RunTrace(
            run_id=run_id,
            task_id=task.task_id,
            graph_id=graph.graph_id,
            condition=condition,
            attack_node=attack_node,
            adversarial_answer_fingerprint=adversarial_answer_fingerprint,
            target_answer=target_answer,
            initial_assignment_id=(
                initial_assignment.assignment_id if initial_assignment else None
            ),
            initial_assignment_seed=(
                initial_assignment.assignment_seed if initial_assignment else None
            ),
            structural_node_to_replica=(
                initial_assignment.structural_node_to_replica if initial_assignment else None
            ),
            seed=seed,
            prompt_version=PROMPT_VERSION,
            execution_settings=self.settings,
            schedule=schedule,
            turns=tuple(turns),
            messages=tuple(messages),
            final_raw_output=final_turn.raw_output,
            final_parsed_answer=final_turn.parsed_answer,
            final_answer_state=final_turn.answer_state,
            total_model_calls=model_calls,
            total_backend_calls=backend_calls,
            state_replay_cache_hits=state_replay_cache_hits,
            total_input_tokens=(known_input_tokens if input_tokens_complete else None),
            total_output_tokens=(known_output_tokens if output_tokens_complete else None),
        )

    @staticmethod
    def _validate_run(
        *,
        graph: GraphSpec,
        task: TaskInstance,
        condition: RunCondition,
        attack_node: int | None,
        adversarial_answer: AdversarialAnswer | None,
        round_zero_records: tuple[RoundZeroRecord, ...] | None,
        initial_assignment: InitialStateAssignment | None,
    ) -> None:
        if task.oracle_type != "numeric":
            raise ValueError("the first execution engine supports numeric tasks only")
        if condition is RunCondition.CLEAN and attack_node is not None:
            raise ValueError("clean execution cannot specify attack_node")
        if condition is RunCondition.ATTACK:
            if attack_node is None:
                raise ValueError("attack execution requires attack_node")
            if not 0 <= attack_node < graph.node_count:
                raise ValueError("attack_node lies outside the graph")
            if adversarial_answer is None or not adversarial_answer.accepted:
                raise ValueError("attack execution requires an oracle-accepted target error")
            if adversarial_answer.task_id != task.task_id:
                raise ValueError("target error belongs to a different task")
        if (round_zero_records is None) != (initial_assignment is None):
            raise ValueError("round_zero_records and initial_assignment must be provided together")

    @staticmethod
    def _assigned_initial_records(
        *,
        graph: GraphSpec,
        task: TaskInstance,
        seed: int,
        round_zero_records: tuple[RoundZeroRecord, ...] | None,
        initial_assignment: InitialStateAssignment | None,
    ) -> dict[int, RoundZeroRecord]:
        if round_zero_records is None or initial_assignment is None:
            return {}
        if initial_assignment.node_count != graph.node_count:
            raise ValueError("initial assignment node count differs from graph")
        eligible = {
            record.replica_slot: record
            for record in round_zero_records
            if record.task_id == task.task_id and record.experiment_seed == seed
        }
        if len(eligible) != graph.node_count or set(eligible) != set(range(graph.node_count)):
            raise ValueError("round-zero cache does not contain exactly one record per replica")
        if any(record.prompt_version != PROMPT_VERSION for record in eligible.values()):
            raise ValueError("round-zero prompt version differs from execution prompt")
        return {
            node_id: eligible[initial_assignment.replica_for_node(node_id)]
            for node_id in range(graph.node_count)
        }
