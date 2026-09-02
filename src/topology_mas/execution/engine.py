"""Strictly synchronous, round-aware execution over a fixed directed graph."""

from __future__ import annotations

import hashlib

from topology_mas.execution.answers import classify_numeric_answer
from topology_mas.execution.assignments import InitialStateAssignment
from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.protocols import GSM8K_PROTOCOL, NodeExecutionProtocol
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
        protocol: NodeExecutionProtocol | None = None,
    ) -> None:
        self._generator = generator
        self.settings = settings or ExecutionSettings()
        self.protocol = protocol or GSM8K_PROTOCOL

    @property
    def prompt_version(self) -> str:
        return self.protocol.prompt_version

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
            self.prompt_version,
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
        previous_output_tokens: dict[int, int | None] = {}
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
                previous_tokens = previous_output_tokens.get(node_id)
                expected_prompt_messages = self.protocol.build_messages(
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
                    adversarial_formatter = getattr(
                        self.protocol, "adversarial_output", None
                    )
                    adversarial_raw = (
                        adversarial_formatter(adversarial_answer)
                        if callable(adversarial_formatter)
                        else adversarial_answer.rationale
                    )
                    completion = TextGenerationResult(
                        raw_text=adversarial_raw,
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
                            **cached.provider_metadata,
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
                    backend_call_count = completion.metadata.get("backend_call_count")
                    backend_called = completion.metadata.get("backend_called", True)
                    cache_hit = completion.metadata.get("state_replay_cache_hit", False)
                    if not isinstance(backend_called, bool) or not isinstance(cache_hit, bool):
                        raise ValueError("generator cache metadata must contain booleans")
                    if backend_call_count is None:
                        backend_call_count = int(backend_called)
                    if not isinstance(backend_call_count, int) or backend_call_count < 0:
                        raise ValueError("backend_call_count must be a nonnegative integer")
                    backend_calls += backend_call_count
                    state_replay_cache_hits += int(cache_hit)

                parsed = self.protocol.parse_answer(
                    completion.raw_text,
                    finish_reason=completion.finish_reason,
                )
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
                        "prompt_version": self.prompt_version,
                        "stochastic_stream_slot": stochastic_stream_slot,
                        "communication_audit": {
                            "receiver_id": node_id,
                            "round": round_index,
                            "own_previous_response_tokens": previous_tokens,
                            "own_previous_response_chars": (
                                len(previous) if previous is not None else None
                            ),
                            "own_previous_response_sha256": (
                                hashlib.sha256(previous.encode("utf-8")).hexdigest()
                                if previous is not None
                                else None
                            ),
                            "incoming_responses": [
                                {
                                    "sender_id": message.sender,
                                    "message_id": message.message_id,
                                    "tokens": message.output_tokens,
                                    "chars": len(message.raw_text),
                                    "sha256": hashlib.sha256(
                                        message.raw_text.encode("utf-8")
                                    ).hexdigest(),
                                }
                                for message in incoming
                            ],
                            "total_prompt_tokens": completion.input_tokens,
                            "generated_tokens": completion.output_tokens,
                            "stop_reason": completion.finish_reason,
                            "context_overflow": completion.metadata.get(
                                "context_window_adjustment"
                            )
                            is not None,
                            "context_truncation": completion.metadata.get(
                                "context_window_adjustment"
                            )
                            is not None,
                            "summarization": self.settings.generation_pipeline
                            in {
                                "aime-private-solve-public-summary-v1",
                                "single-pass-dual-channel-v1",
                            },
                            "message_compression": self.settings.generation_pipeline
                            in {
                                "aime-private-solve-public-summary-v1",
                                "single-pass-dual-channel-v1",
                            },
                        },
                    },
                )
                turns.append(turn)
                round_outputs[node_id] = turn
                previous_outputs[node_id] = completion.raw_text
                previous_output_tokens[node_id] = completion.output_tokens

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
                public_text = self.protocol.public_message(source_turn.raw_output)
                message = MessageRecord(
                    message_id=f"{run_id}-m{round_index}-n{sender}",
                    run_id=run_id,
                    task_id=task.task_id,
                    graph_id=graph.graph_id,
                    round_index=round_index,
                    sender=sender,
                    recipients=tuple(sorted(recipients)),
                    raw_text=public_text,
                    parsed_answer=source_turn.parsed_answer,
                    answer_state=source_turn.answer_state,
                    output_tokens=source_turn.metadata.get(
                        "public_output_tokens", source_turn.output_tokens
                    ),
                    metadata={
                        "broadcast_copy": True,
                        "raw_output_sha256": hashlib.sha256(
                            source_turn.raw_output.encode("utf-8")
                        ).hexdigest(),
                        "public_message_equals_raw_output": (
                            public_text == source_turn.raw_output
                        ),
                        "raw_solution_sha256": source_turn.metadata.get(
                            "raw_solution_sha256"
                        ),
                        "public_summary_sha256": source_turn.metadata.get(
                            "public_summary_sha256"
                        ),
                        "raw_solution_tokens": source_turn.metadata.get(
                            "raw_solution_tokens"
                        ),
                        "public_summary_tokens": source_turn.metadata.get(
                            "public_output_tokens"
                        ),
                        "raw_parsed_answer": source_turn.metadata.get(
                            "raw_parsed_answer"
                        ),
                        "public_parsed_answer": source_turn.metadata.get(
                            "public_parsed_answer"
                        ),
                        "summary_answer_matches_raw": source_turn.metadata.get(
                            "summary_answer_matches_raw"
                        ),
                        "summary_validation_passed": source_turn.metadata.get(
                            "summary_validation_passed"
                        ),
                        "summary_mode": source_turn.metadata.get("summary_mode"),
                    },
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
            prompt_version=self.prompt_version,
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

    def _validate_run(
        self,
        *,
        graph: GraphSpec,
        task: TaskInstance,
        condition: RunCondition,
        attack_node: int | None,
        adversarial_answer: AdversarialAnswer | None,
        round_zero_records: tuple[RoundZeroRecord, ...] | None,
        initial_assignment: InitialStateAssignment | None,
    ) -> None:
        if task.oracle_type not in self.protocol.supported_oracle_types:
            raise ValueError(
                f"protocol {self.prompt_version!r} does not support "
                f"oracle_type={task.oracle_type!r}"
            )
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

    def _assigned_initial_records(
        self,
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
        if any(
            record.prompt_version != self.prompt_version for record in eligible.values()
        ):
            raise ValueError("round-zero prompt version differs from execution prompt")
        return {
            node_id: eligible[initial_assignment.replica_for_node(node_id)]
            for node_id in range(graph.node_count)
        }
