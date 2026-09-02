from pathlib import Path

from topology_mas.execution.scalable_round_zero import (
    SCALABLE_PROTOCOL_VERSION,
    ScalableRoundZeroPoolConfig,
    ScalableRoundZeroPoolGenerator,
    ScalableRoundZeroPoolStore,
    assign_draw_to_graph,
    build_round_zero_draws,
    default_numeric_parser,
    materialize_engine_inputs,
)
from topology_mas.execution.schemas import (
    ChatMessage,
    TextGenerationRequest,
    TextGenerationResult,
)
from topology_mas.models import AnswerState, DirectedEdge, GraphSpec, TaskInstance


class SlotGenerator:
    def __init__(self) -> None:
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        slot = len(self.requests)
        self.requests.append(request)
        raw = (
            "No parseable final answer"
            if slot == 1
            else f"Work for slot {slot}.\nFINAL_ANSWER: {42 + slot}"
        )
        return TextGenerationResult(
            raw_text=raw,
            model_name="returned-model",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5 + slot,
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


def prompt_builder(item: TaskInstance) -> tuple[ChatMessage, ...]:
    return (ChatMessage(role="user", content=item.prompt),)


def graph(graph_id: str, edges: tuple[tuple[int, int], ...]) -> GraphSpec:
    return GraphSpec(
        graph_id=graph_id,
        node_count=3,
        edges=tuple(DirectedEdge(source=source, target=target) for source, target in edges),
        readout_node=2,
        max_rounds=2,
    )


def test_pool_retains_every_slot_including_unparsed_and_resumes(tmp_path: Path) -> None:
    backend = SlotGenerator()
    config = ScalableRoundZeroPoolConfig(
        responses_per_task=4,
        requested_model="requested-model",
        expected_returned_model="returned-model",
        prompt_version=SCALABLE_PROTOCOL_VERSION,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_output_tokens=128,
    )
    generator = ScalableRoundZeroPoolGenerator(
        backend,
        config=config,
        store=ScalableRoundZeroPoolStore(tmp_path),
        prompt_builder=prompt_builder,
        answer_parser=default_numeric_parser,
        max_workers=1,
    )

    first = generator.generate((task(),))
    second = generator.generate((task(),))

    assert first == second
    assert len(first) == 4
    assert len(backend.requests) == 4
    assert [record.pool_slot for record in first] == [0, 1, 2, 3]
    assert first[0].answer_state is AnswerState.CORRECT
    assert first[1].answer_state is AnswerState.UNPARSED
    assert first[1].raw_response == "No parseable final answer"
    assert all(len(record.content_hash) == 64 for record in first)


def test_draw_is_shared_across_graphs_but_node_assignment_depends_on_graph(
    tmp_path: Path,
) -> None:
    config = ScalableRoundZeroPoolConfig(
        responses_per_task=8,
        requested_model="requested-model",
        prompt_version=SCALABLE_PROTOCOL_VERSION,
        temperature=0.7,
        max_output_tokens=128,
    )
    responses = ScalableRoundZeroPoolGenerator(
        SlotGenerator(),
        config=config,
        store=ScalableRoundZeroPoolStore(tmp_path),
        prompt_builder=prompt_builder,
        answer_parser=default_numeric_parser,
    ).generate((task(),))
    pool_version = responses[0].pool_version
    draws = build_round_zero_draws(
        pool_version=pool_version,
        task_id="task-1",
        node_count=3,
        replicate_count=10,
        pool_responses=responses,
        draw_seed=17,
        fresh_audit_fraction=0.2,
    )
    pooled = next(draw for draw in draws if draw.mode == "pooled")
    graph_a = graph("chain", ((0, 1), (1, 2)))
    graph_b = graph("star", ((0, 2), (1, 2)))
    assignment_a = assign_draw_to_graph(pooled, graph_a)
    assignment_b = assign_draw_to_graph(pooled, graph_b)

    assert len([draw for draw in draws if draw.mode == "fresh_audit"]) == 2
    assert set(assignment_a.node_to_pool_response_id) == set(
        pooled.selected_pool_response_ids
    )
    assert set(assignment_b.node_to_pool_response_id) == set(
        pooled.selected_pool_response_ids
    )
    assert assignment_a.graph_fingerprint != assignment_b.graph_fingerprint
    assert assignment_a.assignment_seed != assignment_b.assignment_seed

    records, engine_assignment = materialize_engine_inputs(
        draw=pooled,
        graph_assignment=assignment_a,
        pool_responses=responses,
        experiment_seed=3,
    )
    assert len(records) == 3
    assert set(engine_assignment.structural_node_to_replica) == {0, 1, 2}
    by_slot = {record.replica_slot: record for record in records}
    node_ids = assignment_a.node_to_pool_response_id
    for node, response_id in enumerate(node_ids):
        slot = engine_assignment.replica_for_node(node)
        assert by_slot[slot].record_id == response_id
        assert by_slot[slot].provider_metadata["draw_id"] == pooled.draw_id
