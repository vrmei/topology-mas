from pathlib import Path

import pytest
from pydantic import ValidationError

from topology_mas.models import DirectedEdge, GraphSpec
from topology_mas.topology.graph_ops import (
    build_causal_schedule,
    candidate_edge_pairs,
    distances_to_readout,
    graph_constraint_violations,
    has_directed_cycle,
    source_nodes,
)
from topology_mas.topology.io import (
    GraphArtifactConflictError,
    read_graphs_jsonl,
    write_graph_collection,
)
from topology_mas.topology.sampling import ConstrainedDirectedGraphSampler
from topology_mas.topology.schemas import GraphSamplingConfig


def graph(
    edges: tuple[tuple[int, int], ...],
    *,
    max_rounds: int = 3,
) -> GraphSpec:
    return GraphSpec(
        graph_id="fixture",
        node_count=4,
        edges=tuple(DirectedEdge(source=u, target=v) for u, v in edges),
        readout_node=3,
        max_rounds=max_rounds,
    )


def test_candidate_edges_exclude_self_loops_and_readout_outgoing_edges() -> None:
    edges = candidate_edge_pairs(5, 4)

    assert len(edges) == 16
    assert all(source != target for source, target in edges)
    assert all(source != 4 for source, _ in edges)


def test_reverse_distances_sources_and_cycles() -> None:
    cyclic = graph(((0, 1), (1, 0), (1, 2), (2, 3)))

    assert distances_to_readout(cyclic) == (3, 2, 1, 0)
    assert source_nodes(cyclic) == ()
    assert has_directed_cycle(cyclic) is True
    assert graph_constraint_violations(cyclic) == ()


def test_constraints_distinguish_unreachable_and_round_limit() -> None:
    unreachable = graph(((0, 1), (1, 3), (2, 0)))
    too_deep = graph(((0, 1), (1, 2), (2, 3)), max_rounds=2)

    assert graph_constraint_violations(unreachable) == ()
    assert graph_constraint_violations(too_deep) == ("round_limit_exceeded",)

    actually_unreachable = graph(((0, 1), (1, 0), (2, 3)))
    assert graph_constraint_violations(actually_unreachable) == ("unreachable_node",)


def test_causal_schedule_stops_nodes_and_edges_outside_final_cone() -> None:
    chain = graph(((0, 1), (1, 2), (2, 3)))

    schedule = build_causal_schedule(chain)

    assert schedule.active_nodes_by_round == (
        (0, 1, 2, 3),
        (1, 2, 3),
        (2, 3),
        (3,),
    )
    assert tuple(len(edges) for edges in schedule.active_edges_by_round) == (3, 2, 1)
    assert schedule.message_opportunities == 6


def test_sampling_config_rejects_impossible_edge_counts() -> None:
    with pytest.raises(ValidationError, match="at least 4"):
        GraphSamplingConfig(
            node_count=5,
            edge_count=3,
            readout_node=4,
            max_rounds=3,
            graph_count=1,
        )

    with pytest.raises(ValidationError, match="cannot exceed 16"):
        GraphSamplingConfig(
            node_count=5,
            edge_count=17,
            readout_node=4,
            max_rounds=3,
            graph_count=1,
        )

    with pytest.raises(ValidationError, match="1 distinct"):
        GraphSamplingConfig(
            node_count=5,
            edge_count=16,
            readout_node=4,
            max_rounds=3,
            graph_count=2,
        )


def test_sampler_is_deterministic_unique_and_constraint_valid() -> None:
    config = GraphSamplingConfig(
        node_count=5,
        edge_count=4,
        readout_node=4,
        max_rounds=4,
        graph_count=5,
        seed=17,
    )

    first = ConstrainedDirectedGraphSampler(config).sample()
    second = ConstrainedDirectedGraphSampler(config).sample()

    assert first == second
    assert len({graph.graph_id for graph in first.graphs}) == 5
    assert all(len(graph.edges) == 4 for graph in first.graphs)
    assert all(not graph_constraint_violations(graph) for graph in first.graphs)
    assert first.summary.accepted_graphs == 5
    assert first.summary.proposal_attempts >= 5
    assert (
        first.summary.accepted_graphs
        + first.summary.rejected_unreachable
        + first.summary.rejected_round_limit
        + first.summary.rejected_duplicate
        == first.summary.proposal_attempts
    )


def test_graph_collection_storage_is_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    config = GraphSamplingConfig(
        node_count=4,
        edge_count=3,
        readout_node=3,
        max_rounds=3,
        graph_count=2,
        seed=9,
    )
    collection = ConstrainedDirectedGraphSampler(config).sample()

    graphs_path, _ = write_graph_collection(tmp_path, collection)
    write_graph_collection(tmp_path, collection)

    assert read_graphs_jsonl(graphs_path) == collection.graphs

    different = ConstrainedDirectedGraphSampler(config.model_copy(update={"seed": 10})).sample()
    with pytest.raises(GraphArtifactConflictError, match="new output directory"):
        write_graph_collection(tmp_path, different)
