"""Dependency-free directed-graph validation and round-aware scheduling."""

from __future__ import annotations

from collections import deque

from topology_mas.models import GraphSpec
from topology_mas.topology.schemas import CausalSchedule


def candidate_edge_pairs(node_count: int, readout_node: int) -> tuple[tuple[int, int], ...]:
    """All permitted directed edges before reachability conditioning."""

    if node_count < 2:
        raise ValueError("node_count must be at least two")
    if not 0 <= readout_node < node_count:
        raise ValueError("readout_node lies outside the graph")
    return tuple(
        (source, target)
        for source in range(node_count)
        if source != readout_node
        for target in range(node_count)
        if source != target
    )


def relabel_graph(
    graph: GraphSpec,
    *,
    old_node_to_new_node: tuple[int, ...],
    graph_id: str,
) -> GraphSpec:
    """Return the graph induced by an explicit node-label permutation."""

    if len(old_node_to_new_node) != graph.node_count or set(old_node_to_new_node) != set(
        range(graph.node_count)
    ):
        raise ValueError("old_node_to_new_node must be a permutation")
    return GraphSpec(
        graph_id=graph_id,
        node_count=graph.node_count,
        edges=tuple(
            edge.model_copy(
                update={
                    "source": old_node_to_new_node[edge.source],
                    "target": old_node_to_new_node[edge.target],
                }
            )
            for edge in graph.edges
        ),
        readout_node=old_node_to_new_node[graph.readout_node],
        max_rounds=graph.max_rounds,
        sampling_seed=graph.sampling_seed,
        metadata={
            **graph.metadata,
            "relabeling_source_graph_id": graph.graph_id,
            "old_node_to_new_node": old_node_to_new_node,
        },
    )


def distances_to_readout(graph: GraphSpec) -> tuple[int | None, ...]:
    """Shortest directed distances, computed by reverse BFS from readout."""

    predecessors: list[list[int]] = [[] for _ in range(graph.node_count)]
    for edge in graph.edges:
        predecessors[edge.target].append(edge.source)

    distances: list[int | None] = [None] * graph.node_count
    distances[graph.readout_node] = 0
    queue: deque[int] = deque([graph.readout_node])
    while queue:
        target = queue.popleft()
        target_distance = distances[target]
        assert target_distance is not None
        for source in predecessors[target]:
            if distances[source] is None:
                distances[source] = target_distance + 1
                queue.append(source)
    return tuple(distances)


def graph_constraint_violations(graph: GraphSpec) -> tuple[str, ...]:
    """Return machine-readable reasons a graph falls outside the experiment space."""

    distances = distances_to_readout(graph)
    reasons: list[str] = []
    if any(distance is None for distance in distances):
        reasons.append("unreachable_node")
    if any(
        distance is not None and distance > graph.max_rounds for distance in distances
    ):
        reasons.append("round_limit_exceeded")
    return tuple(reasons)


def source_nodes(graph: GraphSpec) -> tuple[int, ...]:
    indegree = [0] * graph.node_count
    for edge in graph.edges:
        indegree[edge.target] += 1
    return tuple(node for node, degree in enumerate(indegree) if degree == 0)


def has_directed_cycle(graph: GraphSpec) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(graph.node_count)]
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)

    state = [0] * graph.node_count

    def visit(node: int) -> bool:
        if state[node] == 1:
            return True
        if state[node] == 2:
            return False
        state[node] = 1
        if any(visit(neighbor) for neighbor in adjacency[node]):
            return True
        state[node] = 2
        return False

    return any(visit(node) for node in range(graph.node_count) if state[node] == 0)


def build_causal_schedule(graph: GraphSpec) -> CausalSchedule:
    """Prune turns and sends that cannot reach readout by the final round."""

    violations = graph_constraint_violations(graph)
    if violations:
        raise ValueError(f"cannot schedule invalid graph: {', '.join(violations)}")
    raw_distances = distances_to_readout(graph)
    distances = tuple(distance for distance in raw_distances if distance is not None)
    assert len(distances) == graph.node_count

    active_nodes_by_round = tuple(
        tuple(
            node
            for node, distance in enumerate(distances)
            if round_index + distance <= graph.max_rounds
        )
        for round_index in range(graph.max_rounds + 1)
    )
    active_edges_by_round = tuple(
        tuple(
            edge
            for edge in graph.edges
            if round_index + 1 + distances[edge.target] <= graph.max_rounds
        )
        for round_index in range(graph.max_rounds)
    )
    return CausalSchedule(
        distances_to_readout=distances,
        active_nodes_by_round=active_nodes_by_round,
        active_edges_by_round=active_edges_by_round,
        message_opportunities=sum(len(edges) for edges in active_edges_by_round),
    )
