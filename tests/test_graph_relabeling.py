import pytest

from topology_mas.models import DirectedEdge, GraphSpec
from topology_mas.topology import distances_to_readout, relabel_graph


def test_graph_relabeling_preserves_directed_structure_and_distances() -> None:
    graph = GraphSpec(
        graph_id="source",
        node_count=4,
        edges=(
            DirectedEdge(source=0, target=1),
            DirectedEdge(source=1, target=3),
            DirectedEdge(source=2, target=3),
        ),
        readout_node=3,
        max_rounds=2,
    )
    old_to_new = (2, 0, 3, 1)

    relabeled = relabel_graph(
        graph,
        old_node_to_new_node=old_to_new,
        graph_id="relabeled",
    )

    assert relabeled.readout_node == 1
    assert {
        (edge.source, edge.target) for edge in relabeled.edges
    } == {(2, 0), (0, 1), (3, 1)}
    original_distances = distances_to_readout(graph)
    relabeled_distances = distances_to_readout(relabeled)
    for old_node, new_node in enumerate(old_to_new):
        assert relabeled_distances[new_node] == original_distances[old_node]


def test_graph_relabeling_requires_a_permutation() -> None:
    graph = GraphSpec(
        graph_id="source",
        node_count=3,
        edges=(DirectedEdge(source=0, target=2), DirectedEdge(source=1, target=2)),
        readout_node=2,
        max_rounds=1,
    )

    with pytest.raises(ValueError, match="permutation"):
        relabel_graph(
            graph,
            old_node_to_new_node=(0, 0, 2),
            graph_id="invalid",
        )
