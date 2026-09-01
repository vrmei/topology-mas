import importlib
import sys
from pathlib import Path

from topology_mas.models import DirectedEdge, GraphSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
module = importlib.import_module("prepare_aime_clean_mas_graph_extension")


def test_rooted_signature_ignores_nonreadout_relabeling() -> None:
    first = GraphSpec(
        graph_id="first",
        node_count=5,
        edges=(
            DirectedEdge(source=0, target=1),
            DirectedEdge(source=1, target=4),
            DirectedEdge(source=2, target=1),
            DirectedEdge(source=3, target=4),
        ),
        readout_node=4,
        max_rounds=3,
    )
    second = GraphSpec(
        graph_id="second",
        node_count=5,
        edges=(
            DirectedEdge(source=3, target=2),
            DirectedEdge(source=2, target=4),
            DirectedEdge(source=0, target=2),
            DirectedEdge(source=1, target=4),
        ),
        readout_node=4,
        max_rounds=3,
    )

    assert module.rooted_canonical_signature(first) == module.rooted_canonical_signature(second)
