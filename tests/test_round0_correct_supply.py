import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_round0_correct_supply")


def test_shortest_distances_follow_directed_edges_to_readout() -> None:
    graph = {
        "node_count": 5,
        "readout_node": 4,
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 4},
            {"source": 2, "target": 1},
            {"source": 3, "target": 2},
        ],
    }
    assert module.shortest_distances_to_readout(graph) == (2, 1, 2, 3, 0)


def test_endpoint_supply_excludes_readout_and_respects_horizon() -> None:
    clean_cases = pd.DataFrame(
        [
            {
                "task_id": "task",
                "graph_id": "graph",
                "n": 5,
                "m": 4,
                "readout_node": 4,
                "horizon": 3,
                "initial_states": (0, 2, 0, 2, 2),
            }
        ]
    )
    endpoints = pd.DataFrame(
        [
            {
                "task_id": "task",
                "graph_id": "graph",
                "n": 5,
                "m": 4,
                "utility": 1,
                "u0": 0,
            }
        ]
    )
    graphs = {
        "graph": {
            "node_count": 5,
            "readout_node": 4,
            "edges": [
                {"source": 0, "target": 1},
                {"source": 1, "target": 4},
                {"source": 2, "target": 4},
                {"source": 3, "target": 2},
            ],
        }
    }

    result = module.build_endpoint_frame(clean_cases, endpoints, graphs).iloc[0]

    assert result.k_correct_total == 2
    assert result.k_correct_within_h1 == 1
    assert result.k_correct_within_h2 == 2
    assert result.k_correct_within_h3 == 2
    assert result.earliest_correct_arrival == 1
    assert result.round0_readout_correct == 0
    assert result.final_readout_correct == 1
