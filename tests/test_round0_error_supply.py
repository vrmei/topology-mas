import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_round0_error_supply")


def test_error_supply_separates_parsed_wrong_and_unparsed() -> None:
    endpoints = pd.DataFrame(
        [
            {
                "task_id": "task",
                "graph_id": "graph",
                "n": 5,
                "m": 4,
                "readout_node": 4,
                "horizon": 3,
                "initial_states": (1, 2, 3, 0, 0),
                "round0_readout_correct": 1,
                "final_readout_correct": 0,
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

    result = module.add_error_supply(endpoints, graphs).iloc[0]

    assert result.k_noncorrect_total == 3
    assert result.k_parsed_wrong_total == 2
    assert result.k_unparsed_total == 1
    assert result.k_noncorrect_within_h1 == 2
    assert result.k_noncorrect_within_h2 == 3
    assert result.earliest_noncorrect_arrival == 1
    assert result.final_readout_corrupted == 1
