from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_ctou_joint_residuals")


def test_causal_cone_respects_round_horizon() -> None:
    incoming = [set(), {0}, {1}, {0, 2}]
    assert module.causal_cone(incoming, 3, 0) == {3}
    assert module.causal_cone(incoming, 3, 1) == {0, 2, 3}
    assert module.causal_cone(incoming, 3, 2) == {0, 1, 2, 3}


def test_pair_features_separate_immediate_and_recursive_overlap() -> None:
    graph = {
        "node_count": 5,
        "edges": [
            {"source": 0, "target": 1},
            {"source": 0, "target": 2},
            {"source": 1, "target": 3},
            {"source": 2, "target": 4},
        ],
    }
    round_one = module.pair_topology_features(graph, 0, 1, 3, 4)
    round_two = module.pair_topology_features(graph, 0, 2, 3, 4)
    assert round_one["immediate_shared_count"] == 0
    assert round_one["causal_shared_count"] == 0
    assert round_two["causal_shared_count"] == 1
    assert round_two["attacker_in_both_cones"] == 1


def test_cross_graph_adjustment_uses_other_graphs_only() -> None:
    frame = pd.DataFrame(
        {
            "task_id": ["q", "q", "q"],
            "round_index": [1, 1, 1],
            "graph_id": ["a", "a", "b"],
        }
    )
    residual = np.array([1.0, 3.0, 5.0])
    adjusted, support = module.cross_graph_adjusted_residual(
        frame, residual, np.array([0, 0, 0])
    )
    np.testing.assert_allclose(adjusted, [-4.0, -2.0, 3.0])
    np.testing.assert_array_equal(support, [1, 1, 2])


def test_fixed_effect_slope_removes_event_intercepts() -> None:
    pairs = pd.DataFrame(
        {
            "event_id": [0, 0, 1, 1],
            "task_id": ["a", "a", "b", "b"],
            "graph_id": ["g1", "g1", "g2", "g2"],
        }
    )
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([10.0, 12.0, -5.0, -3.0])
    result = module.fixed_effect_slope(
        pairs,
        x,
        y,
        np.ones(4, dtype=bool),
        replicates=100,
        seed=7,
    )
    assert result is not None
    assert result["slope"] == 2.0
    assert result["varying_events"] == 2
