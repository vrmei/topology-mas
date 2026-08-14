import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_ctou_recursive_rollout")


def test_degroot_recursive_rollout_matches_one_round_closed_form() -> None:
    graph = {
        "graph_id": "smoke",
        "node_count": 3,
        "edges": [
            {"source": 0, "target": 2},
            {"source": 1, "target": 2},
        ],
        "readout_node": 2,
        "max_rounds": 1,
    }
    initial = (
        module.STATE_INDEX["target"],
        module.STATE_INDEX["correct"],
        module.STATE_INDEX["correct"],
    )

    probability = module.mean_field_rollout(
        graph=graph,
        initial_states=initial,
        attack_node=0,
        model="degroot_equal",
        lookup=None,
    )

    assert probability[module.STATE_INDEX["correct"]] == pytest.approx(2 / 3)
    assert probability[module.STATE_INDEX["target"]] == pytest.approx(1 / 3)


def test_particle_rollout_approximates_degroot_closed_form() -> None:
    graph = {
        "graph_id": "smoke",
        "node_count": 3,
        "edges": [
            {"source": 0, "target": 2},
            {"source": 1, "target": 2},
        ],
        "readout_node": 2,
        "max_rounds": 1,
    }
    initial = (
        module.STATE_INDEX["target"],
        module.STATE_INDEX["correct"],
        module.STATE_INDEX["correct"],
    )

    probability = module.particle_rollout(
        graph=graph,
        initial_states=initial,
        attack_node=0,
        model="degroot_equal",
        lookup=None,
        particles=100_000,
        seed=1,
    )

    assert np.isclose(probability[module.STATE_INDEX["target"]], 1 / 3, atol=0.01)
