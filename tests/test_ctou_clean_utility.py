import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
clean_module = importlib.import_module("analyze_ctou_clean_utility")
rollout_module = importlib.import_module("analyze_ctou_recursive_rollout")


def test_stored_state_normalizes_execution_labels() -> None:
    assert clean_module.stored_state({"answer_state": "target_error"}) == "target"
    assert clean_module.stored_state({"answer_state": "other_error"}) == "other"


def test_clean_degroot_rollout_has_no_clamped_attacker() -> None:
    graph = {
        "graph_id": "clean-smoke",
        "node_count": 3,
        "edges": [
            {"source": 0, "target": 2},
            {"source": 1, "target": 2},
        ],
        "readout_node": 2,
        "max_rounds": 1,
    }
    initial = (
        rollout_module.STATE_INDEX["target"],
        rollout_module.STATE_INDEX["correct"],
        rollout_module.STATE_INDEX["correct"],
    )
    probability = rollout_module.mean_field_rollout(
        graph=graph,
        initial_states=initial,
        attack_node=None,
        model="degroot_equal",
        lookup=None,
    )
    assert probability[rollout_module.STATE_INDEX["correct"]] == pytest.approx(2 / 3)
    assert probability[rollout_module.STATE_INDEX["target"]] == pytest.approx(1 / 3)


def test_clean_correlated_draw_preserves_vector_composition() -> None:
    pool = np.asarray([[0, 0, 1, 2], [1, 1, 1, 3]], dtype=np.int8)
    particles = clean_module.draw_clean_particles(
        pool,
        particles=1_000,
        rng=np.random.default_rng(3),
    )
    expected = {tuple(np.bincount(row, minlength=4)) for row in pool}
    observed = {tuple(np.bincount(row, minlength=4)) for row in particles}
    assert observed == expected


def test_pooled_weights_balance_clean_and_attack_mass() -> None:
    frame = pd.DataFrame({"condition": ["clean"] * 2 + ["attack"] * 8})
    weights = clean_module.balanced_pooled_weights(frame)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())
