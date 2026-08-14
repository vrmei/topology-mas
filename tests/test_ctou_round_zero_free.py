import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_ctou_round_zero_free")


def test_iid_initialization_clamps_attacker_and_matches_marginal() -> None:
    probability = np.asarray([0.7, 0.1, 0.15, 0.05])
    particles = module.draw_initial_particles(
        n=5,
        attack_node=2,
        mode="iid_empirical",
        particles=100_000,
        iid_probability=probability,
        correlated_pool=np.zeros((1, 4), dtype=np.int8),
        rng=np.random.default_rng(1),
    )
    assert np.all(particles[:, 2] == module.STATE_INDEX["target"])
    benign = particles[:, [0, 1, 3, 4]].ravel()
    observed = np.bincount(benign, minlength=4) / len(benign)
    assert np.allclose(observed, probability, atol=0.005)


def test_correlated_initialization_preserves_each_vector_composition() -> None:
    pool = np.asarray([[0, 0, 1, 2], [1, 1, 1, 3]], dtype=np.int8)
    particles = module.draw_initial_particles(
        n=5,
        attack_node=4,
        mode="correlated_empirical",
        particles=1_000,
        iid_probability=np.full(4, 0.25),
        correlated_pool=pool,
        rng=np.random.default_rng(2),
    )
    expected = {tuple(np.bincount(row, minlength=4)) for row in pool}
    observed = {
        tuple(np.bincount(row[:4], minlength=4)) for row in particles
    }
    assert observed == expected
    assert np.all(particles[:, 4] == module.STATE_INDEX["target"])


def test_split_half_noise_ceiling_is_one_for_identical_graph_order() -> None:
    rows = []
    for task in range(4):
        for graph, correct in (("a", 0.0), ("b", 0.5), ("c", 1.0)):
            for attack in range(2):
                rows.append(
                    {
                        "n": 3,
                        "task_id": f"t{task}",
                        "graph_id": graph,
                        "actual_target": 1.0 - correct,
                        "actual_correct": correct,
                        "attack_node": attack,
                    }
                )
    _, summary = module.split_half_noise_ceiling(
        pd.DataFrame(rows), replicates=1_000, seed=3
    )
    assert np.allclose(summary.split_half_median, 1.0)
