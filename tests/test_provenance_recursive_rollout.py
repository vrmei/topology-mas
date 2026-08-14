import importlib
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_provenance_recursive_rollout")


def test_initial_target_origin_distinguishes_attacker_and_normal_node() -> None:
    initial = (
        module.STATE_INDEX["target"],
        module.STATE_INDEX["target"],
        module.STATE_INDEX["correct"],
    )

    provenance = module.initialize_provenance_states(initial, attack_node=0)

    assert provenance[0] == module.P_STATE_INDEX["target_attack"]
    assert provenance[1] == module.P_STATE_INDEX["target_natural"]
    assert provenance[2] == module.P_STATE_INDEX["correct"]


def test_collapsing_provenance_adds_natural_and_attack_target_mass() -> None:
    extended = np.array([0.2, 0.1, 0.3, 0.25, 0.15])

    collapsed = module.collapse_provenance(extended)

    assert np.allclose(collapsed, [0.2, 0.4, 0.25, 0.15])


def test_particle_rollout_propagates_attack_lineage_across_chain() -> None:
    graph = {
        "graph_id": "chain",
        "node_count": 4,
        "readout_node": 3,
        "max_rounds": 3,
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
        ],
    }
    query = module.provenance_query(maximum_neighbors=1, horizon=3)
    probability = np.zeros((len(query), len(module.P_STATES)), dtype=float)
    for index, row in enumerate(query.itertuples(index=False)):
        attack_exposure = row.direct_target_count + row.relayed_target_count
        if attack_exposure:
            next_state = "target_attack"
        elif row.natural_target_count:
            next_state = "target_natural"
        else:
            next_state = str(row.previous_provenance_state)
        probability[index, module.P_STATE_INDEX[next_state]] = 1.0
    lookup = module.dense_provenance_lookup(
        query,
        probability,
        horizon=3,
        maximum_neighbors=1,
    )
    initial = tuple(module.STATE_INDEX["correct"] for _ in range(4))

    endpoint = module.provenance_particle_rollout(
        graph=graph,
        initial_states=initial,
        attack_node=0,
        lookup=lookup,
        particles=512,
        seed=1,
    )

    assert endpoint[module.STATE_INDEX["target"]] == 1.0


def test_composition_enumerator_covers_all_six_state_counts() -> None:
    rows = module.count_compositions(dimensions=6, maximum=2)

    assert len(rows) == 28
    assert all(len(row) == 6 and sum(row) <= 2 for row in rows)
