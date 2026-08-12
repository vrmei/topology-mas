import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_node_round_transitions")


def _row(previous: str, current: str, *, exposed: bool = True) -> dict:
    return {
        "regime": "fixed_t3",
        "stratum": "n5_m4",
        "task_id": "task",
        "graph_id": "graph",
        "attack_node": 0,
        "receiver_node": 1,
        "round_index": 1,
        "receiver_distance_to_readout": 1,
        "graph_depth": 2,
        "previous_attack_state": previous,
        "current_attack_state": current,
        "current_clean_state": "correct",
        "previous_induced_target_state": int(previous == "target"),
        "received_target": int(exposed),
        "received_induced_target": int(exposed),
    }


def test_primary_transition_rates_have_expected_denominators() -> None:
    frame = pd.DataFrame(
        [
            _row("correct", "target"),
            _row("correct", "other"),
            _row("target", "correct"),
            _row("target", "target"),
        ]
    )
    stats = module.sufficient_statistics(frame)
    rates = module.summarize_rates(
        stats,
        rng=module.np.random.default_rng(0),
        replicates=1_000,
    ).set_index("metric")

    assert rates.loc["c_to_t_given_target_exposed", "estimate"] == pytest.approx(0.5)
    assert rates.loc["c_to_o_given_target_exposed", "estimate"] == pytest.approx(0.5)
    assert rates.loc["t_to_c", "estimate"] == pytest.approx(0.5)
    assert rates.loc["t_to_t", "estimate"] == pytest.approx(0.5)


def test_graph_depth_regime_keeps_only_causally_relevant_updates() -> None:
    near = _row("correct", "target")
    far = _row("correct", "target")
    near.update(round_index=1, receiver_distance_to_readout=1, graph_depth=2)
    far.update(round_index=2, receiver_distance_to_readout=1, graph_depth=2)

    result = module.add_regimes(pd.DataFrame([near, far]))

    assert len(result[result.regime == "fixed_t3"]) == 2
    assert len(result[result.regime == "graph_depth"]) == 1


def test_chunk_outputs_are_aggregated_before_rates_are_computed() -> None:
    matrix_part = pd.DataFrame(
        [
            {
                "regime": "fixed_t3",
                "stratum": "n5_m4",
                "previous_attack_state": "correct",
                "current_attack_state": "target",
                "updates": 2,
                "tasks": 1,
                "row_total": 2,
                "transition_probability": 1.0,
            },
            {
                "regime": "fixed_t3",
                "stratum": "n5_m4",
                "previous_attack_state": "correct",
                "current_attack_state": "correct",
                "updates": 6,
                "tasks": 1,
                "row_total": 6,
                "transition_probability": 1.0,
            },
        ]
    )
    matrix = module.combine_transition_matrices([matrix_part])
    target = matrix[matrix.current_attack_state.eq("target")].iloc[0]
    assert target.transition_probability == pytest.approx(0.25)

    reach_parts = [
        pd.DataFrame(
            [
                {
                    "regime": "fixed_t3",
                    "stratum": "n5_m4",
                    "attack_conditions": 1,
                    "mean_unique_eligible_receivers": 4.0,
                    "mean_unique_target_exposed_receivers": 4.0,
                    "mean_unique_induced_target_exposed_receivers": 4.0,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "regime": "fixed_t3",
                    "stratum": "n5_m4",
                    "attack_conditions": 3,
                    "mean_unique_eligible_receivers": 4.0,
                    "mean_unique_target_exposed_receivers": 0.0,
                    "mean_unique_induced_target_exposed_receivers": 0.0,
                }
            ]
        ),
    ]
    reach = module.combine_exposure_reach(reach_parts).iloc[0]
    assert reach.mean_unique_induced_target_exposed_receivers == pytest.approx(1.0)
    assert reach.induced_target_reach_fraction == pytest.approx(0.25)
