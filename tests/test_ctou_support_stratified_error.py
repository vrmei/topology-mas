from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_ctou_support_stratified_error")


def test_observed_path_support_counts_unseen_and_low_cells() -> None:
    base = {
        "previous_attack_state": ["correct", "target"],
        "round_index": [1, 1],
        "incoming_correct_count": [1, 0],
        "incoming_target_count": [0, 1],
        "incoming_other_count": [0, 0],
        "incoming_unparsed_count": [0, 0],
        "task_id": ["a", "a"],
        "graph_id": ["g", "g"],
        "attack_node": [0, 0],
    }
    train = pd.concat([pd.DataFrame(base).iloc[[0]]] * 6, ignore_index=True)
    test = pd.DataFrame(base)

    result = module.observed_path_support(train, test).iloc[0]

    assert result.actual_transition_visits == 2
    assert result.actual_unseen_visits == 1
    assert result.actual_support_lt_5_visits == 1
    assert result.actual_support_lt_10_visits == 2


def test_support_strata_keep_realized_and_expected_sources_separate() -> None:
    frame = pd.DataFrame(
        {
            "expected_support_lt_20_fraction": [0.0, 0.03, 0.10, 0.30],
            "actual_support_lt_20_visits": [0, 2, 3, 4],
            "actual_unseen_visits": [0, 0, 1, 2],
        }
    )

    assert module.expected_support_stratum(frame).tolist() == [
        "all_high_support",
        "low_mass_le_5pct",
        "low_mass_5_20pct",
        "low_mass_gt_20pct",
    ]
    assert module.actual_support_stratum(frame).tolist() == [
        "all_high_support",
        "seen_but_low_support",
        "one_unseen",
        "multiple_unseen",
    ]


def test_mean_field_support_mass_matches_single_deterministic_cell() -> None:
    graph = {
        "node_count": 2,
        "readout_node": 1,
        "max_rounds": 1,
        "edges": [{"source": 0, "target": 1}],
    }
    shape = (2, 4, 2, 2, 2, 2)
    probability = np.zeros((*shape, 4), dtype=float)
    probability[..., 0] = 1.0
    support = np.zeros(shape, dtype=float)
    # Readout was C and receives one T message from the clamped attacker.
    support[(1, 0, 0, 1, 0, 0)] = 25

    endpoint, metrics = module.mean_field_rollout_with_support(
        graph=graph,
        initial_states=(1, 0),
        attack_node=0,
        probability_lookup=probability,
        support_lookup=support,
    )

    assert endpoint.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert metrics["expected_transition_visits"] == 1.0
    assert metrics["expected_unseen_fraction"] == 0.0
    assert metrics["expected_support_lt_20_fraction"] == 0.0


def test_pooled_rank_correlation_removes_between_group_shift() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "support": [1.0, 2.0, 11.0, 12.0],
            # Between groups both quantities rise, but within each group they
            # move in opposite directions.
            "error": [2.0, 1.0, 12.0, 11.0],
        }
    )

    result = module.pooled_within_group_rank_correlation(
        frame,
        group_columns=["group"],
        left="support",
        right="error",
    )

    assert result == -1.0
