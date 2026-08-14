from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_ctou_density_extrapolation")


def test_split_boundaries_use_middle_sampled_level_per_node_count() -> None:
    frame = pd.DataFrame(
        {
            "n": [5] * 13 + [8] * 15,
            "m": list(range(4, 17)) + list(range(7, 50, 3)),
        }
    )

    assert module.split_boundaries(frame) == {5: 10, 8: 28}


def test_sparse_mask_applies_node_count_specific_boundary() -> None:
    frame = pd.DataFrame(
        {
            "n": [5, 5, 8, 8],
            "m": [10, 11, 28, 31],
        }
    )

    assert module.sparse_mask(frame, {5: 10, 8: 28}).tolist() == [True, False, True, False]


def test_support_rows_distinguish_exact_and_composition_coverage() -> None:
    columns = {
        "previous_attack_state": ["correct", "target"],
        "round_index": [1, 2],
        "incoming_correct_count": [1, 2],
        "incoming_target_count": [0, 0],
        "incoming_other_count": [0, 0],
        "incoming_unparsed_count": [0, 0],
        "n": [5, 5],
        "m": [4, 4],
    }
    train = pd.DataFrame(columns).iloc[[0]].copy()
    test = pd.DataFrame(columns)
    rows = module.support_rows_for_split(train, test, metadata={"split": "test"})

    assert len(rows) == 1
    assert rows[0]["exact_transition_cell_coverage"] == 0.5
    assert rows[0]["composition_cell_coverage"] == 0.5
    assert rows[0]["mean_incoming_total"] == 1.5


def test_support_rows_allow_composition_seen_but_transition_cell_unseen() -> None:
    train = pd.DataFrame(
        {
            "previous_attack_state": ["correct"],
            "round_index": [1],
            "incoming_correct_count": [1],
            "incoming_target_count": [0],
            "incoming_other_count": [0],
            "incoming_unparsed_count": [0],
            "n": [5],
            "m": [4],
        }
    )
    test = train.assign(previous_attack_state="target")
    rows = module.support_rows_for_split(train, test, metadata={})

    assert rows[0]["exact_transition_cell_coverage"] == 0.0
    assert rows[0]["composition_cell_coverage"] == 1.0
