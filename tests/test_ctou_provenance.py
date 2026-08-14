import importlib
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_ctou_provenance")


def _turn(node: int, round_index: int, state: str, incoming: list[str]) -> dict:
    return {
        "node_id": node,
        "round_index": round_index,
        "answer_state": state,
        "incoming_message_ids": incoming,
    }


def _message(node: int, round_index: int, state: str) -> dict:
    return {
        "message_id": f"m{round_index}-n{node}",
        "sender": node,
        "round_index": round_index,
        "answer_state": state,
    }


def _pair() -> dict:
    return {
        "task_id": "task",
        "graph_id": "graph",
        "attack_run_spec_id": "attack",
        "attack_node": 0,
        "target_answer": "6",
    }


def test_direct_target_origin_is_reconstructed_from_actual_sender() -> None:
    graph = {
        "node_count": 4,
        "readout_node": 3,
        "max_rounds": 1,
        "edges": [
            {"source": 0, "target": 3},
            {"source": 1, "target": 3},
            {"source": 2, "target": 3},
        ],
    }
    turns = [_turn(node, 0, "target_error" if node == 0 else "correct", []) for node in range(4)]
    turns.append(_turn(3, 1, "correct", ["m0-n0", "m0-n1", "m0-n2"]))
    stored = {"trace": {"turns": turns, "messages": [_message(node, 0, "target_error" if node == 0 else "correct") for node in range(4)]}}

    rows, errors = module.provenance_trace_rows(
        pair=_pair(),
        graph=graph,
        task={"reference_answer": "5"},
        attack_stored=stored,
        stratum="n4-m3",
    )

    assert not errors
    row = rows[0]
    assert row["incoming_correct_count"] == 2
    assert row["incoming_target_count"] == 1
    assert row["direct_target_count"] == 1
    assert row["relayed_target_count"] == 0
    assert row["target_origin"] == "direct_only"


def test_common_parent_of_correct_messages_is_detected() -> None:
    graph = {
        "node_count": 5,
        "readout_node": 3,
        "max_rounds": 2,
        "edges": [
            {"source": 0, "target": 3},
            {"source": 1, "target": 3},
            {"source": 2, "target": 3},
            {"source": 4, "target": 1},
            {"source": 4, "target": 2},
        ],
    }
    turns = [_turn(node, 0, "target_error" if node == 0 else "correct", []) for node in range(5)]
    turns.extend(
        [
            _turn(0, 1, "target_error", []),
            _turn(1, 1, "correct", ["m0-n4"]),
            _turn(2, 1, "correct", ["m0-n4"]),
            _turn(3, 1, "correct", ["m0-n0", "m0-n1", "m0-n2"]),
            _turn(3, 2, "correct", ["m1-n0", "m1-n1", "m1-n2"]),
        ]
    )
    messages = [_message(node, 0, "target_error" if node == 0 else "correct") for node in range(5)]
    messages.extend(
        [
            _message(0, 1, "target_error"),
            _message(1, 1, "correct"),
            _message(2, 1, "correct"),
            _message(3, 1, "correct"),
        ]
    )

    rows, errors = module.provenance_trace_rows(
        pair=_pair(),
        graph=graph,
        task={"reference_answer": "5"},
        attack_stored={"trace": {"turns": turns, "messages": messages}},
        stratum="n5-m5",
    )

    assert not errors
    row = next(item for item in rows if item["receiver_node"] == 3 and item["round_index"] == 2)
    assert row["incoming_correct_count"] == 2
    assert row["incoming_target_count"] == 1
    assert row["immediate_correct_overlap"] == 1
    assert row["immediate_correct_max_overlap"] == 1
    assert row["recursive_correct_overlap"] == 1


def test_common_support_standardization_does_not_mix_ctou_cells() -> None:
    records = []
    for target_count, direct_rate, relay_rate in [(1, 0.1, 0.2), (2, 0.8, 0.6)]:
        for group, rate in [("direct", direct_rate), ("relay", relay_rate)]:
            for index in range(10):
                records.append(
                    {
                        "previous_state": "correct",
                        "round_index": 2,
                        "incoming_correct_count": 2,
                        "incoming_target_count": target_count,
                        "incoming_other_count": 0,
                        "incoming_unparsed_count": 0,
                        "group": group,
                        "outcome": int(index < rate * 10),
                    }
                )
    frame = module.pd.DataFrame(records)

    summary, cells = module._point_effect(
        frame,
        group_column="group",
        group_a="direct",
        group_b="relay",
        outcome_column="outcome",
        minimum_cell_group_rows=10,
    )

    assert summary["matched_cells"] == 2
    assert summary["risk_difference"] == pytest.approx(0.05)
    assert len(cells) == 2
