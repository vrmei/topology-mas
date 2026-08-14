import importlib
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_node_round_adoption")


def test_trace_category_prefers_execution_oracle_over_numeric_surface_form() -> None:
    record = {
        "answer_state": "target_error",
        "parsed_answer": "11/5",
    }

    assert module.trace_category(record, reference="2", target="2.2") == "target"


def _turn(node: int, round_index: int, answer: str, incoming: list[str]) -> dict:
    return {
        "node_id": node,
        "round_index": round_index,
        "parsed_answer": answer,
        "incoming_message_ids": incoming,
    }


def _message(run: str, node: int, round_index: int, answer: str) -> dict:
    return {
        "message_id": f"{run}-m{round_index}-n{node}",
        "sender": node,
        "round_index": round_index,
        "parsed_answer": answer,
    }


def _stored(run: str, *, attack: bool) -> dict:
    first = "6" if attack else "5"
    second = "6" if attack else "5"
    third = "6" if attack else "5"
    return {
        "trace": {
            "task_id": "task",
            "graph_id": "graph",
            "seed": 0,
            "prompt_version": "test",
            "initial_assignment_id": "assignment",
            "initial_assignment_seed": 0,
            "structural_node_to_replica": [0, 1, 2],
            "turns": [
                _turn(0, 0, first, []),
                _turn(1, 0, "5", []),
                _turn(2, 0, "5", []),
                _turn(1, 1, second, [f"{run}-m0-n0"]),
                _turn(2, 1, "5", [f"{run}-m0-n1"]),
                _turn(2, 2, third, [f"{run}-m1-n1"]),
            ],
            "messages": [
                _message(run, 0, 0, first),
                _message(run, 1, 0, "5"),
                _message(run, 2, 0, "5"),
                _message(run, 1, 1, second),
                _message(run, 2, 1, "5"),
            ],
        }
    }


def test_degroot_target_mass_uses_equal_self_and_incoming_weights() -> None:
    graph = {
        "node_count": 3,
        "readout_node": 2,
        "max_rounds": 2,
        "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}],
    }
    round_zero = {
        0: {"parsed_answer": "6"},
        1: {"parsed_answer": "5"},
        2: {"parsed_answer": "5"},
    }

    masses = module.degroot_target_masses(graph, round_zero, attack_node=0, target="6")

    assert masses[(1, 1)] == pytest.approx(0.5)
    assert masses[(2, 1)] == 0.0
    assert masses[(2, 2)] == pytest.approx(0.25)


def test_paired_trace_rows_detect_new_induced_adoption() -> None:
    graph = {
        "node_count": 3,
        "readout_node": 2,
        "max_rounds": 2,
        "edges": [{"source": 0, "target": 1}, {"source": 1, "target": 2}],
    }
    pair = {
        "task_id": "task",
        "graph_id": "graph",
        "attack_node": 0,
        "target_answer": "6",
        "clean_run_spec_id": "clean-spec",
        "attack_run_spec_id": "attack-spec",
    }

    rows, errors = module.paired_trace_rows(
        pair=pair,
        graph=graph,
        task={"reference_answer": "5"},
        clean_stored=_stored("clean", attack=False),
        attack_stored=_stored("attack", attack=True),
        stratum="n3_m2",
    )

    assert errors == []
    first = next(row for row in rows if row["receiver_node"] == 1)
    assert first["outcome"] == 1
    assert first["received_induced_target"] == 1
    assert first["incoming_induced_target_count"] == 1
    assert first["degroot_receiver_target_mass"] == pytest.approx(0.5)
    assert first["previous_attack_state"] == "correct"
    assert first["current_attack_state"] == "target"
    assert first["current_clean_state"] == "correct"
    assert first["previous_induced_target_state"] == 0
    assert first["graph_depth"] == 2
    readout_final = next(row for row in rows if row["round_index"] == 2)
    assert readout_final["outcome"] == 1
    assert readout_final["incoming_induced_target_count"] == 1
