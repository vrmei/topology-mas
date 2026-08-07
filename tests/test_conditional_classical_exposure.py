from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_conditional_classical_exposure import (  # noqa: E402
    degroot_target_exposure,
    state_features,
)


def test_degroot_target_exposure_matches_hand_computed_chain() -> None:
    graph = {
        "node_count": 3,
        "readout_node": 2,
        "max_rounds": 2,
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
        ],
    }

    exposure = degroot_target_exposure(
        graph,
        ("A", "B", "B"),
        attack_node=0,
        target_answer="A",
    )

    assert exposure == pytest.approx(0.25)


def test_state_features_exclude_attacker_from_benign_summary() -> None:
    features = state_features(
        ("wrong", "correct", "correct", "other"),
        reference_answer="correct",
        target_answer="wrong",
        attack_node=0,
        readout_node=2,
    )

    assert features["readout_initial_correct"] == 1.0
    assert features["attacker_initial_target"] == 1.0
    assert features["benign_correct_fraction"] == pytest.approx(2 / 3)
    assert features["benign_target_fraction"] == 0.0
    assert features["benign_distinct_answer_fraction"] == pytest.approx(2 / 3)
    assert features["benign_largest_consensus_fraction"] == pytest.approx(2 / 3)
