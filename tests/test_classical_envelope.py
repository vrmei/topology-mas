import importlib
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
degroot_trajectory_features = importlib.import_module(
    "analyze_classical_envelope"
).degroot_trajectory_features


def test_degroot_trajectory_tracks_target_mass_round_by_round() -> None:
    graph = {
        "node_count": 2,
        "readout_node": 1,
        "max_rounds": 3,
        "edges": [{"source": 0, "target": 1}],
    }

    features = degroot_trajectory_features(
        graph,
        ("42", "42"),
        attack_node=0,
        target_answer="41",
    )

    assert features["readout_target_mass_r0"] == 0.0
    assert features["readout_target_mass_r1"] == pytest.approx(0.5)
    assert features["readout_target_mass_r2"] == pytest.approx(0.75)
    assert features["readout_target_mass_r3"] == pytest.approx(0.875)
    assert features["benign_mean_target_mass_r3"] == pytest.approx(0.875)
    assert features["readout_target_mass_peak"] == pytest.approx(0.875)


def test_degroot_trajectory_rejects_unfrozen_round_horizon() -> None:
    graph = {
        "node_count": 2,
        "readout_node": 1,
        "max_rounds": 2,
        "edges": [{"source": 0, "target": 1}],
    }

    with pytest.raises(ValueError, match="expected 3 rounds"):
        degroot_trajectory_features(
            graph,
            ("42", "42"),
            attack_node=0,
            target_answer="41",
        )
