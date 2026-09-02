import importlib
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_gsm8k_difficulty_gain")


def test_cross_n_difficulty_uses_only_calibration_system_size() -> None:
    frame = pd.DataFrame(
        [
            {"task_id": "floor", "graph_id": "n5-a", "n": 5, "u0": 1, "utility": 1},
            {"task_id": "middle", "graph_id": "n5-b", "n": 5, "u0": 0, "utility": 1},
            {"task_id": "ceiling", "graph_id": "n5-c", "n": 5, "u0": 0, "utility": 0},
            {"task_id": "floor", "graph_id": "n8-a", "n": 8, "u0": 0, "utility": 0},
            {"task_id": "middle", "graph_id": "n8-b1", "n": 8, "u0": 0, "utility": 0},
            {"task_id": "middle", "graph_id": "n8-b2", "n": 8, "u0": 1, "utility": 1},
            {"task_id": "ceiling", "graph_id": "n8-c", "n": 8, "u0": 1, "utility": 1},
        ]
    )

    crossed = module.cross_n_frame(
        frame,
        evaluation_n=5,
        calibration_n=8,
        floor_max=0.10,
        ceiling_min=0.90,
    ).set_index("task_id")

    # The n=5 values deliberately contradict the n=8 calibration for floor and
    # ceiling. Correct labels therefore demonstrate that evaluation rows did not
    # leak into the difficulty estimate.
    assert crossed.loc["floor", "difficulty"] == 0.0
    assert crossed.loc["floor", "difficulty_band"] == "floor"
    assert crossed.loc["middle", "difficulty"] == 0.5
    assert crossed.loc["middle", "difficulty_band"] == "intermediate"
    assert crossed.loc["ceiling", "difficulty"] == 1.0
    assert crossed.loc["ceiling", "difficulty_band"] == "ceiling"


def test_gain_decomposition_counts_correction_and_corruption() -> None:
    frame = pd.DataFrame(
        {
            "task_id": ["task"] * 4,
            "difficulty": [0.5] * 4,
            "difficulty_band": ["intermediate"] * 4,
            "u0": [1, 1, 0, 0],
            "utility": [1, 0, 1, 0],
        }
    )

    result = module.summarize(module.task_sufficient_statistics(frame))

    assert result["u0"] == 0.5
    assert result["utility"] == 0.5
    assert result["delta_u"] == 0.0
    assert result["correct_preservation_C0_to_C3"] == 0.5
    assert result["wrong_correction_notC0_to_C3"] == 0.5
    assert result["correct_corruption_C0_to_notC3"] == 0.5
    assert result["corrected_runs"] == 1
    assert result["corrupted_runs"] == 1
