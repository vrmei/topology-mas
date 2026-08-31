import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
module = importlib.import_module("analyze_aime_clean_mas")


def transitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": ["a", "a", "b", "b", "c", "c"],
            "initial_state": ["C", "C", "O", "O", "U", "U"],
            "final_state": ["C", "O", "C", "O", "C", "U"],
            "initial_correct": [1, 1, 0, 0, 0, 0],
            "final_correct": [1, 0, 1, 0, 1, 0],
            "paired_delta": [0, -1, 1, 0, 1, 0],
        }
    )


def test_transition_summary_keeps_other_error_and_unparsed_separate() -> None:
    summary = module.transition_summary(transitions(), {"scope": "test"})

    assert summary["initial_utility"] == pytest.approx(1 / 3)
    assert summary["final_utility"] == pytest.approx(1 / 2)
    assert summary["paired_delta"] == pytest.approx(1 / 6)
    assert summary["correct_preservation_C_to_C"] == pytest.approx(1 / 2)
    assert summary["correct_corruption_C_to_not_C"] == pytest.approx(1 / 2)
    assert summary["other_error_correction_O_to_C"] == pytest.approx(1 / 2)
    assert summary["unparsed_correction_U_to_C"] == pytest.approx(1 / 2)
    assert summary["count_C_to_O"] == 1
    assert summary["count_O_to_C"] == 1
    assert summary["count_U_to_C"] == 1


def test_task_cluster_bootstrap_is_reproducible() -> None:
    first = module.bootstrap_delta(transitions(), samples=200, seed=7)
    second = module.bootstrap_delta(transitions(), samples=200, seed=7)

    assert first == second
    assert first[0] <= transitions().paired_delta.mean() <= first[1]


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.0, "floor"),
        (0.1, "floor"),
        (0.2, "informative"),
        (0.8, "informative"),
        (0.9, "ceiling"),
        (1.0, "ceiling"),
    ],
)
def test_external_difficulty_bands_are_frozen(rate: float, expected: str) -> None:
    assert module.difficulty_band(rate) == expected
