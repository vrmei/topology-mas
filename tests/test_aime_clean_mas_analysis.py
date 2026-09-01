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


def test_json_safe_replaces_nonfinite_values() -> None:
    assert module.json_safe({"value": float("nan"), "nested": [float("inf")]}) == {
        "value": None,
        "nested": [None],
    }


def crossed_frame(*, edge_count: int, values: list[list[float]]) -> pd.DataFrame:
    rows = []
    for graph_index, graph_values in enumerate(values):
        for task_index, value in enumerate(graph_values):
            rows.append(
                {
                    "graph_id": f"g-{edge_count}-{graph_index}",
                    "task_id": f"t-{task_index}",
                    "edge_count": edge_count,
                    "score": value,
                }
            )
    return pd.DataFrame(rows)


def test_hierarchical_bootstrap_is_reproducible_for_crossed_design() -> None:
    frame = crossed_frame(edge_count=4, values=[[0, 1, 1], [1, 0, 1]])

    first = module.hierarchical_bootstrap_mean(
        frame, value="score", samples=500, seed=9
    )
    second = module.hierarchical_bootstrap_mean(
        frame, value="score", samples=500, seed=9
    )

    assert first == second
    assert first[0] <= frame.score.mean() <= first[1]


def test_group_difference_preserves_paired_task_axis() -> None:
    lower = crossed_frame(edge_count=4, values=[[0, 0, 0], [0, 0, 0]])
    higher = crossed_frame(edge_count=8, values=[[1, 1, 1], [1, 1, 1]])

    low, high = module.bootstrap_group_difference(
        higher, lower, value="score", samples=100, seed=3
    )

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_density_slope_reports_probability_direction() -> None:
    frame = pd.concat(
        [
            crossed_frame(edge_count=4, values=[[0, 0, 0], [0, 0, 0]]),
            crossed_frame(edge_count=8, values=[[1, 1, 1], [1, 1, 1]]),
            crossed_frame(edge_count=12, values=[[2, 2, 2], [2, 2, 2]]),
        ],
        ignore_index=True,
    )

    result = module.bootstrap_density_slope(
        frame,
        value="score",
        edge_counts=(4, 8, 12),
        samples=100,
        seed=4,
    )

    assert result["observed_slope_per_edge"] == pytest.approx(0.25)
    assert result["bootstrap_probability_positive"] == pytest.approx(1.0)


def test_conditional_bootstrap_handles_crossed_denominators() -> None:
    frame = crossed_frame(edge_count=4, values=[[1, 1, 0], [1, 0, 0]])
    frame["initial_state"] = ["C", "C", "O", "C", "O", "O"]
    frame["final_state"] = ["C", "O", "C", "C", "O", "C"]

    low, high = module.hierarchical_bootstrap_conditional_rate(
        frame,
        initial_state="C",
        final_state="C",
        samples=500,
        seed=5,
    )

    observed = module.conditional_rate(frame, "C", "C")
    assert observed is not None
    assert low <= observed <= high


def test_structure_permutation_detects_graph_fixed_signal() -> None:
    rows = []
    for task in range(8):
        for graph, value in (("g-low", 0.0), ("g-high", 1.0)):
            rows.append(
                {
                    "task_id": f"t-{task}",
                    "graph_id": graph,
                    "edge_count": 4,
                    "score": value,
                }
            )
    frame = pd.DataFrame(rows)

    result = module.permutation_structure_test(
        frame,
        value="score",
        reduced_columns=("task_id",),
        full_columns=("task_id", "graph_id"),
        permutations=499,
        seed=6,
    )

    assert result["partial_r_squared"] == pytest.approx(1.0)
    assert result["permutation_p_value"] <= 0.01


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
