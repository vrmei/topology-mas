import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_ctou_scale_transfer")


def test_atomic_json_serializes_numpy_scalars(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    module.atomic_json(path, {"value": np.int64(7)})
    assert '"value": 7' in path.read_text(encoding="utf-8")


def test_round_summary_uses_node_round_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "condition": "attack",
                "n": 6,
                "round_index": 1,
                "receiver_scope": "readout",
                "actual_state": "target",
                "actual_correct": 0,
                "actual_target": 1,
                "p_correct": 0.25,
                "p_target": 0.75,
                "task_id": "task-1",
                "state_brier": 0.125,
                "state_log_loss": 0.3,
                "state_error": 0,
                "composition_count_mae": 0.0,
                "composition_tv": 0.0,
            }
        ]
    )
    result = module.aggregate_rounds(frame).iloc[0]
    assert result.observed_correct == 0
    assert result.observed_target == 1


def test_graph_outputs_preserve_round0_and_delta_utility() -> None:
    rows = []
    for model in ("ctou_table", "persistence"):
        rows.extend(
            [
                {
                    "condition": "clean",
                    "model": model,
                    "graph_id": "g1",
                    "n": 6,
                    "m": 10,
                    "rho": 0.4,
                    "actual_correct": 1,
                    "p_correct": 0.8,
                    "actual_target": 0,
                    "p_target": 0.1,
                    "round0_correct": 0,
                },
                {
                    "condition": "attack",
                    "model": model,
                    "graph_id": "g1",
                    "n": 6,
                    "m": 10,
                    "rho": 0.4,
                    "actual_correct": 0,
                    "p_correct": 0.6,
                    "actual_target": 1,
                    "p_target": 0.3,
                    "round0_correct": float("nan"),
                },
            ]
        )
    graph, _, _ = module.graph_and_curve_outputs(pd.DataFrame(rows))
    assert graph.observed_u0.eq(0).all()
    assert graph.observed_delta_utility.eq(1).all()
    assert graph.predicted_delta_utility.eq(0.8).all()


def test_same_cell_scale_stability_keeps_conditions_separate() -> None:
    def updates(sizes: tuple[int, ...], condition: str) -> pd.DataFrame:
        rows = []
        for n in sizes:
            for outcome in ("correct", "target"):
                rows.append(
                    {
                        "n": n,
                        "previous_attack_state": "correct",
                        "round_index": 1,
                        "incoming_correct_count": 1,
                        "incoming_target_count": 1,
                        "incoming_other_count": 0,
                        "incoming_unparsed_count": 0,
                        "current_attack_state": outcome,
                        "condition": condition,
                    }
                )
        return pd.DataFrame(rows)

    train = {
        "clean_updates": updates((5, 8), "clean"),
        "attack_updates": updates((5, 8), "attack"),
    }
    test = {
        "clean_updates": updates((6, 7), "clean"),
        "attack_updates": updates((6, 7), "attack"),
    }
    cells, metrics = module.same_cell_scale_stability(train, test)
    assert set(cells.n) == {5, 6, 7, 8}
    assert set(metrics.condition) == {"clean", "attack"}
    assert metrics.weighted_mean_tv.eq(0).all()
