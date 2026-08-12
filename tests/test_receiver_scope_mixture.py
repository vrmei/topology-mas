import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
module = importlib.import_module("analyze_receiver_scope_mixture")


def test_target_share_bins_have_frozen_boundaries() -> None:
    assert module.target_share_bin(0.25) == "(0,.25]"
    assert module.target_share_bin(1 / 3) == "(.25,.5)"
    assert module.target_share_bin(0.5) == ".5"
    assert module.target_share_bin(2 / 3) == "(.5,.75)"
    assert module.target_share_bin(0.75) == "[.75,1)"
    assert module.target_share_bin(1.0) == "1"


def test_scope_and_parsed_target_share_are_separate_from_unparsed() -> None:
    frame = pd.DataFrame(
        {
            "receiver_is_readout": [0.0, 1.0],
            "incoming_correct_count": [2, 0],
            "incoming_target_count": [1, 1],
            "incoming_other_count": [0, 1],
            "incoming_unparsed_count": [3, 0],
        }
    )
    result = module.add_scope_and_mixture(frame)
    assert result.receiver_scope.tolist() == ["internal", "readout"]
    assert result.target_share_parsed.tolist() == pytest.approx([1 / 3, 1 / 2])


def test_mixture_decomposition_reweights_common_transition_law() -> None:
    rows = []
    # In both strata, adoption is deterministic within each exact composition.
    # Only the mixture differs, so the pooled-law prediction matches observations.
    for stratum, low_count, high_count in [("n5_m4", 9, 1), ("n5_m8", 1, 9)]:
        for _ in range(low_count):
            rows.append(
                {
                    "regime": "fixed_t3",
                    "receiver_scope": "readout",
                    "stratum": stratum,
                    "round_index": 1,
                        "incoming_target_count": 1,
                        "incoming_correct_count": 3,
                        "incoming_other_count": 0,
                        "incoming_unparsed_count": 0,
                    "adopted": 0,
                }
            )
        for _ in range(high_count):
            rows.append(
                {
                    "regime": "fixed_t3",
                    "receiver_scope": "readout",
                    "stratum": stratum,
                    "round_index": 1,
                        "incoming_target_count": 1,
                        "incoming_correct_count": 0,
                        "incoming_other_count": 0,
                        "incoming_unparsed_count": 0,
                    "adopted": 1,
                }
            )
    estimates, contrasts = module.mixture_decomposition(pd.DataFrame(rows))
    assert (estimates.within_mixture_residual.abs() < 1e-12).all()
    contrast = contrasts.iloc[0]
    assert contrast.observed_delta == pytest.approx(0.8)
    assert contrast.mixture_predicted_delta == pytest.approx(0.8)
    assert contrast.residual_delta == pytest.approx(0.0)


def test_scope_sufficient_statistics_can_preserve_round() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["fixed_t3", "fixed_t3"],
            "stratum": ["n5_m4", "n5_m4"],
            "receiver_scope": ["readout", "readout"],
            "task_id": ["task", "task"],
            "round_index": [1, 2],
            "previous_induced_target_state": [0, 1],
            "previous_attack_state": ["correct", "target"],
            "received_induced_target": [1, 1],
            "current_attack_state": ["target", "correct"],
            "current_clean_state": ["correct", "correct"],
            "received_target": [1, 1],
        }
    )
    stats = module.scope_sufficient_statistics(frame, include_round=True)
    adoption = stats[stats.metric.eq("induced_c_to_t_given_induced_target_exposed")]
    assert set(adoption.round_index) == {1, 2}
    assert adoption.numerator.sum() == 1
