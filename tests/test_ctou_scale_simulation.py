from __future__ import annotations

import numpy as np
import pandas as pd

from topology_mas.simulation.ctou_scale import (
    LOCAL_LAW_VARIANTS,
    HierarchicalRoundZeroModel,
    ctou_design_matrix,
    extract_round_zero_groups,
    fit_hierarchical_round_zero,
    local_law_feature_names,
)
from topology_mas.simulation.graph_sampling import (
    normalized_density_edge_levels,
    sample_backbone_augmented_graph,
)
from topology_mas.simulation.rollout import (
    expected_composition_rollout,
    particle_composition_rollout,
    sample_round_zero_states,
)
from topology_mas.topology.graph_ops import graph_constraint_violations


def _transition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "previous_attack_state": ["correct", "other", "unparsed"],
            "round_index": [1, 2, 3],
            "incoming_correct_count": [2, 4, 0],
            "incoming_target_count": [1, 2, 0],
            "incoming_other_count": [0, 0, 0],
            "incoming_unparsed_count": [0, 0, 0],
        }
    )


def test_local_law_features_are_finite_and_named() -> None:
    frame = _transition_frame()
    for variant in LOCAL_LAW_VARIANTS:
        matrix = ctou_design_matrix(frame, variant)
        assert matrix.shape == (len(frame), len(local_law_feature_names(variant)))
        assert np.isfinite(matrix).all()


def test_saturating_volume_is_bounded_and_ratio_is_preserved() -> None:
    frame = _transition_frame().iloc[:2]
    matrix = ctou_design_matrix(frame, "proportions_saturating_volume_k2")
    names = local_law_feature_names("proportions_saturating_volume_k2")
    volume = matrix[:, names.index("evidence_volume")]
    correct_share = matrix[:, names.index("incoming_correct_proportion")]
    target_share = matrix[:, names.index("incoming_target_proportion")]
    assert np.all((volume >= 0) & (volume < 1))
    assert np.allclose(correct_share, [2 / 3, 2 / 3])
    assert np.allclose(target_share, [1 / 3, 1 / 3])
    assert volume[1] > volume[0]


def test_round_zero_extraction_keeps_one_task_graph_group() -> None:
    cases = pd.DataFrame(
        {
            "task_id": ["q1", "q1", "q1"],
            "graph_id": ["g1", "g1", "g2"],
            "n": [5, 5, 5],
            "initial_states": [
                (0, 0, 0, 2, 3),
                (0, 0, 0, 2, 3),
                (0, 0, 2, 2, 3),
            ],
        }
    )
    groups = extract_round_zero_groups(cases)
    assert len(groups) == 2
    assert groups[["correct_count", "other_count", "unparsed_count"]].sum(axis=1).eq(5).all()


def test_hierarchical_round_zero_fit_is_valid_and_correlated() -> None:
    groups = pd.DataFrame(
        {
            "task_id": ["easy"] * 4 + ["hard"] * 4,
            "graph_id": [f"g{i}" for i in range(8)],
            "n": [5] * 8,
            "correct_count": [5, 5, 4, 5, 1, 0, 1, 0],
            "other_count": [0, 0, 1, 0, 4, 5, 4, 5],
            "unparsed_count": [0] * 8,
        }
    )
    model = fit_hierarchical_round_zero(groups)
    assert model.concentration > 0
    assert np.isclose(sum(model.global_mean), 1.0)
    assert model.mean_for_task("easy")[0] > model.mean_for_task("hard")[0]
    draws = model.sample_counts(
        task_id="easy",
        node_count=10,
        draws=50,
        rng=np.random.default_rng(7),
    )
    assert draws.shape == (50, 3)
    assert np.all(draws.sum(axis=1) == 10)


def test_scale_graph_sampler_preserves_fixed_m_and_horizon() -> None:
    graph, audit = sample_backbone_augmented_graph(
        node_count=8,
        edge_count=19,
        horizon=3,
        seed=11,
        sample_index=0,
        swap_steps=20,
    )
    assert len(graph.edges) == 19
    assert graph.readout_node == 7
    assert not graph_constraint_violations(graph)
    assert audit.proposed_swaps == 20
    assert 0 <= audit.acceptance_rate <= 1


def test_density_levels_are_deduplicated_after_rounding() -> None:
    levels = normalized_density_edge_levels(5, tuple(np.linspace(0, 1, 20)))
    assert levels[0][0] == 4
    assert levels[-1][0] == 16
    assert len({edge_count for edge_count, _ in levels}) == len(levels)


def test_expected_composition_rollout_respects_persistent_attacker() -> None:
    graph, _ = sample_backbone_augmented_graph(
        node_count=5,
        edge_count=8,
        horizon=3,
        seed=7,
        sample_index=1,
        swap_steps=0,
    )
    initial = np.zeros((2, 5, 4), dtype=float)
    initial[:, :, 0] = 1.0

    def predictor(frame: pd.DataFrame) -> np.ndarray:
        counts = frame[
            [
                "incoming_correct_count",
                "incoming_target_count",
                "incoming_other_count",
                "incoming_unparsed_count",
            ]
        ].to_numpy(float)
        totals = counts.sum(axis=1, keepdims=True)
        previous = np.zeros_like(counts)
        previous[:, 0] = 1.0
        return np.divide(counts, totals, out=previous, where=totals > 0)

    result = expected_composition_rollout(
        graph=graph,
        initial_marginals=initial,
        attack_nodes=np.asarray([-1, 0]),
        predictor=predictor,
    )
    assert result.shape == (2, 4)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert result[0, 0] == 1.0


def test_round_zero_particle_sampler_preserves_ctou_support() -> None:
    initializer = HierarchicalRoundZeroModel(
        global_mean=(0.6, 0.3, 0.1),
        task_means={"q": (0.7, 0.2, 0.1)},
        concentration=20.0,
        smoothing_strength=3.0,
    )
    states = sample_round_zero_states(
        initializer=initializer,
        task_ids=["q"],
        node_count=5,
        particles=100,
        rng=np.random.default_rng(7),
    )
    assert states.shape == (100, 5)
    assert set(np.unique(states)) <= {0, 2, 3}
    assert abs(np.mean(states == 0) - 0.7) < 0.1


def test_particle_rollout_respects_persistent_attacker() -> None:
    graph, _ = sample_backbone_augmented_graph(
        node_count=5,
        edge_count=4,
        horizon=3,
        seed=9,
        sample_index=0,
        swap_steps=0,
    )

    def copy_target_when_exposed(frame: pd.DataFrame) -> np.ndarray:
        target = frame["incoming_target_count"].to_numpy() > 0
        probability = np.zeros((len(frame), 4), dtype=float)
        probability[:, 0] = ~target
        probability[:, 1] = target
        return probability

    initial = np.zeros((8, 5), dtype=np.int8)
    endpoint = particle_composition_rollout(
        graph=graph,
        initial_states=initial,
        attack_nodes=np.zeros(8, dtype=int),
        predictor=copy_target_when_exposed,
        rng=np.random.default_rng(4),
    )
    assert np.all(endpoint == 1)
