import pandas as pd

from scripts.analyze_topology_ranking_stability import (
    fractional_top_overlap,
    metric_tables,
)


def test_metric_tables_keep_mean_and_worst_attack_per_graph() -> None:
    attacks = pd.DataFrame(
        [
            {
                "stratum": "n3_m2",
                "task_id": task,
                "graph_id": "g1",
                "attack_node": node,
                "clean_correct": True,
                "attack_correct": value,
                "induced_readout_target": not value,
                "paired_accuracy_drop": int(not value),
            }
            for task, node, value in (
                ("t1", 0, True),
                ("t1", 1, False),
                ("t2", 0, True),
                ("t2", 1, True),
            )
        ]
    )

    graph, node = metric_tables(attacks, {"t1", "t2"})

    assert graph.loc[0, "clean_utility"] == 1.0
    assert graph.loc[0, "mean_attack_accuracy"] == 0.75
    assert graph.loc[0, "worst_node_attack_accuracy"] == 0.5
    assert node.set_index("attack_node").loc[1, "induced_target_rate"] == 0.5


def test_fractional_top_overlap_handles_ties() -> None:
    first = pd.Series({0: 1.0, 1: 1.0, 2: 0.0})
    second = pd.Series({0: 0.0, 1: 1.0, 2: 1.0})

    assert fractional_top_overlap(first, second) == 0.5
