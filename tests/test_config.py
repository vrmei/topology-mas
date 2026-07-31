from pathlib import Path

import pytest
from pydantic import ValidationError

from topology_mas.config import ExperimentConfig, GraphConfig


def test_example_config_loads() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/pilot.example.yaml"))

    assert config.experiment_id == "gsm8k_n5_pilot"
    assert config.graph.edge_counts == (4, 8, 12)
    assert config.graph.readout_node == 4
    assert config.tasks[0].oracle_type == "numeric"


def test_graph_budget_rejects_too_few_edges() -> None:
    with pytest.raises(ValidationError, match="minimum is 4"):
        GraphConfig(node_count=5, edge_counts=(3,), readout_node=4, max_rounds=3)


def test_graph_budget_rejects_too_many_edges() -> None:
    with pytest.raises(ValidationError, match="maximum 16"):
        GraphConfig(node_count=5, edge_counts=(17,), readout_node=4, max_rounds=3)

