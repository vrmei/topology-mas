import json
from pathlib import Path

from topology_mas.execution.batch import BatchExecutionManifest
from topology_mas.execution.schemas import ExecutionSettings
from topology_mas.models import DirectedEdge, GraphSpec


def test_reuse_manifest_separates_exact_traces_from_prefix_replay(
    tmp_path: Path,
) -> None:
    from scripts.prepare_graph_depth_reuse import build_reuse_manifest

    batch = tmp_path / "batch"
    inputs = batch / "inputs"
    inputs.mkdir(parents=True)
    graphs = (
        GraphSpec(
            graph_id="depth-three",
            node_count=4,
            edges=(
                DirectedEdge(source=0, target=1),
                DirectedEdge(source=1, target=2),
                DirectedEdge(source=2, target=3),
            ),
            readout_node=3,
            max_rounds=3,
        ),
        GraphSpec(
            graph_id="depth-one",
            node_count=4,
            edges=(
                DirectedEdge(source=0, target=3),
                DirectedEdge(source=1, target=3),
                DirectedEdge(source=2, target=3),
            ),
            readout_node=3,
            max_rounds=3,
        ),
    )
    (inputs / "graphs.jsonl").write_text(
        "".join(graph.model_dump_json() + "\n" for graph in graphs), encoding="utf-8"
    )
    manifest = BatchExecutionManifest(
        config={
            "experiment_seeds": [0],
            "assignment_seeds": [0],
            "requested_model": "model",
        },
        execution_settings=ExecutionSettings(),
        node_count=4,
        readout_node=3,
        max_rounds=3,
        task_ids=("a", "b"),
        graph_ids=("depth-three", "depth-one"),
        task_collection_fingerprint="a" * 64,
        graph_collection_fingerprint="b" * 64,
        round_zero_fingerprint="c" * 64,
        round_zero_index_fingerprint="d" * 64,
        adversarial_answers_fingerprint="e" * 64,
        plan_fingerprint="f" * 64,
        expected_run_count=16,
    )
    (batch / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
    )

    reuse = build_reuse_manifest(batch)

    assert reuse["exact_trace_reuse_count"] == 8
    assert reuse["state_replay_prefix_count"] == 8
    assert [item["reuse_class"] for item in reuse["graphs"]] == [
        "exact_trace",
        "state_replay_prefix",
    ]
