"""Classify fixed-horizon traces that can be reused by graph-depth Experiment C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from topology_mas.execution.batch import BatchExecutionManifest
from topology_mas.topology.graph_ops import graph_depth_to_readout
from topology_mas.topology.io import read_graphs_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reuse_manifest(source_batch: Path) -> dict[str, object]:
    manifest_path = source_batch / "manifest.json"
    graphs_path = source_batch / "inputs" / "graphs.jsonl"
    source = BatchExecutionManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if source.execution_settings.horizon_policy != "fixed":
        raise ValueError("source batch is not a fixed-horizon execution")
    graphs = read_graphs_jsonl(graphs_path)
    conditions_per_task = source.expected_run_count // (
        len(source.task_ids) * len(source.graph_ids)
    )
    records = []
    exact_trace_count = 0
    prefix_replay_count = 0
    for graph in graphs:
        depth = graph_depth_to_readout(graph)
        exact = depth == graph.max_rounds
        run_count = len(source.task_ids) * conditions_per_task
        exact_trace_count += run_count if exact else 0
        prefix_replay_count += 0 if exact else run_count
        records.append(
            {
                "graph_id": graph.graph_id,
                "configured_horizon": graph.max_rounds,
                "graph_depth_horizon": depth,
                "reuse_class": "exact_trace" if exact else "state_replay_prefix",
                "run_count": run_count,
            }
        )
    return {
        "schema_version": 1,
        "source_batch": str(source_batch.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_horizon_policy": "fixed",
        "target_horizon_policy": "graph_depth",
        "task_count": len(source.task_ids),
        "graph_count": len(graphs),
        "expected_run_count": source.expected_run_count,
        "exact_trace_reuse_count": exact_trace_count,
        "state_replay_prefix_count": prefix_replay_count,
        "graphs": records,
        "claim": (
            "exact_trace is permitted only when graph depth equals configured horizon; "
            "otherwise only prompt-identical cached transitions may be replayed"
        ),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_reuse_manifest(args.source_batch)
    atomic_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
