"""Prepare the held-out n=6/n=7 dense-50 topology scale pilot.

The pilot deliberately reuses the frozen task IDs and audited target errors from the
existing dense-50 experiment. Only topology size changes. Each non-complete edge-count
stratum contains five independently sampled labeled directed graphs; the unique complete
graph is retained once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.inputs import load_adversarial_answer_index
from topology_mas.topology.io import write_graph_collection
from topology_mas.topology.sampling import ConstrainedDirectedGraphSampler
from topology_mas.topology.schemas import GraphSamplingConfig


EDGE_COUNTS = {
    6: tuple(range(5, 26, 2)),
    7: tuple(range(6, 37, 2)),
}
DEFAULT_GRAPH_COUNT = 5
DEFAULT_MAX_ROUNDS = 3
DEFAULT_GRAPH_SEED = 20_260_815


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-prepared-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--graphs-per-stratum", type=int, default=DEFAULT_GRAPH_COUNT)
    result.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    result.add_argument("--graph-seed", type=int, default=DEFAULT_GRAPH_SEED)
    return result


def main() -> None:
    args = parser().parse_args()
    source_root = args.source_prepared_root.resolve()
    output_root = args.output_root.resolve()
    if args.graphs_per_stratum < 1:
        raise ValueError("graphs-per-stratum must be positive")

    source_inputs = source_root / "inputs"
    source_tasks = source_inputs / "tasks50-fixed.jsonl"
    source_answers = source_inputs / "adversarial50-fixed.jsonl"
    source_ids = source_inputs / "task_ids50-fixed.json"
    for path in (source_tasks, source_answers, source_ids):
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = read_tasks_jsonl(source_tasks)
    task_ids = [task.task_id for task in tasks]
    frozen_ids = json.loads(source_ids.read_text(encoding="utf-8"))
    if task_ids != frozen_ids:
        raise ValueError("task file order does not match the frozen task-ID manifest")
    answer_index = load_adversarial_answer_index(source_answers)
    missing_answers = [task_id for task_id in task_ids if task_id not in answer_index]
    if missing_answers:
        raise ValueError(f"missing target errors for: {missing_answers[:5]}")

    inputs = output_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    copied_inputs = {}
    for source in (source_tasks, source_answers, source_ids):
        destination = inputs / source.name
        shutil.copy2(source, destination)
        copied_inputs[source.name] = {
            "path": str(destination),
            "sha256": sha256_file(destination),
        }

    strata = []
    for node_count, edge_counts in EDGE_COUNTS.items():
        maximum_edges = (node_count - 1) ** 2
        for edge_count in edge_counts:
            graph_count = (
                1 if edge_count == maximum_edges else args.graphs_per_stratum
            )
            stratum_seed = args.graph_seed + node_count * 10_000 + edge_count
            config = GraphSamplingConfig(
                node_count=node_count,
                edge_count=edge_count,
                readout_node=node_count - 1,
                max_rounds=args.max_rounds,
                graph_count=graph_count,
                seed=stratum_seed,
                max_attempts_per_graph=1_000_000,
            )
            collection = ConstrainedDirectedGraphSampler(config).sample()
            destination = output_root / "graphs" / f"n{node_count}_m{edge_count}"
            graphs_path, graph_manifest_path = write_graph_collection(
                destination, collection
            )
            strata.append(
                {
                    "key": f"n{node_count}_m{edge_count}",
                    "n": node_count,
                    "m": edge_count,
                    "normalized_density": edge_count / maximum_edges,
                    "stratum_seed": stratum_seed,
                    "requested_graphs": args.graphs_per_stratum,
                    "sampled_graphs": graph_count,
                    "complete_graph_unique_anchor": edge_count == maximum_edges,
                    "graphs_path": str(graphs_path),
                    "graphs_sha256": sha256_file(graphs_path),
                    "graph_manifest_path": str(graph_manifest_path),
                    "graph_ids": [graph.graph_id for graph in collection.graphs],
                    "sampling_summary": collection.summary.model_dump(mode="json"),
                }
            )

    manifest = {
        "schema_version": 1,
        "pilot": "llama31-8b-scale-n6n7-dense50-v1",
        "purpose": (
            "held-out node-count evaluation using the exact dense-50 tasks and target "
            "errors from the n=5/n=8 experiment"
        ),
        "source_prepared_root": str(source_root),
        "task_count": len(tasks),
        "task_ids": task_ids,
        "inputs": copied_inputs,
        "node_counts": sorted(EDGE_COUNTS),
        "edge_counts": {str(n): list(values) for n, values in EDGE_COUNTS.items()},
        "max_rounds": args.max_rounds,
        "graph_seed": args.graph_seed,
        "graphs_per_noncomplete_stratum": args.graphs_per_stratum,
        "total_graphs": sum(item["sampled_graphs"] for item in strata),
        "strata": strata,
    }
    atomic_json(output_root / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
