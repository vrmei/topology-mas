"""Freeze the 50-task subset and sample dense fixed-m pilot topology strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from topology_mas.data.gsm8k import read_tasks_jsonl, write_tasks_jsonl
from topology_mas.execution.inputs import load_adversarial_answer_index
from topology_mas.topology.io import write_graph_collection
from topology_mas.topology.sampling import ConstrainedDirectedGraphSampler
from topology_mas.topology.schemas import GraphSamplingConfig

N5_EDGE_COUNTS = tuple(range(4, 17))
N8_EDGE_COUNTS = tuple(range(7, 50, 3))
DEFAULT_SELECTION_COUNT = 50
DEFAULT_GRAPH_COUNT = 5
DEFAULT_MAX_ROUNDS = 3


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
    result.add_argument("--old-tasks", type=Path, required=True)
    result.add_argument("--old-adversarial-answers", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--task-count", type=int, default=DEFAULT_SELECTION_COUNT)
    result.add_argument("--graphs-per-stratum", type=int, default=DEFAULT_GRAPH_COUNT)
    result.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    result.add_argument("--graph-seed", type=int, default=20_260_812)
    return result


def main() -> None:
    args = parser().parse_args()
    source_tasks = read_tasks_jsonl(args.old_tasks)
    if args.task_count < 1 or args.task_count > len(source_tasks):
        raise ValueError("task count is outside the frozen old-task collection")

    # Preserve the historical 500-task order. This is deliberately not a new GSM8K draw.
    selected_tasks = source_tasks[: args.task_count]
    selected_ids = tuple(task.task_id for task in selected_tasks)
    source_answers = load_adversarial_answer_index(args.old_adversarial_answers)
    missing = [task_id for task_id in selected_ids if task_id not in source_answers]
    if missing:
        raise ValueError(f"historical mutations are missing for: {missing[:5]}")
    selected_answers = tuple(source_answers[task_id] for task_id in selected_ids)

    inputs = args.output_root / "inputs"
    tasks_path = inputs / "tasks50-fixed.jsonl"
    answers_path = inputs / "adversarial50-fixed.jsonl"
    ids_path = inputs / "task_ids50-fixed.json"
    write_tasks_jsonl(tasks_path, selected_tasks)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_text(
        "".join(answer.model_dump_json() + "\n" for answer in selected_answers),
        encoding="utf-8",
    )
    atomic_json(ids_path, list(selected_ids))

    strata = []
    for n, edge_counts in ((5, N5_EDGE_COUNTS), (8, N8_EDGE_COUNTS)):
        maximum_edges = (n - 1) ** 2
        for m in edge_counts:
            # The complete graph is unique under fixed node labels and readout constraints.
            graph_count = 1 if m == maximum_edges else args.graphs_per_stratum
            config = GraphSamplingConfig(
                node_count=n,
                edge_count=m,
                readout_node=n - 1,
                max_rounds=args.max_rounds,
                graph_count=graph_count,
                seed=args.graph_seed + n * 10_000 + m,
                max_attempts_per_graph=1_000_000,
            )
            collection = ConstrainedDirectedGraphSampler(config).sample()
            destination = args.output_root / "graphs" / f"n{n}_m{m}"
            graphs_path, manifest_path = write_graph_collection(destination, collection)
            strata.append(
                {
                    "key": f"n{n}_m{m}",
                    "n": n,
                    "m": m,
                    "normalized_density": m / maximum_edges,
                    "requested_graphs": args.graphs_per_stratum,
                    "sampled_graphs": graph_count,
                    "complete_graph_unique_anchor": m == maximum_edges,
                    "graphs_path": str(graphs_path.resolve()),
                    "graph_manifest_path": str(manifest_path.resolve()),
                    "graph_ids": [graph.graph_id for graph in collection.graphs],
                    "sampling_summary": collection.summary.model_dump(mode="json"),
                }
            )

    manifest = {
        "schema_version": 1,
        "pilot": "qwen3-4b-2507-dense-m-robustness-pilot",
        "selection_policy": "first 50 records in the frozen historical 500-task order",
        "not_a_new_gsm8k_draw": True,
        "source_task_count": len(source_tasks),
        "selected_task_count": len(selected_tasks),
        "selected_task_ids": list(selected_ids),
        "old_tasks": str(args.old_tasks.resolve()),
        "old_tasks_sha256": sha256_file(args.old_tasks),
        "old_adversarial_answers": str(args.old_adversarial_answers.resolve()),
        "old_adversarial_answers_sha256": sha256_file(args.old_adversarial_answers),
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": sha256_file(tasks_path),
        "adversarial_answers_path": str(answers_path.resolve()),
        "adversarial_answers_sha256": sha256_file(answers_path),
        "task_ids_path": str(ids_path.resolve()),
        "max_rounds": args.max_rounds,
        "graphs_per_noncomplete_stratum": args.graphs_per_stratum,
        "complete_stratum_note": (
            "m=(n-1)^2 contains one unique labeled graph, so it is retained once rather "
            "than duplicated five times"
        ),
        "strata": strata,
    }
    atomic_json(args.output_root / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
