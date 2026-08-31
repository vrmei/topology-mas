#!/usr/bin/env python3
"""Freeze the n=5, H=3 graph set for the first 2026 AIME clean-MAS pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.topology.sampling import (
    ConstrainedDirectedGraphSampler,
    graph_collection_fingerprint,
)
from topology_mas.topology.schemas import GraphSamplingConfig

GRAPH_LEVELS = ((4, 3), (8, 3), (12, 3), (16, 1))
PREPARATION_VERSION = "aime-clean-mas-pilot-n5-h3-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/aime/original_2026.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aime/clean_mas_n5_h3_v1"),
    )
    parser.add_argument("--base-seed", type=int, default=20260901)
    return parser.parse_args()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    tasks = load_aime_jsonl(args.tasks, split="test")
    if len(tasks) != 30:
        raise ValueError(f"expected all 30 frozen 2026 AIME tasks, found {len(tasks)}")

    graphs = []
    strata = []
    for edge_count, graph_count in GRAPH_LEVELS:
        config = GraphSamplingConfig(
            node_count=5,
            edge_count=edge_count,
            readout_node=4,
            max_rounds=3,
            graph_count=graph_count,
            seed=args.base_seed + edge_count,
        )
        collection = ConstrainedDirectedGraphSampler(config).sample()
        graphs.extend(collection.graphs)
        strata.append(
            {
                "edge_count": edge_count,
                "graph_count": graph_count,
                "config": config.model_dump(mode="json"),
                "sampling_summary": collection.summary.model_dump(mode="json"),
                "graph_ids": [graph.graph_id for graph in collection.graphs],
            }
        )

    graphs_tuple = tuple(graphs)
    if len(graphs_tuple) != 10 or len({graph.graph_id for graph in graphs_tuple}) != 10:
        raise ValueError("pilot graph collection must contain ten unique labeled graphs")
    if sum(len(graph.edges) == 16 for graph in graphs_tuple) != 1:
        raise ValueError("pilot graph collection must contain exactly one complete graph")

    graphs_text = "".join(graph.model_dump_json() + "\n" for graph in graphs_tuple)
    graphs_sha256 = hashlib.sha256(graphs_text.encode("utf-8")).hexdigest()
    calls_per_graph = {
        graph.graph_id: sum(graph.metadata["active_node_count_by_round"])
        for graph in graphs_tuple
    }
    manifest = {
        "schema_version": 1,
        "preparation_version": PREPARATION_VERSION,
        "task_count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
        "task_selection": "all frozen 2026 AIME I and II tasks; no difficulty filtering",
        "node_count": 5,
        "readout_node": 4,
        "horizon": 3,
        "edge_levels": [edge_count for edge_count, _ in GRAPH_LEVELS],
        "graph_count": len(graphs_tuple),
        "strata": strata,
        "graph_collection_fingerprint": graph_collection_fingerprint(graphs_tuple),
        "graphs_lf_sha256": graphs_sha256,
        "logical_node_calls_per_task_graph": calls_per_graph,
        "total_planned_clean_runs": len(tasks) * len(graphs_tuple),
        "total_planned_logical_model_calls": len(tasks) * sum(calls_per_graph.values()),
        "round_zero_policy": "independent_per_graph_run",
        "cross_graph_generation_reuse": False,
    }
    atomic_text(args.output_dir / "graphs.jsonl", graphs_text)
    atomic_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
