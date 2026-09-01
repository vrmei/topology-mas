#!/usr/bin/env python3
"""Freeze the existing 16 rooted-nonisomorphic n=5 AIME graphs in one pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from itertools import permutations
from pathlib import Path

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.models import GraphSpec
from topology_mas.topology.io import read_graphs_jsonl
from topology_mas.topology.sampling import graph_collection_fingerprint

EXPECTED_COUNTS = {4: 5, 8: 5, 12: 5, 16: 1}
PREPARATION_VERSION = "aime-clean-mas-n5-full-rationale-graph-pack-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/aime/original_2026.jsonl"))
    parser.add_argument(
        "--base-graphs",
        type=Path,
        default=Path("data/aime/clean_mas_n5_h3_v1/graphs.jsonl"),
    )
    parser.add_argument(
        "--extension-graphs",
        type=Path,
        default=Path("data/aime/clean_mas_n5_h3_extension_v1/graphs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aime/clean_mas_n5_h3_full_rationale_v1"),
    )
    return parser.parse_args()


def rooted_signature(graph: GraphSpec) -> tuple[tuple[int, int], ...]:
    movable = tuple(node for node in range(graph.node_count) if node != graph.readout_node)
    candidates = []
    for permuted in permutations(movable):
        mapping = dict(zip(movable, permuted, strict=True))
        mapping[graph.readout_node] = graph.readout_node
        candidates.append(
            tuple(sorted((mapping[edge.source], mapping[edge.target]) for edge in graph.edges))
        )
    return min(candidates)


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
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    tasks = load_aime_jsonl(args.tasks, split="test")
    graphs = read_graphs_jsonl(args.base_graphs) + read_graphs_jsonl(args.extension_graphs)
    if len(tasks) != 30:
        raise ValueError(f"expected 30 AIME tasks, found {len(tasks)}")
    if len(graphs) != 16 or len({graph.graph_id for graph in graphs}) != 16:
        raise ValueError("expected 16 distinct graph IDs")
    observed_counts = {
        edge_count: sum(len(graph.edges) == edge_count for graph in graphs)
        for edge_count in EXPECTED_COUNTS
    }
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected m strata: {observed_counts}")
    if any(
        graph.node_count != 5 or graph.readout_node != 4 or graph.max_rounds != 3
        for graph in graphs
    ):
        raise ValueError("all graphs must use n=5, readout=4, H=3")
    for edge_count in EXPECTED_COUNTS:
        level = tuple(graph for graph in graphs if len(graph.edges) == edge_count)
        signatures = {rooted_signature(graph) for graph in level}
        if len(signatures) != len(level):
            raise ValueError(f"m={edge_count} contains rooted-isomorphic duplicates")

    graph_text = "".join(graph.model_dump_json() + "\n" for graph in graphs)
    calls = {
        graph.graph_id: sum(graph.metadata["active_node_count_by_round"])
        for graph in graphs
    }
    manifest = {
        "schema_version": 1,
        "preparation_version": PREPARATION_VERSION,
        "task_count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
        "graph_count": len(graphs),
        "graph_ids": [graph.graph_id for graph in graphs],
        "edge_level_counts": observed_counts,
        "graph_collection_fingerprint": graph_collection_fingerprint(graphs),
        "graphs_lf_sha256": hashlib.sha256(graph_text.encode()).hexdigest(),
        "node_count": 5,
        "readout_node": 4,
        "horizon": 3,
        "total_planned_clean_runs": len(tasks) * len(graphs),
        "logical_node_calls_per_task_graph": calls,
        "total_planned_logical_model_calls": len(tasks) * sum(calls.values()),
        "total_planned_physical_model_calls": len(tasks) * sum(calls.values()),
        "round_zero_policy": "independent_per_graph_run",
        "cross_graph_generation_reuse": False,
        "communication_protocol": "one-stage-full-raw-response",
        "rooted_isomorphism_policy": "readout-fixed canonical nonisomorphism",
    }
    atomic_text(args.output_dir / "graphs.jsonl", graph_text)
    atomic_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
