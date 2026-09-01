#!/usr/bin/env python3
"""Add two rooted-nonisomorphic graphs to each AIME pilot density stratum."""

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
from topology_mas.topology.sampling import (
    ConstrainedDirectedGraphSampler,
    graph_collection_fingerprint,
)
from topology_mas.topology.schemas import GraphSamplingConfig

EDGE_LEVELS = (4, 8, 12)
NEW_GRAPHS_PER_LEVEL = 2
PREPARATION_VERSION = "aime-clean-mas-n5-density-extension-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/aime/original_2026.jsonl"))
    parser.add_argument(
        "--base-graphs",
        type=Path,
        default=Path("data/aime/clean_mas_n5_h3_v1/graphs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aime/clean_mas_n5_h3_extension_v1"),
    )
    parser.add_argument("--base-seed", type=int, default=20260920)
    return parser.parse_args()


def rooted_canonical_signature(graph: GraphSpec) -> tuple[tuple[int, int], ...]:
    """Canonicalize a small directed graph while fixing the readout node."""

    movable = tuple(node for node in range(graph.node_count) if node != graph.readout_node)
    if len(movable) > 7:
        raise ValueError("factorial canonicalization is restricted to at most seven movable nodes")
    candidates = []
    for permuted in permutations(movable):
        mapping = dict(zip(movable, permuted, strict=True))
        mapping[graph.readout_node] = graph.readout_node
        candidates.append(
            tuple(
                sorted((mapping[edge.source], mapping[edge.target]) for edge in graph.edges)
            )
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
        raise ValueError(f"expected 30 frozen AIME tasks, found {len(tasks)}")
    base_graphs = read_graphs_jsonl(args.base_graphs)
    selected: list[GraphSpec] = []
    strata = []
    for edge_count in EDGE_LEVELS:
        base_level = tuple(graph for graph in base_graphs if len(graph.edges) == edge_count)
        if len(base_level) != 3:
            raise ValueError(f"expected three base graphs for m={edge_count}")
        seen = {rooted_canonical_signature(graph) for graph in base_level}
        if len(seen) != len(base_level):
            raise ValueError(f"base m={edge_count} graphs contain rooted isomorphs")
        config = GraphSamplingConfig(
            node_count=5,
            edge_count=edge_count,
            readout_node=4,
            max_rounds=3,
            graph_count=100,
            seed=args.base_seed + edge_count,
        )
        candidates = ConstrainedDirectedGraphSampler(config).sample()
        accepted = []
        for graph in candidates.graphs:
            canonical = rooted_canonical_signature(graph)
            if canonical in seen:
                continue
            seen.add(canonical)
            accepted.append(graph)
            if len(accepted) == NEW_GRAPHS_PER_LEVEL:
                break
        if len(accepted) != NEW_GRAPHS_PER_LEVEL:
            raise RuntimeError(f"could not find two new rooted-nonisomorphic m={edge_count} graphs")
        selected.extend(accepted)
        strata.append(
            {
                "edge_count": edge_count,
                "base_graph_ids": [graph.graph_id for graph in base_level],
                "new_graph_ids": [graph.graph_id for graph in accepted],
                "candidate_config": config.model_dump(mode="json"),
            }
        )

    selected_tuple = tuple(selected)
    graph_text = "".join(graph.model_dump_json() + "\n" for graph in selected_tuple)
    calls = {
        graph.graph_id: sum(graph.metadata["active_node_count_by_round"])
        for graph in selected_tuple
    }
    manifest = {
        "schema_version": 1,
        "preparation_version": PREPARATION_VERSION,
        "task_count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
        "base_graph_collection_fingerprint": graph_collection_fingerprint(base_graphs),
        "new_graph_collection_fingerprint": graph_collection_fingerprint(selected_tuple),
        "new_graphs_lf_sha256": hashlib.sha256(graph_text.encode()).hexdigest(),
        "node_count": 5,
        "readout_node": 4,
        "horizon": 3,
        "new_graph_count": len(selected_tuple),
        "strata": strata,
        "logical_node_calls_per_task_graph": calls,
        "total_planned_clean_runs": len(tasks) * len(selected_tuple),
        "total_planned_logical_model_calls": len(tasks) * sum(calls.values()),
        "total_planned_backend_calls": 2 * len(tasks) * sum(calls.values()),
        "round_zero_policy": "independent_per_graph_run",
        "cross_graph_generation_reuse": False,
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
