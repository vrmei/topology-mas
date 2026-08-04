"""Sample a fixed-edge stratum of valid labeled directed graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.topology.io import write_graph_collection
from topology_mas.topology.sampling import ConstrainedDirectedGraphSampler
from topology_mas.topology.schemas import GraphSamplingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-count", type=int, required=True)
    parser.add_argument("--edge-count", type=int, required=True)
    parser.add_argument("--readout-node", type=int)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--graph-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts-per-graph", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    readout_node = (
        args.readout_node if args.readout_node is not None else args.node_count - 1
    )
    config = GraphSamplingConfig(
        node_count=args.node_count,
        edge_count=args.edge_count,
        readout_node=readout_node,
        max_rounds=args.max_rounds,
        graph_count=args.graph_count,
        seed=args.seed,
        max_attempts_per_graph=args.max_attempts_per_graph,
    )
    collection = ConstrainedDirectedGraphSampler(config).sample()
    graphs_path, manifest_path = write_graph_collection(args.output_dir, collection)
    output = {
        "graphs_path": str(graphs_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        **collection.summary.model_dump(mode="json"),
        "collection_fingerprint": collection.collection_fingerprint,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
