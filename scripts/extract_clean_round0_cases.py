#!/usr/bin/env python3
"""Extract compact clean endpoint and Round-0 state data from raw MAS traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_ctou_clean_utility import load_clean_data
from analyze_node_round_adoption import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = read_json(args.run_root / "orchestrator_status.json")
    cases, _updates, graphs, audit = load_clean_data(args.run_root, status, args.folds)
    if not audit["passed"]:
        raise RuntimeError(f"clean trace audit failed: {audit['errors'][:10]}")
    columns = [
        "task_id",
        "graph_id",
        "n",
        "m",
        "readout_node",
        "horizon",
        "initial_states",
        "round0_state",
        "round0_correct",
        "actual_state",
        "actual_correct",
    ]
    cases[columns].to_pickle(args.output_dir / "clean_round0_cases.pkl")
    (args.output_dir / "graphs.json").write_text(
        json.dumps(graphs) + "\n", encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
