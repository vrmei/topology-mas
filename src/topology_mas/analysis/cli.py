"""Analyze one complete paired topology-MAS batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.analysis.artifacts import write_analysis
from topology_mas.analysis.loader import load_complete_batch
from topology_mas.analysis.metrics import analyze_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (args.batch_dir / "analysis-v1")
    result = analyze_batch(load_complete_batch(args.batch_dir))
    write_analysis(output_dir, result)
    print(json.dumps(result.manifest.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
