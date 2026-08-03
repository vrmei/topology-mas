"""Prepare disjoint GSM8K calibration and main-study task manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.data.gsm8k import (
    GSM8K_COMMIT,
    GSM8K_SPECS,
    download_gsm8k,
    load_gsm8k,
    select_deterministically,
    task_collection_fingerprint,
    write_tasks_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/gsm8k"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared/gsm8k"))
    parser.add_argument("--calibration-count", type=int, default=20)
    parser.add_argument("--main-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.no_download:
        download_gsm8k(args.data_dir)

    train = load_gsm8k(args.data_dir / "train.jsonl", split="train")
    test = load_gsm8k(args.data_dir / "test.jsonl", split="test")
    calibration = select_deterministically(
        train,
        count=args.calibration_count,
        seed=args.seed,
        namespace="mutation-calibration",
    )
    main_study = select_deterministically(
        test,
        count=args.main_count,
        seed=args.seed,
        namespace="topology-main-study",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tasks_jsonl(args.output_dir / "calibration.jsonl", calibration)
    write_tasks_jsonl(args.output_dir / "main.jsonl", main_study)
    manifest = {
        "dataset": "gsm8k",
        "source_commit": GSM8K_COMMIT,
        "source_specs": {
            split: {
                "url": spec.url,
                "sha256": spec.sha256,
                "expected_lines": spec.expected_lines,
            }
            for split, spec in GSM8K_SPECS.items()
        },
        "selection_seed": args.seed,
        "calibration": {
            "source_split": "train",
            "namespace": "mutation-calibration",
            "count": len(calibration),
            "fingerprint": task_collection_fingerprint(calibration),
            "task_ids": [task.task_id for task in calibration],
        },
        "main": {
            "source_split": "test",
            "namespace": "topology-main-study",
            "count": len(main_study),
            "fingerprint": task_collection_fingerprint(main_study),
            "task_ids": [task.task_id for task in main_study],
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
