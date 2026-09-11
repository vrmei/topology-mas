#!/usr/bin/env python3
"""Freeze a deterministic random K64 subset from every task's K80 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from topology_mas.execution.scalable_round_zero import ScalableRoundZeroPoolStore
from topology_mas.execution.seeding import stable_integer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--public-token-limit", type=int, default=3000)
    parser.add_argument("--minimum-per-task", type=int, default=5)
    args = parser.parse_args()
    manifest, rows = ScalableRoundZeroPoolStore(args.pool).load_available()
    if manifest.config.responses_per_task != 80:
        raise ValueError("source pool must contain exactly 80 responses per task")
    by_task = {task_id: [] for task_id in manifest.task_ids}
    discarded_over_limit = {task_id: 0 for task_id in manifest.task_ids}
    for row in rows:
        public_tokens = row.provider_metadata.get("public_output_tokens")
        if not isinstance(public_tokens, int) or public_tokens > args.public_token_limit:
            discarded_over_limit[row.task_id] += 1
            continue
        by_task[row.task_id].append(row)
    selected = {}
    available_counts = {}
    for task_id, candidates in by_task.items():
        available_counts[task_id] = len(candidates)
        if len(candidates) < args.minimum_per_task:
            raise ValueError(
                f"{task_id} has only {len(candidates)} usable candidates; "
                f"minimum is {args.minimum_per_task}"
            )
        ordered = sorted(
            candidates,
            key=lambda row: (
                stable_integer(
                    "gsm8k-k80-to-k64", args.selection_seed, task_id, row.pool_response_id
                ),
                row.pool_response_id,
            ),
        )
        selected[task_id] = [row.pool_response_id for row in ordered[: args.k]]
    payload = {
        "protocol": "gsm8k-k80-to-k64-v1",
        "source_pool_version": manifest.pool_version,
        "source_attempts_per_task": 80,
        "public_token_limit": args.public_token_limit,
        "maximum_selected_responses_per_task": args.k,
        "minimum_required_responses_per_task": args.minimum_per_task,
        "available_counts_by_task": available_counts,
        "discarded_over_limit_by_task": discarded_over_limit,
        "selected_counts_by_task": {task_id: len(ids) for task_id, ids in selected.items()},
        "selection_seed": args.selection_seed,
        "task_ids": list(manifest.task_ids),
        "selected_pool_response_ids": selected,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks": len(selected),
                "maximum_k": args.k,
                "minimum_available": min(available_counts.values()),
                "maximum_available": max(available_counts.values()),
                "minimum_selected": min(len(ids) for ids in selected.values()),
                "maximum_selected": max(len(ids) for ids in selected.values()),
                "discarded_over_limit": sum(discarded_over_limit.values()),
                "fingerprint": payload["fingerprint"],
            }
        )
    )


if __name__ == "__main__":
    main()
