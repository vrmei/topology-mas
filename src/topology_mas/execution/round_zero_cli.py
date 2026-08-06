"""CLI for graph-independent round-zero generation and caching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.round_zero import (
    RoundZeroCache,
    RoundZeroCacheConfig,
    RoundZeroGenerator,
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replica-count", type=int, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OHMYGPT_API_KEY")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="send no Authorization header (private local servers only)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = read_tasks_jsonl(args.tasks)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be at least one")
        tasks = tasks[: args.limit]
    config = RoundZeroCacheConfig(
        replica_count=args.replica_count,
        seeds=args.seeds,
        requested_model=args.model,
        expected_returned_model=args.expected_returned_model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    cache = RoundZeroCache(args.output_dir)
    with OpenAICompatibleTextGenerator(
        model=args.model,
        expected_returned_model=args.expected_returned_model,
        base_url=args.base_url,
        api_key_env=None if args.no_auth else args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    ) as generator:
        result = RoundZeroGenerator(
            generator,
            config=config,
            cache=cache,
            max_workers=args.max_workers,
        ).generate(tasks)
    records = result.records
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "task_count": len(tasks),
                "replica_count": config.replica_count,
                "seeds": config.seeds,
                "record_count": len(records),
                "generated_count": result.generated_count,
                "reused_count": result.reused_count,
                "max_workers": args.max_workers,
                "parsed_count": sum(record.parsed_answer is not None for record in records),
                "correct_count": sum(record.is_correct for record in records),
                "returned_models": sorted(
                    {record.returned_model for record in records if record.returned_model}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
