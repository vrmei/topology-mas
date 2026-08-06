"""Run or resume one strictly paired topology-MAS execution batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.batch import BatchExecutionConfig, BatchExecutionRunner
from topology_mas.execution.engine import SynchronousExecutionEngine
from topology_mas.execution.inputs import (
    load_adversarial_answer_index,
    load_round_zero_collection,
    load_selected_adversarial_answers,
)
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.schemas import ExecutionSettings
from topology_mas.topology.io import read_graphs_jsonl


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
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--round-zero-dir", type=Path, required=True)
    parser.add_argument("--mutations-dir", type=Path)
    parser.add_argument(
        "--adversarial-answers",
        type=Path,
        help="audited selected_adversarial_answers.jsonl; preferred over a raw mutation directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-seeds", type=_parse_seeds)
    parser.add_argument("--assignment-seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--base-url", default="https://api.ohmygpt.com/v1")
    parser.add_argument("--api-key-env", default="OHMYGPT_API_KEY")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="send no Authorization header (private local servers only)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--message-order-seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = read_tasks_jsonl(args.tasks)
    graphs = read_graphs_jsonl(args.graphs)
    round_zero_manifest, round_zero_records = load_round_zero_collection(args.round_zero_dir)
    experiment_seeds = args.experiment_seeds or round_zero_manifest.config.seeds
    model = args.model or round_zero_manifest.config.requested_model
    expected_returned_model = (
        args.expected_returned_model or round_zero_manifest.config.expected_returned_model
    )
    if args.mutations_dir is not None and args.adversarial_answers is not None:
        raise ValueError("use only one of --mutations-dir and --adversarial-answers")
    if not args.clean_only and args.mutations_dir is None and args.adversarial_answers is None:
        raise ValueError(
            "--adversarial-answers or --mutations-dir is required unless --clean-only is set"
        )
    if args.clean_only:
        adversarial_answers = {}
    elif args.adversarial_answers is not None:
        adversarial_answers = load_adversarial_answer_index(args.adversarial_answers)
    else:
        adversarial_answers = load_selected_adversarial_answers(args.mutations_dir)

    settings = ExecutionSettings(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        message_order_seed=args.message_order_seed,
    )
    config = BatchExecutionConfig(
        experiment_seeds=experiment_seeds,
        assignment_seeds=args.assignment_seeds,
        include_attacks=not args.clean_only,
        requested_model=model,
        expected_returned_model=expected_returned_model,
        provider_base_url=args.base_url,
    )
    with OpenAICompatibleTextGenerator(
        model=model,
        expected_returned_model=expected_returned_model,
        base_url=args.base_url,
        api_key_env=None if args.no_auth else args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    ) as generator:
        runner = BatchExecutionRunner(
            SynchronousExecutionEngine(generator, settings=settings),
            config=config,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
        )
        _, summary = runner.run(
            tasks=tasks,
            graphs=graphs,
            round_zero_records=round_zero_records,
            adversarial_answers=adversarial_answers,
        )
    print(json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
