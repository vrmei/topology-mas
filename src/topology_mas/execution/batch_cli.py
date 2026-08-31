"""Run or resume one strictly paired topology-MAS execution batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.batch import BatchExecutionConfig, BatchExecutionRunner
from topology_mas.execution.engine import SynchronousExecutionEngine
from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.inputs import (
    load_adversarial_answer_index,
    load_round_zero_collection,
    load_selected_adversarial_answers,
)
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.protocols import AIME_BOUNDED_PROTOCOL, GSM8K_PROTOCOL
from topology_mas.execution.round_zero import RoundZeroRecord
from topology_mas.execution.schemas import ExecutionSettings
from topology_mas.execution.state_replay import StateConsistentReplayGenerator
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
    parser.add_argument(
        "--task-format",
        choices=("gsm8k-prepared", "aime-free-response"),
        default="gsm8k-prepared",
        help="select the task loader and node communication protocol",
    )
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--round-zero-dir", type=Path)
    parser.add_argument(
        "--independent-round-zero",
        action="store_true",
        help=(
            "generate every normal Round-zero node inside its graph/condition run; "
            "forbids shared Round-zero and state-replay caches"
        ),
    )
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
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--message-order-seed", type=int, default=0)
    parser.add_argument(
        "--horizon-policy",
        choices=("fixed", "graph_depth"),
        default="fixed",
        help=(
            "fixed uses every graph's configured max_rounds; graph_depth ends each graph "
            "at its maximum shortest-path distance to readout"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--state-replay-cache-dir",
        type=Path,
        help="enable exact state-consistent replay using this persistent cache",
    )
    parser.add_argument(
        "--state-replay-model-fingerprint",
        help="required 64-character content fingerprint when state replay is enabled",
    )
    parser.add_argument("--state-replay-namespace")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.task_format == "aime-free-response":
        tasks = load_aime_jsonl(args.tasks, split="test")
        protocol = AIME_BOUNDED_PROTOCOL
    else:
        tasks = read_tasks_jsonl(args.tasks)
        protocol = GSM8K_PROTOCOL
    graphs = read_graphs_jsonl(args.graphs)
    if args.independent_round_zero:
        if args.round_zero_dir is not None:
            raise ValueError("--independent-round-zero forbids --round-zero-dir")
        if args.experiment_seeds is None or args.model is None:
            raise ValueError(
                "independent Round zero requires explicit --experiment-seeds and --model"
            )
        round_zero_records: tuple[RoundZeroRecord, ...] = ()
        experiment_seeds = args.experiment_seeds
        model = args.model
        expected_returned_model = args.expected_returned_model
    else:
        if args.round_zero_dir is None:
            raise ValueError("--round-zero-dir is required unless Round zero is independent")
        round_zero_manifest, round_zero_records = load_round_zero_collection(
            args.round_zero_dir
        )
        experiment_seeds = args.experiment_seeds or round_zero_manifest.config.seeds
        model = args.model or round_zero_manifest.config.requested_model
        expected_returned_model = (
            args.expected_returned_model
            or round_zero_manifest.config.expected_returned_model
        )
    replay_enabled = args.state_replay_cache_dir is not None
    if replay_enabled and (
        args.state_replay_model_fingerprint is None or args.state_replay_namespace is None
    ):
        raise ValueError(
            "state replay requires --state-replay-model-fingerprint and "
            "--state-replay-namespace"
        )
    if not replay_enabled and (
        args.state_replay_model_fingerprint is not None or args.state_replay_namespace is not None
    ):
        raise ValueError("state replay identity options require --state-replay-cache-dir")
    if args.independent_round_zero and replay_enabled:
        raise ValueError("independent Round zero forbids state replay")
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
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_output_tokens=args.max_output_tokens,
        initial_state_policy=(
            "independent_per_run"
            if args.independent_round_zero
            else "shared_round_zero_cache"
        ),
        message_order_seed=args.message_order_seed,
        horizon_policy=args.horizon_policy,
        state_transition_policy=(
            "state-consistent-replay-v1" if replay_enabled else "independent-resampling"
        ),
    )
    config = BatchExecutionConfig(
        experiment_seeds=experiment_seeds,
        assignment_seeds=args.assignment_seeds,
        include_attacks=not args.clean_only,
        initial_state_policy=(
            "independent_per_run"
            if args.independent_round_zero
            else "shared_round_zero_cache"
        ),
        requested_model=model,
        expected_returned_model=expected_returned_model,
        provider_base_url=args.base_url,
        state_replay_model_fingerprint=args.state_replay_model_fingerprint,
        state_replay_namespace=args.state_replay_namespace,
    )
    with OpenAICompatibleTextGenerator(
        model=model,
        expected_returned_model=expected_returned_model,
        base_url=args.base_url,
        api_key_env=None if args.no_auth else args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    ) as backend:
        generator: TextGenerator = (
            StateConsistentReplayGenerator(
                backend,
                cache_dir=args.state_replay_cache_dir,
                requested_model=model,
                expected_returned_model=expected_returned_model,
                model_fingerprint=args.state_replay_model_fingerprint,
                namespace=args.state_replay_namespace,
            )
            if replay_enabled
            else backend
        )
        runner = BatchExecutionRunner(
            SynchronousExecutionEngine(
                generator,
                settings=settings,
                protocol=protocol,
            ),
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
