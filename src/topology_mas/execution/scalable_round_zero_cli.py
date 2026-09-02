"""Generate an unfiltered Round-0 pool for homogeneous-mas-scalable-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import (
    HuggingFaceTokenCounter,
    SinglePassDualChannelGenerator,
    scalable_aime_protocol,
    scalable_gsm8k_protocol,
)
from topology_mas.execution.scalable_round_zero import (
    ScalableRoundZeroPoolConfig,
    ScalableRoundZeroPoolGenerator,
    ScalableRoundZeroPoolStore,
)
from topology_mas.models import AnswerState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--task-format", choices=("gsm8k", "aime"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--responses-per-task", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OHMYGPT_API_KEY")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-public-tokens", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = (
        load_aime_jsonl(args.tasks, split="test")
        if args.task_format == "aime"
        else read_tasks_jsonl(args.tasks)
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    token_counter = HuggingFaceTokenCounter(
        args.tokenizer,
        cache_dir=(
            str(args.tokenizer_cache_dir.resolve())
            if args.tokenizer_cache_dir is not None
            else None
        ),
    )
    protocol = (
        scalable_aime_protocol(token_counter)
        if args.task_format == "aime"
        else scalable_gsm8k_protocol(token_counter)
    )
    config = ScalableRoundZeroPoolConfig(
        responses_per_task=args.responses_per_task,
        base_seed=args.base_seed,
        requested_model=args.model,
        expected_returned_model=args.expected_returned_model,
        prompt_version=protocol.prompt_version,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_output_tokens=args.max_output_tokens,
    )
    with OpenAICompatibleTextGenerator(
        model=args.model,
        expected_returned_model=args.expected_returned_model,
        base_url=args.base_url,
        api_key_env=None if args.no_auth else args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        allow_context_window_adjustment=False,
    ) as backend:
        generator = SinglePassDualChannelGenerator(
            backend,
            answer_parser=protocol.answer_parser,
            token_counter=token_counter,
            max_public_tokens=args.max_public_tokens,
            strict_validation=False,
        )
        records = ScalableRoundZeroPoolGenerator(
            generator,
            config=config,
            store=ScalableRoundZeroPoolStore(args.output_dir),
            prompt_builder=lambda task: protocol.build_messages(
                task, previous_output=None, incoming_messages=()
            ),
            answer_parser=lambda raw, finish: protocol.parse_answer(
                raw, finish_reason=finish
            ),
            max_workers=args.max_workers,
        ).generate(tasks)
    state_counts = {
        state.value: sum(record.answer_state is state for record in records)
        for state in (
            AnswerState.CORRECT,
            AnswerState.OTHER_ERROR,
            AnswerState.UNPARSED,
        )
    }
    print(
        json.dumps(
            {
                "protocol_version": config.protocol_version,
                "output_dir": str(args.output_dir.resolve()),
                "task_count": len(tasks),
                "responses_per_task": args.responses_per_task,
                "response_count": len(records),
                "state_counts": state_counts,
                "one_backend_call_per_response": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
