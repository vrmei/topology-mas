#!/usr/bin/env python3
"""Generate the frozen K80 Llama/GSM8K solve-then-summary Round-0 pool."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.endpoint_task_pool import EndpointTaskPoolTextGenerator
from topology_mas.execution.numeric_summary_protocol import (
    NUMERIC_FULL_MAX_TOKENS,
    NUMERIC_SUMMARY_MODEL,
    NUMERIC_SUMMARY_PROTOCOL,
    SolveThenSummarizeNumericGenerator,
    numeric_summary_protocol,
)
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import HuggingFaceTokenCounter
from topology_mas.execution.scalable_round_zero import (
    ScalableRoundZeroPoolConfig,
    ScalableRoundZeroPoolGenerator,
    ScalableRoundZeroPoolStore,
)
from topology_mas.execution.summary_protocol_v3 import SummaryProtocolV3Cache
from topology_mas.models import AnswerState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default=NUMERIC_SUMMARY_MODEL)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--responses-per-task", type=int, default=80)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-seed", type=int, default=20260911)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--slots-per-backend", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()

    tasks = read_tasks_jsonl(args.tasks)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    counter = HuggingFaceTokenCounter(args.tokenizer)
    protocol = numeric_summary_protocol(counter)
    config = ScalableRoundZeroPoolConfig(
        responses_per_task=args.responses_per_task,
        base_seed=args.base_seed,
        requested_model=args.model,
        expected_returned_model=args.expected_returned_model,
        prompt_version=protocol.prompt_version,
        temperature=0.6,
        top_p=0.9,
        top_k=None,
        min_p=None,
        presence_penalty=None,
        max_output_tokens=NUMERIC_FULL_MAX_TOKENS,
    )
    with ExitStack() as stack:
        backends = tuple(
            stack.enter_context(
                OpenAICompatibleTextGenerator(
                    model=args.model,
                    expected_returned_model=args.expected_returned_model,
                    base_url=url,
                    api_key_env=None,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=3,
                    allow_context_window_adjustment=False,
                )
            )
            for url in args.base_url
        )
        backend_pool = EndpointTaskPoolTextGenerator(
            backends,
            slots_per_backend=args.slots_per_backend,
        )
        if args.max_workers < backend_pool.total_slots:
            raise ValueError("max-workers must be at least the total endpoint slots")
        generator = SolveThenSummarizeNumericGenerator(
            backend_pool,
            cache=SummaryProtocolV3Cache(args.cache_dir),
            token_counter=counter,
        )
        store = ScalableRoundZeroPoolStore(args.output_dir)
        generation_error = None
        try:
            records = ScalableRoundZeroPoolGenerator(
                generator,
                config=config,
                store=store,
                prompt_builder=lambda task: protocol.build_messages(
                    task, previous_output=None, incoming_messages=()
                ),
                answer_parser=lambda raw, finish: protocol.parse_answer(raw, finish_reason=finish),
                max_workers=args.max_workers,
            ).generate(tasks)
        except Exception as exc:
            generation_error = {"type": type(exc).__name__, "message": str(exc)}
            _, records = store.load_available()
    available_by_task = {
        task.task_id: sum(row.task_id == task.task_id for row in records) for task in tasks
    }
    print(
        json.dumps(
            {
                "protocol": NUMERIC_SUMMARY_PROTOCOL,
                "tasks": len(tasks),
                "responses": len(records),
                "intended_responses": len(tasks) * args.responses_per_task,
                "responses_per_task": args.responses_per_task,
                "available_by_task": available_by_task,
                "minimum_available_per_task": min(available_by_task.values()),
                "states": {
                    state.value: sum(row.answer_state is state for row in records)
                    for state in (
                        AnswerState.CORRECT,
                        AnswerState.OTHER_ERROR,
                        AnswerState.UNPARSED,
                    )
                },
                "all_summaries_valid": all(
                    row.provider_metadata.get("summary_validation_passed") is True
                    for row in records
                ),
                "endpoint_task_pool": backend_pool.snapshot(),
                "generation_error": generation_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
