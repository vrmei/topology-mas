#!/usr/bin/env python3
"""Generate a resumable AIME Round-0 pool under summary-protocol-v2."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import HuggingFaceTokenCounter
from topology_mas.execution.scalable_round_zero import (
    ScalableRoundZeroPoolConfig,
    ScalableRoundZeroPoolGenerator,
    ScalableRoundZeroPoolStore,
)
from topology_mas.execution.scalable_round_zero_cli import RoundRobinTextGenerator
from topology_mas.execution.summary_protocol_v2 import (
    SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
    SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
    SUMMARY_PROTOCOL_V2_MODEL,
    SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
    SolveThenSummarizeGeneratorV2,
    SummaryProtocolV2Cache,
    require_summary_protocol_v2_settings,
    summary_protocol_v2,
)
from topology_mas.models import AnswerState


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--cache-dir", type=Path, required=True)
    value.add_argument("--base-url", action="append", required=True)
    value.add_argument("--model", default=SUMMARY_PROTOCOL_V2_MODEL)
    value.add_argument("--expected-returned-model")
    value.add_argument("--tokenizer", required=True)
    value.add_argument("--tokenizer-cache-dir", type=Path)
    value.add_argument("--responses-per-task", type=int, default=64)
    value.add_argument("--base-seed", type=int, default=20260903)
    value.add_argument("--max-workers", type=int, default=24)
    value.add_argument("--timeout-seconds", type=float, default=3600.0)
    value.add_argument("--provider-max-attempts", type=int, default=3)
    return value


def main() -> None:
    args = parser().parse_args()
    require_summary_protocol_v2_settings(
        model=args.model,
        full_temperature=0.7,
        full_top_p=0.8,
        full_top_k=20,
        full_max_output_tokens=SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
        summary_max_output_tokens=SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
        summary_max_attempts=SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
    )
    tasks = load_aime_jsonl(args.tasks, split="test")
    token_counter = HuggingFaceTokenCounter(
        args.tokenizer,
        cache_dir=(str(args.tokenizer_cache_dir.resolve()) if args.tokenizer_cache_dir else None),
    )
    protocol = summary_protocol_v2(token_counter)
    config = ScalableRoundZeroPoolConfig(
        responses_per_task=args.responses_per_task,
        base_seed=args.base_seed,
        requested_model=args.model,
        expected_returned_model=args.expected_returned_model,
        prompt_version=protocol.prompt_version,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=None,
        presence_penalty=None,
        max_output_tokens=SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
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
                    max_attempts=args.provider_max_attempts,
                    allow_context_window_adjustment=False,
                )
            )
            for url in args.base_url
        )
        generator = SolveThenSummarizeGeneratorV2(
            RoundRobinTextGenerator(backends),
            cache=SummaryProtocolV2Cache(args.cache_dir),
            token_counter=token_counter,
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
    print(
        json.dumps(
            {
                "prompt_version": protocol.prompt_version,
                "responses": len(records),
                "states": {
                    state.value: sum(record.answer_state is state for record in records)
                    for state in (
                        AnswerState.CORRECT,
                        AnswerState.OTHER_ERROR,
                        AnswerState.UNPARSED,
                    )
                },
                "all_summaries_validated": all(
                    record.provider_metadata.get("summary_validation_passed") is True
                    for record in records
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

