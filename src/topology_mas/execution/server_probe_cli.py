"""Probe a self-hosted OpenAI-compatible model server before experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.server_probe import (
    ServerProbeConfig,
    run_server_probe,
    write_server_probe_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ServerProbeConfig(
        requested_model=args.model,
        expected_returned_model=args.expected_returned_model,
        repetitions=args.repetitions,
        seed=args.seed,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    with OpenAICompatibleTextGenerator(
        model=args.model,
        expected_returned_model=args.expected_returned_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    ) as generator:
        report = run_server_probe(generator, base_url=args.base_url, config=config)
    output = write_server_probe_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "exact_repeat_observed": report.exact_repeat_observed,
                "returned_model_stable": report.returned_model_stable,
                "returned_model_matches_expectation": (
                    report.returned_model_matches_expectation
                ),
                "all_outputs_parseable": report.all_outputs_parseable,
                "token_usage_complete": report.token_usage_complete,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
