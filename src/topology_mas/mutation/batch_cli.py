"""Batch target-error generation with resume-safe cache validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.mutation.batch import BatchMutationRunner
from topology_mas.mutation.pipeline import MutationPipeline
from topology_mas.mutation.schemas import MutationPipelineConfig
from topology_mas.mutation.storage import MutationArtifactStore
from topology_mas.providers import OpenAICompatibleJSONClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.ohmygpt.com/v1")
    parser.add_argument("--api-key-env", default="OHMYGPT_API_KEY")
    parser.add_argument("--generator-model", default="gpt-5.6-sol")
    parser.add_argument("--plausibility-model", default="deepseek-chat")
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = read_tasks_jsonl(args.tasks)
    config = MutationPipelineConfig(
        generator_model=args.generator_model,
        plausibility_model=args.plausibility_model,
        candidate_count=args.candidate_count,
    )
    task_artifacts = args.output_dir / "tasks"
    with OpenAICompatibleJSONClient(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        pipeline = MutationPipeline(
            client,
            config=config,
            artifact_store=MutationArtifactStore(task_artifacts),
        )
        runner = BatchMutationRunner(
            pipeline,
            output_dir=args.output_dir,
            fail_fast=args.fail_fast,
            max_workers=args.max_workers,
        )
        _, summary = runner.run(tasks, source_path=args.tasks)
    print(json.dumps(summary.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
