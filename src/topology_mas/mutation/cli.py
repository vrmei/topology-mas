"""Command-line entry point for offline mutation of one task record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology_mas.models import TaskInstance
from topology_mas.mutation.pipeline import MutationPipeline
from topology_mas.mutation.schemas import MutationPipelineConfig
from topology_mas.mutation.storage import MutationArtifactStore
from topology_mas.providers import OpenAICompatibleJSONClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="TaskInstance JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mutations"))
    parser.add_argument("--base-url", default="https://api.ohmygpt.com/v1")
    parser.add_argument("--api-key-env", default="OHMYGPT_API_KEY")
    parser.add_argument("--generator-model", default="gpt-5.6-sol")
    parser.add_argument("--plausibility-model", default="deepseek-chat")
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task = TaskInstance.model_validate_json(args.task.read_text(encoding="utf-8"))
    config = MutationPipelineConfig(
        generator_model=args.generator_model,
        plausibility_model=args.plausibility_model,
        candidate_count=args.candidate_count,
    )
    store = MutationArtifactStore(args.output_dir)

    with OpenAICompatibleJSONClient(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        pipeline = MutationPipeline(client, config=config, artifact_store=store)
        result = pipeline.run(task)

    summary = {
        "task_id": task.task_id,
        "candidate_count": len(result.evaluations),
        "objective_passed": sum(item.objective.passed for item in result.evaluations),
        "plausibility_passed": sum(item.eligible for item in result.evaluations),
        "selected_candidate_id": result.selected_candidate_id,
        "artifact_dir": str(args.output_dir / task.task_id.replace("/", "_")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
