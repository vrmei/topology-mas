#!/usr/bin/env python3
"""Run or resume original-AIME Round-0 utility calibration against vLLM."""

from __future__ import annotations

import argparse
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.execution.aime import parse_aime_answer
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.schemas import ChatMessage, TextGenerationRequest
from topology_mas.experiments.aime_utility import (
    AIME_UTILITY_VERSION,
    atomic_json,
    build_round_zero_plan,
    canonical_fingerprint,
    result_path,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--max-output-tokens", type=int, default=3072)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_path = args.tasks.resolve()
    output = args.output_dir.resolve()
    tasks = load_aime_jsonl(tasks_path, split="original-2025-calibration")
    task_by_id = {task.task_id: task for task in tasks}
    plan = build_round_zero_plan(tasks, replicates=args.replicates)
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "max_output_tokens": args.max_output_tokens,
    }
    identity = {
        "experiment_version": AIME_UTILITY_VERSION,
        "tasks_sha256": canonical_fingerprint(
            [task.model_dump(mode="json") for task in tasks]
        ),
        "request_plan_fingerprint": canonical_fingerprint(plan),
        "model": args.model,
        "expected_returned_model": args.expected_returned_model,
        "sampling": sampling,
    }
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    existing = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else None
    )
    if existing is not None and existing.get("identity") != identity:
        raise ValueError("existing output belongs to a different frozen experiment")
    status: dict[str, Any] = existing or {
        "status": "running",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "identity": identity,
        "task_ids": [task.task_id for task in tasks],
        "task_count": len(tasks),
        "replicates": args.replicates,
        "expected": len(plan),
        "completed": 0,
        "failed": 0,
    }
    status.update(status="running", host=socket.gethostname(), pid=os.getpid())
    atomic_json(status_path, status)
    write_jsonl(output / "request_plan.jsonl", plan)

    cached: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in plan:
        path = result_path(output, row)
        if not path.exists():
            pending.append(row)
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("request_id") != row["request_id"]:
            raise ValueError(f"cached result identity mismatch: {path}")
        cached.append(item)
    status.update(completed=len(cached), failed=0, pending=len(pending))
    atomic_json(status_path, status)

    generator = OpenAICompatibleTextGenerator(
        model=args.model,
        base_url=args.base_url,
        expected_returned_model=args.expected_returned_model,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        task = task_by_id[str(row["task_id"])]
        messages = tuple(ChatMessage.model_validate(item) for item in row["messages"])
        completion = generator.generate(
            TextGenerationRequest(
                request_id=str(row["request_id"]),
                messages=messages,
                seed=int(row["generation_seed"]),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                max_output_tokens=args.max_output_tokens,
            )
        )
        parsed = parse_aime_answer(completion.raw_text)
        valid_answer = parsed is not None and completion.finish_reason != "length"
        result = {
            **row,
            "raw_output": completion.raw_text,
            "parsed_answer": parsed,
            "gold_answer": task.reference_answer,
            "is_correct": valid_answer and parsed == task.reference_answer,
            "is_parsed": valid_answer,
            "raw_parser_found_answer": parsed is not None,
            "requested_model": args.model,
            "returned_model": completion.model_name,
            "finish_reason": completion.finish_reason,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "latency_ms": completion.latency_ms,
            "completed_at": utc_now(),
        }
        atomic_json(result_path(output, row), result)
        return result

    failures: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(execute, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    generated.append(future.result())
                except Exception as exc:  # noqa: BLE001 - persist failures and continue
                    failures.append(
                        {
                            "request_id": row["request_id"],
                            "task_id": row["task_id"],
                            "replicate": row["replicate"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                if index % 20 == 0 or index == len(pending):
                    status.update(
                        completed=len(cached) + len(generated),
                        failed=len(failures),
                        pending=len(plan) - len(cached) - len(generated),
                        updated_at=utc_now(),
                    )
                    atomic_json(status_path, status)
                    print(
                        f"completed={status['completed']}/{len(plan)} "
                        f"failed={len(failures)}",
                        flush=True,
                    )
    finally:
        generator.close()

    outcomes = cached + generated
    outcomes.sort(key=lambda item: (str(item["task_id"]), int(item["replicate"])))
    write_jsonl(output / "outcomes.jsonl", outcomes)
    write_jsonl(output / "failures.jsonl", failures)
    complete = len(outcomes) == len(plan) and not failures
    status.update(
        status="completed" if complete else "incomplete",
        completed=len(outcomes),
        failed=len(failures),
        pending=len(plan) - len(outcomes),
        ended_at=utc_now(),
    )
    atomic_json(status_path, status)
    if not complete:
        raise RuntimeError(f"AIME utility run incomplete: {status}")


if __name__ == "__main__":
    main()
