#!/usr/bin/env python3
"""Run the prepared evidence-volume one-step intervention against vLLM."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.answers import classify_numeric_answer, parse_numeric_answer
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.schemas import TextGenerationRequest
from topology_mas.experiments.evidence_volume import (
    EXPERIMENT_VERSION,
    atomic_json,
    content_fingerprint,
    read_jsonl,
    render_request_messages,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--expected-returned-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--max-workers", type=int, default=96)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--tokenizer")
    parser.add_argument("--server-context", type=int, default=8192)
    parser.add_argument(
        "--previous-mode",
        choices=("include", "omit"),
        default="include",
        help="Whether the receiver sees its frozen previous solution.",
    )
    return parser.parse_args()


def result_path(output_dir: Path, request_id: str) -> Path:
    return output_dir / "results" / f"{request_id}.json"


def audit_rendered_prompts(
    *,
    plan: list[dict[str, Any]],
    tasks: dict[str, Any],
    stimuli: dict[str, dict[str, Any]],
    previous_mode: str,
    tokenizer_path: str,
    maximum_input_tokens: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    counts: list[int] = []
    previous_heading_leaks = 0
    previous_id_peer_leaks = 0
    incidental_previous_text_matches: list[str] = []
    for row in plan:
        messages = render_request_messages(
            task=tasks[str(row["task_id"])],
            plan_row=row,
            stimuli=stimuli,
            include_previous=previous_mode == "include",
        )
        rendered = [{"role": item.role, "content": item.content} for item in messages]
        count = len(
            tokenizer.apply_chat_template(rendered, tokenize=True, add_generation_prompt=True)
        )
        counts.append(count)
        if count > maximum_input_tokens:
            raise ValueError(
                f"{row['request_id']} needs {count} input tokens; "
                f"maximum is {maximum_input_tokens}"
            )
        if previous_mode == "omit":
            user_prompt = messages[-1].content
            previous_id = str(row["previous_stimulus_id"])
            previous_text = str(
                stimuli[previous_id]["raw_text"]
            )
            old_update_section = (
                "\n\nReconsider the problem using your own work and the candidate peer "
                "reasoning.\n\nOUTPUT REQUIREMENT:"
            )
            if "YOUR_PREVIOUS_SOLUTION:\n" in user_prompt or old_update_section in user_prompt:
                previous_heading_leaks += 1
            if previous_id in row["peer_stimulus_ids"]:
                previous_id_peer_leaks += 1
            if previous_text in user_prompt:
                incidental_previous_text_matches.append(str(row["request_id"]))
    if previous_heading_leaks or previous_id_peer_leaks:
        raise ValueError(
            "previous solution leaked into no-self prompts: "
            f"heading={previous_heading_leaks}, peer_id={previous_id_peer_leaks}"
        )
    ordered = sorted(counts)
    return {
        "requests": len(counts),
        "previous_mode": previous_mode,
        "previous_heading_leaks": previous_heading_leaks,
        "previous_id_peer_leaks": previous_id_peer_leaks,
        "incidental_previous_text_match_count": len(incidental_previous_text_matches),
        "incidental_previous_text_match_request_ids": incidental_previous_text_matches,
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "maximum": ordered[-1],
        "maximum_allowed_input_tokens": maximum_input_tokens,
        "tokenizer": tokenizer_path,
    }


def atomic_result(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)


def main() -> None:
    args = parse_args()
    prepared = args.prepared_dir.resolve()
    output = args.output_dir.resolve()
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("prepared experiment version mismatch")
    frozen_sampling = manifest["sampling"]
    requested_sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
    }
    if frozen_sampling != requested_sampling:
        raise ValueError(
            f"sampling differs from frozen manifest: {requested_sampling} != {frozen_sampling}"
        )
    tasks = {task.task_id: task for task in read_tasks_jsonl(prepared / "tasks.jsonl")}
    targets = {
        str(row["task_id"]): str(row["target_answer"])
        for row in read_jsonl(prepared / "adversarial_answers.jsonl")
    }
    stimuli_rows = read_jsonl(prepared / "stimuli.jsonl")
    stimuli = {str(row["stimulus_id"]): row for row in stimuli_rows}
    plan = read_jsonl(prepared / "requests.jsonl")
    if content_fingerprint(stimuli_rows) != manifest["stimuli_fingerprint"]:
        raise ValueError("stimulus fingerprint mismatch")
    if content_fingerprint(plan) != manifest["request_plan_fingerprint"]:
        raise ValueError("request-plan fingerprint mismatch")

    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    if args.tokenizer:
        prompt_audit = audit_rendered_prompts(
            plan=plan,
            tasks=tasks,
            stimuli=stimuli,
            previous_mode=args.previous_mode,
            tokenizer_path=args.tokenizer,
            maximum_input_tokens=args.server_context - args.max_output_tokens,
        )
        atomic_json(output / "prompt_audit.json", prompt_audit)
    status_path = output / "status.json"
    existing = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "request_plan_fingerprint": manifest["request_plan_fingerprint"],
        "stimuli_fingerprint": manifest["stimuli_fingerprint"],
        "model": args.model,
        "sampling": requested_sampling,
        "previous_mode": args.previous_mode,
    }
    if existing and existing.get("identity") != identity:
        raise ValueError("existing output belongs to a different experiment")
    status: dict[str, Any] = existing or {
        "status": "running",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "identity": identity,
        "expected": len(plan),
        "completed": 0,
        "failed": 0,
    }
    status.update(status="running", pid=os.getpid(), host=socket.gethostname())
    lock = threading.Lock()

    cached: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in plan:
        path = result_path(output, str(row["request_id"]))
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("request_id") != row["request_id"]:
                raise ValueError(f"cached result identity mismatch: {path}")
            cached.append(item)
        else:
            pending.append(row)
    status["completed"] = len(cached)
    status["failed"] = 0
    atomic_json(status_path, status)

    generator = OpenAICompatibleTextGenerator(
        model=args.model,
        base_url=args.base_url,
        expected_returned_model=args.expected_returned_model,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        task = tasks[str(row["task_id"])]
        messages = render_request_messages(
            task=task,
            plan_row=row,
            stimuli=stimuli,
            include_previous=args.previous_mode == "include",
        )
        request = TextGenerationRequest(
            request_id=str(row["request_id"]),
            messages=messages,
            seed=int(row["generation_seed"]),
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
        )
        generated = generator.generate(request)
        parsed = parse_numeric_answer(generated.raw_text)
        state = classify_numeric_answer(
            parsed,
            reference_answer=task.reference_answer,
            target_answer=targets[task.task_id],
        )
        result = {
            **row,
            "raw_output": generated.raw_text,
            "parsed_answer": parsed,
            "next_state": state.value,
            "is_primary_outcome": state.value == row["primary_state"],
            "is_correct": state.value == "correct",
            "is_target": state.value == "target_error",
            "is_other": state.value == "other_error",
            "is_unparsed": state.value == "unparsed",
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "latency_ms": generated.latency_ms,
            "model_name": generated.model_name,
            "finish_reason": generated.finish_reason,
            "previous_mode": args.previous_mode,
            "completed_at": utc_now(),
        }
        atomic_result(result_path(output, str(row["request_id"])), result)
        return result

    failures: list[dict[str, Any]] = []
    generated_results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(execute, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    generated_results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - persist per-request failure
                    failures.append(
                        {
                            "request_id": row["request_id"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                with lock:
                    status["completed"] = len(cached) + len(generated_results)
                    status["failed"] = len(failures)
                    if index % 50 == 0 or index == len(pending):
                        status["updated_at"] = utc_now()
                        atomic_json(status_path, status)
                        print(
                            f"completed={status['completed']}/{len(plan)} failed={len(failures)}",
                            flush=True,
                        )
    finally:
        generator.close()

    all_results = cached + generated_results
    all_results.sort(key=lambda row: str(row["request_id"]))
    write_jsonl(output / "outcomes.jsonl", all_results)
    write_jsonl(output / "failures.jsonl", failures)
    status["status"] = (
        "completed" if len(all_results) == len(plan) and not failures else "incomplete"
    )
    status["completed"] = len(all_results)
    status["failed"] = len(failures)
    status["ended_at"] = utc_now()
    atomic_json(status_path, status)
    if status["status"] != "completed":
        raise RuntimeError(f"intervention incomplete: {status}")


if __name__ == "__main__":
    main()
