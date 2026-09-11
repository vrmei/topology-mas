#!/usr/bin/env python3
"""Run the frozen paired round-phase receiver replay against local vLLM."""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_round_phase_receiver_replay import EXPERIMENT_VERSION, fingerprint, render
from topology_mas.execution.answers import classify_numeric_answer, parse_numeric_answer
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.schemas import TextGenerationRequest
from topology_mas.models import TaskInstance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--expected-returned-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    prepared = args.prepared_dir.resolve()
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["experiment_version"] != EXPERIMENT_VERSION:
        raise ValueError("prepared experiment version mismatch")
    plan = read_jsonl(prepared / "requests.jsonl")
    stimuli_rows = read_jsonl(prepared / "stimuli.jsonl")
    if fingerprint(plan) != manifest["request_fingerprint"]:
        raise ValueError("request fingerprint mismatch")
    if fingerprint(stimuli_rows) != manifest["stimuli_fingerprint"]:
        raise ValueError("stimuli fingerprint mismatch")
    stimuli = {str(row["stimulus_id"]): row for row in stimuli_rows}
    tasks = {
        str(row["task_id"]): TaskInstance.model_validate(row)
        for row in read_jsonl(prepared / "tasks.jsonl")
    }
    targets = {
        str(row["task_id"]): str(row["target_answer"])
        for row in read_jsonl(prepared / "adversarial_answers.jsonl")
    }
    sampling = manifest["sampling"]
    args.out.mkdir(parents=True, exist_ok=True)
    result_dir = args.out / "results"
    failure_dir = args.out / "failures"
    result_dir.mkdir(exist_ok=True)
    failure_dir.mkdir(exist_ok=True)
    pending = [row for row in plan if not (result_dir / f"{row['request_id']}.json").exists()]
    lock = threading.Lock()
    status = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "expected": len(plan),
        "cached": len(plan) - len(pending),
        "completed": len(plan) - len(pending),
        "failed": 0,
        "model": args.model,
        "sampling": sampling,
    }
    atomic_json(args.out / "status.json", status)
    generator = OpenAICompatibleTextGenerator(
        model=args.model,
        base_url=args.base_url,
        expected_returned_model=args.expected_returned_model,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        messages = render(tasks[row["task_id"]], row, stimuli)
        generated = generator.generate(
            TextGenerationRequest(
                request_id=row["request_id"],
                messages=messages,
                seed=int(row["generation_seed"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                max_output_tokens=int(sampling["max_output_tokens"]),
            )
        )
        parsed = parse_numeric_answer(generated.raw_text)
        state = classify_numeric_answer(
            parsed,
            reference_answer=tasks[row["task_id"]].reference_answer,
            target_answer=targets[row["task_id"]],
        )
        payload = {
            **row,
            "raw_output": generated.raw_text,
            "parsed_answer": parsed,
            "next_state": state.value,
            "is_correct": state.value == "correct",
            "is_target": state.value == "target_error",
            "is_other": state.value == "other_error",
            "is_unparsed": state.value == "unparsed",
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "latency_ms": generated.latency_ms,
            "finish_reason": generated.finish_reason,
            "model_name": generated.model_name,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(result_dir / f"{row['request_id']}.json", payload)
        return payload

    completed = status["completed"]
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(execute, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    future.result()
                    completed += 1
                except Exception as exc:  # persist every technical failure
                    failed += 1
                    atomic_json(
                        failure_dir / f"{row['request_id']}.json",
                        {"request": row, "error_type": type(exc).__name__, "error": str(exc)},
                    )
                if index % 25 == 0 or index == len(pending):
                    with lock:
                        status.update(
                            completed=completed,
                            failed=failed,
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        )
                        atomic_json(args.out / "status.json", status)
                        print(f"completed={completed}/{len(plan)} failed={failed}", flush=True)
    finally:
        generator.close()
    status.update(
        status="completed" if completed == len(plan) and failed == 0 else "incomplete",
        completed=completed,
        failed=failed,
        ended_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(args.out / "status.json", status)
    if status["status"] != "completed":
        raise RuntimeError(f"replay incomplete: {status}")


if __name__ == "__main__":
    main()
