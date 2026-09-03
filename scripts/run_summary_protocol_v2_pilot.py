#!/usr/bin/env python3
"""Run the frozen 150-solve, difficulty-stratified summary-protocol-v2 gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import HuggingFaceTokenCounter
from topology_mas.execution.scalable_round_zero_cli import RoundRobinTextGenerator
from topology_mas.execution.schemas import TextGenerationRequest
from topology_mas.execution.seeding import stable_id, stable_integer
from topology_mas.execution.summary_protocol_v2 import (
    SUMMARY_PROTOCOL_V2,
    SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
    SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
    SUMMARY_PROTOCOL_V2_MODEL,
    SUMMARY_PROTOCOL_V2_PROMPT_VERSION,
    SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
    SolveThenSummarizeGeneratorV2,
    SummaryProtocolV2Cache,
    parse_summary_envelope_v2,
    require_summary_protocol_v2_settings,
    summary_protocol_v2,
    validate_public_summary_v2,
)

PILOT_VERSION = "summary-protocol-v2-stratified-pilot-v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--difficulty-csv", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--cache-dir", type=Path, required=True)
    value.add_argument("--base-url", action="append", required=True)
    value.add_argument("--model", default=SUMMARY_PROTOCOL_V2_MODEL)
    value.add_argument("--expected-returned-model")
    value.add_argument("--tokenizer", required=True)
    value.add_argument("--tokenizer-cache-dir", type=Path)
    value.add_argument("--responses-per-task", type=int, default=5)
    value.add_argument("--base-seed", type=int, default=20260903)
    value.add_argument("--max-workers", type=int, default=24)
    value.add_argument("--timeout-seconds", type=float, default=3600.0)
    value.add_argument("--provider-max-attempts", type=int, default=1)
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_bands(path: Path) -> dict[str, str]:
    aliases = {"informative": "intermediate", "middle": "intermediate"}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    bands: dict[str, str] = {}
    for row in rows:
        task_id = row.get("task_id")
        band = row.get("difficulty_band")
        if task_id and band:
            bands[task_id] = aliases.get(band, band)
    return bands


def job_path(root: Path, task_id: str, slot: int) -> Path:
    safe = task_id.replace("/", "_").replace("\\", "_")
    return root / "jobs" / safe / f"response_{slot:02d}.json"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def one(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        successful = [row for row in rows if row["status"] == "validated"]
        parseable = [row for row in successful if row["full_answer"] is not None]
        unparsed = [row for row in successful if row["full_answer"] is None]
        attempted_retry = [
            row for row in rows
            if row.get("summary_attempt_count", 0) > 1
            or row.get("status") == "failed"
        ]
        return {
            "jobs": total,
            "validated": len(successful),
            "structure_success_rate": len(successful) / total if total else None,
            "parseable_full_jobs": len(parseable),
            "parseable_answer_preservation_rate": (
                sum(row["answer_preserved"] for row in parseable) / len(parseable)
                if parseable else None
            ),
            "state_preservation_rate": (
                sum(row["state_preserved"] for row in successful) / len(successful)
                if successful else None
            ),
            "unparsed_full_jobs": len(unparsed),
            "u_to_parsed_count": sum(row["u_to_parsed"] for row in successful),
            "u_to_parsed_rate": (
                sum(row["u_to_parsed"] for row in unparsed) / len(unparsed)
                if unparsed else None
            ),
            "full_length_stop_count": sum(
                row.get("full_finish_reason") == "length" for row in rows
            ),
            "full_length_stop_rate": (
                sum(row.get("full_finish_reason") == "length" for row in rows) / total
                if total else None
            ),
            "summary_retry_job_count": len(attempted_retry),
            "summary_retry_job_rate": len(attempted_retry) / total if total else None,
            "failed_jobs": total - len(successful),
        }

    overall = one(records)
    by_band = {
        band: one([row for row in records if row["difficulty_band"] == band])
        for band in ("floor", "intermediate", "ceiling")
    }
    gate = {
        "structure_success_at_least_0_99": overall["structure_success_rate"] >= 0.99,
        "parseable_preservation_at_least_0_99": (
            overall["parseable_answer_preservation_rate"] is not None
            and overall["parseable_answer_preservation_rate"] >= 0.99
        ),
        "no_u_to_parsed": overall["u_to_parsed_count"] == 0,
    }
    gate["passed"] = all(gate.values())
    return {
        "pilot_version": PILOT_VERSION,
        "protocol": SUMMARY_PROTOCOL_V2,
        "prompt_version": SUMMARY_PROTOCOL_V2_PROMPT_VERSION,
        "overall": overall,
        "by_difficulty_band": by_band,
        "gate": gate,
    }


def main() -> None:
    args = parser().parse_args()
    if args.responses_per_task != 5:
        raise ValueError("the frozen pilot requires exactly 5 responses per task")
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
    if len(tasks) != 30:
        raise ValueError(f"the frozen pilot requires all 30 AIME tasks; found {len(tasks)}")
    bands = load_bands(args.difficulty_csv)
    missing_bands = sorted(task.task_id for task in tasks if task.task_id not in bands)
    if missing_bands:
        raise ValueError(f"difficulty bands missing for tasks: {missing_bands}")
    band_counts = {
        band: sum(bands[task.task_id] == band for task in tasks)
        for band in ("floor", "intermediate", "ceiling")
    }
    if band_counts != {"floor": 9, "intermediate": 12, "ceiling": 9}:
        raise ValueError(f"unexpected frozen difficulty stratification: {band_counts}")

    token_counter = HuggingFaceTokenCounter(
        args.tokenizer,
        cache_dir=(str(args.tokenizer_cache_dir.resolve()) if args.tokenizer_cache_dir else None),
    )
    protocol = summary_protocol_v2(token_counter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
            "pilot_version": PILOT_VERSION,
            "protocol": SUMMARY_PROTOCOL_V2,
            "prompt_version": protocol.prompt_version,
            "tasks": len(tasks),
            "responses_per_task": args.responses_per_task,
            "planned_jobs": len(tasks) * args.responses_per_task,
            "difficulty_band_counts": band_counts,
            "model": args.model,
            "provider_max_attempts": args.provider_max_attempts,
            "full_sampling": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_output_tokens": SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
            },
            "summary_sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "max_output_tokens": SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
                "max_attempts": SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
            },
            "gate": {
                "structure_success_rate": 0.99,
                "parseable_answer_preservation_rate": 0.99,
                "u_to_parsed_count": 0,
            },
        }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("pilot manifest differs; use a new output directory")
    else:
        atomic_json(manifest_path, manifest)

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

        def execute(task: Any, slot: int) -> dict[str, Any]:
            path = job_path(args.output_dir, task.task_id, slot)
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            seed = stable_integer(PILOT_VERSION, args.base_seed, task.task_id, slot)
            request = TextGenerationRequest(
                request_id=stable_id(PILOT_VERSION, task.task_id, slot, seed),
                messages=protocol.build_messages(
                    task, previous_output=None, incoming_messages=()
                ),
                seed=seed,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_output_tokens=SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
            )
            try:
                result = generator.generate(request)
                envelope = parse_summary_envelope_v2(result.raw_text)
                full_answer = protocol.parse_answer(
                    result.raw_text, finish_reason=result.finish_reason
                )
                validated = validate_public_summary_v2(
                    envelope.public_summary,
                    full_answer=full_answer,
                    finish_reason="stop",
                    token_counter=token_counter,
                )
                full_state = (
                    "U" if full_answer is None else
                    "C" if full_answer == task.reference_answer else "O"
                )
                summary_state = (
                    "U" if validated.parsed_answer is None else
                    "C" if validated.parsed_answer == task.reference_answer else "O"
                )
                record = {
                    "status": "validated",
                    "task_id": task.task_id,
                    "slot": slot,
                    "difficulty_band": bands[task.task_id],
                    "seed": seed,
                    "raw_response": result.raw_text,
                    "full_answer": full_answer,
                    "summary_answer": validated.parsed_answer,
                    "full_state": full_state,
                    "summary_state": summary_state,
                    "answer_preserved": full_answer == validated.parsed_answer,
                    "state_preserved": full_state == summary_state,
                    "u_to_parsed": full_answer is None and validated.parsed_answer is not None,
                    "full_finish_reason": envelope.full_finish_reason,
                    "summary_tokens": validated.token_count,
                    "summary_attempt_count": result.metadata["summary_attempt_count"],
                    "summary_retry_count": result.metadata["summary_retry_count"],
                    "metadata": result.metadata,
                }
            except Exception as exc:  # every failed interface call is retained
                payload = (
                    exc.to_failure_payload()
                    if callable(getattr(exc, "to_failure_payload", None))
                    else {"message": str(exc)}
                )
                record = {
                    "status": "failed",
                    "task_id": task.task_id,
                    "slot": slot,
                    "difficulty_band": bands[task.task_id],
                    "seed": seed,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "failure_payload": payload,
                    "full_finish_reason": payload.get("full_completion", {}).get(
                        "finish_reason"
                    ),
                    "summary_attempt_count": len(payload.get("summary_attempts", [])),
                }
            atomic_json(path, record)
            return record

        jobs = [(task, slot) for task in tasks for slot in range(args.responses_per_task)]
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures: dict[Future[dict[str, Any]], tuple[str, int]] = {
                executor.submit(execute, task, slot): (task.task_id, slot)
                for task, slot in jobs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if completed % 10 == 0 or completed == len(futures):
                    report = summarize(records)
                    report["completed_jobs"] = completed
                    report["planned_jobs"] = len(futures)
                    atomic_json(args.output_dir / "progress.json", report)
                    print(json.dumps(report["overall"], sort_keys=True), flush=True)

    records.sort(key=lambda row: (row["task_id"], row["slot"]))
    report = summarize(records)
    atomic_json(args.output_dir / "records.json", records)
    atomic_json(args.output_dir / "gate_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["gate"]["passed"]:
        raise RuntimeError("summary-protocol-v2 did not pass the frozen pilot gate")


if __name__ == "__main__":
    main()
