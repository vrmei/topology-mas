#!/usr/bin/env python3
"""Build a fixed-size v2 Round-0 pool using only technical replacement.

Every candidate receives one solve call and at most one summary call.  C/O/U is
never used for selection.  A candidate is replaced only when the frozen public
summary interface rejects it technically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.execution.answers import classify_numeric_answer
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import HuggingFaceTokenCounter
from topology_mas.execution.scalable_round_zero import (
    ScalableRoundZeroPoolConfig,
    ScalableRoundZeroPoolResponse,
    ScalableRoundZeroPoolStore,
)
from topology_mas.execution.scalable_round_zero_cli import RoundRobinTextGenerator
from topology_mas.execution.schemas import TextGenerationRequest
from topology_mas.execution.seeding import stable_id, stable_integer
from topology_mas.execution.summary_protocol_v2 import (
    SUMMARY_PROTOCOL_V2,
    SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
    SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
    SUMMARY_PROTOCOL_V2_MODEL,
    SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
    SolveThenSummarizeGeneratorV2,
    SummaryProtocolV2Cache,
    require_summary_protocol_v2_settings,
    summary_protocol_v2,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
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


def fingerprint(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--cache-dir", type=Path, required=True)
    value.add_argument("--base-url", action="append", required=True)
    value.add_argument("--model", default=SUMMARY_PROTOCOL_V2_MODEL)
    value.add_argument("--expected-returned-model")
    value.add_argument("--tokenizer", required=True)
    value.add_argument("--responses-per-task", type=int, default=5)
    value.add_argument("--base-seed", type=int, default=20260903)
    value.add_argument("--max-candidates-per-task", type=int, default=20)
    value.add_argument("--max-workers", type=int, default=24)
    value.add_argument("--timeout-seconds", type=float, default=3600.0)
    value.add_argument("--provider-max-attempts", type=int, default=1)
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
    counter = HuggingFaceTokenCounter(args.tokenizer)
    protocol = summary_protocol_v2(counter)
    config = ScalableRoundZeroPoolConfig(
        responses_per_task=args.responses_per_task,
        base_seed=args.base_seed,
        requested_model=args.model,
        expected_returned_model=args.expected_returned_model,
        prompt_version=protocol.prompt_version,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_output_tokens=SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
    )
    store = ScalableRoundZeroPoolStore(args.output_dir)
    manifest = store.initialize(config=config, tasks=tasks)
    accepted = {
        task.task_id: sum(
            store.load(task_id=task.task_id, pool_slot=slot) is not None
            for slot in range(args.responses_per_task)
        )
        for task in tasks
    }

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
            token_counter=counter,
        )

        def generate_candidate(task: Any, candidate: int) -> tuple[Any, int, Any, Any]:
            seed = stable_integer(
                "summary-v2-validated-pool", args.base_seed, task.task_id, candidate
            )
            messages = protocol.build_messages(
                task, previous_output=None, incoming_messages=()
            )
            request = TextGenerationRequest(
                request_id=stable_id(
                    "summary-v2-validated-pool", task.task_id, candidate, seed
                ),
                messages=messages,
                seed=seed,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_output_tokens=SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
            )
            result = generator.generate(request)
            return task, candidate, messages, result

        for candidate in range(args.max_candidates_per_task):
            pending = [
                task for task in tasks if accepted[task.task_id] < args.responses_per_task
            ]
            if not pending:
                break
            with ThreadPoolExecutor(max_workers=min(args.max_workers, len(pending))) as pool:
                futures = {
                    pool.submit(generate_candidate, task, candidate): task
                    for task in pending
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        _, candidate_index, messages, result = future.result()
                        parsed = protocol.parse_answer(
                            result.raw_text, finish_reason=result.finish_reason
                        )
                        slot = accepted[task.task_id]
                        identity = {
                            "pool_version": manifest.pool_version,
                            "task_id": task.task_id,
                            "pool_slot": slot,
                            "candidate_index": candidate_index,
                            "generation_seed": stable_integer(
                                "summary-v2-validated-pool",
                                args.base_seed,
                                task.task_id,
                                candidate_index,
                            ),
                            "messages": [message.model_dump() for message in messages],
                            "config": config.model_dump(mode="json"),
                        }
                        response = ScalableRoundZeroPoolResponse(
                            pool_version=manifest.pool_version,
                            task_id=task.task_id,
                            pool_response_id=stable_id(
                                "pool-response", manifest.pool_version, task.task_id, slot
                            ),
                            pool_slot=slot,
                            generation_seed=identity["generation_seed"],
                            raw_response=result.raw_text,
                            parsed_answer=parsed,
                            answer_state=classify_numeric_answer(
                                parsed,
                                reference_answer=task.reference_answer,
                                target_answer=None,
                            ),
                            output_tokens=result.output_tokens,
                            input_tokens=result.input_tokens,
                            finish_reason=result.finish_reason,
                            latency_ms=result.latency_ms,
                            content_hash=hashlib.sha256(
                                result.raw_text.encode("utf-8")
                            ).hexdigest(),
                            requested_model=args.model,
                            returned_model=result.model_name,
                            prompt_version=protocol.prompt_version,
                            prompt_messages=tuple(
                                message.model_dump() for message in messages
                            ),
                            request_fingerprint=fingerprint(identity),
                            provider_metadata={
                                **result.metadata,
                                "generation_pipeline": SUMMARY_PROTOCOL_V2,
                                "summary_validation_passed": True,
                                "technical_replacement_only": True,
                                "candidate_index": candidate_index,
                            },
                        )
                        store.save(response)
                        accepted[task.task_id] += 1
                    except Exception as exc:
                        failure = {
                            "task_id": task.task_id,
                            "candidate_index": candidate,
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "payload": (
                                exc.to_failure_payload()
                                if callable(getattr(exc, "to_failure_payload", None))
                                else None
                            ),
                        }
                        atomic_json(
                            args.output_dir
                            / "technical_failures"
                            / task.task_id
                            / f"candidate_{candidate:04d}.json",
                            failure,
                        )
            progress = {
                "candidate_round": candidate,
                "accepted": sum(accepted.values()),
                "required": len(tasks) * args.responses_per_task,
                "per_task": accepted,
            }
            atomic_json(args.output_dir / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    missing = {
        task_id: args.responses_per_task - count
        for task_id, count in accepted.items()
        if count < args.responses_per_task
    }
    if missing:
        raise RuntimeError(f"validated Round-0 pool remains incomplete: {missing}")
    loaded_manifest, responses = store.load_complete()
    print(
        json.dumps(
            {
                "pool_version": loaded_manifest.pool_version,
                "responses": len(responses),
                "technical_failures": len(
                    list((args.output_dir / "technical_failures").rglob("*.json"))
                ),
                "summary_attempts_per_candidate": 1,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
