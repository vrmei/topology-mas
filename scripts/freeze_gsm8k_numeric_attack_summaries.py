#!/usr/bin/env python3
"""Create one validated public summary for every frozen GSM8K target error."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from topology_mas.execution.answers import parse_numeric_answer
from topology_mas.execution.endpoint_task_pool import EndpointTaskPoolTextGenerator
from topology_mas.execution.inputs import load_adversarial_answer_index
from topology_mas.execution.numeric_summary_protocol import (
    NUMERIC_SUMMARY_MAX_TOKENS,
    NUMERIC_SUMMARY_MODEL,
    NUMERIC_SUMMARY_SYSTEM_PROMPT,
    serialize_numeric_public_summary,
    validate_numeric_public_summary,
)
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import HuggingFaceTokenCounter
from topology_mas.execution.schemas import ChatMessage, TextGenerationRequest
from topology_mas.execution.seeding import stable_integer
from topology_mas.execution.summary_protocol_v3 import (
    SummaryProtocolV3Cache,
    sanitize_retry_summary_body_v3,
    validate_summary_body_v3,
)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default=NUMERIC_SUMMARY_MODEL)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--slots-per-backend", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()

    answers = load_adversarial_answer_index(args.input)
    counter = HuggingFaceTokenCounter(args.tokenizer)
    cache = SummaryProtocolV3Cache(args.cache_dir)
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
        pool = EndpointTaskPoolTextGenerator(
            backends,
            slots_per_backend=args.slots_per_backend,
        )
        if args.max_workers < pool.total_slots:
            raise ValueError("max-workers must be at least the total endpoint slots")

        def freeze(task_id: str):
            answer = answers[task_id]
            parsed = parse_numeric_answer(answer.rationale)
            if parsed != answer.target_answer:
                raise ValueError(f"{task_id}: rationale does not parse to frozen target")
            request = TextGenerationRequest(
                request_id=f"attack-summary-{task_id}",
                messages=(
                    ChatMessage(role="system", content=NUMERIC_SUMMARY_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=(
                            f"IMMUTABLE_FULL_SOLUTION:\n{answer.rationale}\n\n"
                            f"FROZEN_FULL_PARSER_STATE: PARSED({answer.target_answer})"
                        ),
                    ),
                ),
                seed=stable_integer("gsm8k-fixed-attack-summary-v1", task_id),
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                presence_penalty=0.0,
                max_output_tokens=NUMERIC_SUMMARY_MAX_TOKENS,
            )
            key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            with cache.lock_for("numeric-attack-summary:" + key):
                result = cache.load_result("numeric-attack-summary", key)
                cache_hit = result is not None
                if result is None:
                    result = pool.generate(request)
                    cache.save_result("numeric-attack-summary", key, result)
            sanitized_markers: tuple[str, ...] = ()
            try:
                body = validate_summary_body_v3(
                    result.raw_text,
                    finish_reason=result.finish_reason,
                    token_counter=counter,
                ).text
            except ValueError as exc:
                if "forbidden protocol marker" not in str(exc):
                    raise
                sanitized, sanitized_markers = sanitize_retry_summary_body_v3(result.raw_text)
                body = validate_summary_body_v3(
                    sanitized,
                    finish_reason=result.finish_reason,
                    token_counter=counter,
                ).text
            public, public_answer, token_count = validate_numeric_public_summary(
                serialize_numeric_public_summary(body, answer.target_answer),
                full_answer=answer.target_answer,
                token_counter=counter,
            )
            if public_answer != answer.target_answer:
                raise AssertionError("validated attack summary changed target")
            return answer.model_copy(
                update={
                    "public_summary": public,
                    "public_summary_tokens": token_count,
                    "public_summary_hash": hashlib.sha256(public.encode()).hexdigest(),
                    "metadata": {
                        **answer.metadata,
                        "public_summary_protocol": "numeric-summary-protocol-v1",
                        "public_summary_temperature": 0.0,
                        "public_summary_top_p": 1.0,
                        "public_summary_top_k": -1,
                        "public_summary_max_tokens": NUMERIC_SUMMARY_MAX_TOKENS,
                        "public_summary_cache_hit": cache_hit,
                        "public_summary_request_sha256": key,
                        "public_summary_model": result.model_name,
                        "public_summary_sanitized_markers": list(sanitized_markers),
                    },
                }
            )

        resolved = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(freeze, task_id): task_id for task_id in answers}
            for future in as_completed(futures):
                task_id = futures[future]
                resolved[task_id] = future.result()
                print(
                    json.dumps(
                        {
                            "completed": len(resolved),
                            "total": len(answers),
                            "task_id": task_id,
                            "task_pool": pool.snapshot(),
                        }
                    ),
                    flush=True,
                )

    rows = "".join(resolved[task_id].model_dump_json() + "\n" for task_id in answers)
    _atomic_write(args.output, rows)
    print(
        json.dumps(
            {
                "answers": len(resolved),
                "all_preserved": all(
                    parse_numeric_answer(row.public_summary or "") == row.target_answer
                    for row in resolved.values()
                ),
                "output": str(args.output.resolve()),
                "task_pool": pool.snapshot(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
