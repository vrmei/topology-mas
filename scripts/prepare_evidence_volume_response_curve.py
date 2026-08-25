#!/usr/bin/env python3
"""Freeze the extended evidence-volume curve and token-matched request plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.experiments.evidence_volume import (
    content_fingerprint,
    read_jsonl,
    render_request_messages,
    write_jsonl,
)
from topology_mas.experiments.evidence_volume_curve import (
    EXPERIMENT_VERSION,
    build_curve_request_plan,
    build_token_matched_plan,
    select_supported_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--server-context", type=int, default=32768)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--task-count", type=int, default=40)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], quantile: float) -> int:
    return int(np.quantile(np.asarray(values, dtype=float), quantile, method="nearest"))


def main() -> None:
    args = parse_args()
    source = args.source_prepared_dir.resolve()
    output = args.output_dir.resolve()
    if args.server_context <= args.max_output_tokens:
        raise ValueError("server context must exceed the output allowance")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("experiment_version") != "evidence-volume-intervention-v1":
        raise ValueError("unexpected source intervention version")

    all_tasks = {task.task_id: task for task in read_tasks_jsonl(source / "tasks.jsonl")}
    adversarial = {
        str(row["task_id"]): row
        for row in read_jsonl(source / "adversarial_answers.jsonl")
    }
    stimulus_rows = read_jsonl(source / "stimuli.jsonl")
    stimuli = {str(row["stimulus_id"]): row for row in stimulus_rows}
    pool_by_task_state: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in stimulus_rows:
        pool_by_task_state[(str(row["task_id"]), str(row["state"]))].append(
            str(row["stimulus_id"])
        )

    selected_tasks = select_supported_tasks(
        pool_by_task_state,
        count=args.task_count,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_lengths: dict[str, int] = {}
    for row in stimulus_rows:
        item_id = str(row["stimulus_id"])
        token_lengths[item_id] = len(
            tokenizer.encode(str(row["raw_text"]), add_special_tokens=False)
        )

    curve_plan = build_curve_request_plan(
        task_ids=selected_tasks,
        pool_by_task_state=pool_by_task_state,
    )
    token_plan = build_token_matched_plan(
        task_ids=selected_tasks,
        pool_by_task_state=pool_by_task_state,
        token_lengths=token_lengths,
        skip_unsupported_tasks=True,
    )
    token_task_ids = sorted({str(row["task_id"]) for row in token_plan})
    if len(token_task_ids) < 30:
        raise ValueError(
            f"token-matched control supports only {len(token_task_ids)} tasks; need 30"
        )
    plan = [*curve_plan, *token_plan]
    expected = len(curve_plan) + len(token_task_ids) * 2 * 5
    if len(plan) != expected:
        raise ValueError(f"expected {expected} requests, built {len(plan)}")

    maximum_input = args.server_context - args.max_output_tokens
    prompt_tokens: list[int] = []
    kind_tokens: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(plan, start=1):
        row["peer_message_tokens"] = sum(
            token_lengths[str(item)] for item in row["peer_stimulus_ids"]
        )
        messages = render_request_messages(
            task=all_tasks[str(row["task_id"])],
            plan_row=row,
            stimuli=stimuli,
            include_previous=str(row["previous_mode"]) == "include",
        )
        rendered = [{"role": item.role, "content": item.content} for item in messages]
        count = len(
            tokenizer.apply_chat_template(
                rendered,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        row["estimated_input_tokens"] = count
        if count > maximum_input:
            raise ValueError(
                f"{row['request_id']} needs {count} input tokens; maximum is {maximum_input}"
            )
        prompt_tokens.append(count)
        kind_tokens[str(row["request_kind"])].append(count)
        if index % 1000 == 0:
            print(f"token-audited {index}/{len(plan)} requests", flush=True)

    pairs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in token_plan:
        pairs[str(row["token_match_pair_id"])].append(row)
    actual_pair_differences: list[int] = []
    for pair_id, rows in pairs.items():
        if len(rows) != 2:
            raise ValueError(f"incomplete token pair: {pair_id}")
        by_condition = {str(row["token_match_condition"]): row for row in rows}
        long = by_condition["four_long"]
        short = by_condition["eight_short"]
        difference = abs(
            int(long["estimated_input_tokens"]) - int(short["estimated_input_tokens"])
        )
        tolerance = max(
            128,
            0.10
            * (
                (int(long["estimated_input_tokens"]) + int(short["estimated_input_tokens"]))
                / 2
            ),
        )
        if difference > tolerance:
            raise ValueError(
                f"rendered token match failed for {pair_id}: difference={difference}, "
                f"tolerance={tolerance:.1f}"
            )
        actual_pair_differences.append(difference)

    referenced_ids = {
        str(item)
        for row in plan
        for item in (
            *row["peer_stimulus_ids"],
            *(
                [row["previous_stimulus_id"]]
                if row.get("previous_stimulus_id") is not None
                else []
            ),
        )
    }
    retained_stimuli = [
        {**row, "message_tokens": token_lengths[str(row["stimulus_id"])]}
        for row in stimulus_rows
        if str(row["stimulus_id"]) in referenced_ids
    ]
    retained_stimuli.sort(key=lambda row: str(row["stimulus_id"]))
    retained_tasks = [all_tasks[task_id] for task_id in selected_tasks]
    retained_targets = [adversarial[task_id] for task_id in selected_tasks]

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output / "tasks.jsonl",
        (task.model_dump(mode="json") for task in retained_tasks),
    )
    write_jsonl(output / "adversarial_answers.jsonl", retained_targets)
    write_jsonl(output / "stimuli.jsonl", retained_stimuli)
    write_jsonl(output / "requests.jsonl", plan)
    (output / "selected_task_ids.txt").write_text(
        "\n".join(selected_tasks) + "\n", encoding="utf-8"
    )

    token_audit = {
        "tokenizer": args.tokenizer,
        "requests": len(plan),
        "maximum_allowed_input_tokens": maximum_input,
        "minimum": min(prompt_tokens),
        "median": percentile(prompt_tokens, 0.5),
        "p95": percentile(prompt_tokens, 0.95),
        "maximum": max(prompt_tokens),
        "by_request_kind": {
            kind: {
                "requests": len(values),
                "median": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "maximum": max(values),
            }
            for kind, values in sorted(kind_tokens.items())
        },
        "token_matched_rendered_difference": {
            "pairs": len(actual_pair_differences),
            "median": percentile(actual_pair_differences, 0.5),
            "p95": percentile(actual_pair_differences, 0.95),
            "maximum": max(actual_pair_differences),
        },
    }
    manifest: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "prepared_before_outcomes",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment_version": source_manifest["experiment_version"],
        "source_stimuli_fingerprint": source_manifest["stimuli_fingerprint"],
        "task_selection_rule": "top 40 by min(T pool, O pool), stable task-ID tie break",
        "task_count": len(selected_tasks),
        "task_ids": selected_tasks,
        "token_matched_task_count": len(token_task_ids),
        "token_matched_task_ids": token_task_ids,
        "response_curve_requests": len(curve_plan),
        "token_matched_requests": len(token_plan),
        "expected_requests": len(plan),
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.9,
            "max_output_tokens": args.max_output_tokens,
        },
        "server_context": args.server_context,
        "token_audit": token_audit,
        "stimuli_fingerprint": content_fingerprint(retained_stimuli),
        "request_plan_fingerprint": content_fingerprint(plan),
        "tasks_sha256": sha256_file(output / "tasks.jsonl"),
        "adversarial_answers_sha256": sha256_file(output / "adversarial_answers.jsonl"),
        "information_boundary": (
            "one anonymous homogeneous receiver update; no topology, source identity, "
            "receiver identity, role, or attack metadata"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "requests": len(plan),
                "tasks": len(selected_tasks),
                "token_matched_tasks": len(token_task_ids),
                "token_audit": token_audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
