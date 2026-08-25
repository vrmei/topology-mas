#!/usr/bin/env python3
"""Freeze trace-derived stimuli and a paired evidence-volume request plan."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.prompts import PROMPT_VERSION
from topology_mas.experiments.evidence_volume import (
    EXPERIMENT_VERSION,
    SCENARIOS,
    BoundedStimulusPool,
    atomic_json,
    build_request_plan,
    content_fingerprint,
    normalize_stimulus_text,
    render_request_messages,
    source_node_count,
    stimulus_id,
    valid_stimulus_text,
    write_jsonl,
)
from topology_mas.models import AnswerState


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--adversarial-answers", type=Path, required=True)
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--pool-capacity", type=int, default=512)
    parser.add_argument("--tokenizer")
    parser.add_argument("--server-context", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    return parser.parse_args()


def load_targets(path: Path) -> dict[str, str]:
    targets: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            targets[str(row["task_id"])] = str(row["target_answer"])
    return targets


def collect_stimuli(
    paths: list[Path],
    *,
    task_ids: set[str],
    targets: dict[str, str],
    capacity: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[tuple[str, str], BoundedStimulusPool] = defaultdict(
        lambda: BoundedStimulusPool(capacity)
    )
    source_sizes: set[int] = set()
    rejected_delimiters = 0
    turns_seen = 0
    for index, path in enumerate(paths, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace = payload["trace"]
        task_id = str(trace["task_id"])
        if task_id not in task_ids:
            continue
        if str(trace.get("target_answer") or targets[task_id]) != targets[task_id]:
            raise ValueError(f"target mismatch in {path}")
        attack_node = trace.get("attack_node")
        graph_id = str(trace["graph_id"])
        node_count = source_node_count(graph_id)
        if node_count is not None:
            source_sizes.add(node_count)
        for turn in trace["turns"]:
            turns_seen += 1
            if attack_node is not None and int(turn["node_id"]) == int(attack_node):
                continue
            state = str(turn["answer_state"])
            if state not in {
                AnswerState.CORRECT.value,
                AnswerState.TARGET_ERROR.value,
                AnswerState.OTHER_ERROR.value,
            }:
                continue
            if state == AnswerState.TARGET_ERROR.value and trace["condition"] != "attack":
                continue
            raw_text = normalize_stimulus_text(str(turn["raw_output"]))
            if not valid_stimulus_text(raw_text):
                rejected_delimiters += 1
                continue
            item_id = stimulus_id(task_id, state, raw_text)
            pools[(task_id, state)].add(
                {
                    "stimulus_id": item_id,
                    "task_id": task_id,
                    "state": state,
                    "raw_text": raw_text,
                    "parsed_answer": turn.get("parsed_answer"),
                    "source_condition": trace["condition"],
                    "source_graph_id": graph_id,
                    "source_n": node_count,
                    "source_run_id": trace["run_id"],
                    "source_node": turn["node_id"],
                    "source_round": turn["round_index"],
                }
            )
        if index % 10000 == 0:
            print(f"scanned {index}/{len(paths)} traces", flush=True)

    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    requirements = {
        AnswerState.CORRECT.value: 13,
        AnswerState.TARGET_ERROR.value: 3,
        AnswerState.OTHER_ERROR.value: 4,
    }
    for task_id in sorted(task_ids):
        counts[task_id] = {}
        for state, minimum in requirements.items():
            selected = pools[(task_id, state)].records()
            counts[task_id][state] = len(selected)
            if len(selected) < minimum:
                raise ValueError(
                    f"{task_id} has only {len(selected)} {state} stimuli; need {minimum}"
                )
            records.extend(selected)
    records.sort(key=lambda row: str(row["stimulus_id"]))
    audit = {
        "source_trace_files": len(paths),
        "turns_seen": turns_seen,
        "source_sizes": sorted(source_sizes),
        "rejected_delimiter_texts": rejected_delimiters,
        "pool_capacity_per_task_state": capacity,
        "counts_by_task_state": counts,
        "minimum_by_state": {
            state: min(counts[task][state] for task in counts) for state in requirements
        },
    }
    return records, audit


def audit_tokens(
    *,
    tokenizer_path: str,
    tasks: dict[str, Any],
    plan: list[dict[str, Any]],
    stimuli: dict[str, dict[str, Any]],
    maximum_input: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_counts: list[int] = []
    for index, row in enumerate(plan, start=1):
        messages = render_request_messages(
            task=tasks[str(row["task_id"])], plan_row=row, stimuli=stimuli
        )
        rendered = [message.model_dump() for message in messages]
        count = len(
            tokenizer.apply_chat_template(rendered, tokenize=True, add_generation_prompt=True)
        )
        row["estimated_input_tokens"] = count
        token_counts.append(count)
        if count > maximum_input:
            raise ValueError(
                f"{row['request_id']} needs {count} input tokens; maximum is {maximum_input}"
            )
        if index % 1000 == 0:
            print(f"token-audited {index}/{len(plan)} requests", flush=True)
    ordered = sorted(token_counts)
    return {
        "tokenizer": tokenizer_path,
        "requests": len(token_counts),
        "maximum_allowed_input_tokens": maximum_input,
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "maximum": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    tasks_tuple = read_tasks_jsonl(args.tasks)
    tasks = {task.task_id: task for task in tasks_tuple}
    targets = load_targets(args.adversarial_answers)
    if set(tasks) != set(targets):
        raise ValueError("task and target task IDs differ")
    trace_paths = sorted(
        {Path(path).resolve() for pattern in args.trace_glob for path in glob.glob(pattern)}
    )
    if not trace_paths:
        raise ValueError("trace globs matched no files")

    records, pool_audit = collect_stimuli(
        trace_paths,
        task_ids=set(tasks),
        targets=targets,
        capacity=args.pool_capacity,
    )
    stimuli = {str(row["stimulus_id"]): row for row in records}
    pool_by_task_state: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in records:
        pool_by_task_state[(str(row["task_id"]), str(row["state"]))].append(str(row["stimulus_id"]))
    plan = build_request_plan(
        task_ids=sorted(tasks),
        pool_by_task_state=pool_by_task_state,
        replicates=args.replicates,
    )
    expected = len(tasks) * len(SCENARIOS) * 5 * 3 * args.replicates
    if len(plan) != expected:
        raise ValueError(f"expected {expected} requests, built {len(plan)}")

    token_audit = None
    if args.tokenizer:
        token_audit = audit_tokens(
            tokenizer_path=args.tokenizer,
            tasks=tasks,
            plan=plan,
            stimuli=stimuli,
            maximum_input=args.server_context - args.max_output_tokens,
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "stimuli.jsonl", records)
    write_jsonl(output / "requests.jsonl", plan)
    shutil.copy2(args.tasks, output / "tasks.jsonl")
    shutil.copy2(args.adversarial_answers, output / "adversarial_answers.jsonl")
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "prepared_before_outcomes",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "task_count": len(tasks),
        "task_ids": sorted(tasks),
        "scenarios": SCENARIOS,
        "ratios": 5,
        "volume_multipliers": [1, 2, 3],
        "message_set_replicates": args.replicates,
        "expected_requests": len(plan),
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.9,
            "max_output_tokens": args.max_output_tokens,
        },
        "server_context": args.server_context,
        "pool_audit": pool_audit,
        "token_audit": token_audit,
        "stimuli_fingerprint": content_fingerprint(records),
        "request_plan_fingerprint": content_fingerprint(plan),
        "tasks_sha256": sha256_file(args.tasks),
        "adversarial_answers_sha256": sha256_file(args.adversarial_answers),
        "source_trace_globs": args.trace_glob,
        "source_trace_files": len(trace_paths),
        "information_boundary": (
            "one receiver update; task-matched previous output and distinct natural peer "
            "rationales; no topology or source identity shown"
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "requests": len(plan), "token_audit": token_audit}))


if __name__ == "__main__":
    main()
