#!/usr/bin/env python3
"""Freeze the 600-call controlled argument-completion receiver intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from topology_mas.execution.answers import parse_numeric_answer
from topology_mas.execution.prompts import PROMPT_VERSION, build_node_messages
from topology_mas.execution.schemas import ChatMessage
from topology_mas.execution.seeding import stable_id, stable_integer
from topology_mas.models import AnswerState, MessageRecord, TaskInstance


EXPERIMENT_VERSION = "argument-completion-replay-v1"
SAMPLING = {"temperature": 0.6, "top_p": 0.9, "max_output_tokens": 768}
CONDITIONS = ("claim", "length_control", "claim_evidence", "claim_warrant", "claim_evidence_warrant")
NEUTRAL_SENTENCES = (
    "I reviewed the wording and kept the quantities in view.",
    "I considered the setup from several angles before recording my answer.",
    "The statement contains several details that deserve careful attention.",
    "I checked the presentation and organized the information consistently.",
    "This note records that the problem was considered carefully.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prepared-dir", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--server-context", type=int, default=16384)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--correct-peer-count", type=int, action="append", default=[])
    parser.add_argument("--length-tolerance", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.split())


def evidence_fragments(source: str, target: str) -> list[str]:
    cleaned = re.sub(r"(?im)^\s*(?:FINAL_ANSWER\s*:|####)\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$", "", source)
    candidates = [normalize(part) for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if normalize(part)]
    selected = []
    for sentence in candidates:
        has_number = bool(re.search(r"\d", sentence))
        has_relation = bool(re.search(r"[=+*/]|\d\s*-\s*\d|\b(?:times|twice|half|total|left|remaining|cost|each)\b", sentence, re.I))
        if has_number and has_relation:
            selected.append(sentence)
        if len(selected) == 4:
            break
    if not selected:
        selected = candidates[:2]
    if not selected:
        raise ValueError(f"cannot extract evidence for target={target!r}")
    return selected


def target_messages(source: str, target: str) -> dict[str, str]:
    evidence = evidence_fragments(source, target)
    evidence_block = "\n".join(f"- {item}" for item in evidence)
    claim = f"My answer is {target}.\nFINAL_ANSWER: {target}"
    claim_evidence = (
        f"Task-specific quantities and calculations:\n{evidence_block}\n"
        f"My answer is {target}.\nFINAL_ANSWER: {target}"
    )
    claim_warrant = (
        "The requested quantity should be the value obtained by consistently applying "
        "the relationships and constraints stated in the problem. A value produced by "
        "that chain is the required result.\n"
        f"My answer is {target}.\nFINAL_ANSWER: {target}"
    )
    claim_evidence_warrant = (
        f"Task-specific quantities and calculations:\n{evidence_block}\n"
        "These calculations instantiate the stated relationships and determine the "
        f"quantity requested by the problem. Therefore my answer is {target}.\n"
        f"FINAL_ANSWER: {target}"
    )
    return {
        "claim": claim,
        "claim_evidence": claim_evidence,
        "claim_warrant": claim_warrant,
        "claim_evidence_warrant": claim_evidence_warrant,
    }


def length_matched_control(tokenizer: Any, target: str, target_tokens: int) -> str:
    suffix = f"\nMy answer is {target}.\nFINAL_ANSWER: {target}"
    pieces: list[str] = []
    index = 0
    while len(tokenizer.encode(" ".join(pieces) + suffix, add_special_tokens=False)) < target_tokens:
        pieces.append(NEUTRAL_SENTENCES[index % len(NEUTRAL_SENTENCES)])
        index += 1
    candidate = " ".join(pieces) + suffix
    ids = tokenizer.encode(candidate, add_special_tokens=False)
    if len(ids) > target_tokens:
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        prefix_budget = max(0, target_tokens - len(suffix_ids))
        neutral = tokenizer.decode(ids[:prefix_budget], skip_special_tokens=True).rstrip()
        candidate = neutral + suffix
    return candidate


def render(task: TaskInstance, row: dict[str, Any], stimuli: dict[str, dict[str, Any]]) -> tuple[ChatMessage, ...]:
    incoming = []
    for index, stimulus_id in enumerate(row["peer_stimulus_ids"]):
        item = stimuli[stimulus_id]
        incoming.append(
            MessageRecord(
                message_id=stimulus_id,
                run_id=EXPERIMENT_VERSION,
                task_id=task.task_id,
                graph_id="single-receiver",
                round_index=0,
                sender=index,
                recipients=(99,),
                raw_text=item["raw_text"],
                parsed_answer=None,
                answer_state=AnswerState(item["state"]),
            )
        )
    return build_node_messages(
        task,
        previous_output=stimuli[row["previous_stimulus_id"]]["raw_text"],
        incoming_messages=tuple(incoming),
    )


def main() -> None:
    args = parse_args()
    peer_counts = args.correct_peer_count or [1, 2]
    if peer_counts != sorted(set(peer_counts)):
        raise ValueError("correct-peer-count values must be unique and sorted")
    source = args.source_prepared_dir.resolve()
    observation = args.observation_dir.resolve()
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    observation_manifest = json.loads((observation / "manifest.json").read_text(encoding="utf-8"))
    task_ids = list(observation_manifest["selected_task_ids"])
    if len(task_ids) != 20:
        raise ValueError(f"frozen intervention requires 20 tasks, got {len(task_ids)}")
    source_tasks = {str(row["task_id"]): row for row in read_jsonl(source / "tasks.jsonl")}
    source_stimuli = {str(row["stimulus_id"]): row for row in read_jsonl(source / "stimuli.jsonl")}
    source_requests = read_jsonl(source / "requests.jsonl")
    missing = sorted(set(task_ids) - set(source_tasks))
    if missing:
        raise ValueError(f"source prepared receiver contexts lack tasks: {missing}")

    direct_by_task: dict[str, dict[str, Any]] = {}
    for pair in read_jsonl(observation / "private_source_pairs.jsonl"):
        direct_by_task.setdefault(str(pair["direct"]["task_id"]), pair["direct"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    contexts: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in source_requests:
        if row.get("mechanism") != "target_laundering" or row.get("condition") != "direct":
            continue
        task_id = str(row["task_id"])
        if task_id not in task_ids:
            continue
        count = int(row["correct_peer_count"])
        replicate = int(row["replicate"])
        if count not in peer_counts or replicate >= args.replicates:
            continue
        target_slot = int(row["target_slot"])
        peer_ids = list(row["peer_stimulus_ids"])
        del peer_ids[target_slot]
        if len(peer_ids) != count or any(source_stimuli[item]["state"] != AnswerState.CORRECT.value for item in peer_ids):
            raise ValueError(f"invalid background peers for {task_id}/{count}/{replicate}")
        contexts[(task_id, count, replicate)] = {
            "previous_stimulus_id": row["previous_stimulus_id"],
            "background_peer_ids": peer_ids,
            "target_slot": target_slot,
            "generation_seed": stable_integer(EXPERIMENT_VERSION, task_id, count, replicate, "generation"),
        }

    expected_contexts = len(task_ids) * len(peer_counts) * args.replicates
    if len(contexts) != expected_contexts:
        raise ValueError(f"need {expected_contexts} paired contexts, found {len(contexts)}")

    stimuli = dict(source_stimuli)
    requests: list[dict[str, Any]] = []
    length_rows = []
    for task_id in task_ids:
        target = str(direct_by_task[task_id]["target_answer"])
        base = target_messages(str(direct_by_task[task_id]["raw_text"]), target)
        ew_tokens = len(tokenizer.encode(base["claim_evidence_warrant"], add_special_tokens=False))
        base["length_control"] = length_matched_control(tokenizer, target, ew_tokens)
        for condition, text in base.items():
            parsed = parse_numeric_answer(text)
            if parsed != target:
                raise ValueError(f"condition {condition} failed target preservation for {task_id}: {parsed!r} != {target!r}")
            stimulus_id = stable_id(EXPERIMENT_VERSION, task_id, condition, text)
            stimuli[stimulus_id] = {
                "stimulus_id": stimulus_id,
                "task_id": task_id,
                "kind": condition,
                "state": AnswerState.TARGET_ERROR.value,
                "raw_text": text,
                "target_answer": target,
                "construction": "deterministic edit of frozen direct target rationale",
            }
        condition_ids = {
            condition: stable_id(EXPERIMENT_VERSION, task_id, condition, text)
            for condition, text in base.items()
        }
        token_counts = {key: len(tokenizer.encode(value, add_special_tokens=False)) for key, value in base.items()}
        if abs(token_counts["length_control"] - token_counts["claim_evidence_warrant"]) > args.length_tolerance:
            raise ValueError(f"length control outside tolerance for {task_id}: {token_counts}")
        length_rows.append({"task_id": task_id, **token_counts})
        for count in peer_counts:
            for replicate in range(args.replicates):
                context = contexts[(task_id, count, replicate)]
                block_id = stable_id(EXPERIMENT_VERSION, task_id, count, replicate)
                for condition in CONDITIONS:
                    peers = list(context["background_peer_ids"])
                    peers.insert(context["target_slot"], condition_ids[condition])
                    requests.append(
                        {
                            "request_id": stable_id(block_id, condition),
                            "block_id": block_id,
                            "condition": condition,
                            "task_id": task_id,
                            "replicate": replicate,
                            "correct_peer_count": count,
                            "previous_stimulus_id": context["previous_stimulus_id"],
                            "peer_stimulus_ids": peers,
                            "target_slot": context["target_slot"],
                            "generation_seed": context["generation_seed"],
                        }
                    )

    requests.sort(key=lambda row: row["request_id"])
    used_ids = {
        item for row in requests
        for item in [row["previous_stimulus_id"], *row["peer_stimulus_ids"]]
    }
    selected_stimuli = sorted((stimuli[item] for item in used_ids), key=lambda row: row["stimulus_id"])
    tasks = [source_tasks[task_id] for task_id in task_ids]
    targets = [{"task_id": task_id, "target_answer": direct_by_task[task_id]["target_answer"]} for task_id in task_ids]
    task_models = {str(row["task_id"]): TaskInstance.model_validate(row) for row in tasks}
    prompt_counts = []
    for row in requests:
        rendered = [message.model_dump() for message in render(task_models[row["task_id"]], row, {x["stimulus_id"]: x for x in selected_stimuli})]
        count = len(tokenizer.apply_chat_template(rendered, tokenize=True, add_generation_prompt=True))
        if count + SAMPLING["max_output_tokens"] > args.server_context:
            raise ValueError(f"context overflow in {row['request_id']}: input={count}")
        row["estimated_input_tokens"] = count
        prompt_counts.append(count)

    if len(requests) != 600:
        raise ValueError(f"frozen design requires 600 requests, got {len(requests)}")
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "tasks.jsonl", tasks)
    write_jsonl(args.out / "adversarial_answers.jsonl", targets)
    write_jsonl(args.out / "stimuli.jsonl", selected_stimuli)
    write_jsonl(args.out / "requests.jsonl", requests)
    write_jsonl(args.out / "length_audit.jsonl", length_rows)
    prompt_counts.sort()
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "prepared_before_receiver_outcomes",
        "prompt_version": PROMPT_VERSION,
        "source_prepared_dir": str(source),
        "observation_dir": str(observation),
        "source_experiment_version": source_manifest.get("experiment_version"),
        "observation_version": observation_manifest.get("version"),
        "task_count": len(task_ids),
        "selected_task_ids": task_ids,
        "replicates": args.replicates,
        "correct_peer_counts": peer_counts,
        "conditions": list(CONDITIONS),
        "requests": len(requests),
        "paired_blocks": len(requests) // len(CONDITIONS),
        "sampling": SAMPLING,
        "length_tolerance_tokens": args.length_tolerance,
        "tokenizer": args.tokenizer,
        "server_context": args.server_context,
        "prompt_token_audit": {
            "minimum": min(prompt_counts),
            "median": prompt_counts[len(prompt_counts) // 2],
            "p95": prompt_counts[int(0.95 * (len(prompt_counts) - 1))],
            "maximum": max(prompt_counts),
        },
        "request_fingerprint": fingerprint(requests),
        "stimuli_fingerprint": fingerprint(selected_stimuli),
        "claim_boundary": (
            "Stimuli are deterministically constructed and receiver outcomes remain unseen. "
            "E/W construct validity still requires blinded human validation before result claims."
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "requests": len(requests), "blocks": len(requests) // 5, "prompt_tokens": manifest["prompt_token_audit"]}))


if __name__ == "__main__":
    main()
