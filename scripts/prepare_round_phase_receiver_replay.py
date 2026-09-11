#!/usr/bin/env python3
"""Freeze paired receiver replays for target laundering and history maturity."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_ctou_provenance import _target_origin_builder
from analyze_node_round_adoption import read_json, read_jsonl, trace_category
from topology_mas.execution.prompts import PROMPT_VERSION, build_node_messages
from topology_mas.execution.schemas import ChatMessage
from topology_mas.execution.seeding import stable_id, stable_integer
from topology_mas.models import AnswerState, MessageRecord, TaskInstance


EXPERIMENT_VERSION = "round-phase-receiver-replay-v1"
SAMPLING = {"temperature": 0.6, "top_p": 0.9, "max_output_tokens": 768}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--correct-peer-count", type=int, action="append", default=[])
    parser.add_argument("--pool-capacity", type=int, default=40)
    parser.add_argument("--tokenizer")
    parser.add_argument("--server-context", type=int, default=16384)
    return parser.parse_args()


def normalized(text: str) -> str:
    return " ".join(text.split())


def fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def add_pool(
    pools: dict[tuple[str, str], dict[str, dict[str, Any]]],
    task_id: str,
    kind: str,
    *,
    raw_text: str,
    state: str,
    source: dict[str, Any],
    capacity: int,
) -> None:
    text = normalized(raw_text)
    if not text or "<peer_message" in text.lower():
        return
    item_id = stable_id("round-phase-stimulus", task_id, kind, text)
    bucket = pools[(task_id, kind)]
    bucket.setdefault(
        item_id,
        {
            "stimulus_id": item_id,
            "task_id": task_id,
            "kind": kind,
            "state": state,
            "raw_text": text,
            **source,
        },
    )
    if len(bucket) > capacity:
        for key in sorted(bucket)[capacity:]:
            del bucket[key]


def collect(run_roots: list[Path], capacity: int) -> tuple[dict, dict, dict, dict]:
    pools: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    tasks: dict[str, dict[str, Any]] = {}
    targets: dict[str, str] = {}
    audit = {"pairs": 0, "traces": 0, "errors": []}
    for run_root in run_roots:
        status = read_json(run_root / "orchestrator_status.json")
        for descriptor in status["strata"]:
            stratum = str(descriptor["key"])
            root = run_root / "strata" / stratum
            graph_path = root / "selected_graphs.jsonl"
            if not graph_path.exists():
                graph_path = root / "batch" / "inputs" / "graphs.jsonl"
            graphs = {str(row["graph_id"]): row for row in read_jsonl(graph_path)}
            stratum_tasks = {
                str(row["task_id"]): row
                for row in read_jsonl(root / "batch" / "inputs" / "tasks.jsonl")
            }
            for task_id, task in stratum_tasks.items():
                if task_id in tasks and tasks[task_id] != task:
                    raise ValueError(f"inconsistent task record: {task_id}")
                tasks[task_id] = task
            trace_root = root / "batch" / "traces"
            for pair in read_jsonl(root / "analysis-v1" / "paired_attacks.jsonl"):
                task_id = str(pair["task_id"])
                graph_id = str(pair["graph_id"])
                attack_node = int(pair["attack_node"])
                target = str(pair["target_answer"])
                if task_id in targets and targets[task_id] != target:
                    raise ValueError(f"inconsistent target answer: {task_id}")
                targets[task_id] = target
                path = trace_root / f"{pair['attack_run_spec_id']}.json"
                if not path.exists():
                    audit["errors"].append(f"missing trace: {path}")
                    continue
                stored = read_json(path)
                trace = stored["trace"]
                reference = str(stratum_tasks[task_id]["reference_answer"])
                turns = {
                    (int(turn["node_id"]), int(turn["round_index"])): turn
                    for turn in trace["turns"]
                }
                messages = {str(message["message_id"]): message for message in trace["messages"]}
                target_origin = _target_origin_builder(
                    turns,
                    messages,
                    reference=reference,
                    target=target,
                    attack_node=attack_node,
                )
                for (node, round_index), turn in turns.items():
                    state = trace_category(turn, reference=reference, target=target)
                    source = {
                        "source_graph_id": graph_id,
                        "source_n": int(graphs[graph_id]["node_count"]),
                        "source_attack_node": attack_node,
                        "source_node": node,
                        "source_round": round_index,
                        "source_run_spec_id": str(pair["attack_run_spec_id"]),
                    }
                    text = str(turn.get("raw_output", ""))
                    if node == attack_node and state == "target":
                        add_pool(
                            pools, task_id, "direct_target", raw_text=text,
                            state=AnswerState.TARGET_ERROR.value, source=source, capacity=capacity,
                        )
                    elif node != attack_node and state == "target" and target_origin(node, round_index) == "relayed":
                        add_pool(
                            pools, task_id, "relayed_target", raw_text=text,
                            state=AnswerState.TARGET_ERROR.value, source=source, capacity=capacity,
                        )
                    if node != attack_node and state == "correct":
                        add_pool(
                            pools, task_id, "correct_peer", raw_text=text,
                            state=AnswerState.CORRECT.value, source=source, capacity=capacity,
                        )
                        kind = "initial_correct" if round_index == 0 else "deliberated_correct"
                        add_pool(
                            pools, task_id, kind, raw_text=text,
                            state=AnswerState.CORRECT.value, source=source, capacity=capacity,
                        )
                audit["pairs"] += 1
                audit["traces"] += 1
    audit["errors"] = audit["errors"][:100]
    return pools, tasks, targets, audit


def shuffled_ids(
    pools: dict[tuple[str, str], dict[str, dict[str, Any]]],
    task_id: str,
    kind: str,
    *seed_parts: object,
) -> list[str]:
    ids = sorted(pools[(task_id, kind)])
    random.Random(stable_integer(EXPERIMENT_VERSION, task_id, kind, *seed_parts)).shuffle(ids)
    return ids


def insert_at(values: list[str], item: str, slot: int) -> list[str]:
    result = list(values)
    result.insert(slot, item)
    return result


def choose_text_distinct(
    candidate_ids: list[str],
    pools: dict[tuple[str, str], dict[str, dict[str, Any]]],
    task_id: str,
    kind: str,
    excluded_texts: set[str],
) -> str:
    bucket = pools[(task_id, kind)]
    for candidate_id in candidate_ids:
        if bucket[candidate_id]["raw_text"] not in excluded_texts:
            return candidate_id
    raise ValueError(f"insufficient text-distinct {kind} stimuli for {task_id}")


def build_plan(
    pools: dict[tuple[str, str], dict[str, dict[str, Any]]],
    task_ids: list[str],
    replicates: int,
    peer_counts: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for correct_count in peer_counts:
            for replicate in range(replicates):
                initial = shuffled_ids(pools, task_id, "initial_correct", correct_count, replicate)
                deliberated = shuffled_ids(pools, task_id, "deliberated_correct", correct_count, replicate)
                correct = shuffled_ids(pools, task_id, "correct_peer", correct_count, replicate)
                relayed = shuffled_ids(pools, task_id, "relayed_target", correct_count, replicate)
                direct = shuffled_ids(pools, task_id, "direct_target", correct_count, replicate)
                previous_initial = initial[0]
                initial_text = pools[(task_id, "initial_correct")][previous_initial]["raw_text"]
                previous_deliberated = choose_text_distinct(
                    deliberated, pools, task_id, "deliberated_correct", {initial_text}
                )
                deliberated_text = pools[(task_id, "deliberated_correct")][previous_deliberated]["raw_text"]
                direct_id = direct[0]
                direct_text = pools[(task_id, "direct_target")][direct_id]["raw_text"]
                relayed_id = choose_text_distinct(
                    relayed, pools, task_id, "relayed_target", {direct_text}
                )
                excluded_texts = {initial_text, deliberated_text, direct_text,
                                  pools[(task_id, "relayed_target")][relayed_id]["raw_text"]}
                background = [
                    item for item in correct
                    if pools[(task_id, "correct_peer")][item]["raw_text"] not in excluded_texts
                ][:correct_count]
                if len(background) != correct_count:
                    raise ValueError(f"insufficient distinct correct peers for {task_id}")
                slot = stable_integer(EXPERIMENT_VERSION, task_id, correct_count, replicate, "slot") % (correct_count + 1)
                pair_seed = stable_integer(EXPERIMENT_VERSION, task_id, correct_count, replicate, "generation")
                pair_id = stable_id("laundering-pair", task_id, correct_count, replicate)
                for condition, target_id in (("direct", direct_id), ("relayed", relayed_id)):
                    rows.append({
                        "request_id": stable_id(pair_id, condition),
                        "pair_id": pair_id,
                        "mechanism": "target_laundering",
                        "condition": condition,
                        "task_id": task_id,
                        "replicate": replicate,
                        "correct_peer_count": correct_count,
                        "previous_stimulus_id": previous_initial,
                        "peer_stimulus_ids": insert_at(background, target_id, slot),
                        "target_slot": slot,
                        "generation_seed": pair_seed,
                    })
                pair_id = stable_id("history-pair", task_id, correct_count, replicate)
                fixed_peers = insert_at(background, relayed_id, slot)
                for condition, previous_id in (
                    ("initial", previous_initial),
                    ("deliberated", previous_deliberated),
                ):
                    rows.append({
                        "request_id": stable_id(pair_id, condition),
                        "pair_id": pair_id,
                        "mechanism": "history_maturity",
                        "condition": condition,
                        "task_id": task_id,
                        "replicate": replicate,
                        "correct_peer_count": correct_count,
                        "previous_stimulus_id": previous_id,
                        "peer_stimulus_ids": fixed_peers,
                        "target_slot": slot,
                        "generation_seed": pair_seed,
                    })
    return sorted(rows, key=lambda row: row["request_id"])


def render(
    task: TaskInstance,
    row: dict[str, Any],
    stimuli: dict[str, dict[str, Any]],
) -> tuple[ChatMessage, ...]:
    messages = []
    for index, stimulus_id in enumerate(row["peer_stimulus_ids"]):
        item = stimuli[stimulus_id]
        messages.append(
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
        incoming_messages=tuple(messages),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    peer_counts = args.correct_peer_count or [1, 2]
    pools, tasks, targets, collection_audit = collect(args.run_root, args.pool_capacity)
    requirements = {
        "direct_target": 1,
        "relayed_target": args.replicates,
        "initial_correct": args.replicates,
        "deliberated_correct": args.replicates,
        "correct_peer": max(peer_counts) + 2,
    }
    support = {
        task_id: {kind: len(pools[(task_id, kind)]) for kind in requirements}
        for task_id in sorted(tasks)
    }
    eligible = [
        task_id for task_id in sorted(tasks)
        if all(support[task_id][kind] >= minimum for kind, minimum in requirements.items())
    ]
    if len(eligible) < args.task_count:
        raise ValueError(f"only {len(eligible)} eligible tasks; need {args.task_count}")
    selected_tasks = eligible[: args.task_count]
    plan = build_plan(pools, selected_tasks, args.replicates, peer_counts)
    stimulus_ids = {
        stimulus_id for row in plan
        for stimulus_id in [row["previous_stimulus_id"], *row["peer_stimulus_ids"]]
    }
    stimuli = {
        stimulus_id: item
        for bucket in pools.values()
        for stimulus_id, item in bucket.items()
        if stimulus_id in stimulus_ids
    }
    token_audit = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        counts = []
        task_models = {task_id: TaskInstance.model_validate(tasks[task_id]) for task_id in selected_tasks}
        for row in plan:
            rendered = [message.model_dump() for message in render(task_models[row["task_id"]], row, stimuli)]
            count = len(tokenizer.apply_chat_template(rendered, tokenize=True, add_generation_prompt=True))
            if count + SAMPLING["max_output_tokens"] > args.server_context:
                raise ValueError(f"context overflow in prepared request {row['request_id']}: {count}")
            row["estimated_input_tokens"] = count
            counts.append(count)
        counts.sort()
        token_audit = {
            "tokenizer": args.tokenizer,
            "minimum": counts[0],
            "median": counts[len(counts) // 2],
            "p95": counts[int(0.95 * (len(counts) - 1))],
            "maximum": counts[-1],
            "server_context": args.server_context,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "tasks.jsonl", [tasks[task_id] for task_id in selected_tasks])
    write_jsonl(
        args.out / "adversarial_answers.jsonl",
        [{"task_id": task_id, "target_answer": targets[task_id]} for task_id in selected_tasks],
    )
    write_jsonl(args.out / "stimuli.jsonl", sorted(stimuli.values(), key=lambda row: row["stimulus_id"]))
    write_jsonl(args.out / "requests.jsonl", plan)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "prepared_before_outcomes",
        "prompt_version": PROMPT_VERSION,
        "run_roots": [str(path) for path in args.run_root],
        "task_selection": "first task IDs meeting preregistered stimulus-support requirements",
        "eligible_task_count": len(eligible),
        "selected_task_ids": selected_tasks,
        "task_count": len(selected_tasks),
        "replicates": args.replicates,
        "correct_peer_counts": peer_counts,
        "requests": len(plan),
        "pairs": len(plan) // 2,
        "sampling": SAMPLING,
        "server_context": args.server_context,
        "support_requirements": requirements,
        "support_by_task": support,
        "collection_audit": collection_audit,
        "token_audit": token_audit,
        "stimuli_fingerprint": fingerprint(sorted(stimuli.values(), key=lambda row: row["stimulus_id"])),
        "request_fingerprint": fingerprint(plan),
        "information_boundary": (
            "anonymous homogeneous receiver; the prompt never exposes attacker identity or "
            "the direct/relayed provenance label"
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(args.out), "tasks": len(selected_tasks), "requests": len(plan), "token_audit": token_audit}))


if __name__ == "__main__":
    main()
