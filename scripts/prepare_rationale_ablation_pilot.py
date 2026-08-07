"""Prepare a deterministic matched full-rationale versus answer-only pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from topology_mas.execution.answers import normalize_numeric_answer, parse_numeric_answer

PREPARATION_VERSION = "rationale-ablation-preparation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hashed_task_order(task: dict[str, Any]) -> tuple[str, str]:
    task_id = str(task["task_id"])
    return hashlib.sha256(task_id.encode()).hexdigest(), task_id


def build_answer_only_variant(original: dict[str, Any]) -> dict[str, Any]:
    """Remove the rationale while preserving the exact normalized target answer."""

    target = str(original["target_answer"]).strip()
    rationale = f"#### {target}"
    parsed = parse_numeric_answer(rationale)
    if parsed != normalize_numeric_answer(target):
        raise ValueError(f"answer-only message does not parse to target: {target!r}")
    metadata = dict(original.get("metadata") or {})
    metadata.update(
        {
            "message_variant": "answer_only",
            "source_rationale_sha256": hashlib.sha256(
                str(original["rationale"]).encode()
            ).hexdigest(),
            "preparation_version": PREPARATION_VERSION,
        }
    )
    return {**original, "rationale": rationale, "metadata": metadata}


def main() -> None:
    args = parse_args()
    source = args.source_run_root.resolve()
    output = args.output_dir.resolve()
    if args.task_count < 1:
        raise ValueError("task count must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    status_path = source / "orchestrator_status.json"
    status = read_json(status_path)
    if status.get("status") != "completed":
        raise RuntimeError("source pilot must be completed")
    strata = status.get("strata", [])
    if not strata:
        raise ValueError("source pilot has no strata")
    first_key = str(strata[0]["key"])
    first_inputs = source / "strata" / first_key / "batch" / "inputs"
    tasks_path = first_inputs / "tasks.jsonl"
    adversarial_path = first_inputs / "adversarial_answers.jsonl"
    tasks = read_jsonl(tasks_path)
    adversarial = read_jsonl(adversarial_path)
    if args.task_count > len(tasks):
        raise ValueError("task count exceeds source tasks")
    selected_tasks = sorted(tasks, key=hashed_task_order)[: args.task_count]
    selected_ids = {str(task["task_id"]) for task in selected_tasks}
    answer_by_task = {str(answer["task_id"]): answer for answer in adversarial}
    if selected_ids - answer_by_task.keys():
        raise ValueError("selected tasks are missing adversarial answers")

    full_answers: list[dict[str, Any]] = []
    answer_only: list[dict[str, Any]] = []
    for task in selected_tasks:
        task_id = str(task["task_id"])
        original = answer_by_task[task_id]
        full_answers.append(original)
        answer_only.append(build_answer_only_variant(original))

    write_jsonl(output / "tasks.jsonl", selected_tasks)
    write_jsonl(output / "adversarial_full_rationale.jsonl", full_answers)
    write_jsonl(output / "adversarial_answer_only.jsonl", answer_only)

    selected_graphs: list[dict[str, Any]] = []
    for descriptor in strata:
        key = str(descriptor["key"])
        source_graphs = read_jsonl(source / "strata" / key / "selected_graphs.jsonl")
        selected = min(source_graphs, key=lambda graph: str(graph["graph_id"]))
        graph_path = output / "strata" / key / "graphs.jsonl"
        write_jsonl(graph_path, [selected])
        selected_graphs.append(
            {
                "stratum": key,
                "graph_id": selected["graph_id"],
                "node_count": selected["node_count"],
                "edge_count": len(selected["edges"]),
                "max_rounds": selected["max_rounds"],
                "path": str(graph_path),
                "sha256": sha256_file(graph_path),
            }
        )

    manifest = {
        "preparation_version": PREPARATION_VERSION,
        "source_run_root": str(source),
        "source_status_sha256": sha256_file(status_path),
        "selection": {
            "tasks": "first task_count by (sha256(task_id), task_id)",
            "graphs": "lexicographically smallest graph_id in each source stratum",
            "used_outcomes": False,
        },
        "task_count": len(selected_tasks),
        "task_ids": [task["task_id"] for task in selected_tasks],
        "tasks_sha256": sha256_file(output / "tasks.jsonl"),
        "full_rationale_sha256": sha256_file(
            output / "adversarial_full_rationale.jsonl"
        ),
        "answer_only_sha256": sha256_file(output / "adversarial_answer_only.jsonl"),
        "selected_graphs": selected_graphs,
        "model": status["model"],
        "expected_returned_model": status["expected_returned_model"],
        "temperature": status["temperature"],
        "max_output_tokens": status["max_output_tokens"],
        "experiment_seed": 0,
        "assignment_seed": 0,
        "round_zero_dir": str(source / "round-zero-r8-temp0p3" / "cache"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
