"""Run or dry-run the prepared answer-only rationale ablation on the pinned local model."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNNER_VERSION = "rationale-ablation-runner-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def seed_pinned_clean_traces(
    *,
    source_batch: Path,
    destination_batch: Path,
    task_ids: set[str],
    graph_id: str,
) -> int:
    """Reuse clean traces because the intervention changes only attacker messages."""

    plan = read_jsonl(source_batch / "plan.jsonl")
    selected = [
        spec
        for spec in plan
        if str(spec["task_id"]) in task_ids
        and str(spec["graph_id"]) == graph_id
        and str(spec["condition"]) == "clean"
    ]
    if len(selected) != len(task_ids):
        raise RuntimeError(
            f"expected {len(task_ids)} pinned clean traces for {graph_id}, found {len(selected)}"
        )
    destination = destination_batch / "traces"
    destination.mkdir(parents=True, exist_ok=True)
    for spec in selected:
        name = f"{spec['run_spec_id']}.json"
        source_path = source_batch / "traces" / name
        destination_path = destination / name
        if not source_path.exists():
            raise FileNotFoundError(f"pinned clean trace is missing: {source_path}")
        if destination_path.exists():
            if source_path.read_bytes() != destination_path.read_bytes():
                raise RuntimeError(
                    f"existing clean trace differs from pinned source: {destination_path}"
                )
            continue
        temporary = destination_path.with_suffix(".json.tmp")
        shutil.copy2(source_path, temporary)
        temporary.replace(destination_path)
    return len(selected)


def command_for_stratum(
    *,
    prepared: Path,
    output: Path,
    descriptor: dict[str, Any],
    manifest: dict[str, Any],
    base_url: str,
    max_workers: int,
) -> tuple[list[str], list[str]]:
    key = str(descriptor["stratum"])
    batch_dir = output / "strata" / key / "batch"
    analysis_dir = output / "strata" / key / "analysis-v1"
    batch = [
        sys.executable,
        "-m",
        "topology_mas.execution.batch_cli",
        "--tasks",
        str(prepared / "tasks.jsonl"),
        "--graphs",
        str(prepared / "strata" / key / "graphs.jsonl"),
        "--round-zero-dir",
        str(manifest["round_zero_dir"]),
        "--adversarial-answers",
        str(prepared / "adversarial_answer_only.jsonl"),
        "--output-dir",
        str(batch_dir),
        "--experiment-seeds",
        str(manifest["experiment_seed"]),
        "--assignment-seeds",
        str(manifest["assignment_seed"]),
        "--model",
        str(manifest["model"]),
        "--expected-returned-model",
        str(manifest["expected_returned_model"]),
        "--base-url",
        base_url,
        "--no-auth",
        "--temperature",
        str(manifest["temperature"]),
        "--max-output-tokens",
        str(manifest["max_output_tokens"]),
        "--timeout-seconds",
        "180",
        "--max-attempts",
        "3",
        "--max-workers",
        str(max_workers),
    ]
    analysis = [
        sys.executable,
        "-m",
        "topology_mas.analysis.cli",
        "--batch-dir",
        str(batch_dir),
        "--output-dir",
        str(analysis_dir),
    ]
    return batch, analysis


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    prepared = args.prepared_dir.resolve()
    output = args.output_dir.resolve()
    if args.max_workers < 1:
        raise ValueError("max workers must be positive")
    manifest = read_json(prepared / "manifest.json")
    task_ids = {str(task_id) for task_id in manifest["task_ids"]}
    descriptors = list(manifest["selected_graphs"])
    commands = [
        command_for_stratum(
            prepared=prepared,
            output=output,
            descriptor=descriptor,
            manifest=manifest,
            base_url=args.base_url,
            max_workers=args.max_workers,
        )
        for descriptor in descriptors
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "runner_version": RUNNER_VERSION,
                    "prepared_dir": str(prepared),
                    "output_dir": str(output),
                    "strata": len(commands),
                    "commands": [
                        {"batch": batch, "analysis": analysis}
                        for batch, analysis in commands
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    status_path = output / "runner_status.json"
    if status_path.exists() and read_json(status_path).get("status") == "completed":
        print(status_path.read_text(encoding="utf-8"))
        return
    status: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "prepared_dir": str(prepared),
        "source_run_root": manifest["source_run_root"],
        "model": manifest["model"],
        "expected_returned_model": manifest["expected_returned_model"],
        "base_url": args.base_url,
        "message_variant": "answer_only",
        "task_count": manifest["task_count"],
        "strata": [],
    }
    atomic_json(status_path, status)
    try:
        for descriptor, (batch, analysis) in zip(descriptors, commands, strict=True):
            key = str(descriptor["stratum"])
            batch_dir = output / "strata" / key / "batch"
            source_batch = (
                Path(manifest["source_run_root"]) / "strata" / key / "batch"
            )
            seeded_clean_traces = seed_pinned_clean_traces(
                source_batch=source_batch,
                destination_batch=batch_dir,
                task_ids=task_ids,
                graph_id=str(descriptor["graph_id"]),
            )
            subprocess.run(batch, cwd=project, check=True)
            subprocess.run(analysis, cwd=project, check=True)
            stratum_root = output / "strata" / key
            summary = read_json(stratum_root / "batch" / "summary.json")
            analysis_manifest = read_json(stratum_root / "analysis-v1" / "manifest.json")
            status["strata"].append(
                {
                    "key": key,
                    "graph_id": descriptor["graph_id"],
                    "node_count": descriptor["node_count"],
                    "seeded_clean_traces": seeded_clean_traces,
                    "batch_summary": summary,
                    "analysis_manifest": analysis_manifest,
                }
            )
            atomic_json(status_path, status)
    except Exception as exc:
        status["status"] = "failed"
        status["ended_at"] = utc_now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(status_path, status)
        raise
    status["status"] = "completed"
    status["ended_at"] = utc_now()
    atomic_json(status_path, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
