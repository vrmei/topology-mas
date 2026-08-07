"""Run a resumable, remotely hosted multi-stratum topology pilot.

This process is intended to be launched with ``nohup`` on the GPU host. It selects a
fixed prefix of graphs from every prepared (n, m) stratum, creates one shared Round-zero
cache large enough for the largest graph, executes each stratum sequentially, analyzes
completed batches, and records wall-clock/GPU/disk telemetry while it runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.models import GraphSpec
from topology_mas.topology.graph_ops import build_causal_schedule
from topology_mas.topology.io import read_graphs_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {
        "available": True,
        "gpu_utilization_percent": float(values[0]),
        "memory_utilization_percent": float(values[1]),
        "memory_used_mib": float(values[2]),
        "memory_total_mib": float(values[3]),
        "power_draw_watts": float(values[4]),
    }


def progress_count(kind: str, progress_root: Path) -> int:
    if kind == "round_zero":
        records = progress_root / "records"
        return sum(1 for _ in records.rglob("replica_*.json")) if records.exists() else 0
    traces = progress_root / "traces"
    return sum(1 for _ in traces.glob("*.json")) if traces.exists() else 0


def run_stage(
    *,
    stage_name: str,
    command: list[str],
    project_root: Path,
    run_root: Path,
    progress_kind: str,
    progress_root: Path,
    expected_items: int,
    poll_seconds: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    stage_dir = run_root / "stage-logs" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    started_wall = time.time()
    started_at = utc_now()
    stage = {
        "stage_name": stage_name,
        "status": "running",
        "started_at": started_at,
        "command": command,
        "expected_items": expected_items,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    manifest["current_stage"] = stage
    manifest["status"] = "running"
    atomic_json(run_root / "orchestrator_status.json", manifest)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    with (
        stdout_path.open("a", encoding="utf-8") as stdout_handle,
        stderr_path.open("a", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        while True:
            return_code = process.poll()
            completed = progress_count(progress_kind, progress_root)
            disk = shutil.disk_usage(project_root)
            telemetry = {
                "timestamp": utc_now(),
                "stage_name": stage_name,
                "stage_pid": process.pid,
                "elapsed_seconds": time.time() - started_wall,
                "completed_items": completed,
                "expected_items": expected_items,
                "gpu": gpu_snapshot(),
                "disk_free_bytes": disk.free,
                "return_code": return_code,
            }
            append_jsonl(run_root / "telemetry.jsonl", telemetry)
            stage.update(
                {
                    "stage_pid": process.pid,
                    "elapsed_seconds": telemetry["elapsed_seconds"],
                    "completed_items": completed,
                    "last_telemetry_at": telemetry["timestamp"],
                }
            )
            manifest["current_stage"] = stage
            atomic_json(run_root / "orchestrator_status.json", manifest)
            if return_code is not None:
                break
            time.sleep(poll_seconds)

    stage.update(
        {
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "ended_at": utc_now(),
            "elapsed_seconds": time.time() - started_wall,
            "completed_items": progress_count(progress_kind, progress_root),
        }
    )
    manifest.setdefault("stages", []).append(stage)
    manifest["current_stage"] = None
    if return_code != 0:
        manifest["status"] = "failed"
        manifest["failed_stage"] = stage_name
        atomic_json(run_root / "orchestrator_status.json", manifest)
        raise RuntimeError(f"stage {stage_name} failed with return code {return_code}")
    atomic_json(run_root / "orchestrator_status.json", manifest)
    return stage


def parse_stratum(path: Path) -> tuple[int, int]:
    parts = path.parent.name.split("_")
    return int(parts[0].removeprefix("n")), int(parts[1].removeprefix("m"))


def inference_calls_per_task(graph: GraphSpec) -> int:
    schedule = build_causal_schedule(graph)
    runtime_rounds = schedule.active_nodes_by_round[1:]
    clean_calls = sum(len(nodes) for nodes in runtime_rounds)
    attack_calls = sum(
        clean_calls - sum(node_id in nodes for nodes in runtime_rounds)
        for node_id in range(graph.node_count)
        if node_id != graph.readout_node
    )
    return clean_calls + attack_calls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--adversarial-answers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-returned-model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--graphs-per-stratum", type=int, default=5)
    parser.add_argument("--round-zero-replicas", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--state-replay-model-fingerprint")
    parser.add_argument("--state-replay-namespace")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    tasks_path = (project_root / args.tasks).resolve()
    graph_root = (project_root / args.graph_root).resolve()
    adversarial_path = (project_root / args.adversarial_answers).resolve()
    run_root = (project_root / args.output_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks_jsonl(tasks_path)
    if args.graphs_per_stratum < 1 or args.round_zero_replicas < 2:
        raise ValueError("invalid graph or replica count")
    replay_enabled = args.state_replay_model_fingerprint is not None
    if replay_enabled != (args.state_replay_namespace is not None):
        raise ValueError(
            "state replay model fingerprint and namespace must be provided together"
        )

    graph_files = sorted(graph_root.glob("*/graphs.jsonl"), key=lambda path: parse_stratum(path))
    if not graph_files:
        raise ValueError("no graph strata found")
    strata: list[dict[str, Any]] = []
    max_node_count = 0
    total_expected_traces = 0
    total_expected_calls = 0
    for graph_file in graph_files:
        n, m = parse_stratum(graph_file)
        available = read_graphs_jsonl(graph_file)
        selected = available[: args.graphs_per_stratum]
        if not selected:
            raise ValueError(f"empty graph stratum: {graph_file}")
        max_node_count = max(max_node_count, n)
        expected_traces = len(tasks) * len(selected) * n
        expected_calls = len(tasks) * sum(inference_calls_per_task(g) for g in selected)
        total_expected_traces += expected_traces
        total_expected_calls += expected_calls
        strata.append(
            {
                "key": f"n{n}_m{m}",
                "n": n,
                "m": m,
                "source_path": str(graph_file),
                "source_sha256": sha256_file(graph_file),
                "available_graphs": len(available),
                "selected_graphs": len(selected),
                "selected_graph_ids": [graph.graph_id for graph in selected],
                "expected_traces": expected_traces,
                "expected_local_inference_calls": expected_calls,
                "selection_note": (
                    "all available graphs selected; this stratum has fewer than requested"
                    if len(selected) < args.graphs_per_stratum
                    else "fixed prefix selected before execution"
                ),
            }
        )

    if args.round_zero_replicas < max_node_count:
        raise ValueError("Round-zero replica count is smaller than the largest graph")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "code_commit": git_commit(project_root) or os.getenv("TOPOLOGY_MAS_CODE_COMMIT"),
        "python": sys.version,
        "tasks": str(tasks_path),
        "task_count": len(tasks),
        "task_sha256": sha256_file(tasks_path),
        "adversarial_answers": str(adversarial_path),
        "adversarial_answers_sha256": sha256_file(adversarial_path),
        "model": args.model,
        "expected_returned_model": args.expected_returned_model,
        "base_url": args.base_url,
        "graphs_per_stratum": args.graphs_per_stratum,
        "round_zero_replicas": args.round_zero_replicas,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_workers": args.max_workers,
        "poll_seconds": args.poll_seconds,
        "state_transition_policy": (
            "state-consistent-replay-v1" if replay_enabled else "independent-resampling"
        ),
        "state_replay_model_fingerprint": args.state_replay_model_fingerprint,
        "state_replay_namespace": args.state_replay_namespace,
        "strata": strata,
        "expected_total_traces": total_expected_traces,
        "expected_total_local_inference_calls": total_expected_calls,
        "stages": [],
        "current_stage": None,
    }
    atomic_json(run_root / "orchestrator_status.json", manifest)

    round_zero_dir = run_root / "round-zero-r8-temp0p3" / "cache"
    run_stage(
        stage_name="round_zero_r8",
        command=[
            sys.executable,
            "-m",
            "topology_mas.execution.round_zero_cli",
            "--tasks",
            str(tasks_path),
            "--output-dir",
            str(round_zero_dir),
            "--replica-count",
            str(args.round_zero_replicas),
            "--seeds",
            "0",
            "--model",
            args.model,
            "--expected-returned-model",
            args.expected_returned_model,
            "--base-url",
            args.base_url,
            "--no-auth",
            "--temperature",
            str(args.temperature),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--timeout-seconds",
            "180",
            "--max-attempts",
            "3",
            "--max-workers",
            str(args.max_workers),
        ],
        project_root=project_root,
        run_root=run_root,
        progress_kind="round_zero",
        progress_root=round_zero_dir,
        expected_items=len(tasks) * args.round_zero_replicas,
        poll_seconds=args.poll_seconds,
        manifest=manifest,
    )

    for stratum, graph_file in zip(strata, graph_files, strict=True):
        key = stratum["key"]
        selected_graphs = read_graphs_jsonl(graph_file)[: args.graphs_per_stratum]
        stratum_root = run_root / "strata" / key
        selected_path = stratum_root / "selected_graphs.jsonl"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(
            "".join(graph.model_dump_json() + "\n" for graph in selected_graphs),
            encoding="utf-8",
        )
        batch_root = stratum_root / "batch"
        batch_command = [
            sys.executable,
            "-m",
            "topology_mas.execution.batch_cli",
            "--tasks",
            str(tasks_path),
            "--graphs",
            str(selected_path),
            "--round-zero-dir",
            str(round_zero_dir),
            "--adversarial-answers",
            str(adversarial_path),
            "--output-dir",
            str(batch_root),
            "--experiment-seeds",
            "0",
            "--assignment-seeds",
            "0",
            "--model",
            args.model,
            "--expected-returned-model",
            args.expected_returned_model,
            "--base-url",
            args.base_url,
            "--no-auth",
            "--temperature",
            str(args.temperature),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--timeout-seconds",
            "180",
            "--max-attempts",
            "3",
            "--max-workers",
            str(args.max_workers),
        ]
        if replay_enabled:
            batch_command.extend(
                [
                    "--state-replay-cache-dir",
                    str(run_root / "state-replay-v1"),
                    "--state-replay-model-fingerprint",
                    args.state_replay_model_fingerprint,
                    "--state-replay-namespace",
                    args.state_replay_namespace,
                ]
            )
        run_stage(
            stage_name=f"batch_{key}",
            command=batch_command,
            project_root=project_root,
            run_root=run_root,
            progress_kind="batch",
            progress_root=batch_root,
            expected_items=int(stratum["expected_traces"]),
            poll_seconds=args.poll_seconds,
            manifest=manifest,
        )
        analysis_root = stratum_root / "analysis-v1"
        run_stage(
            stage_name=f"analysis_{key}",
            command=[
                sys.executable,
                "-m",
                "topology_mas.analysis.cli",
                "--batch-dir",
                str(batch_root),
                "--output-dir",
                str(analysis_root),
            ],
            project_root=project_root,
            run_root=run_root,
            progress_kind="batch",
            progress_root=batch_root,
            expected_items=int(stratum["expected_traces"]),
            poll_seconds=args.poll_seconds,
            manifest=manifest,
        )
        stratum["batch_summary"] = json.loads(
            (batch_root / "summary.json").read_text(encoding="utf-8")
        )
        stratum["analysis_manifest"] = json.loads(
            (analysis_root / "manifest.json").read_text(encoding="utf-8")
        )
        atomic_json(run_root / "orchestrator_status.json", manifest)

    manifest["status"] = "completed"
    manifest["ended_at"] = utc_now()
    manifest["current_stage"] = None
    atomic_json(run_root / "orchestrator_status.json", manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"REMOTE SCALE PILOT FAILED: {exc}", file=sys.stderr, flush=True)
        raise
