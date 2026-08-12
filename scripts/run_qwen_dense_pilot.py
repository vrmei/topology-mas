"""Run the prepared no-reuse Qwen dense-m robustness pilot with telemetry."""

from __future__ import annotations

import argparse
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


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return result.stdout.strip() or None


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return {"available": False, "error": result.stderr.strip()}
    fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {
        "available": True,
        "utilization_percent": float(fields[0]),
        "memory_used_mib": float(fields[1]),
        "memory_total_mib": float(fields[2]),
        "power_draw_watts": float(fields[3]),
    }


def trace_count(batch_root: Path) -> int:
    traces = batch_root / "traces"
    return sum(1 for _ in traces.glob("*.json")) if traces.exists() else 0


def graph_calls_per_task(graph: GraphSpec) -> int:
    rounds = build_causal_schedule(graph).active_nodes_by_round
    clean = sum(len(nodes) for nodes in rounds)
    attacks = sum(
        sum(node_id != attacker for node_id in nodes)
        for attacker in range(graph.node_count)
        if attacker != graph.readout_node
        for nodes in rounds
    )
    return clean + attacks


def run_stage(
    *,
    name: str,
    command: list[str],
    project_root: Path,
    run_root: Path,
    batch_root: Path,
    expected: int,
    poll_seconds: int,
    status: dict[str, Any],
) -> None:
    log_root = run_root / "stage-logs" / name
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stage = {
        "name": name,
        "status": "running",
        "command": command,
        "started_at": utc_now(),
        "expected_traces": expected,
    }
    status["current_stage"] = stage
    atomic_json(run_root / "orchestrator_status.json", status)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    with (log_root / "stdout.log").open("a", encoding="utf-8") as stdout, (
        log_root / "stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
        )
        while process.poll() is None:
            completed = trace_count(batch_root)
            sample = {
                "timestamp": utc_now(),
                "stage": name,
                "pid": process.pid,
                "elapsed_seconds": time.time() - started,
                "completed_traces": completed,
                "expected_traces": expected,
                "gpu": gpu_snapshot(),
                "system_disk_free_bytes": shutil.disk_usage("/").free,
                "data_disk_free_bytes": shutil.disk_usage(run_root).free,
            }
            append_jsonl(run_root / "telemetry.jsonl", sample)
            stage.update(sample)
            atomic_json(run_root / "orchestrator_status.json", status)
            time.sleep(poll_seconds)
        return_code = process.wait()
    stage.update(
        {
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "ended_at": utc_now(),
            "elapsed_seconds": time.time() - started,
            "completed_traces": trace_count(batch_root),
        }
    )
    status.setdefault("stages", []).append(stage)
    status["current_stage"] = None
    atomic_json(run_root / "orchestrator_status.json", status)
    if return_code:
        status["status"] = "failed"
        atomic_json(run_root / "orchestrator_status.json", status)
        raise RuntimeError(f"stage {name} failed with return code {return_code}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=Path.cwd())
    result.add_argument("--prepared-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    result.add_argument("--expected-returned-model", default="Qwen/Qwen3-4B-Instruct-2507")
    result.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    result.add_argument("--temperature", type=float, default=0.7)
    result.add_argument("--top-p", type=float, default=0.8)
    result.add_argument("--top-k", type=int)
    result.add_argument("--min-p", type=float)
    result.add_argument(
        "--sampling-source",
        default="explicit experiment configuration",
    )
    result.add_argument("--max-output-tokens", type=int, default=16_384)
    result.add_argument("--max-workers", type=int, default=96)
    result.add_argument("--poll-seconds", type=int, default=30)
    result.add_argument("--only-strata", nargs="*")
    return result


def main() -> None:
    args = parser().parse_args()
    project_root = args.project_root.resolve()
    prepared = args.prepared_root.resolve()
    run_root = args.output_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((prepared / "preparation_manifest.json").read_text())
    tasks = read_tasks_jsonl(prepared / "inputs" / "tasks50-fixed.jsonl")
    strata = preparation["strata"]
    if args.only_strata:
        selected = set(args.only_strata)
        strata = [stratum for stratum in strata if stratum["key"] in selected]
        missing = selected - {stratum["key"] for stratum in strata}
        if missing:
            raise ValueError(f"unknown strata: {sorted(missing)}")

    total_traces = 0
    total_calls = 0
    for stratum in strata:
        graphs = read_graphs_jsonl(stratum["graphs_path"])
        expected_traces = len(tasks) * len(graphs) * stratum["n"]
        expected_calls = len(tasks) * sum(graph_calls_per_task(graph) for graph in graphs)
        stratum["expected_traces"] = expected_traces
        stratum["expected_backend_calls"] = expected_calls
        total_traces += expected_traces
        total_calls += expected_calls

    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "code_commit": git_commit(project_root) or os.getenv("TOPOLOGY_MAS_CODE_COMMIT"),
        "prepared_root": str(prepared),
        "task_ids": [task.task_id for task in tasks],
        "task_count": len(tasks),
        "model": args.model,
        "expected_returned_model": args.expected_returned_model,
        "base_url": args.base_url,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_output_tokens": args.max_output_tokens,
            "source": args.sampling_source,
        },
        "reuse_policy": {
            "round_zero": "independent_per_graph_condition_attacker_run",
            "post_round": "independent_per_run",
            "state_replay": False,
            "within_round_broadcast": "one generation copied to all out-neighbors",
        },
        "horizon_policy": "fixed_t3_with_causal_active-node-pruning",
        "strata": strata,
        "expected_total_traces": total_traces,
        "expected_total_backend_calls": total_calls,
        "stages": [],
        "current_stage": None,
    }
    atomic_json(run_root / "orchestrator_status.json", status)

    for stratum in strata:
        key = stratum["key"]
        stratum_root = run_root / "strata" / key
        batch_root = stratum_root / "batch"
        command = [
            sys.executable,
            "-m",
            "topology_mas.execution.batch_cli",
            "--tasks",
            str(prepared / "inputs" / "tasks50-fixed.jsonl"),
            "--graphs",
            str(stratum["graphs_path"]),
            "--independent-round-zero",
            "--adversarial-answers",
            str(prepared / "inputs" / "adversarial50-fixed.jsonl"),
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
            "--top-p",
            str(args.top_p),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--timeout-seconds",
            "600",
            "--max-attempts",
            "3",
            "--max-workers",
            str(args.max_workers),
            "--horizon-policy",
            "fixed",
        ]
        if args.top_k is not None:
            command.extend(["--top-k", str(args.top_k)])
        if args.min_p is not None:
            command.extend(["--min-p", str(args.min_p)])
        run_stage(
            name=f"batch_{key}",
            command=command,
            project_root=project_root,
            run_root=run_root,
            batch_root=batch_root,
            expected=stratum["expected_traces"],
            poll_seconds=args.poll_seconds,
            status=status,
        )
        analysis_root = stratum_root / "analysis-v1"
        analysis_command = [
            sys.executable,
            "-m",
            "topology_mas.analysis.cli",
            "--batch-dir",
            str(batch_root),
            "--output-dir",
            str(analysis_root),
        ]
        run_stage(
            name=f"analysis_{key}",
            command=analysis_command,
            project_root=project_root,
            run_root=run_root,
            batch_root=batch_root,
            expected=stratum["expected_traces"],
            poll_seconds=args.poll_seconds,
            status=status,
        )
        stratum["batch_summary"] = json.loads((batch_root / "summary.json").read_text())
        stratum["analysis_manifest"] = json.loads(
            (analysis_root / "manifest.json").read_text()
        )
        atomic_json(run_root / "orchestrator_status.json", status)

    status["status"] = "completed"
    status["ended_at"] = utc_now()
    atomic_json(run_root / "orchestrator_status.json", status)


if __name__ == "__main__":
    main()
