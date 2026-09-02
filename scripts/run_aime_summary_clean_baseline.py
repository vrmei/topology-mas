#!/usr/bin/env python3
"""Run the pooled-Round-0, summary-only 2026 AIME clean baseline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from topology_mas.data.aime import load_aime_jsonl
from topology_mas.execution.batch import (
    BatchExecutionStore,
    ExecutionRunSpec,
    content_fingerprint,
)
from topology_mas.execution.engine import SynchronousExecutionEngine
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import (
    HuggingFaceTokenCounter,
    SinglePassDualChannelGenerator,
    scalable_aime_protocol,
)
from topology_mas.execution.scalable_round_zero import (
    ScalableRoundZeroPoolStore,
    assign_draw_to_graph,
    build_round_zero_draws,
    materialize_engine_inputs,
)
from topology_mas.execution.schemas import ExecutionSettings
from topology_mas.execution.seeding import stable_id
from topology_mas.models import RunCondition
from topology_mas.topology.io import read_graphs_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--round-zero-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--workers-per-backend", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--draw-seed", type=int, default=20260903)
    return parser


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle_path, path)
    except Exception:
        if handle_path is not None:
            handle_path.unlink(missing_ok=True)
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle_path = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle_path, path)
    except Exception:
        if handle_path is not None:
            handle_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    if args.workers_per_backend < 1:
        raise ValueError("workers-per-backend must be positive")
    tasks = load_aime_jsonl(args.tasks, split="test")
    graphs = read_graphs_jsonl(args.graphs)
    if len(tasks) != 30:
        raise ValueError(f"expected 30 frozen 2026 AIME tasks, found {len(tasks)}")
    if len(graphs) != 16:
        raise ValueError(f"expected 16 frozen graphs, found {len(graphs)}")
    if any(graph.node_count != 5 or graph.max_rounds != 3 for graph in graphs):
        raise ValueError("all graphs must use n=5 and H=3")
    edge_histogram = {
        edge_count: sum(len(graph.edges) == edge_count for graph in graphs)
        for edge_count in sorted({len(graph.edges) for graph in graphs})
    }
    if edge_histogram != {4: 5, 8: 5, 12: 5, 16: 1}:
        raise ValueError(f"unexpected graph edge histogram: {edge_histogram}")

    pool_manifest, pool_responses = ScalableRoundZeroPoolStore(
        args.round_zero_pool
    ).load_complete()
    if pool_manifest.task_ids != tuple(task.task_id for task in tasks):
        raise ValueError("Round-0 pool task set differs from the requested tasks")
    if pool_manifest.config.responses_per_task < 64:
        raise ValueError("Round-0 pool must contain at least 64 responses per task")

    token_counter = HuggingFaceTokenCounter(
        args.tokenizer,
        cache_dir=(
            str(args.tokenizer_cache_dir.resolve())
            if args.tokenizer_cache_dir is not None
            else None
        ),
    )
    protocol = scalable_aime_protocol(token_counter)
    if pool_manifest.config.prompt_version != protocol.prompt_version:
        raise ValueError("Round-0 pool and execution prompt versions differ")
    if pool_manifest.config.requested_model != args.model:
        raise ValueError("Round-0 pool and execution requested models differ")

    draws = {
        task.task_id: build_round_zero_draws(
            pool_version=pool_manifest.pool_version,
            task_id=task.task_id,
            node_count=5,
            replicate_count=1,
            pool_responses=pool_responses,
            draw_seed=args.draw_seed,
        )[0]
        for task in tasks
    }
    assignments = {
        (task.task_id, graph.graph_id): assign_draw_to_graph(draws[task.task_id], graph)
        for task in tasks
        for graph in graphs
    }
    plan = tuple(
        ExecutionRunSpec(
            run_spec_id=stable_id(
                "aime-summary-clean",
                task.task_id,
                graph.graph_id,
                draws[task.task_id].draw_id,
                assignments[(task.task_id, graph.graph_id)].assignment_id,
            ),
            task_id=task.task_id,
            graph_id=graph.graph_id,
            experiment_seed=draws[task.task_id].selection_seed,
            assignment_seed=assignments[
                (task.task_id, graph.graph_id)
            ].assignment_seed,
            condition=RunCondition.CLEAN,
        )
        for task in tasks
        for graph in graphs
    )
    manifest = {
        "experiment_version": "aime-summary-clean-baseline-v1",
        "prompt_version": protocol.prompt_version,
        "model": args.model,
        "expected_returned_model": args.expected_returned_model,
        "tasks": len(tasks),
        "graphs": len(graphs),
        "expected_runs": len(plan),
        "node_count": 5,
        "max_rounds": 3,
        "edge_histogram": edge_histogram,
        "round_zero_policy": "task_pool_k64_paired_draw_graph_permutation",
        "pool_version": pool_manifest.pool_version,
        "pool_manifest_fingerprint": content_fingerprint(pool_manifest),
        "cross_node_representation": "summary_only",
        "max_public_summary_tokens": protocol.max_public_tokens,
        "self_history_representation": "previous_full_solution",
        "generation_pipeline": "single-pass-dual-channel-v1",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_output_tokens": args.max_output_tokens,
        },
        "base_url_count": len(args.base_url),
        "workers_per_backend": args.workers_per_backend,
        "task_fingerprint": content_fingerprint(tasks),
        "graph_fingerprint": content_fingerprint(graphs),
        "plan_fingerprint": content_fingerprint(plan),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    serialized_manifest = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != serialized_manifest:
        raise ValueError("output manifest differs; use a new output directory")
    if not manifest_path.exists():
        atomic_text(manifest_path, serialized_manifest)
    atomic_text(
        args.output_dir / "plan.jsonl",
        "".join(spec.model_dump_json() + "\n" for spec in plan),
    )
    atomic_text(
        args.output_dir / "draws.jsonl",
        "".join(draw.model_dump_json() + "\n" for draw in draws.values()),
    )
    atomic_text(
        args.output_dir / "graph_assignments.jsonl",
        "".join(value.model_dump_json() + "\n" for value in assignments.values()),
    )

    settings = ExecutionSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_output_tokens=args.max_output_tokens,
        initial_state_policy="shared_round_zero_cache",
        message_order_seed=0,
        horizon_policy="fixed",
        generation_pipeline="single-pass-dual-channel-v1",
    )
    task_by_id = {task.task_id: task for task in tasks}
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    store = BatchExecutionStore(args.output_dir)
    for directory in (store.traces_dir, store.failures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    progress_lock = threading.Lock()
    start = time.monotonic()
    outcomes: list[dict[str, Any]] = []

    with ExitStack() as stack:
        backends = tuple(
            stack.enter_context(
                OpenAICompatibleTextGenerator(
                    model=args.model,
                    expected_returned_model=args.expected_returned_model,
                    base_url=base_url,
                    api_key_env=None,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.max_attempts,
                    allow_context_window_adjustment=False,
                )
            )
            for base_url in args.base_url
        )
        engines = tuple(
            SynchronousExecutionEngine(
                SinglePassDualChannelGenerator(
                    backend,
                    answer_parser=protocol.answer_parser,
                    summary_answer_parser=protocol.summary_answer_parser,
                    token_counter=token_counter,
                    max_public_tokens=protocol.max_public_tokens,
                    strict_validation=True,
                ),
                settings=settings,
                protocol=protocol,
            )
            for backend in backends
        )

        def execute(index: int, spec: ExecutionRunSpec) -> dict[str, Any]:
            cached = store.load(spec)
            if cached is not None:
                return {
                    "run_spec_id": spec.run_spec_id,
                    "status": "cached",
                    "graph_id": spec.graph_id,
                    "task_id": spec.task_id,
                    "backend_index": index % len(engines),
                    "elapsed_seconds": 0.0,
                    "trace_path": str(store.trace_path(spec)),
                    "model_calls": cached.trace.total_model_calls,
                    "backend_calls": cached.trace.total_backend_calls,
                    "input_tokens": cached.trace.total_input_tokens,
                    "output_tokens": cached.trace.total_output_tokens,
                }
            task = task_by_id[spec.task_id]
            graph = graph_by_id[spec.graph_id]
            draw = draws[spec.task_id]
            graph_assignment = assignments[(spec.task_id, spec.graph_id)]
            round_zero_records, initial_assignment = materialize_engine_inputs(
                draw=draw,
                graph_assignment=graph_assignment,
                pool_responses=pool_responses,
                experiment_seed=spec.experiment_seed,
            )
            run_start = time.monotonic()
            try:
                trace = engines[index % len(engines)].run(
                    graph=graph,
                    task=task,
                    condition=RunCondition.CLEAN,
                    seed=spec.experiment_seed,
                    round_zero_records=round_zero_records,
                    initial_assignment=initial_assignment,
                )
                path = store.save(spec, trace)
                return {
                    "run_spec_id": spec.run_spec_id,
                    "status": "generated",
                    "graph_id": spec.graph_id,
                    "task_id": spec.task_id,
                    "backend_index": index % len(engines),
                    "elapsed_seconds": time.monotonic() - run_start,
                    "trace_path": str(path),
                    "model_calls": trace.total_model_calls,
                    "backend_calls": trace.total_backend_calls,
                    "input_tokens": trace.total_input_tokens,
                    "output_tokens": trace.total_output_tokens,
                }
            except Exception as exc:  # noqa: BLE001 - every failed cell is retained
                failure_path = store.save_failure(spec, exc)
                return {
                    "run_spec_id": spec.run_spec_id,
                    "status": "failed",
                    "graph_id": spec.graph_id,
                    "task_id": spec.task_id,
                    "backend_index": index % len(engines),
                    "elapsed_seconds": time.monotonic() - run_start,
                    "failure_path": str(failure_path),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }

        with ThreadPoolExecutor(
            max_workers=len(engines) * args.workers_per_backend
        ) as executor:
            futures: dict[Future[dict[str, Any]], int] = {
                executor.submit(execute, index, spec): index
                for index, spec in enumerate(plan)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                outcome = future.result()
                outcomes.append(outcome)
                if completed % 5 == 0 or completed == len(plan):
                    with progress_lock:
                        status_counts = {
                            status: sum(row["status"] == status for row in outcomes)
                            for status in ("generated", "cached", "failed")
                        }
                        progress = {
                            "completed": completed,
                            "expected": len(plan),
                            "status_counts": status_counts,
                            "elapsed_seconds": time.monotonic() - start,
                        }
                        atomic_json(args.output_dir / "progress.json", progress)
                        print(json.dumps(progress), flush=True)

    outcomes.sort(key=lambda row: row["run_spec_id"])
    summary = {
        "expected_runs": len(plan),
        "completed_runs": len(outcomes),
        "generated_runs": sum(row["status"] == "generated" for row in outcomes),
        "cached_runs": sum(row["status"] == "cached" for row in outcomes),
        "failed_runs": sum(row["status"] == "failed" for row in outcomes),
        "known_input_tokens": sum(row.get("input_tokens") or 0 for row in outcomes),
        "known_output_tokens": sum(row.get("output_tokens") or 0 for row in outcomes),
        "elapsed_seconds": time.monotonic() - start,
    }
    atomic_json(args.output_dir / "outcomes.json", outcomes)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["failed_runs"]:
        raise RuntimeError(f"summary clean baseline has {summary['failed_runs']} failed runs")


if __name__ == "__main__":
    main()
