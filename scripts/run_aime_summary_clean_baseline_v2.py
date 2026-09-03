#!/usr/bin/env python3
"""Run the 480-cell clean baseline only after summary-protocol-v2 passes its gate."""

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
from topology_mas.execution.scalable_protocol import HuggingFaceTokenCounter
from topology_mas.execution.scalable_round_zero import (
    ScalableRoundZeroPoolStore,
    assign_draw_to_graph,
    build_round_zero_draws,
    materialize_engine_inputs,
)
from topology_mas.execution.schemas import ExecutionSettings
from topology_mas.execution.seeding import stable_id
from topology_mas.execution.summary_protocol_v2 import (
    SUMMARY_PROTOCOL_V2,
    SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
    SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
    SUMMARY_PROTOCOL_V2_MODEL,
    SUMMARY_PROTOCOL_V2_PROMPT_VERSION,
    SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
    SolveThenSummarizeGeneratorV2,
    SummaryProtocolV2Cache,
    require_summary_protocol_v2_settings,
    summary_protocol_v2,
)
from topology_mas.models import RunCondition
from topology_mas.topology.io import read_graphs_jsonl


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--tasks", type=Path, required=True)
    value.add_argument("--graphs", type=Path, required=True)
    value.add_argument("--round-zero-pool", type=Path, required=True)
    value.add_argument("--protocol-gate-report", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--cache-dir", type=Path, required=True)
    value.add_argument("--base-url", action="append", required=True)
    value.add_argument("--model", default=SUMMARY_PROTOCOL_V2_MODEL)
    value.add_argument("--expected-returned-model")
    value.add_argument("--tokenizer", required=True)
    value.add_argument("--tokenizer-cache-dir", type=Path)
    value.add_argument("--workers-per-backend", type=int, default=24)
    value.add_argument("--timeout-seconds", type=float, default=3600.0)
    value.add_argument("--provider-max-attempts", type=int, default=1)
    value.add_argument("--draw-seed", type=int, default=20260903)
    return value


def verify_gate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != SUMMARY_PROTOCOL_V2:
        raise ValueError("pilot gate report is not for summary-protocol-v2")
    if report.get("prompt_version") != SUMMARY_PROTOCOL_V2_PROMPT_VERSION:
        raise ValueError("pilot gate prompt version differs from the baseline")
    if report.get("gate", {}).get("passed") is not True:
        raise ValueError("summary-protocol-v2 pilot gate did not pass")
    if report.get("overall", {}).get("jobs", 0) < 100:
        raise ValueError("pilot gate contains fewer than 100 stratified jobs")
    return report


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parser().parse_args()
    gate = verify_gate(args.protocol_gate_report)
    require_summary_protocol_v2_settings(
        model=args.model,
        full_temperature=0.7,
        full_top_p=0.8,
        full_top_k=20,
        full_max_output_tokens=SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
        summary_max_output_tokens=SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
        summary_max_attempts=SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
    )
    tasks = load_aime_jsonl(args.tasks, split="test")
    graphs = read_graphs_jsonl(args.graphs)
    if len(tasks) != 30 or len(graphs) != 16:
        raise ValueError("frozen v2 baseline requires 30 tasks and 16 graphs")
    if any(graph.node_count != 5 or graph.max_rounds != 3 for graph in graphs):
        raise ValueError("all graphs must use n=5 and H=3")
    edge_histogram = {
        m: sum(len(graph.edges) == m for graph in graphs)
        for m in sorted({len(graph.edges) for graph in graphs})
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
    if pool_manifest.config.prompt_version != SUMMARY_PROTOCOL_V2_PROMPT_VERSION:
        raise ValueError("Round-0 pool is not a summary-protocol-v2 pool")
    if not all(
        row.provider_metadata.get("generation_pipeline") == SUMMARY_PROTOCOL_V2
        and row.provider_metadata.get("summary_validation_passed") is True
        for row in pool_responses
    ):
        raise ValueError("Round-0 pool contains an unvalidated v2 public summary")

    token_counter = HuggingFaceTokenCounter(
        args.tokenizer,
        cache_dir=(str(args.tokenizer_cache_dir.resolve()) if args.tokenizer_cache_dir else None),
    )
    protocol = summary_protocol_v2(token_counter)
    draws = {
        task.task_id: build_round_zero_draws(
            pool_version=pool_manifest.pool_version,
            task_id=task.task_id,
            node_count=5,
            replicate_count=1,
            pool_responses=pool_responses,
            draw_seed=args.draw_seed,
            required_generation_pipeline=SUMMARY_PROTOCOL_V2,
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
                "aime-summary-clean-v2",
                task.task_id,
                graph.graph_id,
                draws[task.task_id].draw_id,
                assignments[(task.task_id, graph.graph_id)].assignment_id,
            ),
            task_id=task.task_id,
            graph_id=graph.graph_id,
            experiment_seed=draws[task.task_id].selection_seed,
            assignment_seed=assignments[(task.task_id, graph.graph_id)].assignment_seed,
            condition=RunCondition.CLEAN,
        )
        for task in tasks
        for graph in graphs
    )
    manifest = {
        "experiment_version": "aime-summary-clean-baseline-v2",
        "protocol": SUMMARY_PROTOCOL_V2,
        "prompt_version": protocol.prompt_version,
        "model": args.model,
        "tasks": len(tasks),
        "graphs": len(graphs),
        "expected_runs": len(plan),
        "node_count": 5,
        "max_rounds": 3,
        "edge_histogram": edge_histogram,
        "round_zero_policy": "task_pool_k64_paired_draw_graph_permutation",
        "pool_version": pool_manifest.pool_version,
        "pool_manifest_fingerprint": content_fingerprint(pool_manifest),
        "pilot_gate_fingerprint": content_fingerprint(gate),
        "cross_node_representation": "validated_summary_only",
        "self_history_representation": "previous_full_solution",
        "generation_pipeline": SUMMARY_PROTOCOL_V2,
        "provider_max_attempts": args.provider_max_attempts,
        "full_sampling": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
        },
        "summary_sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_output_tokens": SUMMARY_PROTOCOL_V2_PUBLIC_MAX_TOKENS,
            "max_attempts": SUMMARY_PROTOCOL_V2_MAX_ATTEMPTS,
        },
        "task_fingerprint": content_fingerprint(tasks),
        "graph_fingerprint": content_fingerprint(graphs),
        "plan_fingerprint": content_fingerprint(plan),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError("baseline manifest differs; use a new output directory")
    else:
        write_json(manifest_path, manifest)
    plan_rows = [spec.model_dump(mode="json") for spec in plan]
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan_rows:
            raise ValueError("baseline plan differs; use a new output directory")
    else:
        write_json(plan_path, plan_rows)

    settings = ExecutionSettings(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_output_tokens=SUMMARY_PROTOCOL_V2_FULL_MAX_TOKENS,
        initial_state_policy="shared_round_zero_cache",
        message_order_seed=0,
        horizon_policy="fixed",
        generation_pipeline=SUMMARY_PROTOCOL_V2,
    )
    task_by_id = {task.task_id: task for task in tasks}
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    store = BatchExecutionStore(args.output_dir)
    store.traces_dir.mkdir(parents=True, exist_ok=True)
    store.failures_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    outcomes: list[dict[str, Any]] = []
    progress_lock = threading.Lock()

    with ExitStack() as stack:
        backends = tuple(
            stack.enter_context(
                OpenAICompatibleTextGenerator(
                    model=args.model,
                    expected_returned_model=args.expected_returned_model,
                    base_url=url,
                    api_key_env=None,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.provider_max_attempts,
                    allow_context_window_adjustment=False,
                )
            )
            for url in args.base_url
        )
        cache = SummaryProtocolV2Cache(args.cache_dir)
        engines = tuple(
            SynchronousExecutionEngine(
                SolveThenSummarizeGeneratorV2(
                    backend, cache=cache, token_counter=token_counter
                ),
                settings=settings,
                protocol=protocol,
            )
            for backend in backends
        )

        def execute(index: int, spec: ExecutionRunSpec) -> dict[str, Any]:
            cached = store.load(spec)
            if cached is not None:
                trace = cached.trace
                return {
                    "run_spec_id": spec.run_spec_id,
                    "status": "cached",
                    "model_calls": trace.total_model_calls,
                    "backend_calls": trace.total_backend_calls,
                }
            task = task_by_id[spec.task_id]
            graph = graph_by_id[spec.graph_id]
            round_zero_records, initial_assignment = materialize_engine_inputs(
                draw=draws[spec.task_id],
                graph_assignment=assignments[(spec.task_id, spec.graph_id)],
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
                    "trace_path": str(path),
                    "elapsed_seconds": time.monotonic() - run_start,
                    "model_calls": trace.total_model_calls,
                    "backend_calls": trace.total_backend_calls,
                    "input_tokens": trace.total_input_tokens,
                    "output_tokens": trace.total_output_tokens,
                }
            except Exception as exc:  # retain full v2 completion and partial trace
                failure_path = store.save_failure(spec, exc)
                return {
                    "run_spec_id": spec.run_spec_id,
                    "status": "failed",
                    "failure_path": str(failure_path),
                    "elapsed_seconds": time.monotonic() - run_start,
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
                outcomes.append(future.result())
                if completed % 5 == 0 or completed == len(plan):
                    with progress_lock:
                        progress = {
                            "completed": completed,
                            "expected": len(plan),
                            "generated": sum(row["status"] == "generated" for row in outcomes),
                            "cached": sum(row["status"] == "cached" for row in outcomes),
                            "failed": sum(row["status"] == "failed" for row in outcomes),
                            "elapsed_seconds": time.monotonic() - start,
                        }
                        write_json(args.output_dir / "progress.json", progress)
                        print(json.dumps(progress), flush=True)

    outcomes.sort(key=lambda row: row["run_spec_id"])
    summary = {
        "expected_runs": len(plan),
        "completed_runs": len(outcomes),
        "generated_runs": sum(row["status"] == "generated" for row in outcomes),
        "cached_runs": sum(row["status"] == "cached" for row in outcomes),
        "failed_runs": sum(row["status"] == "failed" for row in outcomes),
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(args.output_dir / "outcomes.json", outcomes)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed_runs"]:
        raise RuntimeError(f"v2 clean baseline has {summary['failed_runs']} failed runs")


if __name__ == "__main__":
    main()
