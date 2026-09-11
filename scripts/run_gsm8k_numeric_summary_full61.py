#!/usr/bin/env python3
"""Run frozen GSM8K graph families under numeric summary v1."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from queue import Queue

from topology_mas.data.gsm8k import read_tasks_jsonl
from topology_mas.execution.batch import (
    BatchExecutionConfig,
    BatchExecutionManifest,
    BatchExecutionStore,
    ExecutionRunSpec,
    RoundZeroRecordReference,
    content_fingerprint,
)
from topology_mas.execution.engine import SynchronousExecutionEngine
from topology_mas.execution.inputs import load_adversarial_answer_index
from topology_mas.execution.numeric_summary_protocol import (
    NUMERIC_FULL_MAX_TOKENS,
    NUMERIC_SUMMARY_MODEL,
    NUMERIC_SUMMARY_PROMPT_VERSION,
    NUMERIC_SUMMARY_PROTOCOL,
    SolveThenSummarizeNumericGenerator,
    numeric_summary_protocol,
)
from topology_mas.execution.openai_compatible import OpenAICompatibleTextGenerator
from topology_mas.execution.scalable_protocol import HuggingFaceTokenCounter
from topology_mas.execution.scalable_round_zero import (
    ScalableRoundZeroPoolStore,
    assign_draw_to_graph,
    build_round_zero_draws,
    materialize_engine_inputs,
)
from topology_mas.execution.schemas import ExecutionSettings
from topology_mas.execution.seeding import stable_id, stable_integer
from topology_mas.execution.summary_protocol_v3 import SummaryProtocolV3Cache
from topology_mas.models import AttackMode, RunCondition
from topology_mas.topology.io import read_graphs_jsonl

EXPECTED_GRAPH_HISTOGRAMS = {
    5: {**{m: 5 for m in range(4, 16)}, 16: 1},
    6: {**{m: 5 for m in range(5, 25, 2)}, 25: 1},
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--round-zero-pool", type=Path, required=True)
    parser.add_argument("--k64-index", type=Path, required=True)
    parser.add_argument("--adversarial-answers", type=Path)
    parser.add_argument("--condition", choices=("clean", "attack", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default=NUMERIC_SUMMARY_MODEL)
    parser.add_argument("--expected-returned-model")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--workers-per-backend", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--provider-max-attempts", type=int, default=3)
    parser.add_argument("--draw-seed", type=int, default=20260911)
    parser.add_argument("--smoke-task-limit", type=int)
    parser.add_argument("--smoke-graph-limit", type=int)
    parser.add_argument("--expected-node-count", type=int, choices=tuple(EXPECTED_GRAPH_HISTOGRAMS))
    args = parser.parse_args()

    tasks = read_tasks_jsonl(args.tasks)
    graphs = read_graphs_jsonl(args.graphs)
    smoke = args.smoke_task_limit is not None or args.smoke_graph_limit is not None
    if args.smoke_task_limit is not None:
        tasks = tasks[: args.smoke_task_limit]
    if args.smoke_graph_limit is not None:
        graphs = graphs[: args.smoke_graph_limit]
    if not smoke and len(tasks) != 50:
        raise ValueError(f"frozen GSM8K experiment requires 50 tasks, got {len(tasks)}")
    if not graphs:
        raise ValueError("graph family is empty")
    node_count = graphs[0].node_count
    readout_node = node_count - 1
    if args.expected_node_count is not None and node_count != args.expected_node_count:
        raise ValueError(
            f"expected n={args.expected_node_count}, but graph family uses n={node_count}"
        )
    if node_count not in EXPECTED_GRAPH_HISTOGRAMS:
        raise ValueError(f"no frozen graph-family contract is registered for n={node_count}")
    if any(graph.node_count != node_count or graph.max_rounds != 3 for graph in graphs):
        raise ValueError(f"every graph must use n={node_count}, H=3")
    if any(graph.readout_node != readout_node for graph in graphs):
        raise ValueError(
            f"the frozen n={node_count} graph family requires readout node {readout_node}"
        )
    edge_histogram = {
        m: sum(len(g.edges) == m for g in graphs) for m in sorted({len(g.edges) for g in graphs})
    }
    expected_histogram = EXPECTED_GRAPH_HISTOGRAMS[node_count]
    if not smoke and edge_histogram != expected_histogram:
        raise ValueError(f"unexpected n={node_count} edge histogram: {edge_histogram}")

    pool_manifest, all_pool_rows = ScalableRoundZeroPoolStore(args.round_zero_pool).load_available()
    task_ids = tuple(task.task_id for task in tasks)
    if not smoke and pool_manifest.task_ids != task_ids:
        raise ValueError("K80 pool task IDs/order differ from the frozen 50 tasks")
    if smoke and any(task_id not in pool_manifest.task_ids for task_id in task_ids):
        raise ValueError("smoke task IDs are not contained in the K80 pool")
    if pool_manifest.config.responses_per_task != 80:
        raise ValueError("source Round-0 pool must contain K80 per task")
    k64 = json.loads(args.k64_index.read_text(encoding="utf-8"))
    if k64.get("source_pool_version") != pool_manifest.pool_version:
        raise ValueError("K64 index belongs to a different K80 pool")
    if k64.get("maximum_selected_responses_per_task") != 64:
        raise ValueError("frozen preselection index must use at most K64")
    if not smoke and tuple(k64.get("task_ids", ())) != task_ids:
        raise ValueError("K64 task IDs/order differ from the frozen 50 tasks")
    per_task_ids = k64.get("selected_pool_response_ids", {})
    if set(per_task_ids) != set(pool_manifest.task_ids) or any(
        len(ids) < node_count or len(ids) > 64 or len(set(ids)) != len(ids)
        for ids in per_task_ids.values()
    ):
        raise ValueError(f"preselection index must contain {node_count}..64 unique IDs per task")
    selected_ids = {response_id for task_id in task_ids for response_id in per_task_ids[task_id]}
    pool_rows = tuple(row for row in all_pool_rows if row.pool_response_id in selected_ids)
    expected_selected = sum(len(per_task_ids[task_id]) for task_id in task_ids)
    if len(pool_rows) != expected_selected:
        raise ValueError("preselection filtering lost one or more selected responses")
    if not all(
        row.provider_metadata.get("generation_pipeline") == NUMERIC_SUMMARY_PROTOCOL
        and row.provider_metadata.get("summary_validation_passed") is True
        for row in pool_rows
    ):
        raise ValueError("K64 contains a response without a validated public summary")

    answers = (
        load_adversarial_answer_index(args.adversarial_answers)
        if args.adversarial_answers is not None
        else {}
    )
    if args.condition in {"attack", "all"}:
        if not {task.task_id for task in tasks}.issubset(answers):
            raise ValueError("attack execution requires one frozen answer for every task")
        if any(answer.public_summary is None for answer in answers.values()):
            raise ValueError("every fixed attacker must have a frozen public summary")

    # A fresh deterministic K64->K5 draw is made for each task×graph.  Clean and
    # all attacker positions within that cell share the same draw and assignment.
    draws = {}
    assignments = {}
    materialized = {}
    for task in tasks:
        for graph in graphs:
            draw = build_round_zero_draws(
                pool_version=pool_manifest.pool_version,
                task_id=task.task_id,
                node_count=node_count,
                replicate_count=1,
                pool_responses=pool_rows,
                draw_seed=stable_integer(
                    "gsm8k-summary-full61-k64-draw",
                    args.draw_seed,
                    task.task_id,
                    graph.graph_id,
                ),
                required_generation_pipeline=NUMERIC_SUMMARY_PROTOCOL,
            )[0]
            assignment = assign_draw_to_graph(draw, graph)
            seed = stable_integer(
                "gsm8k-summary-full61-cell",
                args.draw_seed,
                task.task_id,
                graph.graph_id,
            )
            records, engine_assignment = materialize_engine_inputs(
                draw=draw,
                graph_assignment=assignment,
                pool_responses=pool_rows,
                experiment_seed=seed,
            )
            key = (task.task_id, graph.graph_id)
            draws[key], assignments[key] = draw, assignment
            materialized[key] = (seed, records, engine_assignment)

    conditions = []
    if args.condition in {"clean", "all"}:
        conditions.append((RunCondition.CLEAN, None))
    if args.condition in {"attack", "all"}:
        conditions.extend(
            (RunCondition.ATTACK, node_id)
            for node_id in range(node_count)
            if node_id != readout_node
        )
    plan = tuple(
        ExecutionRunSpec(
            run_spec_id=stable_id(
                "gsm8k-numeric-summary-full61-v1",
                task.task_id,
                graph.graph_id,
                condition.value,
                attack_node,
                draws[(task.task_id, graph.graph_id)].draw_id,
                assignments[(task.task_id, graph.graph_id)].assignment_id,
            ),
            task_id=task.task_id,
            graph_id=graph.graph_id,
            experiment_seed=materialized[(task.task_id, graph.graph_id)][0],
            assignment_seed=assignments[(task.task_id, graph.graph_id)].assignment_seed,
            condition=condition,
            attack_node=attack_node,
        )
        for task in tasks
        for graph in graphs
        for condition, attack_node in conditions
    )
    settings = ExecutionSettings(
        temperature=0.6,
        top_p=0.9,
        top_k=None,
        max_output_tokens=NUMERIC_FULL_MAX_TOKENS,
        initial_state_policy="shared_round_zero_cache",
        message_order_seed=0,
        horizon_policy="fixed",
        generation_pipeline=NUMERIC_SUMMARY_PROTOCOL,
    )
    config = BatchExecutionConfig(
        experiment_seeds=tuple(sorted({spec.experiment_seed for spec in plan})),
        assignment_seeds=tuple(sorted({spec.assignment_seed for spec in plan})),
        include_attacks=args.condition in {"attack", "all"},
        attack_mode=AttackMode.FIXED,
        initial_state_policy="shared_round_zero_cache",
        requested_model=args.model,
        expected_returned_model=args.expected_returned_model,
    )
    refs = tuple(
        RoundZeroRecordReference.from_record(record)
        for key in sorted(materialized)
        for record in materialized[key][1]
    )
    selected_answers = tuple(answers[task.task_id] for task in tasks if task.task_id in answers)
    manifest = BatchExecutionManifest(
        config=config,
        execution_settings=settings,
        prompt_version=NUMERIC_SUMMARY_PROMPT_VERSION,
        node_count=node_count,
        readout_node=readout_node,
        max_rounds=3,
        task_ids=tuple(task.task_id for task in tasks),
        graph_ids=tuple(graph.graph_id for graph in graphs),
        task_collection_fingerprint=content_fingerprint(tasks),
        graph_collection_fingerprint=content_fingerprint(graphs),
        round_zero_fingerprint=content_fingerprint(pool_manifest),
        round_zero_index_fingerprint=content_fingerprint(
            {
                "k64": k64,
                "draws": draws,
                "assignments": assignments,
            }
        ),
        adversarial_answers_fingerprint=content_fingerprint(selected_answers),
        plan_fingerprint=content_fingerprint(plan),
        expected_run_count=len(plan),
    )
    store = BatchExecutionStore(args.output_dir)
    store.initialize(
        manifest=manifest,
        plan=plan,
        tasks=tasks,
        graphs=graphs,
        round_zero_references=refs,
        adversarial_answers=selected_answers,
    )
    _write_json(
        args.output_dir / "k64_draw_audit.json",
        {
            "k80_pool_version": pool_manifest.pool_version,
            "k64_fingerprint": k64["fingerprint"],
            "policy": (
                "independent deterministic K5 draw per task-graph; paired across clean/attack"
            ),
            "draw_seed": args.draw_seed,
            "cells": {
                f"{task_id}|{graph_id}": {
                    "draw": draws[(task_id, graph_id)].model_dump(mode="json"),
                    "assignment": assignments[(task_id, graph_id)].model_dump(mode="json"),
                }
                for task_id, graph_id in sorted(draws)
            },
        },
    )

    task_by_id = {task.task_id: task for task in tasks}
    graph_by_id = {graph.graph_id: graph for graph in graphs}
    token_counter = HuggingFaceTokenCounter(args.tokenizer)
    protocol = numeric_summary_protocol(token_counter)
    progress_lock = threading.Lock()
    outcomes = []
    started = time.monotonic()

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
        cache = SummaryProtocolV3Cache(args.cache_dir)
        engines = tuple(
            SynchronousExecutionEngine(
                SolveThenSummarizeNumericGenerator(
                    backend,
                    cache=cache,
                    token_counter=token_counter,
                ),
                settings=settings,
                protocol=protocol,
            )
            for backend in backends
        )
        slots: Queue[tuple[int, SynchronousExecutionEngine]] = Queue()
        for endpoint_index, engine in enumerate(engines):
            for _ in range(args.workers_per_backend):
                slots.put((endpoint_index, engine))
        active_by_backend = [0 for _ in engines]
        completed_by_backend = [0 for _ in engines]

        def execute(spec: ExecutionRunSpec):
            cached = store.load(spec)
            if cached is not None:
                return {"run_spec_id": spec.run_spec_id, "status": "cached"}
            key = (spec.task_id, spec.graph_id)
            seed, records, initial_assignment = materialized[key]
            endpoint_index, engine = slots.get()
            with progress_lock:
                active_by_backend[endpoint_index] += 1
            run_started = time.monotonic()
            try:
                trace = engine.run(
                    graph=graph_by_id[spec.graph_id],
                    task=task_by_id[spec.task_id],
                    condition=spec.condition,
                    seed=seed,
                    attack_node=spec.attack_node,
                    adversarial_answer=(
                        answers[spec.task_id] if spec.condition is RunCondition.ATTACK else None
                    ),
                    attack_mode=AttackMode.FIXED,
                    round_zero_records=records,
                    initial_assignment=initial_assignment,
                )
                path = store.save(spec, trace)
                return {
                    "run_spec_id": spec.run_spec_id,
                    "status": "generated",
                    "endpoint_index": endpoint_index,
                    "trace_path": str(path),
                    "elapsed_seconds": time.monotonic() - run_started,
                    "model_calls": trace.total_model_calls,
                    "backend_calls": trace.total_backend_calls,
                }
            except Exception as exc:
                path = store.save_failure(spec, exc)
                return {
                    "run_spec_id": spec.run_spec_id,
                    "status": "failed",
                    "endpoint_index": endpoint_index,
                    "failure_path": str(path),
                    "elapsed_seconds": time.monotonic() - run_started,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            finally:
                with progress_lock:
                    active_by_backend[endpoint_index] -= 1
                    completed_by_backend[endpoint_index] += 1
                slots.put((endpoint_index, engine))

        workers = len(engines) * args.workers_per_backend
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute, spec): spec for spec in plan}
            for completed, future in enumerate(as_completed(futures), start=1):
                outcomes.append(future.result())
                if completed % 10 == 0 or completed == len(plan):
                    with progress_lock:
                        progress = {
                            "completed": completed,
                            "expected": len(plan),
                            "generated": sum(x["status"] == "generated" for x in outcomes),
                            "cached": sum(x["status"] == "cached" for x in outcomes),
                            "failed": sum(x["status"] == "failed" for x in outcomes),
                            "elapsed_seconds": time.monotonic() - started,
                            "active_by_backend": list(active_by_backend),
                            "completed_by_backend": list(completed_by_backend),
                            "available_slots": slots.qsize(),
                        }
                        _write_json(args.output_dir / "progress.json", progress)
                        print(json.dumps(progress), flush=True)

    outcomes.sort(key=lambda row: row["run_spec_id"])
    summary = {
        "expected": len(plan),
        "completed": len(outcomes),
        "generated": sum(x["status"] == "generated" for x in outcomes),
        "cached": sum(x["status"] == "cached" for x in outcomes),
        "failed": sum(x["status"] == "failed" for x in outcomes),
        "elapsed_seconds": time.monotonic() - started,
        "completed_by_backend": completed_by_backend,
    }
    _write_json(args.output_dir / "outcomes.json", outcomes)
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
