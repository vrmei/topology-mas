"""Strict loader for self-contained paired batch artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from topology_mas.data.gsm8k import task_collection_fingerprint
from topology_mas.execution.batch import (
    BATCH_EXECUTION_VERSION,
    BatchExecutionConflictError,
    BatchExecutionManifest,
    BatchExecutionStore,
    ExecutionRunSpec,
    RoundZeroRecordReference,
    StoredExecutionRun,
    content_fingerprint,
)
from topology_mas.models import AdversarialAnswer, GraphSpec, TaskInstance
from topology_mas.topology.sampling import graph_collection_fingerprint

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class LoadedBatch:
    root: Path
    manifest: BatchExecutionManifest
    plan: tuple[ExecutionRunSpec, ...]
    tasks: tuple[TaskInstance, ...]
    graphs: tuple[GraphSpec, ...]
    round_zero_references: tuple[RoundZeroRecordReference, ...]
    adversarial_answers: tuple[AdversarialAnswer, ...]
    runs: tuple[StoredExecutionRun, ...]


def _read_jsonl(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    if not path.exists():
        raise BatchExecutionConflictError(f"required batch artifact is missing: {path}")
    values: list[ModelT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                values.append(model.model_validate_json(line))
            except ValueError as exc:
                raise BatchExecutionConflictError(
                    f"invalid {model.__name__} at {path}:{line_number}"
                ) from exc
    return tuple(values)


def load_complete_batch(root: str | Path) -> LoadedBatch:
    batch_root = Path(root)
    store = BatchExecutionStore(batch_root)
    if not store.manifest_path.exists():
        raise BatchExecutionConflictError("batch manifest is missing")
    try:
        manifest = BatchExecutionManifest.model_validate(
            json.loads(store.manifest_path.read_text(encoding="utf-8"))
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise BatchExecutionConflictError("batch manifest is invalid") from exc
    if manifest.runner_version != BATCH_EXECUTION_VERSION:
        raise BatchExecutionConflictError(
            f"analysis requires a self-contained {BATCH_EXECUTION_VERSION} artifact"
        )

    plan = _read_jsonl(store.plan_path, ExecutionRunSpec)
    tasks = _read_jsonl(store.tasks_path, TaskInstance)
    graphs = _read_jsonl(store.graphs_path, GraphSpec)
    references = _read_jsonl(store.round_zero_index_path, RoundZeroRecordReference)
    answers = _read_jsonl(store.adversarial_answers_path, AdversarialAnswer)

    if len(plan) != manifest.expected_run_count:
        raise BatchExecutionConflictError("plan length differs from the manifest")
    if content_fingerprint(plan) != manifest.plan_fingerprint:
        raise BatchExecutionConflictError("plan fingerprint differs from the manifest")
    if tuple(task.task_id for task in tasks) != manifest.task_ids:
        raise BatchExecutionConflictError("task IDs differ from the manifest")
    if tuple(graph.graph_id for graph in graphs) != manifest.graph_ids:
        raise BatchExecutionConflictError("graph IDs differ from the manifest")
    if task_collection_fingerprint(tasks) != manifest.task_collection_fingerprint:
        raise BatchExecutionConflictError("task fingerprint differs from the manifest")
    if graph_collection_fingerprint(graphs) != manifest.graph_collection_fingerprint:
        raise BatchExecutionConflictError("graph fingerprint differs from the manifest")
    if manifest.round_zero_index_fingerprint is None:
        if references or manifest.config.initial_state_policy != "independent_per_run":
            raise BatchExecutionConflictError(
                "missing Round-zero fingerprint is incompatible with the stored index/policy"
            )
    elif content_fingerprint(references) != manifest.round_zero_index_fingerprint:
        raise BatchExecutionConflictError("round-zero index fingerprint differs")
    if content_fingerprint(answers) != manifest.adversarial_answers_fingerprint:
        raise BatchExecutionConflictError("adversarial-answer fingerprint differs")

    expected_trace_names = {f"{spec.run_spec_id}.json" for spec in plan}
    actual_trace_names = (
        {path.name for path in store.traces_dir.glob("*.json")}
        if store.traces_dir.exists()
        else set()
    )
    if actual_trace_names != expected_trace_names:
        missing = sorted(expected_trace_names - actual_trace_names)
        extra = sorted(actual_trace_names - expected_trace_names)
        raise BatchExecutionConflictError(
            f"trace set is incomplete or unexpected; missing={missing[:3]}, extra={extra[:3]}"
        )
    runs = tuple(store.load(spec) for spec in plan)
    if any(run is None for run in runs):
        raise BatchExecutionConflictError("planned trace is missing")
    return LoadedBatch(
        root=batch_root,
        manifest=manifest,
        plan=plan,
        tasks=tasks,
        graphs=graphs,
        round_zero_references=references,
        adversarial_answers=answers,
        runs=tuple(run for run in runs if run is not None),
    )
