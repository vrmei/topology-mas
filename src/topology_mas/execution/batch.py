"""Strictly paired, resumable batch execution over tasks and graph strata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from topology_mas.data.gsm8k import task_collection_fingerprint
from topology_mas.execution.assignments import (
    InitialStateAssignment,
    build_initial_state_assignment,
)
from topology_mas.execution.engine import SynchronousExecutionEngine
from topology_mas.execution.prompts import PROMPT_VERSION
from topology_mas.execution.round_zero import RoundZeroRecord
from topology_mas.execution.schemas import ExecutionSettings, RunTrace
from topology_mas.execution.seeding import stable_fingerprint, stable_id
from topology_mas.models import (
    AdversarialAnswer,
    AnswerState,
    GraphSpec,
    RunCondition,
    TaskInstance,
)
from topology_mas.topology.sampling import graph_collection_fingerprint

BATCH_EXECUTION_VERSION = "paired-batch-v2"


class BatchExecutionConfig(BaseModel):
    """Frozen choices defining one paired execution collection."""

    model_config = ConfigDict(frozen=True)

    experiment_seeds: tuple[int, ...] = Field(min_length=1)
    assignment_seeds: tuple[int, ...] = Field(min_length=1)
    include_attacks: bool = True
    requested_model: str = Field(min_length=1)
    expected_returned_model: str | None = None
    provider_base_url: str | None = None

    @model_validator(mode="after")
    def validate_seeds(self) -> BatchExecutionConfig:
        if len(set(self.experiment_seeds)) != len(self.experiment_seeds):
            raise ValueError("experiment_seeds must be unique")
        if len(set(self.assignment_seeds)) != len(self.assignment_seeds):
            raise ValueError("assignment_seeds must be unique")
        return self


class ExecutionRunSpec(BaseModel):
    """One cell in the paired task-by-graph-by-seed execution matrix."""

    model_config = ConfigDict(frozen=True)

    run_spec_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    experiment_seed: int
    assignment_seed: int
    condition: RunCondition
    attack_node: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_condition(self) -> ExecutionRunSpec:
        if self.condition is RunCondition.CLEAN and self.attack_node is not None:
            raise ValueError("clean run specs cannot contain an attack node")
        if self.condition is RunCondition.ATTACK and self.attack_node is None:
            raise ValueError("attack run specs require an attack node")
        return self


class BatchExecutionManifest(BaseModel):
    """Immutable identity for a complete intended execution matrix."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    runner_version: str = BATCH_EXECUTION_VERSION
    config: BatchExecutionConfig
    execution_settings: ExecutionSettings
    prompt_version: str = PROMPT_VERSION
    node_count: int = Field(ge=2)
    readout_node: int = Field(ge=0)
    max_rounds: int = Field(ge=1)
    task_ids: tuple[str, ...]
    graph_ids: tuple[str, ...]
    task_collection_fingerprint: str = Field(min_length=64, max_length=64)
    graph_collection_fingerprint: str = Field(min_length=64, max_length=64)
    round_zero_fingerprint: str = Field(min_length=64, max_length=64)
    round_zero_index_fingerprint: str = Field(min_length=64, max_length=64)
    adversarial_answers_fingerprint: str = Field(min_length=64, max_length=64)
    plan_fingerprint: str = Field(min_length=64, max_length=64)
    expected_run_count: int = Field(ge=1)


class BatchDisposition(StrEnum):
    GENERATED = "generated"
    CACHED = "cached"


class BatchExecutionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_spec_id: str
    disposition: BatchDisposition
    trace_path: str
    run_id: str
    final_answer_state: AnswerState
    final_parsed_answer: str | None = None
    model_calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class BatchExecutionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_runs: int = Field(ge=1)
    completed_runs: int = Field(ge=0)
    generated_runs: int = Field(ge=0)
    cached_runs: int = Field(ge=0)
    clean_runs: int = Field(ge=0)
    attack_runs: int = Field(ge=0)
    trace_model_calls: int = Field(ge=0)
    new_model_calls: int = Field(ge=0)
    known_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    input_tokens_complete: bool
    output_tokens_complete: bool


class StoredExecutionRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_spec: ExecutionRunSpec
    trace_fingerprint: str = Field(min_length=64, max_length=64)
    trace: RunTrace


class RoundZeroRecordReference(BaseModel):
    """Compact immutable pointer to one cached independent response."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(min_length=1)
    request_fingerprint: str = Field(min_length=64, max_length=64)
    task_id: str = Field(min_length=1)
    replica_slot: int = Field(ge=0)
    experiment_seed: int
    generation_seed: int
    prompt_version: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    returned_model: str | None = None

    @classmethod
    def from_record(cls, record: RoundZeroRecord) -> RoundZeroRecordReference:
        return cls(
            record_id=record.record_id,
            request_fingerprint=record.request_fingerprint,
            task_id=record.task_id,
            replica_slot=record.replica_slot,
            experiment_seed=record.experiment_seed,
            generation_seed=record.generation_seed,
            prompt_version=record.prompt_version,
            requested_model=record.requested_model,
            returned_model=record.returned_model,
        )


class BatchExecutionConflictError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def content_fingerprint(value: Any) -> str:
    """Return the canonical SHA-256 fingerprint used by batch artifacts."""

    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonl(values: tuple[BaseModel, ...]) -> str:
    return "".join(value.model_dump_json() + "\n" for value in values)


def _atomic_write_text(path: Path, content: str) -> None:
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


class BatchExecutionStore:
    """Conflict-safe storage for a manifest, immutable plan, and atomic trace files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.plan_path = self.root / "plan.jsonl"
        self.inputs_dir = self.root / "inputs"
        self.tasks_path = self.inputs_dir / "tasks.jsonl"
        self.graphs_path = self.inputs_dir / "graphs.jsonl"
        self.round_zero_index_path = self.inputs_dir / "round_zero_index.jsonl"
        self.adversarial_answers_path = self.inputs_dir / "adversarial_answers.jsonl"
        self.traces_dir = self.root / "traces"

    def initialize(
        self,
        *,
        manifest: BatchExecutionManifest,
        plan: tuple[ExecutionRunSpec, ...],
        tasks: tuple[TaskInstance, ...],
        graphs: tuple[GraphSpec, ...],
        round_zero_references: tuple[RoundZeroRecordReference, ...],
        adversarial_answers: tuple[AdversarialAnswer, ...],
    ) -> None:
        manifest_text = json.dumps(
            manifest.model_dump(mode="json"), indent=2, sort_keys=True
        ) + "\n"
        artifacts = (
            (self.manifest_path, manifest_text, "batch manifest"),
            (self.plan_path, _jsonl(plan), "batch plan"),
            (self.tasks_path, _jsonl(tasks), "task snapshot"),
            (self.graphs_path, _jsonl(graphs), "graph snapshot"),
            (
                self.round_zero_index_path,
                _jsonl(round_zero_references),
                "round-zero index",
            ),
            (
                self.adversarial_answers_path,
                _jsonl(adversarial_answers),
                "adversarial-answer snapshot",
            ),
        )
        for path, content, label in artifacts:
            if path.exists() and path.read_text(encoding="utf-8") != content:
                raise BatchExecutionConflictError(
                    f"{label} differs; use a new output directory"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        for path, content, _ in artifacts:
            if not path.exists():
                _atomic_write_text(path, content)

    def trace_path(self, spec: ExecutionRunSpec) -> Path:
        return self.traces_dir / f"{spec.run_spec_id}.json"

    def load(self, spec: ExecutionRunSpec) -> StoredExecutionRun | None:
        path = self.trace_path(spec)
        if not path.exists():
            return None
        try:
            stored = StoredExecutionRun.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise BatchExecutionConflictError(f"invalid cached trace at {path}") from exc
        if stored.run_spec != spec:
            raise BatchExecutionConflictError(f"cached run spec differs at {path}")
        if stored.trace_fingerprint != content_fingerprint(stored.trace):
            raise BatchExecutionConflictError(f"cached trace fingerprint differs at {path}")
        return stored

    def save(self, spec: ExecutionRunSpec, trace: RunTrace) -> Path:
        path = self.trace_path(spec)
        stored = StoredExecutionRun(
            run_spec=spec,
            trace_fingerprint=content_fingerprint(trace),
            trace=trace,
        )
        if path.exists():
            existing = self.load(spec)
            if existing != stored:
                raise BatchExecutionConflictError(f"cached trace differs at {path}")
            return path
        _atomic_write_json(path, stored)
        return path

    def write_results(
        self,
        *,
        outcomes: tuple[BatchExecutionOutcome, ...],
        summary: BatchExecutionSummary,
    ) -> None:
        _atomic_write_json(self.root / "outcomes.json", outcomes)
        _atomic_write_json(self.root / "summary.json", summary)


class BatchExecutionRunner:
    """Execute and resume a complete paired matrix without silently skipping cells."""

    def __init__(
        self,
        engine: SynchronousExecutionEngine,
        *,
        config: BatchExecutionConfig,
        output_dir: str | Path,
    ) -> None:
        self.engine = engine
        self.config = config
        self.store = BatchExecutionStore(output_dir)

    def run(
        self,
        *,
        tasks: tuple[TaskInstance, ...],
        graphs: tuple[GraphSpec, ...],
        round_zero_records: tuple[RoundZeroRecord, ...],
        adversarial_answers: dict[str, AdversarialAnswer] | None = None,
    ) -> tuple[tuple[BatchExecutionOutcome, ...], BatchExecutionSummary]:
        adversarial = adversarial_answers or {}
        record_index, node_count, readout_node, max_rounds = self._preflight(
            tasks=tasks,
            graphs=graphs,
            round_zero_records=round_zero_records,
            adversarial_answers=adversarial,
        )
        assignments = {
            seed: build_initial_state_assignment(
                node_count=node_count,
                assignment_seed=seed,
            )
            for seed in self.config.assignment_seeds
        }
        plan = self._build_plan(tasks=tasks, graphs=graphs)
        selected_records = tuple(
            record_index[(task.task_id, experiment_seed, replica_slot)]
            for task in tasks
            for experiment_seed in self.config.experiment_seeds
            for replica_slot in range(node_count)
        )
        selected_adversarial = tuple(
            adversarial[task.task_id] for task in tasks
        ) if self.config.include_attacks else ()
        round_zero_references = tuple(
            RoundZeroRecordReference.from_record(record) for record in selected_records
        )
        manifest = BatchExecutionManifest(
            config=self.config,
            execution_settings=self.engine.settings,
            node_count=node_count,
            readout_node=readout_node,
            max_rounds=max_rounds,
            task_ids=tuple(task.task_id for task in tasks),
            graph_ids=tuple(graph.graph_id for graph in graphs),
            task_collection_fingerprint=task_collection_fingerprint(tasks),
            graph_collection_fingerprint=graph_collection_fingerprint(graphs),
            round_zero_fingerprint=content_fingerprint(selected_records),
            round_zero_index_fingerprint=content_fingerprint(round_zero_references),
            adversarial_answers_fingerprint=content_fingerprint(selected_adversarial),
            plan_fingerprint=content_fingerprint(plan),
            expected_run_count=len(plan),
        )
        self.store.initialize(
            manifest=manifest,
            plan=plan,
            tasks=tasks,
            graphs=graphs,
            round_zero_references=round_zero_references,
            adversarial_answers=selected_adversarial,
        )

        task_by_id = {task.task_id: task for task in tasks}
        graph_by_id = {graph.graph_id: graph for graph in graphs}
        outcomes: list[BatchExecutionOutcome] = []
        for spec in plan:
            assignment = assignments[spec.assignment_seed]
            cached = self.store.load(spec)
            if cached is not None:
                self._validate_trace(
                    spec=spec,
                    trace=cached.trace,
                    assignment=assignment,
                    graph=graph_by_id[spec.graph_id],
                    expected_records=record_index,
                    adversarial_answer=(
                        adversarial[spec.task_id]
                        if spec.condition is RunCondition.ATTACK
                        else None
                    ),
                )
                trace = cached.trace
                disposition = BatchDisposition.CACHED
            else:
                task = task_by_id[spec.task_id]
                trace = self.engine.run(
                    graph=graph_by_id[spec.graph_id],
                    task=task,
                    condition=spec.condition,
                    seed=spec.experiment_seed,
                    attack_node=spec.attack_node,
                    adversarial_answer=(
                        adversarial[spec.task_id]
                        if spec.condition is RunCondition.ATTACK
                        else None
                    ),
                    round_zero_records=selected_records,
                    initial_assignment=assignment,
                )
                self._validate_trace(
                    spec=spec,
                    trace=trace,
                    assignment=assignment,
                    graph=graph_by_id[spec.graph_id],
                    expected_records=record_index,
                    adversarial_answer=(
                        adversarial[spec.task_id]
                        if spec.condition is RunCondition.ATTACK
                        else None
                    ),
                )
                self.store.save(spec, trace)
                disposition = BatchDisposition.GENERATED
            outcomes.append(
                BatchExecutionOutcome(
                    run_spec_id=spec.run_spec_id,
                    disposition=disposition,
                    trace_path=str(self.store.trace_path(spec).resolve()),
                    run_id=trace.run_id,
                    final_answer_state=trace.final_answer_state,
                    final_parsed_answer=trace.final_parsed_answer,
                    model_calls=trace.total_model_calls,
                    input_tokens=trace.total_input_tokens,
                    output_tokens=trace.total_output_tokens,
                )
            )

        outcomes_tuple = tuple(outcomes)
        summary = self._summarize(plan=plan, outcomes=outcomes_tuple)
        if summary.completed_runs != manifest.expected_run_count:
            raise RuntimeError("batch completed without the full planned run matrix")
        self.store.write_results(outcomes=outcomes_tuple, summary=summary)
        return outcomes_tuple, summary

    def _build_plan(
        self,
        *,
        tasks: tuple[TaskInstance, ...],
        graphs: tuple[GraphSpec, ...],
    ) -> tuple[ExecutionRunSpec, ...]:
        specs: list[ExecutionRunSpec] = []
        for task in tasks:
            for graph in graphs:
                for experiment_seed in self.config.experiment_seeds:
                    for assignment_seed in self.config.assignment_seeds:
                        conditions: list[tuple[RunCondition, int | None]] = [
                            (RunCondition.CLEAN, None)
                        ]
                        if self.config.include_attacks:
                            conditions.extend(
                                (RunCondition.ATTACK, node_id)
                                for node_id in range(graph.node_count)
                                if node_id != graph.readout_node
                            )
                        for condition, attack_node in conditions:
                            run_spec_id = stable_id(
                                "run-spec",
                                BATCH_EXECUTION_VERSION,
                                task.task_id,
                                graph.graph_id,
                                experiment_seed,
                                assignment_seed,
                                condition.value,
                                attack_node,
                            )
                            specs.append(
                                ExecutionRunSpec(
                                    run_spec_id=run_spec_id,
                                    task_id=task.task_id,
                                    graph_id=graph.graph_id,
                                    experiment_seed=experiment_seed,
                                    assignment_seed=assignment_seed,
                                    condition=condition,
                                    attack_node=attack_node,
                                )
                            )
        if len({spec.run_spec_id for spec in specs}) != len(specs):
            raise ValueError("run plan contains duplicate identities")
        return tuple(specs)

    def _preflight(
        self,
        *,
        tasks: tuple[TaskInstance, ...],
        graphs: tuple[GraphSpec, ...],
        round_zero_records: tuple[RoundZeroRecord, ...],
        adversarial_answers: dict[str, AdversarialAnswer],
    ) -> tuple[dict[tuple[str, int, int], RoundZeroRecord], int, int, int]:
        if not tasks:
            raise ValueError("batch execution requires at least one task")
        if not graphs:
            raise ValueError("batch execution requires at least one graph")
        task_ids = tuple(task.task_id for task in tasks)
        graph_ids = tuple(graph.graph_id for graph in graphs)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task IDs must be unique")
        if len(set(graph_ids)) != len(graph_ids):
            raise ValueError("graph IDs must be unique")

        node_counts = {graph.node_count for graph in graphs}
        readout_nodes = {graph.readout_node for graph in graphs}
        max_round_values = {graph.max_rounds for graph in graphs}
        if len(node_counts) != 1 or len(readout_nodes) != 1 or len(max_round_values) != 1:
            raise ValueError(
                "one batch must contain a single node_count, readout_node, and max_rounds stratum"
            )
        node_count = next(iter(node_counts))
        readout_node = next(iter(readout_nodes))
        max_rounds = next(iter(max_round_values))

        record_index: dict[tuple[str, int, int], RoundZeroRecord] = {}
        for record in round_zero_records:
            key = (record.task_id, record.experiment_seed, record.replica_slot)
            if key in record_index:
                raise ValueError(f"duplicate round-zero record for {key}")
            record_index[key] = record
        missing_records = [
            (task_id, experiment_seed, replica_slot)
            for task_id in task_ids
            for experiment_seed in self.config.experiment_seeds
            for replica_slot in range(node_count)
            if (task_id, experiment_seed, replica_slot) not in record_index
        ]
        if missing_records:
            preview = ", ".join(map(str, missing_records[:5]))
            raise ValueError(f"round-zero records are missing: {preview}")
        selected_records = [
            record_index[(task_id, experiment_seed, replica_slot)]
            for task_id in task_ids
            for experiment_seed in self.config.experiment_seeds
            for replica_slot in range(node_count)
        ]
        if any(record.prompt_version != PROMPT_VERSION for record in selected_records):
            raise ValueError("round-zero prompt version differs from the execution prompt")
        if any(
            record.requested_model != self.config.requested_model
            for record in selected_records
        ):
            raise ValueError("round-zero requested model differs from the batch model")
        if self.config.expected_returned_model is not None and any(
            record.returned_model != self.config.expected_returned_model
            for record in selected_records
        ):
            raise ValueError("round-zero returned model differs from the pinned batch model")

        if self.config.include_attacks:
            missing_answers = [
                task_id for task_id in task_ids if task_id not in adversarial_answers
            ]
            if missing_answers:
                raise ValueError(
                    "adversarial answers are missing for tasks: " + ", ".join(missing_answers)
                )
            task_by_id = {task.task_id: task for task in tasks}
            for task_id in task_ids:
                answer = adversarial_answers[task_id]
                if answer.task_id != task_id or not answer.accepted:
                    raise ValueError(f"adversarial answer is invalid for {task_id}")
                if answer.target_answer == task_by_id[task_id].reference_answer:
                    raise ValueError(
                        f"adversarial answer equals the reference answer for {task_id}"
                    )
        return record_index, node_count, readout_node, max_rounds

    def _validate_trace(
        self,
        *,
        spec: ExecutionRunSpec,
        trace: RunTrace,
        assignment: InitialStateAssignment,
        graph: GraphSpec,
        expected_records: dict[tuple[str, int, int], RoundZeroRecord],
        adversarial_answer: AdversarialAnswer | None,
    ) -> None:
        if (
            trace.task_id != spec.task_id
            or trace.graph_id != spec.graph_id
            or trace.seed != spec.experiment_seed
            or trace.condition is not spec.condition
            or trace.attack_node != spec.attack_node
        ):
            raise BatchExecutionConflictError(
                f"trace identity differs for run spec {spec.run_spec_id}"
            )
        expected_adversarial_fingerprint = (
            stable_fingerprint(adversarial_answer.model_dump_json())
            if adversarial_answer is not None
            else None
        )
        expected_target = (
            adversarial_answer.target_answer if adversarial_answer is not None else None
        )
        if (
            trace.adversarial_answer_fingerprint != expected_adversarial_fingerprint
            or trace.target_answer != expected_target
        ):
            raise BatchExecutionConflictError(
                f"trace adversarial answer differs for run spec {spec.run_spec_id}"
            )
        if (
            trace.prompt_version != PROMPT_VERSION
            or trace.execution_settings != self.engine.settings
        ):
            raise BatchExecutionConflictError(
                f"trace execution protocol differs for run spec {spec.run_spec_id}"
            )
        if (
            trace.initial_assignment_id != assignment.assignment_id
            or trace.initial_assignment_seed != assignment.assignment_seed
            or trace.structural_node_to_replica != assignment.structural_node_to_replica
        ):
            raise BatchExecutionConflictError(
                f"trace initial assignment differs for run spec {spec.run_spec_id}"
            )
        round_zero_turns = [turn for turn in trace.turns if turn.round_index == 0]
        if len(round_zero_turns) != graph.node_count:
            raise BatchExecutionConflictError(
                f"trace round zero is incomplete for run spec {spec.run_spec_id}"
            )
        for turn in round_zero_turns:
            if spec.condition is RunCondition.ATTACK and turn.node_id == spec.attack_node:
                if not turn.metadata.get("attack_replay"):
                    raise BatchExecutionConflictError(
                        f"attacker round zero was not replayed for {spec.run_spec_id}"
                    )
                continue
            replica_slot = assignment.replica_for_node(turn.node_id)
            expected = expected_records[
                (spec.task_id, spec.experiment_seed, replica_slot)
            ]
            if turn.metadata.get("round_zero_record_id") != expected.record_id:
                raise BatchExecutionConflictError(
                    f"round-zero record differs for run spec {spec.run_spec_id}"
                )
        if self.config.expected_returned_model is not None:
            online_models = {
                turn.model_name
                for turn in trace.turns
                if turn.metadata.get("generator_called")
            }
            if online_models and online_models != {self.config.expected_returned_model}:
                raise BatchExecutionConflictError(
                    f"returned model differs for run spec {spec.run_spec_id}: {online_models}"
                )

    @staticmethod
    def _summarize(
        *,
        plan: tuple[ExecutionRunSpec, ...],
        outcomes: tuple[BatchExecutionOutcome, ...],
    ) -> BatchExecutionSummary:
        return BatchExecutionSummary(
            expected_runs=len(plan),
            completed_runs=len(outcomes),
            generated_runs=sum(
                outcome.disposition is BatchDisposition.GENERATED for outcome in outcomes
            ),
            cached_runs=sum(
                outcome.disposition is BatchDisposition.CACHED for outcome in outcomes
            ),
            clean_runs=sum(spec.condition is RunCondition.CLEAN for spec in plan),
            attack_runs=sum(spec.condition is RunCondition.ATTACK for spec in plan),
            trace_model_calls=sum(outcome.model_calls for outcome in outcomes),
            new_model_calls=sum(
                outcome.model_calls
                for outcome in outcomes
                if outcome.disposition is BatchDisposition.GENERATED
            ),
            known_input_tokens=sum(outcome.input_tokens or 0 for outcome in outcomes),
            known_output_tokens=sum(outcome.output_tokens or 0 for outcome in outcomes),
            input_tokens_complete=all(
                outcome.input_tokens is not None for outcome in outcomes
            ),
            output_tokens_complete=all(
                outcome.output_tokens is not None for outcome in outcomes
            ),
        )
