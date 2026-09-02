"""Dependency-aware global scheduler for node-round MAS generation jobs."""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.scalable_round_zero import SCALABLE_PROTOCOL_VERSION
from topology_mas.execution.schemas import TextGenerationResult
from topology_mas.execution.seeding import stable_id
from topology_mas.models import GraphSpec
from topology_mas.topology.graph_ops import build_causal_schedule


class ReadyJobSpec(BaseModel):
    """One globally schedulable node update and its exact causal dependencies."""

    model_config = ConfigDict(frozen=True)

    protocol_version: str = SCALABLE_PROTOCOL_VERSION
    job_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    node_id: int = Field(ge=0)
    distance_to_readout: int = Field(ge=0)
    dependency_ids: tuple[str, ...] = ()
    estimated_prompt_tokens: int = Field(default=0, ge=0)
    downstream_unlock_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dependencies(self) -> ReadyJobSpec:
        if len(set(self.dependency_ids)) != len(self.dependency_ids):
            raise ValueError("dependency_ids must be unique")
        if self.job_id in self.dependency_ids:
            raise ValueError("a job cannot depend on itself")
        return self


class ReadySchedulerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    protocol_version: str = SCALABLE_PROTOCOL_VERSION
    workers_per_backend: int = Field(default=1, ge=1)
    prompt_bucket_boundaries: tuple[int, ...] = (
        8192,
        16384,
        32768,
        65536,
    )
    priority_policy: Literal["readout_distance_then_round"] = (
        "readout_distance_then_round"
    )

    @model_validator(mode="after")
    def validate_buckets(self) -> ReadySchedulerConfig:
        if any(value < 1 for value in self.prompt_bucket_boundaries):
            raise ValueError("prompt bucket boundaries must be positive")
        if tuple(sorted(set(self.prompt_bucket_boundaries))) != self.prompt_bucket_boundaries:
            raise ValueError("prompt bucket boundaries must be strictly increasing")
        return self


class ReadyJobOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["precompleted", "succeeded", "failed", "blocked"]
    backend_index: int | None = Field(default=None, ge=0)
    prompt_length_bucket: int = Field(ge=0)
    ready_wait_ms: float = Field(ge=0.0)
    execution_ms: float = Field(ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    summary_tokens: int | None = Field(default=None, ge=0)
    prefill_ms: float | None = Field(default=None, ge=0.0)
    decode_ms: float | None = Field(default=None, ge=0.0)
    backend_batch_size: int | None = Field(default=None, ge=1)
    error_type: str | None = None
    error_message: str | None = None


class ReadySchedulerSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_jobs: int = Field(ge=0)
    precompleted_jobs: int = Field(ge=0)
    succeeded_jobs: int = Field(ge=0)
    failed_jobs: int = Field(ge=0)
    blocked_jobs: int = Field(ge=0)
    known_input_tokens: int = Field(ge=0)
    known_output_tokens: int = Field(ge=0)
    input_tokens_complete: bool
    output_tokens_complete: bool
    wall_time_ms: float = Field(ge=0.0)


class ReadySchedulerResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    outcomes: tuple[ReadyJobOutcome, ...]
    outputs: dict[str, TextGenerationResult]
    summary: ReadySchedulerSummary


class ReadyRunCostSummary(BaseModel):
    """Cost and failure audit aggregated at task-graph-run granularity."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    physical_calls: int = Field(ge=0)
    round_zero_pool_hits: int = Field(ge=0)
    round_zero_new_generations: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    summary_tokens: int = Field(ge=0)
    maximum_prompt_tokens: int | None = Field(default=None, ge=0)
    mean_prompt_tokens: float | None = Field(default=None, ge=0.0)
    queue_wait_ms: float = Field(ge=0.0)
    execution_ms: float = Field(ge=0.0)
    unparsed_count: int = Field(ge=0)
    length_stop_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)


class ReadySchedulerPlanError(ValueError):
    pass


JobExecutor = Callable[
    [ReadyJobSpec, Mapping[str, TextGenerationResult], TextGenerator],
    TextGenerationResult,
]


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_positive_int(value: object) -> int | None:
    parsed = _optional_nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _optional_nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _bucket(token_count: int, boundaries: tuple[int, ...]) -> int:
    for boundary in boundaries:
        if token_count <= boundary:
            return boundary
    return boundaries[-1] * 2


def build_causal_ready_jobs(
    *,
    run_id: str,
    task_id: str,
    graph: GraphSpec,
    estimated_prompt_tokens: Mapping[tuple[int, int], int] | None = None,
) -> tuple[ReadyJobSpec, ...]:
    """Build the node-round DAG after the existing causal-cone pruning rule."""

    estimates = estimated_prompt_tokens or {}
    schedule = build_causal_schedule(graph)
    active = {
        (round_index, node_id)
        for round_index, nodes in enumerate(schedule.active_nodes_by_round)
        for node_id in nodes
    }
    incoming_by_round_node: dict[tuple[int, int], list[int]] = defaultdict(list)
    for send_round, edges in enumerate(schedule.active_edges_by_round):
        for edge in edges:
            incoming_by_round_node[(send_round + 1, edge.target)].append(edge.source)

    def job_id(round_index: int, node_id: int) -> str:
        return stable_id("ready-job", run_id, round_index, node_id)

    jobs: list[ReadyJobSpec] = []
    for round_index, nodes in enumerate(schedule.active_nodes_by_round):
        for node_id in nodes:
            dependencies: list[str] = []
            if round_index > 0 and (round_index - 1, node_id) in active:
                dependencies.append(job_id(round_index - 1, node_id))
            dependencies.extend(
                job_id(round_index - 1, sender)
                for sender in sorted(incoming_by_round_node[(round_index, node_id)])
                if (round_index - 1, sender) in active
            )
            jobs.append(
                ReadyJobSpec(
                    job_id=job_id(round_index, node_id),
                    run_id=run_id,
                    task_id=task_id,
                    graph_id=graph.graph_id,
                    round_index=round_index,
                    node_id=node_id,
                    distance_to_readout=schedule.distances_to_readout[node_id],
                    dependency_ids=tuple(dict.fromkeys(dependencies)),
                    estimated_prompt_tokens=estimates.get((round_index, node_id), 0),
                    metadata={
                        "effective_horizon": schedule.effective_horizon,
                        "causal_cone_pruning": True,
                    },
                )
            )
    unlock_counts: dict[str, int] = defaultdict(int)
    for job in jobs:
        for dependency_id in job.dependency_ids:
            unlock_counts[dependency_id] += 1
    return tuple(
        job.model_copy(
            update={"downstream_unlock_count": unlock_counts[job.job_id]}
        )
        for job in jobs
    )


class GlobalReadyScheduler:
    """Execute all task/graph/node/round jobs from one global READY heap.

    Each backend is exposed through ``workers_per_backend`` slots.  Concurrent
    requests let a vLLM server perform its own continuous/dynamic batching; prompt
    buckets keep similarly sized ready requests adjacent without delaying jobs to
    manufacture a batch.
    """

    def __init__(
        self,
        backends: tuple[TextGenerator, ...],
        *,
        config: ReadySchedulerConfig | None = None,
    ) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self.config = config or ReadySchedulerConfig()

    def run(
        self,
        jobs: tuple[ReadyJobSpec, ...],
        *,
        execute_job: JobExecutor,
        precompleted: Mapping[str, TextGenerationResult] | None = None,
    ) -> ReadySchedulerResult:
        started = time.perf_counter()
        precompleted_outputs = dict(precompleted or {})
        job_by_id = {job.job_id: job for job in jobs}
        if len(job_by_id) != len(jobs):
            raise ReadySchedulerPlanError("job IDs must be unique")
        unknown_precompleted = set(precompleted_outputs) - set(job_by_id)
        if unknown_precompleted:
            raise ReadySchedulerPlanError("precompleted output refers to an unknown job")
        for job in jobs:
            unknown = set(job.dependency_ids) - set(job_by_id)
            if unknown:
                raise ReadySchedulerPlanError(
                    f"job {job.job_id} has unknown dependencies: {sorted(unknown)}"
                )
        self._validate_acyclic(jobs)

        dependents: dict[str, list[str]] = defaultdict(list)
        remaining = {job.job_id: len(job.dependency_ids) for job in jobs}
        for job in jobs:
            for dependency in job.dependency_ids:
                dependents[dependency].append(job.job_id)

        outputs: dict[str, TextGenerationResult] = dict(precompleted_outputs)
        statuses: dict[str, str] = {
            job_id: "precompleted" for job_id in precompleted_outputs
        }
        outcomes: dict[str, ReadyJobOutcome] = {}
        ready_since: dict[str, float] = {}
        ready_heap: list[tuple[tuple[int, int, int, int, str], str]] = []

        for job_id, output in precompleted_outputs.items():
            job = job_by_id[job_id]
            outcomes[job_id] = ReadyJobOutcome(
                job_id=job_id,
                status="precompleted",
                prompt_length_bucket=_bucket(
                    job.estimated_prompt_tokens, self.config.prompt_bucket_boundaries
                ),
                ready_wait_ms=0.0,
                execution_ms=0.0,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                finish_reason=output.finish_reason,
                summary_tokens=_optional_nonnegative_int(
                    output.metadata.get("public_output_tokens")
                ),
                prefill_ms=_optional_nonnegative_float(
                    output.metadata.get("prefill_ms")
                ),
                decode_ms=_optional_nonnegative_float(
                    output.metadata.get("decode_ms")
                ),
                backend_batch_size=_optional_positive_int(
                    output.metadata.get("batch_size")
                ),
            )

        for job_id in precompleted_outputs:
            for dependent_id in dependents[job_id]:
                remaining[dependent_id] -= 1

        def enqueue(job_id: str) -> None:
            if job_id in statuses:
                return
            job = job_by_id[job_id]
            bucket = _bucket(
                job.estimated_prompt_tokens, self.config.prompt_bucket_boundaries
            )
            priority = (
                -job.downstream_unlock_count,
                job.distance_to_readout,
                job.round_index,
                bucket,
                job.job_id,
            )
            ready_since[job_id] = time.perf_counter()
            statuses[job_id] = "ready"
            heapq.heappush(ready_heap, (priority, job_id))

        for job in jobs:
            if remaining[job.job_id] == 0:
                enqueue(job.job_id)

        slots = [
            backend_index
            for backend_index in range(len(self.backends))
            for _ in range(self.config.workers_per_backend)
        ]
        available_slots = list(slots)
        running: dict[Future[TextGenerationResult], tuple[str, int, float, float]] = {}

        def run_one(job: ReadyJobSpec, backend_index: int) -> TextGenerationResult:
            dependencies = {dependency: outputs[dependency] for dependency in job.dependency_ids}
            return execute_job(job, dependencies, self.backends[backend_index])

        with ThreadPoolExecutor(max_workers=len(slots)) as executor:
            while len(outcomes) < len(jobs):
                while ready_heap and available_slots:
                    _, job_id = heapq.heappop(ready_heap)
                    statuses[job_id] = "running"
                    backend_index = available_slots.pop(0)
                    submitted = time.perf_counter()
                    future = executor.submit(run_one, job_by_id[job_id], backend_index)
                    running[future] = (
                        job_id,
                        backend_index,
                        ready_since[job_id],
                        submitted,
                    )
                if not running:
                    unresolved = set(job_by_id) - set(outcomes)
                    if unresolved:
                        raise ReadySchedulerPlanError(
                            f"scheduler stalled with unresolved jobs: {sorted(unresolved)}"
                        )
                    break
                done, _ = wait(tuple(running), return_when="FIRST_COMPLETED")
                for future in done:
                    job_id, backend_index, became_ready, submitted = running.pop(future)
                    available_slots.append(backend_index)
                    available_slots.sort()
                    finished = time.perf_counter()
                    job = job_by_id[job_id]
                    bucket = _bucket(
                        job.estimated_prompt_tokens,
                        self.config.prompt_bucket_boundaries,
                    )
                    try:
                        output = future.result()
                    except Exception as exc:  # persist failure while unrelated jobs continue
                        statuses[job_id] = "failed"
                        outcomes[job_id] = ReadyJobOutcome(
                            job_id=job_id,
                            status="failed",
                            backend_index=backend_index,
                            prompt_length_bucket=bucket,
                            ready_wait_ms=(submitted - became_ready) * 1000.0,
                            execution_ms=(finished - submitted) * 1000.0,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    else:
                        outputs[job_id] = output
                        statuses[job_id] = "succeeded"
                        outcomes[job_id] = ReadyJobOutcome(
                            job_id=job_id,
                            status="succeeded",
                            backend_index=backend_index,
                            prompt_length_bucket=bucket,
                            ready_wait_ms=(submitted - became_ready) * 1000.0,
                            execution_ms=(finished - submitted) * 1000.0,
                            input_tokens=output.input_tokens,
                            output_tokens=output.output_tokens,
                            finish_reason=output.finish_reason,
                            summary_tokens=_optional_nonnegative_int(
                                output.metadata.get("public_output_tokens")
                            ),
                            prefill_ms=_optional_nonnegative_float(
                                output.metadata.get("prefill_ms")
                            ),
                            decode_ms=_optional_nonnegative_float(
                                output.metadata.get("decode_ms")
                            ),
                            backend_batch_size=_optional_positive_int(
                                output.metadata.get("batch_size")
                            ),
                        )
                    self._release_dependents(
                        job_id=job_id,
                        dependents=dependents,
                        remaining=remaining,
                        statuses=statuses,
                        outcomes=outcomes,
                        job_by_id=job_by_id,
                        enqueue=enqueue,
                    )

        ordered_outcomes = tuple(outcomes[job.job_id] for job in jobs)
        known_inputs = sum(outcome.input_tokens or 0 for outcome in ordered_outcomes)
        known_outputs = sum(outcome.output_tokens or 0 for outcome in ordered_outcomes)
        terminal = {status: sum(item.status == status for item in ordered_outcomes) for status in (
            "precompleted", "succeeded", "failed", "blocked"
        )}
        successful = tuple(
            outcome
            for outcome in ordered_outcomes
            if outcome.status in {"precompleted", "succeeded"}
        )
        return ReadySchedulerResult(
            outcomes=ordered_outcomes,
            outputs=outputs,
            summary=ReadySchedulerSummary(
                total_jobs=len(jobs),
                precompleted_jobs=terminal["precompleted"],
                succeeded_jobs=terminal["succeeded"],
                failed_jobs=terminal["failed"],
                blocked_jobs=terminal["blocked"],
                known_input_tokens=known_inputs,
                known_output_tokens=known_outputs,
                input_tokens_complete=all(item.input_tokens is not None for item in successful),
                output_tokens_complete=all(item.output_tokens is not None for item in successful),
                wall_time_ms=(time.perf_counter() - started) * 1000.0,
            ),
        )

    def _release_dependents(
        self,
        *,
        job_id: str,
        dependents: Mapping[str, list[str]],
        remaining: dict[str, int],
        statuses: dict[str, str],
        outcomes: dict[str, ReadyJobOutcome],
        job_by_id: Mapping[str, ReadyJobSpec],
        enqueue: Callable[[str], None],
    ) -> None:
        queue = list(dependents[job_id])
        while queue:
            dependent_id = queue.pop(0)
            remaining[dependent_id] -= 1
            if remaining[dependent_id] != 0 or dependent_id in statuses:
                continue
            dependency_statuses = [
                statuses[dependency]
                for dependency in job_by_id[dependent_id].dependency_ids
            ]
            if any(status in {"failed", "blocked"} for status in dependency_statuses):
                statuses[dependent_id] = "blocked"
                job = job_by_id[dependent_id]
                outcomes[dependent_id] = ReadyJobOutcome(
                    job_id=dependent_id,
                    status="blocked",
                    prompt_length_bucket=_bucket(
                        job.estimated_prompt_tokens,
                        self.config.prompt_bucket_boundaries,
                    ),
                    ready_wait_ms=0.0,
                    execution_ms=0.0,
                    error_type="DependencyFailure",
                    error_message="one or more causal dependencies failed",
                )
                queue.extend(dependents[dependent_id])
            else:
                enqueue(dependent_id)

    @staticmethod
    def _validate_acyclic(jobs: tuple[ReadyJobSpec, ...]) -> None:
        remaining = {job.job_id: len(job.dependency_ids) for job in jobs}
        dependents: dict[str, list[str]] = defaultdict(list)
        for job in jobs:
            for dependency in job.dependency_ids:
                dependents[dependency].append(job.job_id)
        queue = [job_id for job_id, count in remaining.items() if count == 0]
        visited = 0
        while queue:
            job_id = queue.pop()
            visited += 1
            for dependent in dependents[job_id]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    queue.append(dependent)
        if visited != len(jobs):
            raise ReadySchedulerPlanError("job dependency graph contains a cycle")


def aggregate_ready_run_costs(
    *,
    jobs: tuple[ReadyJobSpec, ...],
    result: ReadySchedulerResult,
) -> tuple[ReadyRunCostSummary, ...]:
    """Aggregate scheduler telemetry without treating missing provider fields as zero."""

    job_by_id = {job.job_id: job for job in jobs}
    outcomes_by_run: dict[str, list[ReadyJobOutcome]] = defaultdict(list)
    for outcome in result.outcomes:
        outcomes_by_run[job_by_id[outcome.job_id].run_id].append(outcome)

    summaries: list[ReadyRunCostSummary] = []
    for run_id in sorted(outcomes_by_run):
        outcomes = outcomes_by_run[run_id]
        round_zero_pool_hits = sum(
            outcome.status == "precompleted"
            and job_by_id[outcome.job_id].round_index == 0
            for outcome in outcomes
        )
        round_zero_new = sum(
            outcome.status == "succeeded"
            and job_by_id[outcome.job_id].round_index == 0
            for outcome in outcomes
        )
        prompt_tokens = [
            outcome.input_tokens
            for outcome in outcomes
            if outcome.input_tokens is not None
        ]
        unparsed_count = 0
        for outcome in outcomes:
            output = result.outputs.get(outcome.job_id)
            if output is not None and output.metadata.get("raw_parsed_answer", False) is None:
                unparsed_count += 1
        summaries.append(
            ReadyRunCostSummary(
                run_id=run_id,
                physical_calls=sum(outcome.status == "succeeded" for outcome in outcomes),
                round_zero_pool_hits=round_zero_pool_hits,
                round_zero_new_generations=round_zero_new,
                input_tokens=sum(outcome.input_tokens or 0 for outcome in outcomes),
                output_tokens=sum(outcome.output_tokens or 0 for outcome in outcomes),
                summary_tokens=sum(outcome.summary_tokens or 0 for outcome in outcomes),
                maximum_prompt_tokens=max(prompt_tokens) if prompt_tokens else None,
                mean_prompt_tokens=(
                    sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None
                ),
                queue_wait_ms=sum(outcome.ready_wait_ms for outcome in outcomes),
                execution_ms=sum(outcome.execution_ms for outcome in outcomes),
                unparsed_count=unparsed_count,
                length_stop_count=sum(
                    outcome.finish_reason == "length" for outcome in outcomes
                ),
                failed_count=sum(outcome.status == "failed" for outcome in outcomes),
                blocked_count=sum(outcome.status == "blocked" for outcome in outcomes),
            )
        )
    return tuple(summaries)
