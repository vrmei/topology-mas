"""Versioned Round-0 pools and paired graph assignments for scalable MAS runs.

This module is deliberately separate from :mod:`round_zero`.  The historical
cache stores exactly one response per anonymous replica.  The scalable protocol
stores a larger, unfiltered response pool per task and records the later draw and
graph-dependent node assignment as first-class experimental artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from topology_mas.execution.answers import classify_numeric_answer, parse_numeric_answer
from topology_mas.execution.assignments import InitialStateAssignment
from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.round_zero import RoundZeroRecord
from topology_mas.execution.schemas import ChatMessage, TextGenerationRequest
from topology_mas.execution.seeding import stable_id, stable_integer
from topology_mas.models import AnswerState, GraphSpec, TaskInstance

SCALABLE_PROTOCOL_VERSION = "homogeneous-mas-scalable-v1"
ROUND_ZERO_POOL_SCHEMA_VERSION = "round-zero-pool-v1"

PromptBuilder = Callable[[TaskInstance], tuple[ChatMessage, ...]]
AnswerParser = Callable[[str, str | None], str | None]


def _canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class ScalableRoundZeroPoolConfig(BaseModel):
    """Frozen generation conditions for one task-indexed response pool."""

    model_config = ConfigDict(frozen=True)

    protocol_version: Literal["homogeneous-mas-scalable-v1"] = SCALABLE_PROTOCOL_VERSION
    schema_version: Literal["round-zero-pool-v1"] = ROUND_ZERO_POOL_SCHEMA_VERSION
    responses_per_task: int = Field(default=64, ge=2)
    base_seed: int = 0
    requested_model: str = Field(min_length=1)
    expected_returned_model: str | None = None
    prompt_version: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=-1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    max_output_tokens: int = Field(ge=1)


class ScalableRoundZeroPoolResponse(BaseModel):
    """One pool member.  Invalid and unparsed generations are retained."""

    model_config = ConfigDict(frozen=True)

    protocol_version: str = SCALABLE_PROTOCOL_VERSION
    pool_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    pool_response_id: str = Field(min_length=1)
    pool_slot: int = Field(ge=0)
    generation_seed: int
    raw_response: str
    parsed_answer: str | None = None
    answer_state: AnswerState
    output_tokens: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_model: str = Field(min_length=1)
    returned_model: str | None = None
    prompt_version: str = Field(min_length=1)
    prompt_messages: tuple[dict[str, str], ...]
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_hash(self) -> ScalableRoundZeroPoolResponse:
        expected = hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest()
        if self.content_hash != expected:
            raise ValueError("content_hash does not match raw_response")
        return self


class ScalableRoundZeroPoolManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    pool_version: str = Field(min_length=1)
    config: ScalableRoundZeroPoolConfig
    task_ids: tuple[str, ...] = Field(min_length=1)
    task_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    intended_response_count: int = Field(ge=1)


class RoundZeroDraw(BaseModel):
    """A graph-independent draw of ``n`` pool members for one task replicate."""

    model_config = ConfigDict(frozen=True)

    protocol_version: str = SCALABLE_PROTOCOL_VERSION
    pool_version: str = Field(min_length=1)
    draw_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    node_count: int = Field(ge=2)
    replicate_index: int = Field(ge=0)
    mode: Literal["pooled", "fresh_audit"]
    selection_seed: int
    selected_pool_response_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_selected_count(self) -> RoundZeroDraw:
        if self.mode == "pooled":
            if len(self.selected_pool_response_ids) != self.node_count:
                raise ValueError("pooled draws require exactly node_count response IDs")
            if len(set(self.selected_pool_response_ids)) != self.node_count:
                raise ValueError("pooled draws must sample without replacement")
        elif self.selected_pool_response_ids:
            raise ValueError("fresh audit draws cannot contain pooled response IDs")
        return self


class GraphRoundZeroAssignment(BaseModel):
    """Graph-specific permutation of a graph-independent Round-0 draw."""

    model_config = ConfigDict(frozen=True)

    protocol_version: str = SCALABLE_PROTOCOL_VERSION
    pool_version: str = Field(min_length=1)
    draw_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    graph_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignment_id: str = Field(min_length=1)
    assignment_seed: int
    node_to_pool_response_id: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_permutation(self) -> GraphRoundZeroAssignment:
        if len(set(self.node_to_pool_response_id)) != len(self.node_to_pool_response_id):
            raise ValueError("node assignment must not reuse a pool response")
        return self


class ScalableRoundZeroPoolConflictError(RuntimeError):
    pass


class ScalableRoundZeroPoolStore:
    """Atomic, resumable storage for one immutable response pool."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    def initialize(
        self,
        *,
        config: ScalableRoundZeroPoolConfig,
        tasks: tuple[TaskInstance, ...],
    ) -> ScalableRoundZeroPoolManifest:
        if not tasks:
            raise ValueError("at least one task is required")
        task_ids = tuple(task.task_id for task in tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task IDs must be unique")
        pool_version = stable_id(
            "r0-pool",
            config.protocol_version,
            config.model_dump_json(),
            _fingerprint([task.model_dump(mode="json") for task in tasks]),
        )
        manifest = ScalableRoundZeroPoolManifest(
            pool_version=pool_version,
            config=config,
            task_ids=task_ids,
            task_fingerprint=_fingerprint(
                [task.model_dump(mode="json") for task in tasks]
            ),
            intended_response_count=len(tasks) * config.responses_per_task,
        )
        if self.manifest_path.exists():
            existing = ScalableRoundZeroPoolManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if existing != manifest:
                raise ScalableRoundZeroPoolConflictError(
                    "pool manifest differs; use a new output directory"
                )
            return existing
        _atomic_write_json(self.manifest_path, manifest)
        return manifest

    def response_path(self, *, task_id: str, pool_slot: int) -> Path:
        safe_task = task_id.replace("/", "_").replace("\\", "_")
        return self.root / "responses" / safe_task / f"response_{pool_slot:04d}.json"

    def load(self, *, task_id: str, pool_slot: int) -> ScalableRoundZeroPoolResponse | None:
        path = self.response_path(task_id=task_id, pool_slot=pool_slot)
        if not path.exists():
            return None
        return ScalableRoundZeroPoolResponse.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def save(self, response: ScalableRoundZeroPoolResponse) -> Path:
        path = self.response_path(task_id=response.task_id, pool_slot=response.pool_slot)
        if path.exists():
            existing = self.load(task_id=response.task_id, pool_slot=response.pool_slot)
            if existing != response:
                raise ScalableRoundZeroPoolConflictError(f"response conflict at {path}")
            return path
        _atomic_write_json(path, response)
        return path

    def load_complete(
        self,
    ) -> tuple[ScalableRoundZeroPoolManifest, tuple[ScalableRoundZeroPoolResponse, ...]]:
        if not self.manifest_path.exists():
            raise ScalableRoundZeroPoolConflictError("pool manifest is missing")
        manifest = ScalableRoundZeroPoolManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        responses: list[ScalableRoundZeroPoolResponse] = []
        missing: list[str] = []
        for task_id in manifest.task_ids:
            for slot in range(manifest.config.responses_per_task):
                response = self.load(task_id=task_id, pool_slot=slot)
                if response is None:
                    missing.append(f"{task_id}/response_{slot:04d}")
                else:
                    responses.append(response)
        if missing:
            preview = ", ".join(missing[:5])
            raise ScalableRoundZeroPoolConflictError(
                f"pool is incomplete: {preview}"
            )
        if len(responses) != manifest.intended_response_count:
            raise ScalableRoundZeroPoolConflictError(
                "pool response count differs from manifest"
            )
        return manifest, tuple(responses)


class ScalableRoundZeroPoolGenerator:
    """Generate every requested pool slot exactly once; never select on output quality."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        config: ScalableRoundZeroPoolConfig,
        store: ScalableRoundZeroPoolStore,
        prompt_builder: PromptBuilder,
        answer_parser: AnswerParser,
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self.generator = generator
        self.config = config
        self.store = store
        self.prompt_builder = prompt_builder
        self.answer_parser = answer_parser
        self.max_workers = max_workers

    def generate(
        self, tasks: tuple[TaskInstance, ...]
    ) -> tuple[ScalableRoundZeroPoolResponse, ...]:
        manifest = self.store.initialize(config=self.config, tasks=tasks)
        jobs = [
            (task, slot)
            for task in tasks
            for slot in range(self.config.responses_per_task)
        ]
        resolved: list[ScalableRoundZeroPoolResponse | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[Future[ScalableRoundZeroPoolResponse], int] = {
                executor.submit(self._generate_one, manifest.pool_version, task, slot): index
                for index, (task, slot) in enumerate(jobs)
            }
            for future in as_completed(futures):
                resolved[futures[future]] = future.result()
        records = tuple(record for record in resolved if record is not None)
        if len(records) != manifest.intended_response_count:
            raise RuntimeError("pool generation completed with missing response slots")
        return records

    def _generate_one(
        self, pool_version: str, task: TaskInstance, pool_slot: int
    ) -> ScalableRoundZeroPoolResponse:
        generation_seed = stable_integer(
            "scalable-r0-pool", pool_version, task.task_id, pool_slot, self.config.base_seed
        )
        messages = self.prompt_builder(task)
        request_identity = {
            "pool_version": pool_version,
            "task": task.model_dump(mode="json"),
            "pool_slot": pool_slot,
            "generation_seed": generation_seed,
            "messages": [message.model_dump() for message in messages],
            "config": self.config.model_dump(mode="json"),
        }
        request_fingerprint = _fingerprint(request_identity)
        cached = self.store.load(task_id=task.task_id, pool_slot=pool_slot)
        if cached is not None:
            if cached.request_fingerprint != request_fingerprint:
                raise ScalableRoundZeroPoolConflictError(
                    "cached response request differs from current request"
                )
            return cached

        result = self.generator.generate(
            TextGenerationRequest(
                request_id=stable_id("scalable-r0", request_fingerprint),
                messages=messages,
                seed=generation_seed,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                min_p=self.config.min_p,
                presence_penalty=self.config.presence_penalty,
                max_output_tokens=self.config.max_output_tokens,
            )
        )
        parse_error: str | None = None
        try:
            parsed = self.answer_parser(result.raw_text, result.finish_reason)
        except ValueError as exc:
            parsed = None
            parse_error = str(exc)
        state = classify_numeric_answer(
            parsed, reference_answer=task.reference_answer, target_answer=None
        )
        response = ScalableRoundZeroPoolResponse(
            pool_version=pool_version,
            task_id=task.task_id,
            pool_response_id=stable_id(
                "pool-response", pool_version, task.task_id, pool_slot
            ),
            pool_slot=pool_slot,
            generation_seed=generation_seed,
            raw_response=result.raw_text,
            parsed_answer=parsed,
            answer_state=state,
            output_tokens=result.output_tokens,
            input_tokens=result.input_tokens,
            finish_reason=result.finish_reason,
            latency_ms=result.latency_ms,
            content_hash=hashlib.sha256(result.raw_text.encode("utf-8")).hexdigest(),
            requested_model=self.config.requested_model,
            returned_model=result.model_name,
            prompt_version=self.config.prompt_version,
            prompt_messages=tuple(message.model_dump() for message in messages),
            request_fingerprint=request_fingerprint,
            provider_metadata={
                **result.metadata,
                "pool_answer_parse_error": parse_error,
            },
        )
        self.store.save(response)
        return response


def default_numeric_parser(raw_text: str, finish_reason: str | None) -> str | None:
    del finish_reason
    return parse_numeric_answer(raw_text)


def build_round_zero_draws(
    *,
    pool_version: str,
    task_id: str,
    node_count: int,
    replicate_count: int,
    pool_responses: tuple[ScalableRoundZeroPoolResponse, ...],
    draw_seed: int,
    fresh_audit_fraction: float = 0.15,
) -> tuple[RoundZeroDraw, ...]:
    """Create paired draws, reserving a deterministic fraction for fresh audits."""

    if not 0.1 <= fresh_audit_fraction <= 0.2:
        raise ValueError("fresh_audit_fraction must lie in [0.1, 0.2]")
    if replicate_count < 1:
        raise ValueError("replicate_count must be positive")
    eligible = tuple(
        response
        for response in pool_responses
        if response.task_id == task_id and response.pool_version == pool_version
    )
    if len(eligible) < node_count:
        raise ValueError("response pool is smaller than node_count")

    audit_count = round(replicate_count * fresh_audit_fraction)
    if replicate_count >= 5:
        audit_count = max(1, audit_count)
    audit_indices = set(
        sorted(
            range(replicate_count),
            key=lambda index: stable_integer("fresh-audit", task_id, draw_seed, index),
        )[:audit_count]
    )
    draws: list[RoundZeroDraw] = []
    for replicate_index in range(replicate_count):
        selection_seed = stable_integer(
            "pool-draw", pool_version, task_id, node_count, replicate_index, draw_seed
        )
        mode: Literal["pooled", "fresh_audit"] = (
            "fresh_audit" if replicate_index in audit_indices else "pooled"
        )
        selected = ()
        if mode == "pooled":
            selected = tuple(
                response.pool_response_id
                for response in sorted(
                    eligible,
                    key=lambda response: (
                        stable_integer(
                            "pool-sample", selection_seed, response.pool_response_id
                        ),
                        response.pool_response_id,
                    ),
                )[:node_count]
            )
        draws.append(
            RoundZeroDraw(
                pool_version=pool_version,
                draw_id=stable_id(
                    "r0-draw", pool_version, task_id, node_count, replicate_index, draw_seed
                ),
                task_id=task_id,
                node_count=node_count,
                replicate_index=replicate_index,
                mode=mode,
                selection_seed=selection_seed,
                selected_pool_response_ids=selected,
            )
        )
    return tuple(draws)


def assign_draw_to_graph(
    draw: RoundZeroDraw, graph: GraphSpec
) -> GraphRoundZeroAssignment:
    """Permute one fixed draw over structural nodes using the graph fingerprint."""

    if draw.mode != "pooled":
        raise ValueError("fresh audit draws are generated in-run and cannot be pool-assigned")
    if graph.node_count != draw.node_count:
        raise ValueError("graph node count differs from draw node count")
    graph_fingerprint = _fingerprint(graph.model_dump(mode="json"))
    assignment_seed = stable_integer(
        "graph-pool-assignment", draw.draw_id, graph_fingerprint
    )
    node_order = tuple(
        sorted(
            draw.selected_pool_response_ids,
            key=lambda response_id: (
                stable_integer("graph-pool-permutation", assignment_seed, response_id),
                response_id,
            ),
        )
    )
    return GraphRoundZeroAssignment(
        pool_version=draw.pool_version,
        draw_id=draw.draw_id,
        graph_id=graph.graph_id,
        graph_fingerprint=graph_fingerprint,
        assignment_id=stable_id(
            "graph-r0-assignment", draw.draw_id, graph_fingerprint, *node_order
        ),
        assignment_seed=assignment_seed,
        node_to_pool_response_id=node_order,
    )


def materialize_engine_inputs(
    *,
    draw: RoundZeroDraw,
    graph_assignment: GraphRoundZeroAssignment,
    pool_responses: tuple[ScalableRoundZeroPoolResponse, ...],
    experiment_seed: int,
) -> tuple[tuple[RoundZeroRecord, ...], InitialStateAssignment]:
    """Adapt a paired pool draw to the historical synchronous engine interface."""

    if graph_assignment.draw_id != draw.draw_id:
        raise ValueError("assignment belongs to a different draw")
    selected = {
        response.pool_response_id: response
        for response in pool_responses
        if response.pool_response_id in draw.selected_pool_response_ids
    }
    if set(selected) != set(draw.selected_pool_response_ids):
        raise ValueError("pool responses do not cover the complete draw")
    canonical_ids = tuple(sorted(draw.selected_pool_response_ids))
    slot_by_id = {response_id: slot for slot, response_id in enumerate(canonical_ids)}
    records = tuple(
        RoundZeroRecord(
            record_id=selected[response_id].pool_response_id,
            request_fingerprint=selected[response_id].request_fingerprint,
            task_id=draw.task_id,
            replica_slot=slot,
            experiment_seed=experiment_seed,
            generation_seed=selected[response_id].generation_seed,
            prompt_version=selected[response_id].prompt_version,
            prompt_messages=selected[response_id].prompt_messages,
            raw_output=selected[response_id].raw_response,
            parsed_answer=selected[response_id].parsed_answer,
            answer_state=selected[response_id].answer_state,
            is_correct=selected[response_id].answer_state is AnswerState.CORRECT,
            requested_model=selected[response_id].requested_model,
            returned_model=selected[response_id].returned_model,
            finish_reason=selected[response_id].finish_reason,
            input_tokens=selected[response_id].input_tokens,
            output_tokens=selected[response_id].output_tokens,
            latency_ms=selected[response_id].latency_ms,
            provider_metadata={
                **selected[response_id].provider_metadata,
                "scalable_round_zero_pool": True,
                "pool_version": draw.pool_version,
                "pool_response_id": response_id,
                "draw_id": draw.draw_id,
            },
        )
        for slot, response_id in enumerate(canonical_ids)
    )
    mapping = tuple(
        slot_by_id[response_id]
        for response_id in graph_assignment.node_to_pool_response_id
    )
    assignment = InitialStateAssignment(
        assignment_id=graph_assignment.assignment_id,
        node_count=draw.node_count,
        assignment_seed=graph_assignment.assignment_seed,
        structural_node_to_replica=mapping,
    )
    return records, assignment
