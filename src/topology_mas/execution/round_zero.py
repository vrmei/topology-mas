"""Graph-independent round-zero generation with conflict-safe atomic caching."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from topology_mas.data.gsm8k import task_collection_fingerprint
from topology_mas.execution.answers import classify_numeric_answer, parse_numeric_answer
from topology_mas.execution.generation import TextGenerator
from topology_mas.execution.prompts import PROMPT_VERSION, build_node_messages
from topology_mas.execution.schemas import TextGenerationRequest
from topology_mas.execution.seeding import node_round_seed, stable_id
from topology_mas.models import AnswerState, TaskInstance

ROUND_ZERO_CACHE_VERSION = "round-zero-cache-v1"


class RoundZeroCacheConfig(BaseModel):
    """Frozen request conditions shared by one cache collection."""

    model_config = ConfigDict(frozen=True)

    node_count: int = Field(ge=2)
    seeds: tuple[int, ...] = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    expected_returned_model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=768, ge=1)
    prompt_version: str = PROMPT_VERSION
    cache_version: str = ROUND_ZERO_CACHE_VERSION

    @model_validator(mode="after")
    def validate_seeds(self) -> RoundZeroCacheConfig:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        return self


class RoundZeroRecord(BaseModel):
    """One independently generated node state, before any graph communication."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(min_length=1)
    request_fingerprint: str = Field(min_length=64, max_length=64)
    task_id: str = Field(min_length=1)
    node_id: int = Field(ge=0)
    experiment_seed: int
    generation_seed: int
    prompt_version: str = Field(min_length=1)
    prompt_messages: tuple[dict[str, str], ...]
    raw_output: str
    parsed_answer: str | None = None
    answer_state: AnswerState
    is_correct: bool
    requested_model: str = Field(min_length=1)
    returned_model: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class RoundZeroManifest(BaseModel):
    """Identity of the intended cache, written before generation starts."""

    model_config = ConfigDict(frozen=True)

    cache_version: str
    config: RoundZeroCacheConfig
    task_collection_fingerprint: str = Field(min_length=64, max_length=64)
    task_ids: tuple[str, ...]
    intended_record_count: int = Field(ge=1)


class RoundZeroGenerationResult(BaseModel):
    """Records plus an explicit audit of new generation versus cache reuse."""

    model_config = ConfigDict(frozen=True)

    records: tuple[RoundZeroRecord, ...]
    generated_count: int = Field(ge=0)
    reused_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> RoundZeroGenerationResult:
        if self.generated_count + self.reused_count != len(self.records):
            raise ValueError("generated and reused counts must cover every record")
        return self


class RoundZeroCacheConflictError(RuntimeError):
    pass


def _canonical_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class RoundZeroCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    def initialize(
        self,
        *,
        config: RoundZeroCacheConfig,
        tasks: tuple[TaskInstance, ...],
    ) -> RoundZeroManifest:
        if not tasks:
            raise ValueError("round-zero generation requires at least one task")
        task_ids = tuple(task.task_id for task in tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task IDs must be unique")
        manifest = RoundZeroManifest(
            cache_version=ROUND_ZERO_CACHE_VERSION,
            config=config,
            task_collection_fingerprint=task_collection_fingerprint(tasks),
            task_ids=task_ids,
            intended_record_count=len(tasks) * config.node_count * len(config.seeds),
        )
        if self.manifest_path.exists():
            existing = RoundZeroManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if existing != manifest:
                raise RoundZeroCacheConflictError(
                    "round-zero cache manifest differs; use a new output directory"
                )
            return existing
        _atomic_write_json(self.manifest_path, manifest.model_dump(mode="json"))
        return manifest

    def record_path(self, *, task_id: str, seed: int, node_id: int) -> Path:
        safe_task = task_id.replace("/", "_").replace("\\", "_")
        return self.root / "records" / safe_task / f"seed_{seed}" / f"node_{node_id}.json"

    def load(
        self,
        *,
        task_id: str,
        seed: int,
        node_id: int,
    ) -> RoundZeroRecord | None:
        path = self.record_path(task_id=task_id, seed=seed, node_id=node_id)
        if not path.exists():
            return None
        return RoundZeroRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: RoundZeroRecord) -> Path:
        path = self.record_path(
            task_id=record.task_id,
            seed=record.experiment_seed,
            node_id=record.node_id,
        )
        if path.exists():
            existing = RoundZeroRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if existing != record:
                raise RoundZeroCacheConflictError(
                    f"round-zero record conflict at {path}; use a new output directory"
                )
            return path
        _atomic_write_json(path, record.model_dump(mode="json"))
        return path


class RoundZeroGenerator:
    def __init__(
        self,
        generator: TextGenerator,
        *,
        config: RoundZeroCacheConfig,
        cache: RoundZeroCache,
    ) -> None:
        self._generator = generator
        self.config = config
        self.cache = cache

    def generate(self, tasks: tuple[TaskInstance, ...]) -> RoundZeroGenerationResult:
        self.cache.initialize(config=self.config, tasks=tasks)
        records: list[RoundZeroRecord] = []
        generated_count = 0
        reused_count = 0
        for task in tasks:
            if task.oracle_type != "numeric":
                raise ValueError("the first round-zero generator supports numeric tasks only")
            for experiment_seed in self.config.seeds:
                for node_id in range(self.config.node_count):
                    record, generated = self._generate_one(
                        task=task,
                        node_id=node_id,
                        experiment_seed=experiment_seed,
                    )
                    records.append(record)
                    generated_count += int(generated)
                    reused_count += int(not generated)
        return RoundZeroGenerationResult(
            records=tuple(records),
            generated_count=generated_count,
            reused_count=reused_count,
        )

    def _generate_one(
        self,
        *,
        task: TaskInstance,
        node_id: int,
        experiment_seed: int,
    ) -> tuple[RoundZeroRecord, bool]:
        prompt_messages = build_node_messages(
            task,
            previous_output=None,
            incoming_messages=(),
        )
        generation_seed = node_round_seed(
            experiment_seed=experiment_seed,
            task_id=task.task_id,
            node_id=node_id,
            round_index=0,
        )
        identity = {
            "cache_version": ROUND_ZERO_CACHE_VERSION,
            "task": task.model_dump(mode="json"),
            "node_id": node_id,
            "experiment_seed": experiment_seed,
            "generation_seed": generation_seed,
            "prompt_version": self.config.prompt_version,
            "prompt_messages": [message.model_dump() for message in prompt_messages],
            "requested_model": self.config.requested_model,
            "expected_returned_model": self.config.expected_returned_model,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
        }
        fingerprint = _canonical_fingerprint(identity)
        cached = self.cache.load(
            task_id=task.task_id,
            seed=experiment_seed,
            node_id=node_id,
        )
        if cached is not None:
            if cached.request_fingerprint != fingerprint:
                raise RoundZeroCacheConflictError(
                    "cached round-zero request differs; use a new output directory"
                )
            return cached, False

        completion = self._generator.generate(
            TextGenerationRequest(
                request_id=stable_id(
                    "round0", task.task_id, node_id, experiment_seed, fingerprint
                ),
                messages=prompt_messages,
                seed=generation_seed,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
        )
        parsed = parse_numeric_answer(completion.raw_text)
        state = classify_numeric_answer(
            parsed,
            reference_answer=task.reference_answer,
            target_answer=None,
        )
        record = RoundZeroRecord(
            record_id=stable_id(
                "round0-record", task.task_id, node_id, experiment_seed, fingerprint
            ),
            request_fingerprint=fingerprint,
            task_id=task.task_id,
            node_id=node_id,
            experiment_seed=experiment_seed,
            generation_seed=generation_seed,
            prompt_version=self.config.prompt_version,
            prompt_messages=tuple(message.model_dump() for message in prompt_messages),
            raw_output=completion.raw_text,
            parsed_answer=parsed,
            answer_state=state,
            is_correct=state is AnswerState.CORRECT,
            requested_model=self.config.requested_model,
            returned_model=completion.model_name,
            finish_reason=completion.finish_reason,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
            provider_metadata=completion.metadata,
        )
        self.cache.save(record)
        return record, True
