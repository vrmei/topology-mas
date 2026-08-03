"""Sequential, resumable batch mutation with strict cache identity checks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field

from topology_mas.data.gsm8k import task_collection_fingerprint
from topology_mas.models import TaskInstance
from topology_mas.mutation.numeric_oracle import OBJECTIVE_ORACLE_VERSION
from topology_mas.mutation.prompts import (
    GENERATOR_PROMPT_VERSION,
    PLAUSIBILITY_PROMPT_VERSION,
)
from topology_mas.mutation.schemas import MutationPipelineConfig, MutationRunResult
from topology_mas.mutation.storage import fingerprint_jsonable, task_directory_name


class MutationPipelineLike(Protocol):
    config: MutationPipelineConfig

    def run(self, task: TaskInstance) -> MutationRunResult: ...


class BatchDisposition(StrEnum):
    GENERATED_SELECTED = "generated_selected"
    GENERATED_NO_CANDIDATE = "generated_no_candidate"
    CACHED_SELECTED = "cached_selected"
    CACHED_NO_CANDIDATE = "cached_no_candidate"
    ERROR = "error"


class BatchTaskOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    disposition: BatchDisposition
    selected_candidate_id: str | None = None
    candidate_count: int = Field(default=0, ge=0)
    objective_passed: int = Field(default=0, ge=0)
    plausibility_passed: int = Field(default=0, ge=0)
    processing_errors: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None


class MutationBatchSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_count: int = Field(ge=0)
    generated_selected: int = Field(ge=0)
    generated_no_candidate: int = Field(ge=0)
    cached_selected: int = Field(ge=0)
    cached_no_candidate: int = Field(ge=0)
    errors: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    objective_passed_candidates: int = Field(ge=0)
    plausibility_passed_candidates: int = Field(ge=0)
    processing_error_candidates: int = Field(ge=0)
    task_selection_rate: float = Field(ge=0.0, le=1.0)
    objective_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    plausibility_pass_rate_given_objective: float | None = Field(
        default=None, ge=0.0, le=1.0
    )


class BatchCacheConflictError(RuntimeError):
    pass


class BatchMutationRunner:
    """Run one fixed pipeline config over tasks, persisting progress after every item."""

    def __init__(
        self,
        pipeline: MutationPipelineLike,
        *,
        output_dir: str | Path,
        fail_fast: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._output_dir = Path(output_dir)
        self._tasks_dir = self._output_dir / "tasks"
        self._fail_fast = fail_fast

    def run(
        self,
        tasks: Iterable[TaskInstance],
        *,
        source_path: str | Path | None = None,
    ) -> tuple[tuple[BatchTaskOutcome, ...], MutationBatchSummary]:
        task_list = tuple(tasks)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._validate_or_create_manifest(task_list, source_path=source_path)

        outcomes: list[BatchTaskOutcome] = []
        progress_path = self._output_dir / "progress.jsonl"
        with progress_path.open("a", encoding="utf-8", newline="\n") as progress:
            for task in task_list:
                try:
                    outcome = self._run_one(task)
                except BatchCacheConflictError:
                    raise
                except Exception as exc:
                    outcome = BatchTaskOutcome(
                        task_id=task.task_id,
                        disposition=BatchDisposition.ERROR,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    self._append_progress(progress, outcome)
                    outcomes.append(outcome)
                    if self._fail_fast:
                        raise
                    continue
                self._append_progress(progress, outcome)
                outcomes.append(outcome)

        summary = self._summarize(outcomes)
        self._write_json(self._output_dir / "outcomes.json", outcomes)
        self._write_json(self._output_dir / "summary.json", summary)
        return tuple(outcomes), summary

    def _run_one(self, task: TaskInstance) -> BatchTaskOutcome:
        cached = self._load_cached(task)
        if cached is not None:
            disposition = (
                BatchDisposition.CACHED_SELECTED
                if cached.selected_candidate_id is not None
                else BatchDisposition.CACHED_NO_CANDIDATE
            )
            return self._outcome_from_result(task.task_id, cached, disposition)

        result = self._pipeline.run(task)
        disposition = (
            BatchDisposition.GENERATED_SELECTED
            if result.selected_candidate_id is not None
            else BatchDisposition.GENERATED_NO_CANDIDATE
        )
        return self._outcome_from_result(task.task_id, result, disposition)

    def _load_cached(self, task: TaskInstance) -> MutationRunResult | None:
        task_dir = self._tasks_dir / task_directory_name(task.task_id)
        manifest_path = task_dir / "manifest.json"
        result_path = task_dir / "result.json"
        if not manifest_path.exists() or not result_path.exists():
            return None

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_task_fingerprint = fingerprint_jsonable(task)
        if manifest.get("task_fingerprint") != expected_task_fingerprint:
            raise BatchCacheConflictError(
                f"cached task fingerprint differs for {task.task_id}; use a new output directory"
            )
        if manifest.get("generator_prompt_version") != GENERATOR_PROMPT_VERSION:
            raise BatchCacheConflictError(
                f"cached generator prompt differs for {task.task_id}; use a new output directory"
            )
        if manifest.get("plausibility_prompt_version") != PLAUSIBILITY_PROMPT_VERSION:
            raise BatchCacheConflictError(
                f"cached plausibility prompt differs for {task.task_id}; "
                "use a new output directory"
            )
        if manifest.get("objective_oracle_version") != OBJECTIVE_ORACLE_VERSION:
            raise BatchCacheConflictError(
                f"cached objective Oracle differs for {task.task_id}; "
                "use a new output directory"
            )
        result = MutationRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if result.config != self._pipeline.config:
            raise BatchCacheConflictError(
                f"cached mutation config differs for {task.task_id}; use a new output directory"
            )
        return result

    def _validate_or_create_manifest(
        self,
        tasks: tuple[TaskInstance, ...],
        *,
        source_path: str | Path | None,
    ) -> None:
        manifest_path = self._output_dir / "batch_manifest.json"
        identity = {
            "schema_version": 1,
            "pipeline_config": self._pipeline.config,
            "pipeline_config_fingerprint": fingerprint_jsonable(self._pipeline.config),
            "generator_prompt_version": GENERATOR_PROMPT_VERSION,
            "plausibility_prompt_version": PLAUSIBILITY_PROMPT_VERSION,
            "objective_oracle_version": OBJECTIVE_ORACLE_VERSION,
            "task_collection_fingerprint": task_collection_fingerprint(tasks),
            "task_count": len(tasks),
            "task_ids": [task.task_id for task in tasks],
            "source_path": str(Path(source_path).resolve()) if source_path else None,
        }
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            comparable_existing = {
                key: existing.get(key)
                for key in identity
                if key not in {"source_path"}
            }
            comparable_identity = {
                key: value for key, value in identity.items() if key not in {"source_path"}
            }
            serialized_identity = json.loads(
                json.dumps(comparable_identity, default=self._json_default)
            )
            if comparable_existing != serialized_identity:
                raise BatchCacheConflictError(
                    "batch manifest differs from current tasks or config; "
                    "use a new output directory"
                )
            return
        identity["created_at"] = datetime.now(UTC).isoformat()
        self._write_json(manifest_path, identity)

    @staticmethod
    def _outcome_from_result(
        task_id: str,
        result: MutationRunResult,
        disposition: BatchDisposition,
    ) -> BatchTaskOutcome:
        return BatchTaskOutcome(
            task_id=task_id,
            disposition=disposition,
            selected_candidate_id=result.selected_candidate_id,
            candidate_count=len(result.evaluations),
            objective_passed=sum(item.objective.passed for item in result.evaluations),
            plausibility_passed=sum(item.eligible for item in result.evaluations),
            processing_errors=sum(
                item.processing_error is not None for item in result.evaluations
            ),
        )

    @staticmethod
    def _summarize(outcomes: list[BatchTaskOutcome]) -> MutationBatchSummary:
        total_candidates = sum(item.candidate_count for item in outcomes)
        objective_passed = sum(item.objective_passed for item in outcomes)
        plausibility_passed = sum(item.plausibility_passed for item in outcomes)
        selected_tasks = sum(item.selected_candidate_id is not None for item in outcomes)
        return MutationBatchSummary(
            task_count=len(outcomes),
            generated_selected=sum(
                item.disposition is BatchDisposition.GENERATED_SELECTED for item in outcomes
            ),
            generated_no_candidate=sum(
                item.disposition is BatchDisposition.GENERATED_NO_CANDIDATE for item in outcomes
            ),
            cached_selected=sum(
                item.disposition is BatchDisposition.CACHED_SELECTED for item in outcomes
            ),
            cached_no_candidate=sum(
                item.disposition is BatchDisposition.CACHED_NO_CANDIDATE for item in outcomes
            ),
            errors=sum(item.disposition is BatchDisposition.ERROR for item in outcomes),
            total_candidates=total_candidates,
            objective_passed_candidates=objective_passed,
            plausibility_passed_candidates=plausibility_passed,
            processing_error_candidates=sum(item.processing_errors for item in outcomes),
            task_selection_rate=(selected_tasks / len(outcomes) if outcomes else 0.0),
            objective_pass_rate=(
                objective_passed / total_candidates if total_candidates else None
            ),
            plausibility_pass_rate_given_objective=(
                plausibility_passed / objective_passed if objective_passed else None
            ),
        )

    @staticmethod
    def _append_progress(handle: TextIO, outcome: BatchTaskOutcome) -> None:
        handle.write(outcome.model_dump_json() + "\n")
        handle.flush()

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        raise TypeError(f"cannot serialize {type(value).__name__}")

    @classmethod
    def _write_json(cls, path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, default=cls._json_default, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
