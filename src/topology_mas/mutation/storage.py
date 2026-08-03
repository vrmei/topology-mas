"""Artifact persistence for complete, secret-free mutation audit trails."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from topology_mas.models import TaskInstance
from topology_mas.mutation.numeric_oracle import OBJECTIVE_ORACLE_VERSION
from topology_mas.mutation.prompts import (
    GENERATOR_PROMPT_VERSION,
    PLAUSIBILITY_PROMPT_VERSION,
)
from topology_mas.mutation.schemas import CandidateBatch, MutationPipelineConfig, MutationRunResult
from topology_mas.providers import JSONCompletion


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


def fingerprint_jsonable(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_directory_name(task_id: str) -> str:
    """Map an arbitrary task identifier to one safe, stable path component."""

    safe_task_id = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")
    if not safe_task_id:
        raise ValueError("task_id does not contain a usable path component")
    if safe_task_id != task_id:
        suffix = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
        return f"{safe_task_id}-{suffix}"
    return safe_task_id


class MutationArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(self, task: TaskInstance, result: MutationRunResult) -> Path:
        task_dir = self._task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "task": task,
            "task_fingerprint": fingerprint_jsonable(task),
            "config": result.config,
            "generator_prompt_version": GENERATOR_PROMPT_VERSION,
            "plausibility_prompt_version": PLAUSIBILITY_PROMPT_VERSION,
            "objective_oracle_version": OBJECTIVE_ORACLE_VERSION,
            "generator_request_fingerprint": fingerprint_jsonable(result.generator_request),
            "selected_candidate_id": result.selected_candidate_id,
        }
        self._write_json(task_dir / "manifest.json", manifest)
        self._write_json(task_dir / "generator_request.json", result.generator_request)
        self._write_json(task_dir / "generator_response.json", result.generator_response)
        self._write_json(task_dir / "result.json", result)

        candidates_dir = task_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)
        for evaluation in result.evaluations:
            self._write_json(
                candidates_dir / f"{evaluation.candidate.candidate_id}.json",
                evaluation,
            )
        return task_dir

    def save_generation_stage(
        self,
        *,
        task: TaskInstance,
        config: MutationPipelineConfig,
        messages: list[dict[str, str]],
        completion: JSONCompletion,
        batch: CandidateBatch,
    ) -> Path:
        task_dir = self._task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            task_dir / "generation_stage.json",
            {
                "task": task,
                "config": config,
                "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                "plausibility_prompt_version": PLAUSIBILITY_PROMPT_VERSION,
                "objective_oracle_version": OBJECTIVE_ORACLE_VERSION,
                "generator_request": messages,
                "generator_response": completion.raw_response,
                "generator_attempts": completion.raw_attempts,
                "parsed_batch": batch,
            },
        )
        return task_dir

    def save_generation_failure(
        self,
        *,
        task: TaskInstance,
        config: MutationPipelineConfig,
        messages: list[dict[str, str]],
        completion: JSONCompletion,
        error: Exception,
    ) -> Path:
        task_dir = self._task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            task_dir / "generation_failure.json",
            {
                "task": task,
                "config": config,
                "generator_prompt_version": GENERATOR_PROMPT_VERSION,
                "plausibility_prompt_version": PLAUSIBILITY_PROMPT_VERSION,
                "objective_oracle_version": OBJECTIVE_ORACLE_VERSION,
                "generator_request": messages,
                "generator_response": completion.raw_response,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return task_dir

    def _task_dir(self, task: TaskInstance) -> Path:
        return self._root / task_directory_name(task.task_id)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        payload = _jsonable(value)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
