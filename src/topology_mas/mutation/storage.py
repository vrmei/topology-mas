"""Artifact persistence for complete, secret-free mutation audit trails."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from topology_mas.models import TaskInstance
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


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MutationArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def save(self, task: TaskInstance, result: MutationRunResult) -> Path:
        task_dir = self._task_dir(task)
        task_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "task": task,
            "task_fingerprint": _fingerprint(task),
            "config": result.config,
            "generator_prompt_version": GENERATOR_PROMPT_VERSION,
            "plausibility_prompt_version": PLAUSIBILITY_PROMPT_VERSION,
            "generator_request_fingerprint": _fingerprint(result.generator_request),
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
                "generator_request": messages,
                "generator_response": completion.raw_response,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return task_dir

    def _task_dir(self, task: TaskInstance) -> Path:
        safe_task_id = re.sub(r"[^A-Za-z0-9._-]+", "_", task.task_id).strip("._")
        if not safe_task_id:
            raise ValueError("task_id does not contain a usable path component")
        return self._root / safe_task_id

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        payload = _jsonable(value)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
