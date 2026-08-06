"""Validate a completed mutation cache and index its eligible selected answers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from topology_mas.models import AdversarialAnswer
from topology_mas.mutation.pipeline import MutationPipeline
from topology_mas.mutation.schemas import MutationRunResult
from topology_mas.mutation.selection import (
    SELECTION_POLICY_VERSION,
    is_coverage_candidate,
    is_preferred_candidate,
    select_candidate_evaluation,
)
from topology_mas.mutation.storage import fingerprint_jsonable, task_directory_name


class MutationCacheAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 2
    selection_policy_version: str = SELECTION_POLICY_VERSION
    task_count: int = Field(ge=1)
    result_count: int = Field(ge=1)
    selected_task_count: int = Field(ge=0)
    no_candidate_task_count: int = Field(ge=0)
    total_candidates: int = Field(ge=0)
    objective_passed_candidates: int = Field(ge=0)
    eligible_candidates: int = Field(ge=0)
    preferred_candidates: int = Field(ge=0)
    preferred_selected_task_count: int = Field(ge=0)
    coverage_fallback_task_count: int = Field(ge=0)
    processing_error_candidates: int = Field(ge=0)
    selected_answers_fingerprint: str = Field(min_length=64, max_length=64)
    selected_task_ids: tuple[str, ...]
    no_candidate_task_ids: tuple[str, ...]


def audit_mutation_cache(
    mutation_dir: str | Path,
) -> tuple[MutationCacheAudit, tuple[AdversarialAnswer, ...]]:
    root = Path(mutation_dir)
    manifest_path = root / "batch_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"mutation batch manifest is missing at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise ValueError("mutation batch manifest has invalid task_ids")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("mutation batch manifest contains duplicate task_ids")

    results: list[MutationRunResult] = []
    for task_id in task_ids:
        result_path = root / "tasks" / task_directory_name(task_id) / "result.json"
        if not result_path.exists():
            raise ValueError(f"mutation result is missing for {task_id}")
        result = MutationRunResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        if result.task_id != task_id:
            raise ValueError(f"mutation result task mismatch for {task_id}")
        results.append(result)

    answers: list[AdversarialAnswer] = []
    no_candidate_ids: list[str] = []
    preferred_selected_task_count = 0
    coverage_fallback_task_count = 0
    for result in results:
        selected_with_tier = select_candidate_evaluation(result.evaluations, result.config)
        if selected_with_tier is None:
            no_candidate_ids.append(result.task_id)
            continue
        _, selection_tier = selected_with_tier
        preferred_selected_task_count += int(selection_tier == "preferred")
        coverage_fallback_task_count += int(selection_tier == "coverage_fallback")
        answer = MutationPipeline.to_adversarial_answer(result)
        result_fingerprint = fingerprint_jsonable(result)
        answers.append(
            answer.model_copy(
                update={
                    "metadata": {
                        **answer.metadata,
                        "source_result_fingerprint": result_fingerprint,
                    }
                }
            )
        )

    answers_tuple = tuple(answers)
    audit = MutationCacheAudit(
        task_count=len(task_ids),
        result_count=len(results),
        selected_task_count=len(answers_tuple),
        no_candidate_task_count=len(no_candidate_ids),
        total_candidates=sum(len(result.evaluations) for result in results),
        objective_passed_candidates=sum(
            evaluation.objective.passed
            for result in results
            for evaluation in result.evaluations
        ),
        eligible_candidates=sum(
            is_coverage_candidate(evaluation, result.config)
            for result in results
            for evaluation in result.evaluations
        ),
        preferred_candidates=sum(
            is_preferred_candidate(evaluation, result.config)
            for result in results
            for evaluation in result.evaluations
        ),
        preferred_selected_task_count=preferred_selected_task_count,
        coverage_fallback_task_count=coverage_fallback_task_count,
        processing_error_candidates=sum(
            evaluation.processing_error is not None
            for result in results
            for evaluation in result.evaluations
        ),
        selected_answers_fingerprint=fingerprint_jsonable(answers_tuple),
        selected_task_ids=tuple(answer.task_id for answer in answers_tuple),
        no_candidate_task_ids=tuple(no_candidate_ids),
    )
    if audit.result_count != audit.selected_task_count + audit.no_candidate_task_count:
        raise ValueError("mutation audit task partition is inconsistent")
    return audit, answers_tuple


def _atomic_write(path: Path, content: str) -> None:
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


def write_mutation_cache_index(
    output_dir: str | Path,
    audit: MutationCacheAudit,
    answers: tuple[AdversarialAnswer, ...],
) -> None:
    destination = Path(output_dir)
    artifacts = {
        "audit.json": json.dumps(
            audit.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "selected_adversarial_answers.jsonl": "".join(
            answer.model_dump_json() + "\n" for answer in answers
        ),
    }
    for name, content in artifacts.items():
        path = destination / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing mutation cache index differs at {path}")
        if not path.exists():
            _atomic_write(path, content)
