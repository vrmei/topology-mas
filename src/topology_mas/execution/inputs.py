"""Strict loaders that bridge preprocessing artifacts into batch execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from topology_mas.execution.round_zero import (
    ROUND_ZERO_CACHE_VERSION,
    RoundZeroCache,
    RoundZeroManifest,
    RoundZeroRecord,
)
from topology_mas.models import AdversarialAnswer, OracleStatus
from topology_mas.mutation.schemas import MutationRunResult
from topology_mas.mutation.storage import task_directory_name


class ExecutionInputError(ValueError):
    """A preprocessing artifact is incomplete, incompatible, or internally inconsistent."""


def load_adversarial_answer_index(
    path: str | Path,
) -> dict[str, AdversarialAnswer]:
    """Load a compact, audited selected-answer JSONL artifact."""

    source = Path(path)
    if not source.exists():
        raise ExecutionInputError(f"adversarial-answer index is missing at {source}")
    answers: dict[str, AdversarialAnswer] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            answer = AdversarialAnswer.model_validate_json(line)
        except ValueError as exc:
            raise ExecutionInputError(
                f"invalid adversarial answer at {source}:{line_number}"
            ) from exc
        if not answer.accepted:
            raise ExecutionInputError(
                f"adversarial answer is not Oracle-accepted at {source}:{line_number}"
            )
        if answer.task_id in answers:
            raise ExecutionInputError(
                f"duplicate adversarial answer for {answer.task_id!r} at {source}:{line_number}"
            )
        answers[answer.task_id] = answer
    if not answers:
        raise ExecutionInputError(f"adversarial-answer index is empty at {source}")
    return answers


def load_round_zero_collection(
    root: str | Path,
) -> tuple[RoundZeroManifest, tuple[RoundZeroRecord, ...]]:
    """Load every record declared by one complete round-zero manifest."""

    cache = RoundZeroCache(root)
    if not cache.manifest_path.exists():
        raise ExecutionInputError(f"round-zero manifest is missing at {cache.manifest_path}")
    manifest = RoundZeroManifest.model_validate_json(
        cache.manifest_path.read_text(encoding="utf-8")
    )
    if manifest.cache_version != ROUND_ZERO_CACHE_VERSION:
        raise ExecutionInputError(
            f"round-zero cache version {manifest.cache_version!r} is not supported"
        )

    records: list[RoundZeroRecord] = []
    missing: list[str] = []
    for task_id in manifest.task_ids:
        for seed in manifest.config.seeds:
            for replica_slot in range(manifest.config.replica_count):
                record = cache.load(
                    task_id=task_id,
                    seed=seed,
                    replica_slot=replica_slot,
                )
                if record is None:
                    missing.append(f"{task_id}/seed_{seed}/replica_{replica_slot}")
                else:
                    records.append(record)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ExecutionInputError(f"round-zero collection is incomplete: {preview}{suffix}")
    if len(records) != manifest.intended_record_count:
        raise ExecutionInputError(
            "round-zero record count differs from manifest: "
            f"{len(records)} != {manifest.intended_record_count}"
        )
    return manifest, tuple(records)


def load_selected_adversarial_answers(
    mutation_batch_dir: str | Path,
) -> dict[str, AdversarialAnswer]:
    """Convert eligible selected mutation candidates into execution attack records."""

    root = Path(mutation_batch_dir)
    manifest_path = root / "batch_manifest.json"
    if not manifest_path.exists():
        raise ExecutionInputError(f"mutation batch manifest is missing at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
        raise ExecutionInputError("mutation batch manifest does not contain valid task_ids")

    answers: dict[str, AdversarialAnswer] = {}
    missing_selection: list[str] = []
    for task_id in task_ids:
        result_path = root / "tasks" / task_directory_name(task_id) / "result.json"
        if not result_path.exists():
            raise ExecutionInputError(f"mutation result is missing for {task_id}: {result_path}")
        result = MutationRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        if result.task_id != task_id:
            raise ExecutionInputError(f"mutation result task mismatch for {task_id}")
        if result.selected_candidate_id is None:
            missing_selection.append(task_id)
            continue
        selected = [
            evaluation
            for evaluation in result.evaluations
            if evaluation.candidate.candidate_id == result.selected_candidate_id
        ]
        if len(selected) != 1:
            raise ExecutionInputError(
                f"selected candidate {result.selected_candidate_id!r} is not unique for {task_id}"
            )
        evaluation = selected[0]
        if not evaluation.eligible or evaluation.plausibility is None:
            raise ExecutionInputError(
                f"selected mutation candidate is not Oracle-eligible for {task_id}"
            )
        candidate = evaluation.candidate
        answers[task_id] = AdversarialAnswer(
            task_id=task_id,
            target_answer=candidate.final_answer,
            rationale=candidate.full_response,
            mutation_type=candidate.mutation_type,
            oracle_status=OracleStatus.PASSED,
            plausibility_score=evaluation.plausibility.overall_score,
            generator_model=result.config.generator_model,
            metadata={
                "candidate_id": candidate.candidate_id,
                "mutated_step_id": candidate.mutated_step_id,
                "plausibility_model": result.config.plausibility_model,
                "plausibility_returned_model": evaluation.plausibility.returned_model,
                "source_result_fingerprint": hashlib.sha256(
                    result.model_dump_json().encode("utf-8")
                ).hexdigest(),
            },
        )
    if missing_selection:
        preview = ", ".join(missing_selection[:5])
        suffix = (
            "" if len(missing_selection) <= 5 else f" (+{len(missing_selection) - 5} more)"
        )
        raise ExecutionInputError(
            f"tasks have no eligible selected mutation: {preview}{suffix}"
        )
    return answers
