"""Pinned GSM8K download, validation, loading, and deterministic sampling."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import httpx

from topology_mas.models import TaskInstance

GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
_RAW_ROOT = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_COMMIT}/grade_school_math/data"
)


@dataclass(frozen=True)
class GSM8KFileSpec:
    split: Literal["train", "test"]
    url: str
    sha256: str
    expected_lines: int


GSM8K_SPECS: dict[str, GSM8KFileSpec] = {
    "train": GSM8KFileSpec(
        split="train",
        url=f"{_RAW_ROOT}/train.jsonl",
        sha256="17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        expected_lines=7473,
    ),
    "test": GSM8KFileSpec(
        split="test",
        url=f"{_RAW_ROOT}/test.jsonl",
        sha256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        expected_lines=1319,
    ),
}

_FINAL_ANSWER_PATTERN = re.compile(r"####\s*([^\s]+)\s*$")
_CALCULATION_ANNOTATION_PATTERN = re.compile(r"<<[^<>]*>>")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_gsm8k(
    data_dir: str | Path,
    *,
    splits: Iterable[Literal["train", "test"]] = ("train", "test"),
    timeout_seconds: float = 120.0,
) -> dict[str, Path]:
    """Download pinned files atomically and verify them before accepting."""

    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}

    for split in splits:
        spec = GSM8K_SPECS[split]
        target = destination / f"{split}.jsonl"
        if target.exists() and sha256_file(target) == spec.sha256:
            resolved[split] = target
            continue

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{split}.",
            suffix=".tmp",
            dir=destination,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            try:
                with httpx.stream("GET", spec.url, timeout=timeout_seconds) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

        actual_hash = sha256_file(temporary)
        if actual_hash != spec.sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"GSM8K {split} SHA-256 mismatch: {actual_hash} != {spec.sha256}"
            )
        temporary.replace(target)
        resolved[split] = target

    return resolved


def _normalize_answer(value: str) -> str:
    cleaned = value.strip().replace(",", "").removeprefix("$")
    number = Fraction(cleaned)
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def _extract_reference(raw_answer: str) -> tuple[str, str]:
    match = _FINAL_ANSWER_PATTERN.search(raw_answer)
    if match is None:
        raise ValueError("GSM8K answer does not end with a #### numeric marker")
    final_answer = _normalize_answer(match.group(1))
    rationale = raw_answer[: match.start()].rstrip()
    cleaned_rationale = _CALCULATION_ANNOTATION_PATTERN.sub("", rationale)
    return final_answer, cleaned_rationale


def load_gsm8k(
    path: str | Path,
    *,
    split: Literal["train", "test"],
    verify_pinned_hash: bool = True,
) -> tuple[TaskInstance, ...]:
    """Load official JSONL into stable TaskInstance records."""

    source_path = Path(path)
    source_hash = sha256_file(source_path)
    spec = GSM8K_SPECS[split]
    if verify_pinned_hash and source_hash != spec.sha256:
        raise ValueError(
            f"GSM8K {split} SHA-256 mismatch: {source_hash} != {spec.sha256}"
        )

    tasks: list[TaskInstance] = []
    seen_questions: set[str] = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            try:
                record = json.loads(line)
                question = record["question"].strip()
                raw_answer = record["answer"].strip()
            except (json.JSONDecodeError, KeyError, AttributeError) as exc:
                raise ValueError(f"invalid GSM8K record at line {line_index + 1}") from exc
            if not question or not raw_answer:
                raise ValueError(f"empty GSM8K field at line {line_index + 1}")
            if question in seen_questions:
                raise ValueError(f"duplicate GSM8K question at line {line_index + 1}")
            seen_questions.add(question)

            reference_answer, reference_solution = _extract_reference(raw_answer)
            tasks.append(
                TaskInstance(
                    task_id=f"gsm8k-{split}-{line_index:05d}",
                    dataset="gsm8k",
                    split=split,
                    prompt=question,
                    reference_answer=reference_answer,
                    oracle_type="numeric",
                    metadata={
                        "source_commit": GSM8K_COMMIT,
                        "source_sha256": source_hash,
                        "source_line_index": line_index,
                        "reference_solution": reference_solution,
                        "raw_answer": raw_answer,
                    },
                )
            )

    if verify_pinned_hash and len(tasks) != spec.expected_lines:
        raise ValueError(
            f"GSM8K {split} line-count mismatch: {len(tasks)} != {spec.expected_lines}"
        )
    return tuple(tasks)


def select_deterministically(
    tasks: Iterable[TaskInstance],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> tuple[TaskInstance, ...]:
    """Select by stable SHA-256 rank rather than mutable RNG implementation details."""

    task_list = list(tasks)
    if count < 0:
        raise ValueError("count cannot be negative")
    if count > len(task_list):
        raise ValueError(f"requested {count} tasks from a pool of {len(task_list)}")

    def rank(task: TaskInstance) -> tuple[str, str]:
        payload = f"{seed}\0{namespace}\0{task.task_id}".encode()
        return hashlib.sha256(payload).hexdigest(), task.task_id

    selected = sorted(task_list, key=rank)[:count]
    return tuple(selected)


def task_collection_fingerprint(tasks: Iterable[TaskInstance]) -> str:
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda item: item.task_id):
        canonical = task.model_dump_json(exclude_none=False)
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_tasks_jsonl(path: str | Path, tasks: Iterable[TaskInstance]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(task.model_dump_json())
            handle.write("\n")


def read_tasks_jsonl(path: str | Path) -> tuple[TaskInstance, ...]:
    tasks: list[TaskInstance] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                task = TaskInstance.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid TaskInstance at line {line_number}") from exc
            if task.task_id in seen_ids:
                raise ValueError(f"duplicate task_id {task.task_id!r}")
            seen_ids.add(task.task_id)
            tasks.append(task)
    return tuple(tasks)
