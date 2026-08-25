"""Frozen request planning and storage helpers for original-AIME utility tests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from topology_mas.execution.aime import (
    AIME_PROMPT_VERSION,
    build_aime_round_zero_messages,
)
from topology_mas.execution.seeding import stable_id, stable_integer
from topology_mas.models import TaskInstance

AIME_UTILITY_VERSION = "aime-original-round0-utility-v1"


def canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
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
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
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


def build_round_zero_plan(
    tasks: tuple[TaskInstance, ...],
    *,
    replicates: int,
) -> list[dict[str, Any]]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if len(tasks) != 30:
        raise ValueError("original AIME calibration requires exactly 30 tasks")
    if any(task.oracle_type != "aime_integer" for task in tasks):
        raise ValueError("all tasks must use the AIME integer oracle")

    rows: list[dict[str, Any]] = []
    for task in tasks:
        messages = build_aime_round_zero_messages(task)
        visible_messages = [message.model_dump() for message in messages]
        prompt_fingerprint = canonical_fingerprint(visible_messages)
        for replicate in range(replicates):
            generation_seed = stable_integer(
                AIME_UTILITY_VERSION,
                task.task_id,
                replicate,
            )
            request_id = stable_id(
                "aime-r0",
                AIME_UTILITY_VERSION,
                task.task_id,
                replicate,
            )
            rows.append(
                {
                    "request_id": request_id,
                    "task_id": task.task_id,
                    "replicate": replicate,
                    "generation_seed": generation_seed,
                    "prompt_version": AIME_PROMPT_VERSION,
                    "prompt_fingerprint": prompt_fingerprint,
                    "messages": visible_messages,
                }
            )
    if len({row["request_id"] for row in rows}) != len(rows):
        raise RuntimeError("request plan contains duplicate IDs")
    return rows


def result_path(output_dir: Path, row: dict[str, Any]) -> Path:
    return (
        output_dir
        / "results"
        / str(row["task_id"])
        / f"replicate_{int(row['replicate']):02d}.json"
    )
