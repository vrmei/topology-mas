"""Utilities for the paired evidence-composition intervention."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from topology_mas.execution.prompts import build_node_messages
from topology_mas.execution.schemas import ChatMessage
from topology_mas.execution.seeding import anonymous_message_order_key, stable_id
from topology_mas.models import AnswerState, MessageRecord, TaskInstance

EXPERIMENT_VERSION = "evidence-volume-intervention-v1"
SCENARIOS = {
    "attack_adoption": {
        "previous_state": AnswerState.CORRECT.value,
        "error_state": AnswerState.TARGET_ERROR.value,
        "primary_state": AnswerState.TARGET_ERROR.value,
    },
    "benign_correction": {
        "previous_state": AnswerState.OTHER_ERROR.value,
        "error_state": AnswerState.OTHER_ERROR.value,
        "primary_state": AnswerState.CORRECT.value,
    },
}


@dataclass(frozen=True)
class RatioDesign:
    ratio_id: str
    base_correct: int
    base_error: int

    @property
    def correct_share(self) -> float:
        return self.base_correct / (self.base_correct + self.base_error)


RATIO_DESIGNS = (
    RatioDesign("c100_e0", 2, 0),
    RatioDesign("c80_e20", 4, 1),
    RatioDesign("c75_e25", 3, 1),
    RatioDesign("c67_e33", 2, 1),
    RatioDesign("c50_e50", 1, 1),
)
VOLUME_MULTIPLIERS = (1, 2, 3)


def normalize_stimulus_text(text: str) -> str:
    """Normalize only whitespace; do not rewrite model-generated evidence."""

    return " ".join(text.split())


def valid_stimulus_text(text: str) -> bool:
    lowered = text.lower()
    return bool(text.strip()) and "<peer_message" not in lowered and "</peer_message" not in lowered


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def stimulus_id(task_id: str, state: str, raw_text: str) -> str:
    return stable_id(
        "stimulus", EXPERIMENT_VERSION, task_id, state, normalize_stimulus_text(raw_text)
    )


def content_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


class BoundedStimulusPool:
    """Keep a deterministic hash sample without retaining every trace output."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._records: dict[str, dict[str, Any]] = {}

    def add(self, record: Mapping[str, Any]) -> None:
        item = dict(record)
        key = str(item["stimulus_id"])
        if key in self._records:
            return
        if len(self._records) < self.capacity:
            self._records[key] = item
            return
        largest = max(self._records)
        if key < largest:
            del self._records[largest]
            self._records[key] = item

    def records(self) -> list[dict[str, Any]]:
        return [self._records[key] for key in sorted(self._records)]


def select_nested_ids(
    ids: Sequence[str],
    *,
    maximum: int,
    seed_parts: Sequence[object],
    excluded: set[str] | None = None,
) -> tuple[str, ...]:
    available = [item for item in ids if item not in (excluded or set())]
    if len(available) < maximum:
        raise ValueError(f"need {maximum} distinct stimuli, found {len(available)}")
    random.Random(stable_seed(*seed_parts)).shuffle(available)
    return tuple(available[:maximum])


def build_request_plan(
    *,
    task_ids: Sequence[str],
    pool_by_task_state: Mapping[tuple[str, str], Sequence[str]],
    replicates: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for scenario, settings in SCENARIOS.items():
            previous_state = settings["previous_state"]
            error_state = settings["error_state"]
            correct_pool = tuple(pool_by_task_state[(task_id, AnswerState.CORRECT.value)])
            error_pool = tuple(pool_by_task_state[(task_id, error_state)])
            previous_pool = tuple(pool_by_task_state[(task_id, previous_state)])
            for ratio in RATIO_DESIGNS:
                maximum_correct = ratio.base_correct * max(VOLUME_MULTIPLIERS)
                maximum_error = ratio.base_error * max(VOLUME_MULTIPLIERS)
                for replicate in range(replicates):
                    previous_ids = list(previous_pool)
                    random.Random(
                        stable_seed(
                            EXPERIMENT_VERSION, task_id, scenario, ratio.ratio_id, replicate, "prev"
                        )
                    ).shuffle(previous_ids)
                    previous_id = previous_ids[0]
                    correct_ids = select_nested_ids(
                        correct_pool,
                        maximum=maximum_correct,
                        seed_parts=(
                            EXPERIMENT_VERSION,
                            task_id,
                            scenario,
                            ratio.ratio_id,
                            replicate,
                            "correct",
                        ),
                        excluded={previous_id},
                    )
                    error_ids = select_nested_ids(
                        error_pool,
                        maximum=maximum_error,
                        seed_parts=(
                            EXPERIMENT_VERSION,
                            task_id,
                            scenario,
                            ratio.ratio_id,
                            replicate,
                            "error",
                        ),
                        excluded={previous_id},
                    )
                    generation_seed = stable_seed(
                        EXPERIMENT_VERSION,
                        task_id,
                        scenario,
                        ratio.ratio_id,
                        replicate,
                        "generation",
                    )
                    for multiplier in VOLUME_MULTIPLIERS:
                        peer_ids = (
                            *correct_ids[: ratio.base_correct * multiplier],
                            *error_ids[: ratio.base_error * multiplier],
                        )
                        request_id = stable_id(
                            "evidence-request",
                            EXPERIMENT_VERSION,
                            task_id,
                            scenario,
                            ratio.ratio_id,
                            replicate,
                            multiplier,
                        )
                        rows.append(
                            {
                                "request_id": request_id,
                                "task_id": task_id,
                                "scenario": scenario,
                                "previous_state": previous_state,
                                "error_state": error_state,
                                "primary_state": settings["primary_state"],
                                "ratio_id": ratio.ratio_id,
                                "correct_share": ratio.correct_share,
                                "error_share": 1.0 - ratio.correct_share,
                                "multiplier": multiplier,
                                "correct_count": ratio.base_correct * multiplier,
                                "error_count": ratio.base_error * multiplier,
                                "incoming_degree": len(peer_ids),
                                "replicate": replicate,
                                "generation_seed": generation_seed,
                                "previous_stimulus_id": previous_id,
                                "peer_stimulus_ids": list(peer_ids),
                                "peer_set_fingerprint": content_fingerprint(
                                    {"stimulus_id": item} for item in sorted(peer_ids)
                                ),
                            }
                        )
    request_ids = [row["request_id"] for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request plan contains duplicate IDs")
    return rows


def render_request_messages(
    *,
    task: TaskInstance,
    plan_row: Mapping[str, Any],
    stimuli: Mapping[str, Mapping[str, Any]],
    message_order_seed: int = 0,
    include_previous: bool = True,
) -> tuple[ChatMessage, ...]:
    previous = (
        str(stimuli[str(plan_row["previous_stimulus_id"])]["raw_text"])
        if include_previous
        else None
    )
    records: list[MessageRecord] = []
    for stimulus_key in plan_row["peer_stimulus_ids"]:
        item = stimuli[str(stimulus_key)]
        records.append(
            MessageRecord(
                message_id=str(stimulus_key),
                run_id="evidence-volume-intervention",
                task_id=task.task_id,
                graph_id="no-topology",
                round_index=0,
                sender=0,
                recipients=(1,),
                raw_text=str(item["raw_text"]),
                parsed_answer=item.get("parsed_answer"),
                answer_state=AnswerState(str(item["state"])),
            )
        )
    ordered = tuple(
        sorted(
            records,
            key=lambda item: anonymous_message_order_key(
                order_seed=message_order_seed,
                task_id=task.task_id,
                round_index=1,
                raw_text=item.raw_text,
            ),
        )
    )
    return build_node_messages(
        task,
        previous_output=previous,
        incoming_messages=ordered,
        allow_peer_only_update=not include_previous,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


_NODE_COUNT = re.compile(r"(?:^|-)n(?P<n>\d+)(?:-|$)")


def source_node_count(graph_id: str) -> int | None:
    match = _NODE_COUNT.search(graph_id)
    return int(match.group("n")) if match else None
