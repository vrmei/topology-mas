"""Strict free-response AIME task records and JSONL loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from topology_mas.models import TaskInstance

AIME_SCHEMA_VERSION = "aime-free-response-v1"

_ILLUSTRATIVE_IMAGE = re.compile(r"<img\b[^>]*?/?>", re.IGNORECASE)
_INLINE_EMPHASIS = re.compile(r"</?(?:i|em|b|strong)>", re.IGNORECASE)
_ANY_HTML_TAG = re.compile(r"<[^>]+>")


def normalize_aime_text_question(raw_question: str) -> str:
    """Convert known presentational HTML into an explicit text-only prompt."""

    normalized = _INLINE_EMPHASIS.sub("", raw_question)
    normalized = _ILLUSTRATIVE_IMAGE.sub(
        "\n[Illustrative diagram omitted.]\n",
        normalized,
    )
    remaining_tags = _ANY_HTML_TAG.findall(normalized)
    if remaining_tags:
        raise ValueError(f"unsupported HTML tags in AIME question: {remaining_tags}")
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


class AIMERecord(BaseModel):
    """Minimal evaluator-side record; only ``problem`` is model-visible."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    mutation_type: Literal["original", "parameter", "structural"]
    problem: str = Field(min_length=1)
    gold_answer: int = Field(ge=0, le=999)

    def to_task_instance(self, *, split: str) -> TaskInstance:
        return TaskInstance(
            task_id=self.task_id,
            dataset="aime",
            split=split,
            prompt=self.problem,
            reference_answer=str(self.gold_answer),
            oracle_type="aime_integer",
            metadata={
                "schema_version": AIME_SCHEMA_VERSION,
                "family_id": self.family_id,
                "mutation_type": self.mutation_type,
            },
        )


def load_aime_jsonl(path: str | Path, *, split: str) -> tuple[TaskInstance, ...]:
    """Load minimal AIME records and reject duplicate IDs or problem text."""

    records: list[AIMERecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(AIMERecord.model_validate(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid AIME record at line {line_number}") from exc

    task_ids = [record.task_id for record in records]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate AIME task_id")
    problems = [record.problem for record in records]
    if len(problems) != len(set(problems)):
        raise ValueError("duplicate AIME problem text")
    return tuple(record.to_task_instance(split=split) for record in records)
