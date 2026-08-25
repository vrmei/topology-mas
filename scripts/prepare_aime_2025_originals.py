#!/usr/bin/env python3
"""Freeze the 30 original 2025 AIME I/II free-response tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

from topology_mas.data.aime import AIMERecord

SOURCE_URLS = {
    "2025_AIME_I": "https://live.poshenloh.com/past-contests/aime/2025I",
    "2025_AIME_II": "https://live.poshenloh.com/past-contests/aime/2025II",
}


class NextDataParser(HTMLParser):
    """Extract Next.js page data without adding an HTML-parser dependency."""

    def __init__(self) -> None:
        super().__init__()
        self._inside = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self._inside = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._inside:
            self._inside = False

    def handle_data(self, data: str) -> None:
        if self._inside:
            self.parts.append(data)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, content: str) -> None:
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


def extract_questions(html: str) -> list[dict[str, Any]]:
    parser = NextDataParser()
    parser.feed(html)
    if not parser.parts:
        raise ValueError("source page does not contain __NEXT_DATA__")
    payload = json.loads("".join(parser.parts))
    questions = payload["props"]["pageProps"]["baseQuestions"]
    if not isinstance(questions, list):
        raise ValueError("baseQuestions is not a list")
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/aime/original_2025.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/aime/original_2025.manifest.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[AIMERecord] = []
    source_fingerprints: dict[str, str] = {}
    upstream_problem_numbers: dict[str, list[object]] = {}
    with httpx.Client(timeout=args.timeout_seconds, follow_redirects=True) as client:
        for contest_id, url in SOURCE_URLS.items():
            response = client.get(url)
            response.raise_for_status()
            questions = extract_questions(response.text)
            if len(questions) != 15:
                raise ValueError(f"{contest_id} has {len(questions)} rather than 15 tasks")
            # The page renders ``baseQuestions`` in contest order. Its legacy
            # ``aimeProblemNumber`` metadata is not reliable (it contains
            # duplicates), so problem identity is deliberately the frozen list
            # position rather than that auxiliary field.
            upstream_problem_numbers[contest_id] = [
                row.get("aimeProblemNumber") for row in questions
            ]
            source_fingerprints[contest_id] = canonical_sha256(questions)
            for number, row in enumerate(questions, start=1):
                record = AIMERecord(
                    family_id=f"{contest_id}_P{number:02d}",
                    task_id=f"{contest_id}_P{number:02d}",
                    mutation_type="original",
                    problem=str(row["question"]).strip(),
                    gold_answer=int(row["answer"]),
                )
                records.append(record)

    task_ids = [record.task_id for record in records]
    if len(records) != 30 or len(set(task_ids)) != 30:
        raise ValueError("expected exactly 30 unique original AIME tasks")
    jsonl = "".join(record.model_dump_json() + "\n" for record in records)
    dataset_lf_sha256 = hashlib.sha256(jsonl.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset": "2025 AIME I + 2025 AIME II",
        "task_count": len(records),
        "task_ids": task_ids,
        "source_urls": SOURCE_URLS,
        "source_base_questions_sha256": source_fingerprints,
        "source_ordering_note": (
            "Problem numbers use baseQuestions list order; upstream "
            "aimeProblemNumber metadata is non-unique."
        ),
        "upstream_aime_problem_numbers": upstream_problem_numbers,
        # Hash the canonical LF serialization. Git may materialize CRLF on a
        # Windows checkout without changing the logical dataset.
        "output_lf_sha256": dataset_lf_sha256,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "normal_agent_visible_fields": ["problem"],
    }
    atomic_text(args.output, jsonl)
    atomic_text(
        args.manifest,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "manifest": str(args.manifest.resolve()),
                "task_count": len(records),
                "output_lf_sha256": dataset_lf_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
