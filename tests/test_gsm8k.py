import json
from pathlib import Path

import pytest

from topology_mas.data.gsm8k import (
    load_gsm8k,
    read_tasks_jsonl,
    select_deterministically,
    task_collection_fingerprint,
    write_tasks_jsonl,
)
from topology_mas.models import TaskInstance


def write_fixture(path: Path) -> None:
    records = [
        {
            "question": "A has 2 bags with 3 apples each. How many apples?",
            "answer": "Two bags give <<2*3=6>>6 apples.\n#### 6",
        },
        {
            "question": "A box has 10 pens and loses 4. How many remain?",
            "answer": "The remainder is <<10-4=6>>6.\n#### 6",
        },
        {
            "question": "Three groups of four plus one equals what?",
            "answer": "Groups give <<3*4=12>>12, then <<12+1=13>>13.\n#### 13",
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_loader_extracts_answer_and_removes_calculator_annotations(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    write_fixture(source)

    tasks = load_gsm8k(source, split="test", verify_pinned_hash=False)

    assert len(tasks) == 3
    assert tasks[0].task_id == "gsm8k-test-00000"
    assert tasks[0].reference_answer == "6"
    assert "<<" not in tasks[0].metadata["reference_solution"]
    assert tasks[0].metadata["source_line_index"] == 0


def test_loader_rejects_duplicate_questions(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    record = {"question": "Duplicate?", "answer": "Yes.\n#### 1"}
    source.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="duplicate GSM8K question"):
        load_gsm8k(source, split="test", verify_pinned_hash=False)


def test_deterministic_selection_is_order_independent() -> None:
    tasks = tuple(
        TaskInstance(
            task_id=f"t{index}",
            dataset="fixture",
            split="test",
            prompt=f"question {index}",
            reference_answer=str(index),
            oracle_type="numeric",
        )
        for index in range(10)
    )

    selected_a = select_deterministically(tasks, count=4, seed=7, namespace="pilot")
    selected_b = select_deterministically(
        reversed(tasks), count=4, seed=7, namespace="pilot"
    )

    assert [task.task_id for task in selected_a] == [task.task_id for task in selected_b]
    assert task_collection_fingerprint(selected_a) == task_collection_fingerprint(selected_b)


def test_task_jsonl_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    output = tmp_path / "tasks.jsonl"
    write_fixture(source)
    tasks = load_gsm8k(source, split="test", verify_pinned_hash=False)

    write_tasks_jsonl(output, tasks)

    assert read_tasks_jsonl(output) == tasks
