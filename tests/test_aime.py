from __future__ import annotations

import json
from pathlib import Path

import pytest

from topology_mas.data.aime import (
    AIMERecord,
    load_aime_jsonl,
    normalize_aime_text_question,
)
from topology_mas.execution.aime import (
    build_aime_round_zero_messages,
    parse_aime_answer,
)
from topology_mas.experiments.aime_utility import build_round_zero_plan


def test_aime_parser_accepts_explicit_integer_answers() -> None:
    assert parse_aime_answer("work\nFINAL_ANSWER: \\boxed{028}") == "28"
    assert parse_aime_answer("work\n\\boxed{999}") == "999"
    assert parse_aime_answer("FINAL ANSWER: 7") == "7"
    assert parse_aime_answer("the calculation contains 28") is None


def test_aime_prompt_hides_evaluator_fields() -> None:
    record = AIMERecord(
        family_id="family-secret",
        task_id="task-secret",
        mutation_type="parameter",
        problem="What is 14+14?",
        gold_answer=28,
    )
    task = record.to_task_instance(split="test")
    messages = build_aime_round_zero_messages(task)
    visible = "\n".join(message.content for message in messages)
    assert task.prompt in visible
    assert "gold_answer" not in visible
    assert "family-secret" not in visible
    assert "task-secret" not in visible
    assert "28" not in visible
    assert "candidate" not in visible.lower()


def test_aime_text_normalization_is_explicit_and_rejects_unknown_html() -> None:
    raw = (
        "Define an <i>object</i>.\n\n"
        '<img src="/aime/example.svg" style="max-height: 4rem"/>\n\n'
        "Find its size."
    )
    assert normalize_aime_text_question(raw) == (
        "Define an object.\n\n[Illustrative diagram omitted.]\n\nFind its size."
    )
    with pytest.raises(ValueError, match="unsupported HTML"):
        normalize_aime_text_question("A <table><tr></tr></table>")


def test_aime_loader_rejects_extra_fields_and_duplicates(tmp_path: Path) -> None:
    valid = {
        "family_id": "f",
        "task_id": "t",
        "mutation_type": "original",
        "problem": "problem",
        "gold_answer": 1,
    }
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        json.dumps(valid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate AIME task_id"):
        load_aime_jsonl(duplicate, split="test")

    extra = tmp_path / "extra.jsonl"
    extra.write_text(json.dumps({**valid, "target_error": 2}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid AIME record"):
        load_aime_jsonl(extra, split="test")


def test_round_zero_plan_is_complete_deterministic_and_candidate_free() -> None:
    tasks = tuple(
        AIMERecord(
            family_id=f"f-{index}",
            task_id=f"t-{index}",
            mutation_type="original",
            problem=f"Problem number {index}.",
            gold_answer=index,
        ).to_task_instance(split="test")
        for index in range(30)
    )
    first = build_round_zero_plan(tasks, replicates=10)
    second = build_round_zero_plan(tasks, replicates=10)
    assert first == second
    assert len(first) == 300
    assert len({row["request_id"] for row in first}) == 300
    assert len({row["generation_seed"] for row in first}) == 300
    for row in first:
        visible = "\n".join(message["content"] for message in row["messages"])
        assert "candidate" not in visible.lower()
        assert "gold_answer" not in visible
