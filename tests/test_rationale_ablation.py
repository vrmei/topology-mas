import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_rationale_ablation_pilot import (
    PREPARATION_VERSION,
    build_answer_only_variant,
    hashed_task_order,
)
from scripts.run_rationale_ablation_pilot import seed_pinned_clean_traces
from topology_mas.execution.answers import parse_numeric_answer


def test_answer_only_variant_preserves_target_and_source() -> None:
    original = {
        "task_id": "gsm8k-1",
        "target_answer": "1,234",
        "rationale": "A plausible but invalid derivation.\n#### 1,234",
        "mutation_type": "arithmetic_result",
        "metadata": {"audit": "passed"},
    }
    frozen = copy.deepcopy(original)

    variant = build_answer_only_variant(original)

    assert original == frozen
    assert variant["rationale"] == "#### 1,234"
    assert parse_numeric_answer(variant["rationale"]) == "1234"
    assert variant["target_answer"] == original["target_answer"]
    assert variant["metadata"]["audit"] == "passed"
    assert variant["metadata"]["message_variant"] == "answer_only"
    assert variant["metadata"]["preparation_version"] == PREPARATION_VERSION
    assert variant["metadata"]["source_rationale_sha256"] == hashlib.sha256(
        original["rationale"].encode()
    ).hexdigest()


def test_task_selection_order_depends_only_on_task_id() -> None:
    first = {"task_id": "gsm8k-1", "outcome": 0}
    changed = {"task_id": "gsm8k-1", "outcome": 1, "other": "ignored"}

    assert hashed_task_order(first) == hashed_task_order(changed)


def test_seed_pinned_clean_traces_copies_only_selected_clean_runs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "traces").mkdir(parents=True)
    plan = [
        {
            "run_spec_id": "clean-a",
            "task_id": "a",
            "graph_id": "g-1",
            "condition": "clean",
        },
        {
            "run_spec_id": "attack-a",
            "task_id": "a",
            "graph_id": "g-1",
            "condition": "attack",
        },
        {
            "run_spec_id": "clean-b",
            "task_id": "b",
            "graph_id": "g-1",
            "condition": "clean",
        },
    ]
    (source / "plan.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in plan), encoding="utf-8"
    )
    (source / "traces" / "clean-a.json").write_text("a", encoding="utf-8")
    (source / "traces" / "clean-b.json").write_text("b", encoding="utf-8")

    copied = seed_pinned_clean_traces(
        source_batch=source,
        destination_batch=destination,
        task_ids={"a", "b"},
        graph_id="g-1",
    )

    assert copied == 2
    assert (destination / "traces" / "clean-a.json").read_text(encoding="utf-8") == "a"
    assert (destination / "traces" / "clean-b.json").read_text(encoding="utf-8") == "b"
    assert not (destination / "traces" / "attack-a.json").exists()


def test_seed_pinned_clean_traces_rejects_conflicting_resume(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "traces").mkdir(parents=True)
    (destination / "traces").mkdir(parents=True)
    plan = [
        {
            "run_spec_id": "clean-a",
            "task_id": "a",
            "graph_id": "g-1",
            "condition": "clean",
        }
    ]
    (source / "plan.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in plan), encoding="utf-8"
    )
    (source / "traces" / "clean-a.json").write_text("pinned", encoding="utf-8")
    (destination / "traces" / "clean-a.json").write_text("drift", encoding="utf-8")

    with pytest.raises(RuntimeError, match="differs from pinned source"):
        seed_pinned_clean_traces(
            source_batch=source,
            destination_batch=destination,
            task_ids={"a"},
            graph_id="g-1",
        )
