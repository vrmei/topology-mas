from pathlib import Path

import pytest

from topology_mas.execution import TextGenerationRequest, TextGenerationResult
from topology_mas.execution import round_zero as round_zero_module
from topology_mas.execution.prompts import PROMPT_VERSION
from topology_mas.execution.round_zero import (
    RoundZeroCache,
    RoundZeroCacheConfig,
    RoundZeroCacheConflictError,
    RoundZeroGenerator,
)
from topology_mas.execution.seeding import node_round_seed
from topology_mas.models import AnswerState, TaskInstance


class CountingGenerator:
    def __init__(self) -> None:
        self.requests: list[TextGenerationRequest] = []

    def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        self.requests.append(request)
        return TextGenerationResult(
            raw_text="Independent work.\nFINAL_ANSWER: 42",
            model_name="pinned-model",
            finish_reason="stop",
            input_tokens=20,
            output_tokens=8,
            latency_ms=2.0,
            metadata={"provider_request_id": request.request_id},
        )


def tasks() -> tuple[TaskInstance, ...]:
    return (
        TaskInstance(
            task_id="task-1",
            dataset="synthetic",
            split="test",
            prompt="What is 40 + 2?",
            reference_answer="42",
            oracle_type="numeric",
        ),
    )


def config() -> RoundZeroCacheConfig:
    return RoundZeroCacheConfig(
        node_count=3,
        seeds=(0, 1),
        requested_model="model-alias",
        expected_returned_model="pinned-model",
        temperature=0.3,
        max_output_tokens=64,
    )


def test_round_zero_generation_is_graph_independent_and_resume_safe(tmp_path: Path) -> None:
    generator = CountingGenerator()
    cache = RoundZeroCache(tmp_path)
    runner = RoundZeroGenerator(generator, config=config(), cache=cache)

    first_result = runner.generate(tasks())
    second_result = runner.generate(tasks())
    first = first_result.records
    second = second_result.records

    assert first == second
    assert first_result.generated_count == 6
    assert first_result.reused_count == 0
    assert second_result.generated_count == 0
    assert second_result.reused_count == 6
    assert len(first) == 6
    assert len(generator.requests) == 6
    assert all(record.prompt_version == PROMPT_VERSION for record in first)
    assert all(record.answer_state is AnswerState.CORRECT for record in first)
    assert all(record.prompt_messages == first[0].prompt_messages for record in first)
    assert (tmp_path / "manifest.json").exists()
    assert len(tuple((tmp_path / "records").rglob("node_*.json"))) == 6


def test_round_zero_seed_matches_execution_seed_contract(tmp_path: Path) -> None:
    records = RoundZeroGenerator(
        CountingGenerator(),
        config=config().model_copy(update={"seeds": (7,)}),
        cache=RoundZeroCache(tmp_path),
    ).generate(tasks()).records

    for record in records:
        assert record.generation_seed == node_round_seed(
            experiment_seed=7,
            task_id="task-1",
            node_id=record.node_id,
            round_index=0,
        )


def test_manifest_change_fails_instead_of_mixing_cache_conditions(tmp_path: Path) -> None:
    RoundZeroGenerator(
        CountingGenerator(), config=config(), cache=RoundZeroCache(tmp_path)
    ).generate(tasks())
    changed = config().model_copy(update={"temperature": 0.9})

    with pytest.raises(RoundZeroCacheConflictError, match="new output directory"):
        RoundZeroGenerator(
            CountingGenerator(), config=changed, cache=RoundZeroCache(tmp_path)
        ).generate(tasks())


def test_task_change_fails_instead_of_reusing_manifest(tmp_path: Path) -> None:
    RoundZeroGenerator(
        CountingGenerator(), config=config(), cache=RoundZeroCache(tmp_path)
    ).generate(tasks())
    changed_task = tasks()[0].model_copy(update={"prompt": "A changed problem"})

    with pytest.raises(RoundZeroCacheConflictError, match="new output directory"):
        RoundZeroGenerator(
            CountingGenerator(), config=config(), cache=RoundZeroCache(tmp_path)
        ).generate((changed_task,))


def test_atomic_write_never_replaces_good_file_with_partial_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "record.json"
    round_zero_module._atomic_write_json(target, {"state": "complete"})
    original = target.read_text(encoding="utf-8")

    def interrupted_dump(payload: object, handle: object, **_: object) -> None:
        assert payload == {"state": "new"}
        handle.write("{")  # type: ignore[attr-defined]
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(round_zero_module.json, "dump", interrupted_dump)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        round_zero_module._atomic_write_json(target, {"state": "new"})

    assert target.read_text(encoding="utf-8") == original
    assert not tuple(tmp_path.glob("*.tmp"))
