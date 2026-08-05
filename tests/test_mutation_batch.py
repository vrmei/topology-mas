import json
from pathlib import Path

import pytest

from topology_mas.models import TaskInstance
from topology_mas.mutation.batch import (
    BatchCacheConflictError,
    BatchDisposition,
    BatchMutationRunner,
)
from topology_mas.mutation.schemas import (
    ArithmeticStep,
    CandidateEvaluation,
    MutationCandidate,
    MutationPipelineConfig,
    MutationRunResult,
    ObjectiveOracleResult,
    PlausibilityOracleResult,
)
from topology_mas.mutation.storage import MutationArtifactStore, task_directory_name


def task(prompt: str = "What is one plus one?") -> TaskInstance:
    return TaskInstance(
        task_id="gsm8k/test/1",
        dataset="gsm8k",
        split="test",
        prompt=prompt,
        reference_answer="2",
        oracle_type="numeric",
    )


def task_with_id(task_id: str) -> TaskInstance:
    return task().model_copy(update={"task_id": task_id})


def selected_result(task_id: str, config: MutationPipelineConfig) -> MutationRunResult:
    candidate = MutationCandidate(
        candidate_id="c01",
        mutation_type="arithmetic_result",
        mutated_step_id="s1",
        steps=(
            ArithmeticStep(
                step_id="s1",
                expression="1 + 1",
                claimed_result="3",
                explanation="A local addition slip.",
                is_mutated=True,
                depends_on=(),
            ),
            ArithmeticStep(
                step_id="s2",
                expression="s1 + 0",
                claimed_result="3",
                explanation="Carry the intermediate result to the answer.",
                depends_on=("s1",),
            ),
        ),
        final_answer="3",
        full_response="One plus one is three.\n#### 3",
    )
    evaluation = CandidateEvaluation(
        candidate=candidate,
        objective=ObjectiveOracleResult(passed=True),
        plausibility=PlausibilityOracleResult(
            model_plausible=True,
            plausible=True,
            local_error_plausibility=0.8,
            global_coherence=0.8,
            subtlety=0.8,
            minimality=1.0,
            overall_score=0.85,
        ),
    )
    return MutationRunResult(
        task_id=task_id,
        config=config,
        generator_request=(),
        generator_response={},
        evaluations=(evaluation,),
        selected_candidate_id="c01",
    )


class FakePipeline:
    def __init__(self, output_dir: Path, *, selected: bool = True) -> None:
        self.config = MutationPipelineConfig(candidate_count=1)
        self.calls: list[str] = []
        self._selected = selected
        self._store = MutationArtifactStore(output_dir / "tasks")

    def run(self, item: TaskInstance) -> MutationRunResult:
        self.calls.append(item.task_id)
        if self._selected:
            result = selected_result(item.task_id, self.config)
        else:
            result = MutationRunResult(
                task_id=item.task_id,
                config=self.config,
                generator_request=(),
                generator_response={},
                evaluations=(),
            )
        self._store.save(item, result)
        return result


@pytest.mark.parametrize("selected", [True, False])
def test_batch_reuses_terminal_result_without_resampling(
    tmp_path: Path,
    selected: bool,
) -> None:
    pipeline = FakePipeline(tmp_path, selected=selected)
    runner = BatchMutationRunner(pipeline, output_dir=tmp_path)

    first_outcomes, _ = runner.run((task(),))
    second_outcomes, second_summary = runner.run((task(),))

    assert pipeline.calls == ["gsm8k/test/1"]
    assert first_outcomes[0].disposition is (
        BatchDisposition.GENERATED_SELECTED
        if selected
        else BatchDisposition.GENERATED_NO_CANDIDATE
    )
    assert second_outcomes[0].disposition is (
        BatchDisposition.CACHED_SELECTED
        if selected
        else BatchDisposition.CACHED_NO_CANDIDATE
    )
    assert second_summary.errors == 0
    assert (
        tmp_path / "tasks" / task_directory_name("gsm8k/test/1") / "result.json"
    ).exists()


def test_batch_rejects_changed_task_collection_in_same_output_dir(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path)
    runner = BatchMutationRunner(pipeline, output_dir=tmp_path)
    runner.run((task(),))

    with pytest.raises(BatchCacheConflictError, match="new output directory"):
        runner.run((task(prompt="A changed prompt"),))


def test_batch_rejects_cached_result_from_old_oracle_version(tmp_path: Path) -> None:
    pipeline = FakePipeline(tmp_path)
    runner = BatchMutationRunner(pipeline, output_dir=tmp_path)
    runner.run((task(),))
    task_manifest = (
        tmp_path
        / "tasks"
        / task_directory_name("gsm8k/test/1")
        / "manifest.json"
    )
    manifest = json.loads(task_manifest.read_text(encoding="utf-8"))
    manifest["objective_oracle_version"] = "old-oracle"
    task_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BatchCacheConflictError, match="objective Oracle"):
        runner.run((task(),))


def test_parallel_batch_preserves_input_order_and_caches_every_task(
    tmp_path: Path,
) -> None:
    tasks = tuple(task_with_id(f"task-{index}") for index in range(6))
    pipeline = FakePipeline(tmp_path)
    runner = BatchMutationRunner(pipeline, output_dir=tmp_path, max_workers=3)

    outcomes, summary = runner.run(tasks)
    resumed, resumed_summary = runner.run(tasks)

    assert [outcome.task_id for outcome in outcomes] == [item.task_id for item in tasks]
    assert set(pipeline.calls) == {item.task_id for item in tasks}
    assert summary.generated_selected == 6
    assert resumed_summary.cached_selected == 6
    assert all(
        outcome.disposition is BatchDisposition.CACHED_SELECTED
        for outcome in resumed
    )
