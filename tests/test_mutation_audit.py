from pathlib import Path

from topology_mas.models import TaskInstance
from topology_mas.mutation.audit import (
    audit_mutation_cache,
    write_mutation_cache_index,
)
from topology_mas.mutation.batch import BatchMutationRunner
from topology_mas.mutation.schemas import MutationPipelineConfig, MutationRunResult
from topology_mas.mutation.storage import MutationArtifactStore


class FakePipeline:
    def __init__(self, root: Path) -> None:
        self.config = MutationPipelineConfig(candidate_count=1)
        self.store = MutationArtifactStore(root / "tasks")

    def run(self, task: TaskInstance) -> MutationRunResult:
        result = MutationRunResult(
            task_id=task.task_id,
            config=self.config,
            generator_request=(),
            generator_response={},
            evaluations=(),
        )
        self.store.save(task, result)
        return result


def test_audit_indexes_complete_no_candidate_cache(tmp_path: Path) -> None:
    tasks = (
        TaskInstance(
            task_id="task-1",
            dataset="synthetic",
            split="test",
            prompt="One plus one?",
            reference_answer="2",
            oracle_type="numeric",
        ),
    )
    pipeline = FakePipeline(tmp_path)
    BatchMutationRunner(pipeline, output_dir=tmp_path).run(tasks)

    audit, answers = audit_mutation_cache(tmp_path)
    write_mutation_cache_index(tmp_path / "selection-index", audit, answers)

    assert audit.result_count == 1
    assert audit.selected_task_count == 0
    assert audit.no_candidate_task_ids == ("task-1",)
    assert answers == ()
    assert (tmp_path / "selection-index" / "audit.json").exists()
