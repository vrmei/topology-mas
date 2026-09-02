import threading
import time

from topology_mas.execution.ready_scheduler import (
    GlobalReadyScheduler,
    ReadyJobSpec,
    ReadySchedulerConfig,
    aggregate_ready_run_costs,
    build_causal_ready_jobs,
)
from topology_mas.execution.schemas import TextGenerationResult
from topology_mas.models import DirectedEdge, GraphSpec


class NamedBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, request):  # pragma: no cover - executor drives the fake directly
        raise AssertionError(request)


def result(text: str) -> TextGenerationResult:
    return TextGenerationResult(
        raw_text=text,
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        metadata={
            "public_output_tokens": 2,
            "raw_parsed_answer": "42",
            "prefill_ms": 0.5,
            "decode_ms": 0.75,
            "batch_size": 4,
        },
    )


def chain() -> GraphSpec:
    return GraphSpec(
        graph_id="chain",
        node_count=3,
        edges=(DirectedEdge(source=0, target=1), DirectedEdge(source=1, target=2)),
        readout_node=2,
        max_rounds=2,
    )


def test_causal_job_builder_preserves_pruning_and_exact_dependencies() -> None:
    jobs = build_causal_ready_jobs(run_id="run-a", task_id="task-a", graph=chain())
    by_turn = {(job.round_index, job.node_id): job for job in jobs}

    assert set(by_turn) == {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)}
    assert set(by_turn[(1, 1)].dependency_ids) == {
        by_turn[(0, 0)].job_id,
        by_turn[(0, 1)].job_id,
    }
    assert set(by_turn[(2, 2)].dependency_ids) == {
        by_turn[(1, 1)].job_id,
        by_turn[(1, 2)].job_id,
    }
    assert all(job.metadata["causal_cone_pruning"] for job in jobs)
    assert by_turn[(0, 1)].downstream_unlock_count == 2


def test_scheduler_uses_precompleted_round_zero_and_multiple_backends() -> None:
    jobs = build_causal_ready_jobs(run_id="run-a", task_id="task-a", graph=chain())
    precompleted = {
        job.job_id: result(f"pre-{job.node_id}")
        for job in jobs
        if job.round_index == 0
    }
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    lock = threading.Lock()

    def execute(job, dependencies, backend):
        with lock:
            calls.append((job.job_id, backend.name, tuple(sorted(dependencies))))
        time.sleep(0.005)
        assert set(dependencies) == set(job.dependency_ids)
        return result(f"done-{job.job_id}")

    scheduler = GlobalReadyScheduler(
        (NamedBackend("gpu-0"), NamedBackend("gpu-1")),
        config=ReadySchedulerConfig(workers_per_backend=1),
    )
    execution = scheduler.run(jobs, execute_job=execute, precompleted=precompleted)

    assert execution.summary.total_jobs == 6
    assert execution.summary.precompleted_jobs == 3
    assert execution.summary.succeeded_jobs == 3
    assert execution.summary.failed_jobs == 0
    assert execution.summary.blocked_jobs == 0
    assert len(calls) == 3
    assert {backend for _, backend, _ in calls} == {"gpu-0", "gpu-1"}
    assert execution.summary.known_input_tokens == 60
    assert execution.summary.known_output_tokens == 30
    costs = aggregate_ready_run_costs(jobs=jobs, result=execution)
    assert costs[0].physical_calls == 3
    assert costs[0].round_zero_pool_hits == 3
    assert costs[0].round_zero_new_generations == 0
    assert costs[0].summary_tokens == 12
    assert costs[0].maximum_prompt_tokens == 10


def test_failed_job_blocks_only_its_descendants_and_other_runs_continue() -> None:
    jobs_a = (
        ReadyJobSpec(
            job_id="a0",
            run_id="a",
            task_id="task",
            graph_id="graph",
            round_index=0,
            node_id=0,
            distance_to_readout=1,
        ),
        ReadyJobSpec(
            job_id="a1",
            run_id="a",
            task_id="task",
            graph_id="graph",
            round_index=1,
            node_id=1,
            distance_to_readout=0,
            dependency_ids=("a0",),
        ),
    )
    jobs_b = (
        ReadyJobSpec(
            job_id="b0",
            run_id="b",
            task_id="task",
            graph_id="graph",
            round_index=0,
            node_id=0,
            distance_to_readout=1,
            estimated_prompt_tokens=5000,
        ),
    )

    def execute(job, dependencies, backend):
        del dependencies, backend
        if job.job_id == "a0":
            raise RuntimeError("intentional failure")
        return result(job.job_id)

    execution = GlobalReadyScheduler((NamedBackend("gpu-0"),)).run(
        jobs_a + jobs_b,
        execute_job=execute,
    )
    by_id = {outcome.job_id: outcome for outcome in execution.outcomes}

    assert by_id["a0"].status == "failed"
    assert by_id["a1"].status == "blocked"
    assert by_id["b0"].status == "succeeded"
    assert by_id["b0"].prompt_length_bucket == 8192
    assert execution.summary.failed_jobs == 1
    assert execution.summary.blocked_jobs == 1


def test_global_parallel_schedule_preserves_fixed_dependency_result() -> None:
    jobs = build_causal_ready_jobs(run_id="run-a", task_id="task-a", graph=chain())

    def execute(job, dependencies, backend):
        del backend
        dependency_text = "+".join(
            dependencies[dependency].raw_text for dependency in sorted(dependencies)
        )
        return result(f"{job.round_index}:{job.node_id}[{dependency_text}]")

    sequential = GlobalReadyScheduler(
        (NamedBackend("gpu-0"),),
        config=ReadySchedulerConfig(workers_per_backend=1),
    ).run(jobs, execute_job=execute)
    parallel = GlobalReadyScheduler(
        (NamedBackend("gpu-0"), NamedBackend("gpu-1")),
        config=ReadySchedulerConfig(workers_per_backend=2),
    ).run(jobs, execute_job=execute)

    assert {
        job_id: output.raw_text for job_id, output in sequential.outputs.items()
    } == {job_id: output.raw_text for job_id, output in parallel.outputs.items()}
