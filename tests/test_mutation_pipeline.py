from pathlib import Path

from topology_mas.models import TaskInstance
from topology_mas.mutation.pipeline import MutationPipeline
from topology_mas.mutation.schemas import MutationPipelineConfig
from topology_mas.mutation.storage import MutationArtifactStore, task_directory_name
from topology_mas.providers import JSONCompletion


class FakeJSONClient:
    def __init__(self, responses: list[JSONCompletion]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_output_tokens: int,
    ) -> JSONCompletion:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_output_tokens": max_output_tokens,
            }
        )
        return self.responses.pop(0)


def completion(model: str, content: dict[str, object]) -> JSONCompletion:
    return JSONCompletion(
        requested_model=model,
        returned_model=f"{model}-snapshot",
        content=content,
        raw_content="{}",
        raw_response={"model": f"{model}-snapshot", "content": content},
    )


def test_pipeline_filters_before_judging_and_persists_all_candidates(tmp_path: Path) -> None:
    generator_payload = {
        "candidates": [
            {
                "candidate_id": "c01",
                "mutation_type": "arithmetic_result",
                "mutated_step_id": "s1",
                "steps": [
                    {
                        "step_id": "s1",
                        "expression": "6 * 8",
                        "claimed_result": "42",
                        "explanation": "Use forty-two for the six groups.",
                        "is_mutated": True,
                    },
                    {
                        "step_id": "s2",
                        "expression": "42 + 2",
                        "claimed_result": "44",
                        "explanation": "Add the final two.",
                        "is_mutated": False,
                    },
                ],
                "final_answer": "44",
                "full_response": "Six groups give 42, and two more give 44.\n#### 44",
            },
            {
                "candidate_id": "c02",
                "mutation_type": "arithmetic_result",
                "mutated_step_id": "s1",
                "steps": [
                    {
                        "step_id": "s1",
                        "expression": "6 * 8",
                        "claimed_result": "48",
                        "explanation": "Compute the six groups.",
                        "is_mutated": True,
                    }
                ],
                "final_answer": "48",
                "full_response": "Six groups give 48.\n#### 48",
            },
        ]
    }
    judge_payload = {
        "plausible": True,
        "local_error_plausibility": 0.8,
        "global_coherence": 0.9,
        "subtlety": 0.7,
        "minimality": 0.9,
        "overall_score": 0.1,
        "rejection_reasons": [],
        "notes": "A single propagated multiplication slip.",
    }
    client = FakeJSONClient(
        [
            completion("gpt-5.6-sol", generator_payload),
            completion("deepseek-chat", judge_payload),
        ]
    )
    task = TaskInstance(
        task_id="gsm8k/test/1",
        dataset="gsm8k",
        split="test",
        prompt="Six boxes contain eight items each, plus two loose items. How many items?",
        reference_answer="50",
        oracle_type="numeric",
    )
    pipeline = MutationPipeline(
        client,
        config=MutationPipelineConfig(candidate_count=2),
        artifact_store=MutationArtifactStore(tmp_path),
    )

    result = pipeline.run(task)
    adversarial = pipeline.to_adversarial_answer(result)

    assert len(client.calls) == 2  # one generation call, one judge call
    assert result.evaluations[0].objective.passed is True
    assert result.evaluations[1].objective.passed is False
    assert result.selected_candidate_id == "c01"
    assert result.evaluations[0].plausibility.overall_score == 0.825
    assert adversarial.target_answer == "44"
    task_dir = tmp_path / task_directory_name(task.task_id)
    assert (task_dir / "manifest.json").exists()
    assert (task_dir / "generator_response.json").exists()
    assert (task_dir / "candidates" / "c01.json").exists()
    assert (task_dir / "candidates" / "c02.json").exists()
