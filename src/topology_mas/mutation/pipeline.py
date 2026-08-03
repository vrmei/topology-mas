"""End-to-end offline target-error generation for one objectively scored task."""

from __future__ import annotations

from topology_mas.models import AdversarialAnswer, OracleStatus, TaskInstance
from topology_mas.mutation.generator import GenerationValidationError, MutationGenerator
from topology_mas.mutation.numeric_oracle import NumericMutationOracle
from topology_mas.mutation.plausibility import DeepSeekPlausibilityOracle
from topology_mas.mutation.schemas import (
    CandidateEvaluation,
    MutationPipelineConfig,
    MutationRunResult,
)
from topology_mas.mutation.storage import MutationArtifactStore
from topology_mas.providers import InvalidJSONCompletionError, JSONChatClient


class MutationPipeline:
    def __init__(
        self,
        client: JSONChatClient,
        *,
        config: MutationPipelineConfig | None = None,
        artifact_store: MutationArtifactStore | None = None,
    ) -> None:
        self.config = config or MutationPipelineConfig()
        self._generator = MutationGenerator(
            client,
            model=self.config.generator_model,
            candidate_count=self.config.candidate_count,
            max_output_tokens=self.config.generator_max_output_tokens,
        )
        self._objective = NumericMutationOracle()
        self._plausibility = DeepSeekPlausibilityOracle(
            client,
            model=self.config.plausibility_model,
            max_output_tokens=self.config.plausibility_max_output_tokens,
            threshold=self.config.plausibility_threshold,
            minimum_dimension_score=self.config.minimum_dimension_score,
        )
        self._artifact_store = artifact_store

    def run(self, task: TaskInstance) -> MutationRunResult:
        if task.oracle_type != "numeric":
            raise ValueError("the first mutation pipeline supports only numeric tasks")

        try:
            batch, completion, messages = self._generator.generate(task)
        except GenerationValidationError as exc:
            if self._artifact_store is not None:
                self._artifact_store.save_generation_failure(
                    task=task,
                    config=self.config,
                    messages=exc.messages,
                    completion=exc.completion,
                    error=exc.validation_error,
                )
            raise
        if self._artifact_store is not None:
            self._artifact_store.save_generation_stage(
                task=task,
                config=self.config,
                messages=messages,
                completion=completion,
                batch=batch,
            )
        evaluations: list[CandidateEvaluation] = []
        for candidate in batch.candidates:
            objective = self._objective.verify(
                candidate,
                reference_answer=task.reference_answer,
            )
            plausibility = None
            processing_error = None
            if objective.passed:
                try:
                    plausibility = self._plausibility.evaluate(task, candidate)
                except Exception as exc:  # preserve a failed judge without losing other candidates
                    processing_error = self._serialize_processing_error(exc)
            evaluations.append(
                CandidateEvaluation(
                    candidate=candidate,
                    objective=objective,
                    plausibility=plausibility,
                    processing_error=processing_error,
                )
            )

        selected = self._select(evaluations)
        result = MutationRunResult(
            task_id=task.task_id,
            config=self.config,
            generator_request=tuple(messages),
            generator_response={
                "final": completion.raw_response,
                "attempts": completion.raw_attempts,
            },
            evaluations=tuple(evaluations),
            selected_candidate_id=(selected.candidate.candidate_id if selected else None),
        )
        if self._artifact_store is not None:
            self._artifact_store.save(task, result)
        return result

    @staticmethod
    def _serialize_processing_error(exc: Exception) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, InvalidJSONCompletionError):
            payload["raw_attempts"] = exc.raw_attempts
        return payload

    @staticmethod
    def _select(evaluations: list[CandidateEvaluation]) -> CandidateEvaluation | None:
        eligible = [evaluation for evaluation in evaluations if evaluation.eligible]
        if not eligible:
            return None
        return sorted(
            eligible,
            key=lambda evaluation: (
                -evaluation.plausibility.overall_score,  # type: ignore[union-attr]
                -evaluation.plausibility.subtlety,  # type: ignore[union-attr]
                -evaluation.plausibility.minimality,  # type: ignore[union-attr]
                evaluation.candidate.candidate_id,
            ),
        )[0]

    @staticmethod
    def to_adversarial_answer(result: MutationRunResult) -> AdversarialAnswer:
        if result.selected_candidate_id is None:
            raise ValueError("mutation run did not produce an eligible candidate")
        selected = next(
            evaluation
            for evaluation in result.evaluations
            if evaluation.candidate.candidate_id == result.selected_candidate_id
        )
        assert selected.plausibility is not None
        return AdversarialAnswer(
            task_id=result.task_id,
            target_answer=selected.candidate.final_answer,
            rationale=selected.candidate.full_response,
            mutation_type=selected.candidate.mutation_type,
            oracle_status=OracleStatus.PASSED,
            plausibility_score=selected.plausibility.overall_score,
            generator_model=result.config.generator_model,
            metadata={
                "candidate_id": selected.candidate.candidate_id,
                "plausibility_model": result.config.plausibility_model,
                "plausibility_returned_model": selected.plausibility.returned_model,
            },
        )
