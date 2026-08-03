"""Frontier-model candidate generation with strict schema validation."""

from __future__ import annotations

from pydantic import ValidationError

from topology_mas.models import TaskInstance
from topology_mas.mutation.prompts import build_generator_messages
from topology_mas.mutation.schemas import CandidateBatch
from topology_mas.providers import JSONChatClient, JSONCompletion


class GenerationValidationError(ValueError):
    """Schema failure carrying the raw completion for audit persistence."""

    def __init__(
        self,
        *,
        completion: JSONCompletion,
        messages: list[dict[str, str]],
        validation_error: ValidationError,
    ) -> None:
        super().__init__("mutation generator response failed schema validation")
        self.completion = completion
        self.messages = messages
        self.validation_error = validation_error


class MutationGenerator:
    def __init__(
        self,
        client: JSONChatClient,
        *,
        model: str,
        candidate_count: int,
        max_output_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._candidate_count = candidate_count
        self._max_output_tokens = max_output_tokens

    def generate(
        self,
        task: TaskInstance,
    ) -> tuple[CandidateBatch, JSONCompletion, list[dict[str, str]]]:
        messages = build_generator_messages(task, candidate_count=self._candidate_count)
        completion = self._client.complete_json(
            model=self._model,
            messages=messages,
            max_output_tokens=self._max_output_tokens,
        )
        try:
            batch = CandidateBatch.model_validate(completion.content)
        except ValidationError as exc:
            raise GenerationValidationError(
                completion=completion,
                messages=messages,
                validation_error=exc,
            ) from exc
        if len(batch.candidates) != self._candidate_count:
            raise ValueError(
                f"generator returned {len(batch.candidates)} candidates; "
                f"expected {self._candidate_count}"
            )
        return batch, completion, messages
