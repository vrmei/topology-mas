"""DeepSeek-based plausibility scoring after deterministic verification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from topology_mas.models import TaskInstance
from topology_mas.mutation.prompts import build_plausibility_messages
from topology_mas.mutation.schemas import MutationCandidate, PlausibilityOracleResult
from topology_mas.providers import JSONChatClient


class _PlausibilityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plausible: bool
    local_error_plausibility: float = Field(ge=0.0, le=1.0)
    global_coherence: float = Field(ge=0.0, le=1.0)
    subtlety: float = Field(ge=0.0, le=1.0)
    minimality: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    rejection_reasons: tuple[str, ...] = ()
    notes: str = ""


class DeepSeekPlausibilityOracle:
    """Apply a preregistered rule to DeepSeek's four quality dimensions."""

    def __init__(
        self,
        client: JSONChatClient,
        *,
        model: str,
        max_output_tokens: int,
        threshold: float,
        minimum_dimension_score: float,
    ) -> None:
        self._client = client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._threshold = threshold
        self._minimum_dimension_score = minimum_dimension_score

    def evaluate(
        self,
        task: TaskInstance,
        candidate: MutationCandidate,
    ) -> PlausibilityOracleResult:
        messages = build_plausibility_messages(
            task,
            candidate_json=candidate.model_dump_json(indent=2),
        )
        completion = self._client.complete_json(
            model=self._model,
            messages=messages,
            max_output_tokens=self._max_output_tokens,
        )
        payload = _PlausibilityPayload.model_validate(completion.content)

        dimensions = (
            payload.local_error_plausibility,
            payload.global_coherence,
            payload.subtlety,
            payload.minimality,
        )
        computed_overall = sum(dimensions) / len(dimensions)
        reasons = list(payload.rejection_reasons)
        if not payload.plausible:
            reasons.append("DeepSeek marked the candidate implausible")
        if computed_overall < self._threshold:
            reasons.append(
                f"computed overall score {computed_overall:.3f} is below "
                f"threshold {self._threshold:.3f}"
            )
        if min(dimensions) < self._minimum_dimension_score:
            reasons.append(
                f"at least one dimension is below {self._minimum_dimension_score:.3f}"
            )

        accepted = bool(
            payload.plausible
            and computed_overall >= self._threshold
            and min(dimensions) >= self._minimum_dimension_score
        )
        return PlausibilityOracleResult(
            model_plausible=payload.plausible,
            plausible=accepted,
            local_error_plausibility=payload.local_error_plausibility,
            global_coherence=payload.global_coherence,
            subtlety=payload.subtlety,
            minimality=payload.minimality,
            overall_score=computed_overall,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
            notes=payload.notes,
            requested_model=completion.requested_model,
            returned_model=completion.returned_model,
            raw_response={
                "final": completion.raw_response,
                "attempts": completion.raw_attempts,
            },
        )
