"""Validated records for mutation generation, filtering, and selection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ArithmeticStep(BaseModel):
    """One explicitly checkable arithmetic step in a candidate solution."""

    model_config = ConfigDict(frozen=True)

    step_id: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    expression: str = Field(min_length=1)
    claimed_result: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    is_mutated: bool = False
    depends_on: tuple[str, ...]

    @field_validator("claimed_result", mode="before")
    @classmethod
    def stringify_claimed_result(cls, value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        return value


class MutationCandidate(BaseModel):
    """A complete wrong solution containing exactly one declared arithmetic mutation."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(min_length=1)
    mutation_type: Literal["arithmetic_result"]
    mutated_step_id: str = Field(min_length=1)
    steps: tuple[ArithmeticStep, ...] = Field(min_length=1, max_length=10)
    final_answer: str = Field(min_length=1)
    full_response: str = Field(min_length=1)

    @field_validator("final_answer", mode="before")
    @classmethod
    def stringify_final_answer(cls, value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value.strip().removeprefix("####").strip()
        return value

    @model_validator(mode="after")
    def validate_declared_mutation(self) -> MutationCandidate:
        step_ids = [step.step_id for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("step_id values must be unique within a candidate")
        positions = {step_id: index for index, step_id in enumerate(step_ids)}
        for index, step in enumerate(self.steps):
            if len(set(step.depends_on)) != len(step.depends_on):
                raise ValueError(f"step {step.step_id} contains duplicate dependencies")
            for dependency in step.depends_on:
                if dependency not in positions:
                    raise ValueError(
                        f"step {step.step_id} depends on unknown step {dependency}"
                    )
                if positions[dependency] >= index:
                    raise ValueError(
                        f"step {step.step_id} dependencies must reference earlier steps"
                    )
        mutated = [step for step in self.steps if step.is_mutated]
        if len(mutated) != 1:
            raise ValueError("a candidate must declare exactly one mutated step")
        if mutated[0].step_id != self.mutated_step_id:
            raise ValueError("mutated_step_id must identify the declared mutated step")
        return self


class CandidateBatch(BaseModel):
    """Root JSON object returned by the mutation generator."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[MutationCandidate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> CandidateBatch:
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id values must be unique")
        return self


class StepOracleCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str
    expression: str
    computed_result: str | None = None
    claimed_result: str
    matches: bool
    error: str | None = None


class ObjectiveOracleResult(BaseModel):
    """Deterministic verdict; passed means objectively wrong and internally traceable."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    reasons: tuple[str, ...] = ()
    step_checks: tuple[StepOracleCheck, ...] = ()
    parsed_reference_answer: str | None = None
    parsed_final_answer: str | None = None


class PlausibilityOracleResult(BaseModel):
    """DeepSeek judgment over candidates already accepted by the objective oracle."""

    model_config = ConfigDict(frozen=True)

    model_plausible: bool
    plausible: bool
    local_error_plausibility: float = Field(ge=0.0, le=1.0)
    global_coherence: float = Field(ge=0.0, le=1.0)
    subtlety: float = Field(ge=0.0, le=1.0)
    minimality: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    rejection_reasons: tuple[str, ...] = ()
    notes: str = ""
    requested_model: str | None = None
    returned_model: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class CandidateEvaluation(BaseModel):
    """Combined, serializable record used by deterministic candidate selection."""

    model_config = ConfigDict(frozen=True)

    candidate: MutationCandidate
    objective: ObjectiveOracleResult
    plausibility: PlausibilityOracleResult | None = None
    processing_error: dict[str, Any] | None = None

    @property
    def eligible(self) -> bool:
        return bool(
            self.objective.passed
            and self.plausibility is not None
            and self.plausibility.plausible
        )


class MutationPipelineConfig(BaseModel):
    """Frozen preprocessing choices that must be reported with generated artifacts."""

    model_config = ConfigDict(frozen=True)

    generator_model: str = "gpt-5.6-sol"
    plausibility_model: str = "deepseek-chat"
    candidate_count: int = Field(default=8, ge=1)
    generator_max_output_tokens: int = Field(default=12_000, ge=1)
    plausibility_max_output_tokens: int = Field(default=4_096, ge=1)
    plausibility_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    minimum_dimension_score: float = Field(default=0.55, ge=0.0, le=1.0)
    preferred_subtlety_score: float = Field(default=0.55, ge=0.0, le=1.0)


class MutationRunResult(BaseModel):
    """Complete result for one task, including rejected candidates."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    config: MutationPipelineConfig
    generator_request: tuple[dict[str, str], ...]
    generator_response: dict[str, Any]
    evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate_id: str | None = None
