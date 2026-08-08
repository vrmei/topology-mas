"""Validated request, response, and trace records for synchronous MAS execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from topology_mas.models import (
    AnswerState,
    MessageRecord,
    NodeTurnRecord,
    RunCondition,
)
from topology_mas.topology.schemas import CausalSchedule


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class TextGenerationRequest(BaseModel):
    """One provider-neutral node call."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    seed: int
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=1)


class TextGenerationResult(BaseModel):
    """Normalized text completion returned by a local or remote backend."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    model_name: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionSettings(BaseModel):
    """Frozen choices that affect semantic execution rather than graph sampling."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=768, ge=1)
    neighbor_message_order: Literal["content_hash"] = "content_hash"
    message_order_seed: int = 0
    active_node_pruning: Literal[True] = True
    horizon_policy: Literal["fixed", "graph_depth"] = "fixed"
    state_transition_policy: Literal[
        "independent-resampling", "state-consistent-replay-v1"
    ] = "independent-resampling"


class RunTrace(BaseModel):
    """Complete trace for one task, graph, condition, attack position, and seed."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    condition: RunCondition
    attack_node: int | None = Field(default=None, ge=0)
    adversarial_answer_fingerprint: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    target_answer: str | None = None
    initial_assignment_id: str | None = None
    initial_assignment_seed: int | None = None
    structural_node_to_replica: tuple[int, ...] | None = None
    seed: int
    prompt_version: str = Field(min_length=1)
    execution_settings: ExecutionSettings
    schedule: CausalSchedule
    turns: tuple[NodeTurnRecord, ...]
    messages: tuple[MessageRecord, ...]
    final_raw_output: str
    final_parsed_answer: str | None = None
    final_answer_state: AnswerState
    total_model_calls: int = Field(ge=0)
    total_backend_calls: int | None = Field(default=None, ge=0)
    state_replay_cache_hits: int = Field(default=0, ge=0)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_condition(self) -> RunTrace:
        if self.condition is RunCondition.CLEAN and self.attack_node is not None:
            raise ValueError("clean traces cannot specify attack_node")
        if self.condition is RunCondition.CLEAN and (
            self.adversarial_answer_fingerprint is not None or self.target_answer is not None
        ):
            raise ValueError("clean traces cannot specify an adversarial answer")
        if self.condition is RunCondition.ATTACK and self.attack_node is None:
            raise ValueError("attack traces must specify attack_node")
        if self.condition is RunCondition.ATTACK and (
            self.adversarial_answer_fingerprint is None or self.target_answer is None
        ):
            raise ValueError("attack traces must identify the adversarial answer")
        return self
