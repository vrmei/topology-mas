"""Serializable domain records shared by every experiment module.

These models intentionally contain no graph algorithms or LLM-provider behavior. They define
the stable records written to disk and exchanged between later modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerState(StrEnum):
    """Coarse state used by classical dynamics and trace analysis."""

    CORRECT = "correct"
    TARGET_ERROR = "target_error"
    OTHER_ERROR = "other_error"
    UNPARSED = "unparsed"


class OracleStatus(StrEnum):
    """Objective verification status for a mutated answer."""

    UNVERIFIED = "unverified"
    PASSED = "passed"
    FAILED = "failed"


class RunCondition(StrEnum):
    CLEAN = "clean"
    ATTACK = "attack"


class DirectedEdge(BaseModel):
    """A directed communication channel from ``source`` to ``target``."""

    model_config = ConfigDict(frozen=True)

    source: int = Field(ge=0)
    target: int = Field(ge=0)

    @model_validator(mode="after")
    def reject_self_loop(self) -> DirectedEdge:
        if self.source == self.target:
            raise ValueError("self-loops are not supported")
        return self


class GraphSpec(BaseModel):
    """Immutable graph description after sampling and validation."""

    model_config = ConfigDict(frozen=True)

    graph_id: str = Field(min_length=1)
    node_count: int = Field(ge=2)
    edges: tuple[DirectedEdge, ...]
    readout_node: int = Field(ge=0)
    max_rounds: int = Field(ge=1)
    sampling_seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_node_references(self) -> GraphSpec:
        if self.readout_node >= self.node_count:
            raise ValueError("readout_node must be smaller than node_count")

        edge_pairs = {(edge.source, edge.target) for edge in self.edges}
        if len(edge_pairs) != len(self.edges):
            raise ValueError("duplicate directed edges are not supported")

        for edge in self.edges:
            if edge.source >= self.node_count or edge.target >= self.node_count:
                raise ValueError("edge endpoint lies outside the graph")
            if edge.source == self.readout_node:
                raise ValueError("readout_node cannot have outgoing edges")
        return self


class TaskInstance(BaseModel):
    """A benchmark item with an objectively checkable reference answer."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    oracle_type: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdversarialAnswer(BaseModel):
    """A task-specific target error generated before graph execution."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    target_answer: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    mutation_type: str = Field(min_length=1)
    oracle_status: OracleStatus = OracleStatus.UNVERIFIED
    plausibility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    generator_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """Whether the candidate is objectively wrong and eligible for experiments."""

        return self.oracle_status is OracleStatus.PASSED


class MessageRecord(BaseModel):
    """One node output broadcast to one or more out-neighbors."""

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    sender: int = Field(ge=0)
    recipients: tuple[int, ...]
    raw_text: str
    parsed_answer: str | None = None
    answer_state: AnswerState = AnswerState.UNPARSED
    output_tokens: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_recipients(self) -> MessageRecord:
        if len(set(self.recipients)) != len(self.recipients):
            raise ValueError("message recipients must be unique")
        if self.sender in self.recipients:
            raise ValueError("a sender cannot broadcast to itself")
        return self


class NodeTurnRecord(BaseModel):
    """Complete observable state for one node at one synchronous round."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    condition: RunCondition
    attack_node: int | None = Field(default=None, ge=0)
    seed: int
    round_index: int = Field(ge=0)
    node_id: int = Field(ge=0)
    incoming_message_ids: tuple[str, ...] = ()
    previous_raw_output: str | None = None
    prompt_messages: tuple[dict[str, str], ...] = ()
    generation_seed: int | None = None
    raw_output: str
    parsed_answer: str | None = None
    answer_state: AnswerState = AnswerState.UNPARSED
    is_correct: bool | None = None
    matches_target_error: bool | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    model_name: str | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_condition(self) -> NodeTurnRecord:
        if self.condition is RunCondition.CLEAN and self.attack_node is not None:
            raise ValueError("clean records cannot specify attack_node")
        if self.condition is RunCondition.ATTACK and self.attack_node is None:
            raise ValueError("attack records must specify attack_node")
        return self
