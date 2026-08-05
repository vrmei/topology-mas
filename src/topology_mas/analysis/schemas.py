"""Validated records produced by paired topology-MAS analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from topology_mas.models import AnswerState, RunCondition


class RunMetricRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_spec_id: str
    task_id: str
    graph_id: str
    experiment_seed: int
    assignment_seed: int
    condition: RunCondition
    attack_node: int | None = None
    final_answer_state: AnswerState
    final_parsed_answer: str | None = None
    final_correct: bool
    readout_round_zero_state: AnswerState
    readout_round_zero_correct: bool
    model_calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class PairedAttackRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    graph_id: str
    experiment_seed: int
    assignment_seed: int
    attack_node: int = Field(ge=0)
    clean_run_spec_id: str
    attack_run_spec_id: str
    clean_correct: bool
    attack_correct: bool
    paired_accuracy_drop: int = Field(ge=-1, le=1)
    target_answer: str
    clean_final_matches_target: bool
    attack_final_matches_target: bool
    induced_readout_target: bool
    correct_to_target_flip: bool
    clean_error_corrected_under_attack: bool
    induced_target_count_by_round: tuple[int, ...]
    observed_target_count_by_round: tuple[int, ...]
    first_induced_target_round: int | None = Field(default=None, ge=0)
    first_induced_readout_target_round: int | None = Field(default=None, ge=0)
    max_induced_nonattacker_count: int = Field(ge=0)
    attack_model_calls: int = Field(ge=0)
    attack_input_tokens: int | None = Field(default=None, ge=0)
    attack_output_tokens: int | None = Field(default=None, ge=0)


class NodeAttackMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: int = Field(ge=0)
    paired_samples: int = Field(ge=1)
    attack_accuracy: float = Field(ge=0.0, le=1.0)
    paired_accuracy_drop: float = Field(ge=-1.0, le=1.0)
    final_target_match_rate: float = Field(ge=0.0, le=1.0)
    induced_readout_target_rate: float = Field(ge=0.0, le=1.0)
    correct_to_target_flip_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_max_induced_nonattacker_count: float = Field(ge=0.0)


class GraphMetric(BaseModel):
    """Graph-level estimands; no confidence claim is attached here."""

    model_config = ConfigDict(frozen=True)

    graph_id: str
    node_count: int = Field(ge=2)
    edge_count: int = Field(ge=1)
    readout_node: int = Field(ge=0)
    max_rounds: int = Field(ge=1)
    clean_samples: int = Field(ge=1)
    paired_attack_samples: int = Field(ge=1)
    utility: float = Field(ge=0.0, le=1.0)
    readout_round_zero_accuracy: float = Field(ge=0.0, le=1.0)
    communication_correction_rate: float = Field(ge=0.0, le=1.0)
    communication_corruption_rate: float = Field(ge=0.0, le=1.0)
    r_mean: float = Field(ge=0.0, le=1.0)
    r_worst: float = Field(ge=0.0, le=1.0)
    d_mean: float = Field(ge=-1.0, le=1.0)
    d_max: float = Field(ge=-1.0, le=1.0)
    node_attack_accuracy_std: float = Field(ge=0.0)
    final_target_match_rate: float = Field(ge=0.0, le=1.0)
    induced_readout_target_rate: float = Field(ge=0.0, le=1.0)
    correct_to_target_flip_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_max_induced_nonattacker_count: float = Field(ge=0.0)
    clean_mean_model_calls: float = Field(ge=0.0)
    attack_mean_model_calls: float = Field(ge=0.0)
    clean_mean_input_tokens: float | None = Field(default=None, ge=0.0)
    attack_mean_input_tokens: float | None = Field(default=None, ge=0.0)
    clean_mean_output_tokens: float | None = Field(default=None, ge=0.0)
    attack_mean_output_tokens: float | None = Field(default=None, ge=0.0)
    node_metrics: tuple[NodeAttackMetric, ...]


class ClassicalInitialStateRecord(BaseModel):
    """One clean initial state reusable by non-LLM dynamics on the same graph."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    graph_id: str
    experiment_seed: int
    assignment_seed: int
    clean_run_spec_id: str
    reference_answer: str
    target_answer: str
    structural_node_to_replica: tuple[int, ...]
    node_parsed_answers: tuple[str | None, ...]
    node_answer_states: tuple[AnswerState, ...]


class AnalysisManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    analyzer_version: str
    source_batch_runner_version: str
    source_batch_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    expected_runs: int = Field(ge=1)
    analyzed_runs: int = Field(ge=1)
    paired_attacks: int = Field(ge=1)
    graph_count: int = Field(ge=1)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: AnalysisManifest
    run_metrics: tuple[RunMetricRow, ...]
    paired_attacks: tuple[PairedAttackRow, ...]
    graph_metrics: tuple[GraphMetric, ...]
    classical_initial_states: tuple[ClassicalInitialStateRecord, ...]
