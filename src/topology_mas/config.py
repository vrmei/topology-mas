"""Validated, secret-free experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GraphConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_count: int = Field(ge=2)
    edge_counts: tuple[int, ...]
    readout_node: int = Field(ge=0)
    max_rounds: int = Field(ge=1)
    allow_cycles: bool = True
    allow_self_loops: Literal[False] = False

    @model_validator(mode="after")
    def validate_graph_budget(self) -> GraphConfig:
        if self.readout_node >= self.node_count:
            raise ValueError("readout_node must be smaller than node_count")
        if not self.edge_counts:
            raise ValueError("edge_counts cannot be empty")
        if len(set(self.edge_counts)) != len(self.edge_counts):
            raise ValueError("edge_counts must be unique")

        minimum_edges = self.node_count - 1
        maximum_edges = (self.node_count - 1) ** 2
        for edge_count in self.edge_counts:
            if edge_count < minimum_edges:
                raise ValueError(
                    f"edge_count={edge_count} cannot make all nodes reach the readout; "
                    f"minimum is {minimum_edges}"
                )
            if edge_count > maximum_edges:
                raise ValueError(
                    f"edge_count={edge_count} exceeds the maximum {maximum_edges} "
                    "when the readout has no outgoing edges"
                )
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: Literal["openai_compatible", "vllm", "transformers"]
    model_name: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str | None = None
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seeds: tuple[int, ...] = (0,)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)
    active_node_pruning: bool = True
    neighbor_message_order: Literal["content_hash"] = "content_hash"
    message_order_seed: int = 0

    @model_validator(mode="after")
    def validate_seeds(self) -> ExecutionConfig:
        if not self.seeds:
            raise ValueError("seeds cannot be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        return self


class TaskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1)
    oracle_type: Literal["exact_match", "numeric", "symbolic", "unit_tests"]


class AttackConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    attack_readout: bool = False
    candidate_count: int = Field(default=4, ge=1)
    require_oracle_verification: bool = True
    persist_target_error: bool = True


class StorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path = Path("runs")
    save_raw_prompts: bool = True
    save_raw_outputs: bool = True


class ExperimentConfig(BaseModel):
    """Top-level immutable configuration loaded from a versioned YAML file."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    graph: GraphConfig
    model: ModelConfig
    execution: ExecutionConfig
    tasks: tuple[TaskConfig, ...]
    attack: AttackConfig = AttackConfig()
    storage: StorageConfig = StorageConfig()

    @model_validator(mode="after")
    def validate_tasks(self) -> ExperimentConfig:
        if not self.tasks:
            raise ValueError("at least one task configuration is required")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.model_validate(payload)
