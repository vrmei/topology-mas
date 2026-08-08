"""Validated records for constrained topology sampling."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from topology_mas.models import DirectedEdge, GraphSpec


class GraphSamplingConfig(BaseModel):
    """One fixed-budget labeled directed-graph sampling stratum."""

    model_config = ConfigDict(frozen=True)

    node_count: int = Field(ge=2)
    edge_count: int = Field(ge=1)
    readout_node: int = Field(ge=0)
    max_rounds: int = Field(ge=1)
    graph_count: int = Field(ge=1)
    seed: int = 0
    max_attempts_per_graph: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def validate_graph_space(self) -> GraphSamplingConfig:
        if self.readout_node >= self.node_count:
            raise ValueError("readout_node must be smaller than node_count")
        minimum_edges = self.node_count - 1
        maximum_edges = (self.node_count - 1) ** 2
        if self.edge_count < minimum_edges:
            raise ValueError(
                f"edge_count must be at least {minimum_edges} for all nodes to reach readout"
            )
        if self.edge_count > maximum_edges:
            raise ValueError(
                f"edge_count cannot exceed {maximum_edges} under the readout constraints"
            )
        proposal_space_size = math.comb(maximum_edges, self.edge_count)
        if self.graph_count > proposal_space_size:
            raise ValueError(
                f"graph_count cannot exceed the {proposal_space_size} distinct fixed-edge "
                "labeled proposals"
            )
        return self


class CausalSchedule(BaseModel):
    """Nodes and edges whose new messages can still affect the final readout turn."""

    model_config = ConfigDict(frozen=True)

    distances_to_readout: tuple[int, ...]
    effective_horizon: int = Field(ge=1)
    active_nodes_by_round: tuple[tuple[int, ...], ...]
    active_edges_by_round: tuple[tuple[DirectedEdge, ...], ...]
    message_opportunities: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_horizon(cls, value: Any) -> Any:
        """Keep traces written before explicit horizon recording readable."""

        if isinstance(value, dict) and "effective_horizon" not in value:
            rounds = value.get("active_nodes_by_round")
            if isinstance(rounds, (list, tuple)) and len(rounds) >= 2:
                return {**value, "effective_horizon": len(rounds) - 1}
        return value


class GraphSamplingSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    requested_graphs: int = Field(ge=1)
    accepted_graphs: int = Field(ge=0)
    proposal_attempts: int = Field(ge=0)
    rejected_unreachable: int = Field(ge=0)
    rejected_round_limit: int = Field(ge=0)
    rejected_duplicate: int = Field(ge=0)
    proposal_acceptance_rate: float = Field(ge=0.0, le=1.0)


class SampledGraphCollection(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: GraphSamplingConfig
    graphs: tuple[GraphSpec, ...]
    summary: GraphSamplingSummary
    collection_fingerprint: str
