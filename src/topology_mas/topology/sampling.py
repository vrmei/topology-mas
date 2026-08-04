"""Reproducible rejection sampling over fixed-edge labeled directed graphs."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

from topology_mas.models import DirectedEdge, GraphSpec
from topology_mas.topology.graph_ops import (
    build_causal_schedule,
    candidate_edge_pairs,
    graph_constraint_violations,
    has_directed_cycle,
    source_nodes,
)
from topology_mas.topology.schemas import (
    GraphSamplingConfig,
    GraphSamplingSummary,
    SampledGraphCollection,
)

SAMPLER_VERSION = "fixed-m-rejection-v2"


class GraphSamplingExhaustedError(RuntimeError):
    pass


def _proposal_seed(base_seed: int, sample_index: int, attempt: int) -> int:
    payload = f"{base_seed}\0{sample_index}\0{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _edge_signature(edge_pairs: tuple[tuple[int, int], ...]) -> str:
    encoded = json.dumps(edge_pairs, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_collection_fingerprint(graphs: tuple[GraphSpec, ...]) -> str:
    digest = hashlib.sha256()
    for graph in graphs:
        digest.update(graph.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ConstrainedDirectedGraphSampler:
    """Sample uniformly proposed edge subsets, conditioned on experiment constraints."""

    def __init__(self, config: GraphSamplingConfig) -> None:
        self.config = config
        self._candidate_edges = candidate_edge_pairs(
            config.node_count,
            config.readout_node,
        )

    def sample(self) -> SampledGraphCollection:
        accepted: list[GraphSpec] = []
        accepted_signatures: set[tuple[tuple[int, int], ...]] = set()
        rejected: Counter[str] = Counter()
        proposal_attempts = 0

        for sample_index in range(self.config.graph_count):
            for attempt in range(1, self.config.max_attempts_per_graph + 1):
                proposal_attempts += 1
                proposal_seed = _proposal_seed(self.config.seed, sample_index, attempt)
                generator = random.Random(proposal_seed)
                edge_pairs = tuple(
                    sorted(generator.sample(self._candidate_edges, self.config.edge_count))
                )
                if edge_pairs in accepted_signatures:
                    rejected["duplicate"] += 1
                    continue

                signature = _edge_signature(edge_pairs)
                candidate = GraphSpec(
                    graph_id=(
                        f"g-n{self.config.node_count}-m{self.config.edge_count}-"
                        f"{signature[:16]}"
                    ),
                    node_count=self.config.node_count,
                    edges=tuple(
                        DirectedEdge(source=source, target=target)
                        for source, target in edge_pairs
                    ),
                    readout_node=self.config.readout_node,
                    max_rounds=self.config.max_rounds,
                    sampling_seed=proposal_seed,
                )
                violations = graph_constraint_violations(candidate)
                if violations:
                    if "unreachable_node" in violations:
                        rejected["unreachable_node"] += 1
                    else:
                        rejected["round_limit_exceeded"] += 1
                    continue

                schedule = build_causal_schedule(candidate)
                graph = candidate.model_copy(
                    update={
                        "metadata": {
                            "sampler_version": SAMPLER_VERSION,
                            "base_seed": self.config.seed,
                            "sample_index": sample_index,
                            "accepted_attempt": attempt,
                            "edge_signature": signature,
                            "distances_to_readout": list(schedule.distances_to_readout),
                            "source_nodes": list(source_nodes(candidate)),
                            "has_directed_cycle": has_directed_cycle(candidate),
                            "active_node_count_by_round": [
                                len(nodes) for nodes in schedule.active_nodes_by_round
                            ],
                            "active_edge_count_by_round": [
                                len(edges) for edges in schedule.active_edges_by_round
                            ],
                            "message_opportunities": schedule.message_opportunities,
                        }
                    }
                )
                accepted.append(graph)
                accepted_signatures.add(edge_pairs)
                break
            else:
                raise GraphSamplingExhaustedError(
                    f"failed to sample graph {sample_index} after "
                    f"{self.config.max_attempts_per_graph} attempts; "
                    f"rejections={dict(rejected)}"
                )

        accepted_tuple = tuple(accepted)
        summary = GraphSamplingSummary(
            requested_graphs=self.config.graph_count,
            accepted_graphs=len(accepted_tuple),
            proposal_attempts=proposal_attempts,
            rejected_unreachable=rejected["unreachable_node"],
            rejected_round_limit=rejected["round_limit_exceeded"],
            rejected_duplicate=rejected["duplicate"],
            proposal_acceptance_rate=(
                len(accepted_tuple) / proposal_attempts if proposal_attempts else 0.0
            ),
        )
        return SampledGraphCollection(
            config=self.config,
            graphs=accepted_tuple,
            summary=summary,
            collection_fingerprint=graph_collection_fingerprint(accepted_tuple),
        )
