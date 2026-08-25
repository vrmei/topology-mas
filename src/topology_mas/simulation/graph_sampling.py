"""Scalable fixed-edge graph proposals for CTOU model-based simulation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from topology_mas.models import DirectedEdge, GraphSpec
from topology_mas.topology.graph_ops import (
    candidate_edge_pairs,
    graph_constraint_violations,
)


@dataclass(frozen=True)
class GraphMixingAudit:
    proposed_swaps: int
    accepted_swaps: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_swaps / self.proposed_swaps if self.proposed_swaps else 0.0


def normalized_density_edge_levels(
    node_count: int,
    deltas: tuple[float, ...],
) -> tuple[tuple[int, float], ...]:
    """Map normalized excess-density levels to unique feasible edge counts."""

    if node_count < 2:
        raise ValueError("node_count must be at least two")
    minimum = node_count - 1
    maximum = (node_count - 1) ** 2
    observed: dict[int, float] = {}
    for delta in deltas:
        if not 0 <= delta <= 1:
            raise ValueError("density levels must lie in [0, 1]")
        edge_count = int(round(minimum + delta * (maximum - minimum)))
        observed.setdefault(edge_count, float(delta))
    return tuple(sorted(observed.items()))


def _seed(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _graph_from_edges(
    *,
    node_count: int,
    edge_pairs: set[tuple[int, int]],
    readout_node: int,
    horizon: int,
    graph_id: str,
    sampling_seed: int,
    metadata: dict[str, object] | None = None,
) -> GraphSpec:
    return GraphSpec(
        graph_id=graph_id,
        node_count=node_count,
        edges=tuple(
            DirectedEdge(source=source, target=target) for source, target in sorted(edge_pairs)
        ),
        readout_node=readout_node,
        max_rounds=horizon,
        sampling_seed=sampling_seed,
        metadata=metadata or {},
    )


def sample_backbone_augmented_graph(
    *,
    node_count: int,
    edge_count: int,
    horizon: int,
    seed: int,
    sample_index: int,
    swap_steps: int = 200,
) -> tuple[GraphSpec, GraphMixingAudit]:
    """Build a legal graph, then apply symmetric fixed-m legal edge swaps.

    The backbone proposal is not uniform.  Symmetric swaps leave the uniform
    distribution invariant within the reachable legal state component, but a
    finite run is reported as a mixing diagnostic rather than treated as proof
    of exact uniformity.
    """

    readout = node_count - 1
    candidates = tuple(candidate_edge_pairs(node_count, readout))
    if not node_count - 1 <= edge_count <= len(candidates):
        raise ValueError("edge_count lies outside the feasible fixed-m range")
    sampling_seed = _seed(seed, node_count, edge_count, sample_index)
    rng = random.Random(sampling_seed)
    nodes = list(range(node_count - 1))
    rng.shuffle(nodes)
    connected = [readout]
    depth = {readout: 0}
    edges: set[tuple[int, int]] = set()
    for node in nodes:
        parents = [candidate for candidate in connected if depth[candidate] < horizon]
        parent = rng.choice(parents)
        edges.add((node, parent))
        depth[node] = depth[parent] + 1
        connected.append(node)
    extras = [edge for edge in candidates if edge not in edges]
    edges.update(rng.sample(extras, edge_count - len(edges)))

    accepted = 0
    for _step in range(swap_steps):
        if edge_count == len(candidates):
            break
        removed = rng.choice(tuple(edges))
        added = rng.choice(tuple(edge for edge in candidates if edge not in edges))
        proposal = set(edges)
        proposal.remove(removed)
        proposal.add(added)
        candidate = _graph_from_edges(
            node_count=node_count,
            edge_pairs=proposal,
            readout_node=readout,
            horizon=horizon,
            graph_id="proposal",
            sampling_seed=sampling_seed,
        )
        if graph_constraint_violations(candidate):
            continue
        edges = proposal
        accepted += 1

    graph = _graph_from_edges(
        node_count=node_count,
        edge_pairs=edges,
        readout_node=readout,
        horizon=horizon,
        graph_id=f"sim-n{node_count}-m{edge_count}-g{sample_index}",
        sampling_seed=sampling_seed,
        metadata={
            "sampler": "backbone-plus-symmetric-legal-edge-swaps-v1",
            "swap_steps": swap_steps,
            "accepted_swaps": accepted,
        },
    )
    violations = graph_constraint_violations(graph)
    if violations:
        raise RuntimeError(f"simulated graph violates constraints: {violations}")
    return graph, GraphMixingAudit(swap_steps, accepted)
