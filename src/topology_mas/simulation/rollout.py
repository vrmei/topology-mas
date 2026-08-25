"""Vectorized expected-composition mean-field rollout for large CTOU graphs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from topology_mas.models import GraphSpec
from topology_mas.simulation.ctou_scale import (
    COUNT_COLUMNS,
    CTOU_STATES,
    HierarchicalRoundZeroModel,
)
from topology_mas.topology.graph_ops import build_causal_schedule

ProbabilityPredictor = Callable[[pd.DataFrame], np.ndarray]


def sample_round_zero_states(
    *,
    initializer: HierarchicalRoundZeroModel,
    task_ids: list[str],
    node_count: int,
    particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw correlated C/O/U Round-0 states, grouped task then particle."""

    if node_count < 1 or particles < 1:
        raise ValueError("node_count and particles must be positive")
    output = np.empty((len(task_ids), particles, node_count), dtype=np.int8)
    round_zero_to_ctou = np.asarray([0, 2, 3], dtype=np.int8)
    for task_index, task_id in enumerate(task_ids):
        mean = initializer.mean_for_task(task_id)
        theta = rng.dirichlet(initializer.concentration * mean, size=particles)
        cumulative = np.cumsum(theta, axis=1)
        uniforms = rng.random((particles, node_count))
        state_index = (uniforms[:, :, None] > cumulative[:, None, :]).sum(axis=2)
        output[task_index] = round_zero_to_ctou[state_index]
    return output.reshape(len(task_ids) * particles, node_count)


def _sample_categorical(probability: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    normalized = probability / probability.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(normalized, axis=1)
    cumulative[:, -1] = 1.0
    return (rng.random(len(normalized))[:, None] > cumulative).sum(axis=1).astype(np.int8)


def _adjacency(graph: GraphSpec) -> np.ndarray:
    adjacency = np.zeros((graph.node_count, graph.node_count), dtype=float)
    for edge in graph.edges:
        adjacency[edge.source, edge.target] = 1.0
    return adjacency


def expected_composition_rollout(
    *,
    graph: GraphSpec,
    initial_marginals: np.ndarray,
    attack_nodes: np.ndarray,
    predictor: ProbabilityPredictor,
) -> np.ndarray:
    """Propagate batches using expected incoming counts.

    `initial_marginals` has shape `(batch, n, 4)`. `attack_nodes` is `-1` for
    clean rows or the persistent attacker index for attack rows. The method is
    a plug-in mean-field approximation; particle checks are required because
    nonlinear response does not generally commute with composition averaging.
    """

    marginals = np.asarray(initial_marginals, dtype=float).copy()
    attacks = np.asarray(attack_nodes, dtype=int)
    if marginals.ndim != 3 or marginals.shape[1:] != (graph.node_count, 4):
        raise ValueError("initial_marginals must have shape (batch, node_count, 4)")
    if attacks.shape != (len(marginals),):
        raise ValueError("attack_nodes must have one value per batch row")
    if not np.allclose(marginals.sum(axis=2), 1.0):
        raise ValueError("initial marginals do not sum to one")
    one_hot_target = np.eye(4, dtype=float)[1]
    for node in range(graph.node_count):
        mask = attacks == node
        marginals[mask, node] = one_hot_target

    adjacency = _adjacency(graph)
    schedule = build_causal_schedule(graph)
    for round_index in range(1, schedule.effective_horizon + 1):
        updated = marginals.copy()
        active_nodes = np.asarray(schedule.active_nodes_by_round[round_index], dtype=int)
        counts = np.einsum("ij,bis->bjs", adjacency, marginals, optimize=True)
        active_counts = counts[:, active_nodes].reshape(-1, 4)
        result = np.zeros((len(marginals), len(active_nodes), 4), dtype=float)
        for previous_index, previous_state in enumerate(CTOU_STATES):
            frame = pd.DataFrame(
                {
                    "previous_attack_state": previous_state,
                    "round_index": round_index,
                    **{
                        column: active_counts[:, state_index]
                        for state_index, column in enumerate(COUNT_COLUMNS)
                    },
                }
            )
            probability = predictor(frame).reshape(len(marginals), len(active_nodes), 4)
            result += marginals[:, active_nodes, previous_index, None] * probability
        result /= result.sum(axis=2, keepdims=True)
        updated[:, active_nodes] = result
        for node in active_nodes:
            updated[attacks == node, node] = one_hot_target
        marginals = updated
    return marginals[:, graph.readout_node]


def particle_composition_rollout(
    *,
    graph: GraphSpec,
    initial_states: np.ndarray,
    attack_nodes: np.ndarray,
    predictor: ProbabilityPredictor,
    rng: np.random.Generator,
) -> np.ndarray:
    """Roll out discrete CTOU particles using realized incoming compositions."""

    states = np.asarray(initial_states, dtype=np.int8).copy()
    attacks = np.asarray(attack_nodes, dtype=int)
    if states.ndim != 2 or states.shape[1] != graph.node_count:
        raise ValueError("initial_states must have shape (batch, node_count)")
    if attacks.shape != (len(states),):
        raise ValueError("attack_nodes must have one value per batch row")
    if np.any((states < 0) | (states >= len(CTOU_STATES))):
        raise ValueError("initial states contain an invalid CTOU index")
    for node in range(graph.node_count):
        states[attacks == node, node] = 1

    adjacency = _adjacency(graph)
    schedule = build_causal_schedule(graph)
    state_eye = np.eye(len(CTOU_STATES), dtype=np.int8)
    for round_index in range(1, schedule.effective_horizon + 1):
        active_nodes = np.asarray(schedule.active_nodes_by_round[round_index], dtype=int)
        one_hot = state_eye[states]
        counts = np.einsum("ij,bis->bjs", adjacency, one_hot, optimize=True)
        active_counts = counts[:, active_nodes].reshape(-1, len(CTOU_STATES))
        previous = states[:, active_nodes].reshape(-1)
        frame = pd.DataFrame(
            {
                "previous_attack_state": np.asarray(CTOU_STATES, dtype=object)[previous],
                "round_index": round_index,
                **{
                    column: active_counts[:, state_index]
                    for state_index, column in enumerate(COUNT_COLUMNS)
                },
            }
        )
        probability = predictor(frame)
        updated = _sample_categorical(probability, rng).reshape(
            len(states), len(active_nodes)
        )
        states[:, active_nodes] = updated
        for node in active_nodes:
            states[attacks == node, node] = 1
    return states[:, graph.readout_node]
