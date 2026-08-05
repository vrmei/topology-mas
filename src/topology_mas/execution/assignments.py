"""Explicit mappings between anonymous initial replicas and structural graph nodes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from topology_mas.execution.seeding import stable_id, stable_integer


class InitialStateAssignment(BaseModel):
    """A permutation mapping each structural node to one cached replica slot."""

    model_config = ConfigDict(frozen=True)

    assignment_id: str = Field(min_length=1)
    node_count: int = Field(ge=2)
    assignment_seed: int
    structural_node_to_replica: tuple[int, ...]

    @model_validator(mode="after")
    def validate_permutation(self) -> InitialStateAssignment:
        if len(self.structural_node_to_replica) != self.node_count:
            raise ValueError("assignment must contain exactly one entry per structural node")
        if set(self.structural_node_to_replica) != set(range(self.node_count)):
            raise ValueError("structural_node_to_replica must be a permutation")
        return self

    def replica_for_node(self, node_id: int) -> int:
        if not 0 <= node_id < self.node_count:
            raise ValueError("node_id lies outside the assignment")
        return self.structural_node_to_replica[node_id]


def _assignment_id(*, node_count: int, assignment_seed: int, mapping: tuple[int, ...]) -> str:
    return stable_id("assignment", node_count, assignment_seed, *mapping)


def build_initial_state_assignment(
    *, node_count: int, assignment_seed: int
) -> InitialStateAssignment:
    """Create a deterministic graph-independent permutation from an assignment seed."""

    if node_count < 2:
        raise ValueError("node_count must be at least two")
    mapping = tuple(
        sorted(
            range(node_count),
            key=lambda replica: (
                stable_integer("initial-assignment", assignment_seed, replica),
                replica,
            ),
        )
    )
    return InitialStateAssignment(
        assignment_id=_assignment_id(
            node_count=node_count,
            assignment_seed=assignment_seed,
            mapping=mapping,
        ),
        node_count=node_count,
        assignment_seed=assignment_seed,
        structural_node_to_replica=mapping,
    )


def relabel_assignment(
    assignment: InitialStateAssignment,
    *,
    old_node_to_new_node: tuple[int, ...],
) -> InitialStateAssignment:
    """Move initial states with nodes under an explicit graph isomorphism."""

    if len(old_node_to_new_node) != assignment.node_count or set(old_node_to_new_node) != set(
        range(assignment.node_count)
    ):
        raise ValueError("old_node_to_new_node must be a permutation")
    transformed = [0] * assignment.node_count
    for old_node, new_node in enumerate(old_node_to_new_node):
        transformed[new_node] = assignment.replica_for_node(old_node)
    mapping = tuple(transformed)
    return InitialStateAssignment(
        assignment_id=_assignment_id(
            node_count=assignment.node_count,
            assignment_seed=assignment.assignment_seed,
            mapping=mapping,
        ),
        node_count=assignment.node_count,
        assignment_seed=assignment.assignment_seed,
        structural_node_to_replica=mapping,
    )
