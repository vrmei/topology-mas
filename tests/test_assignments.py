import pytest
from pydantic import ValidationError

from topology_mas.execution import (
    InitialStateAssignment,
    build_initial_state_assignment,
    relabel_assignment,
)


def test_assignment_is_a_deterministic_permutation() -> None:
    first = build_initial_state_assignment(node_count=8, assignment_seed=17)
    second = build_initial_state_assignment(node_count=8, assignment_seed=17)

    assert first == second
    assert set(first.structural_node_to_replica) == set(range(8))


def test_assignment_seed_changes_replica_placement() -> None:
    mappings = {
        build_initial_state_assignment(
            node_count=8,
            assignment_seed=seed,
        ).structural_node_to_replica
        for seed in range(5)
    }

    assert len(mappings) > 1


def test_assignment_rejects_missing_or_duplicate_replicas() -> None:
    with pytest.raises(ValidationError, match="permutation"):
        InitialStateAssignment(
            assignment_id="invalid",
            node_count=4,
            assignment_seed=0,
            structural_node_to_replica=(0, 1, 1, 3),
        )


def test_relabel_assignment_moves_each_initial_state_with_its_node() -> None:
    assignment = InitialStateAssignment(
        assignment_id="source",
        node_count=4,
        assignment_seed=0,
        structural_node_to_replica=(3, 0, 2, 1),
    )
    old_to_new = (2, 0, 3, 1)

    relabeled = relabel_assignment(
        assignment,
        old_node_to_new_node=old_to_new,
    )

    for old_node, new_node in enumerate(old_to_new):
        assert relabeled.replica_for_node(new_node) == assignment.replica_for_node(old_node)
