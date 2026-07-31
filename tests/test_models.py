import pytest
from pydantic import ValidationError

from topology_mas.models import (
    AdversarialAnswer,
    DirectedEdge,
    GraphSpec,
    MessageRecord,
    OracleStatus,
    RunCondition,
    NodeTurnRecord,
)


def test_graph_spec_is_validated() -> None:
    graph = GraphSpec(
        graph_id="g1",
        node_count=3,
        edges=(DirectedEdge(source=0, target=2), DirectedEdge(source=1, target=2)),
        readout_node=2,
        max_rounds=1,
    )

    assert len(graph.edges) == 2


def test_graph_spec_rejects_readout_outgoing_edge() -> None:
    with pytest.raises(ValidationError, match="readout_node cannot have outgoing edges"):
        GraphSpec(
            graph_id="g1",
            node_count=3,
            edges=(DirectedEdge(source=2, target=0),),
            readout_node=2,
            max_rounds=1,
        )


def test_adversarial_answer_acceptance_is_oracle_based() -> None:
    answer = AdversarialAnswer(
        task_id="task-1",
        target_answer="41",
        rationale="A single arithmetic carry is omitted.",
        mutation_type="arithmetic_carry",
        oracle_status=OracleStatus.PASSED,
        plausibility_score=0.8,
    )

    assert answer.accepted is True


def test_message_rejects_self_recipient() -> None:
    with pytest.raises(ValidationError, match="cannot broadcast to itself"):
        MessageRecord(
            message_id="m1",
            run_id="r1",
            task_id="t1",
            graph_id="g1",
            round_index=0,
            sender=1,
            recipients=(1,),
            raw_text="answer",
        )


def test_clean_turn_rejects_attack_node() -> None:
    with pytest.raises(ValidationError, match="clean records cannot specify attack_node"):
        NodeTurnRecord(
            run_id="r1",
            task_id="t1",
            graph_id="g1",
            condition=RunCondition.CLEAN,
            attack_node=0,
            seed=0,
            round_index=0,
            node_id=0,
            raw_output="answer",
        )
