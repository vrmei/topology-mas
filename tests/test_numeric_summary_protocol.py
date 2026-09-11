from __future__ import annotations

import hashlib

import pytest

from topology_mas.execution.numeric_summary_protocol import (
    NUMERIC_SUMMARY_PROTOCOL,
    NumericSummaryEnvelope,
    build_numeric_summary_messages,
    numeric_summary_protocol,
    parse_numeric_summary_envelope,
    serialize_numeric_public_summary,
    validate_numeric_public_summary,
)
from topology_mas.models import (
    AdversarialAnswer,
    AnswerState,
    MessageRecord,
    MessageType,
    OracleStatus,
    TaskInstance,
)


def count_words(text: str) -> int:
    return len(text.split())


def task() -> TaskInstance:
    return TaskInstance(
        task_id="gsm8k-x",
        dataset="gsm8k",
        split="test",
        prompt="What is 2+3?",
        reference_answer="5",
        oracle_type="numeric",
    )


def test_envelope_and_public_summary_preserve_numeric_answer() -> None:
    public = serialize_numeric_public_summary("Compute 2+3=5.", "5.0")
    validated, answer, tokens = validate_numeric_public_summary(
        public,
        full_answer="5",
        token_counter=count_words,
    )
    envelope = NumericSummaryEnvelope(
        "Work.\nFINAL_ANSWER: \\boxed{5}", validated, "stop"
    ).serialize()
    parsed = parse_numeric_summary_envelope(envelope)
    assert parsed.public_summary == validated
    assert answer == "5"
    assert tokens == count_words(validated)


@pytest.mark.parametrize(
    "text",
    (
        "FINAL_ANSWER: \\boxed{5}",
        "FINAL_ANSWER: $\\boxed{5}$",
        "#### 5",
    ),
)
def test_numeric_parser_accepts_frozen_explicit_terminal_forms(text: str) -> None:
    from topology_mas.execution.answers import parse_numeric_answer

    assert parse_numeric_answer(text) == "5"


def test_unparsed_summary_cannot_invent_answer() -> None:
    public = serialize_numeric_public_summary("The source did not finish.", None)
    assert (
        validate_numeric_public_summary(
            public,
            full_answer=None,
            token_counter=count_words,
        )[1]
        is None
    )
    with pytest.raises(ValueError):
        validate_numeric_public_summary(
            serialize_numeric_public_summary("The source did not finish.", "7"),
            full_answer=None,
            token_counter=count_words,
        )


def test_update_prompt_uses_self_full_and_peer_summary_only() -> None:
    self_full = "Private details.\nFINAL_ANSWER: \\boxed{5}"
    self_public = serialize_numeric_public_summary("Short self.", "5")
    previous = NumericSummaryEnvelope(self_full, self_public, "stop").serialize()
    peer_public = serialize_numeric_public_summary("Peer computes 2+3=5.", "5")
    peer = MessageRecord(
        message_id="m",
        run_id="r",
        task_id="gsm8k-x",
        graph_id="g",
        round_index=0,
        sender=0,
        recipients=(1,),
        raw_text=peer_public,
        parsed_answer="5",
        answer_state=AnswerState.CORRECT,
        message_type=MessageType.SUMMARY,
        summary_source_response_id="peer-r0",
    )
    messages = build_numeric_summary_messages(
        task(),
        previous_output=previous,
        incoming_messages=(peer,),
    )
    user = messages[1].content
    assert self_full in user
    assert self_public not in user
    assert peer_public in user
    assert NUMERIC_SUMMARY_PROTOCOL not in user


def test_fixed_attacker_requires_validated_same_target_summary() -> None:
    public = serialize_numeric_public_summary("A plausible wrong calculation gives 6.", "6")
    tokens = count_words(public)
    answer = AdversarialAnswer(
        task_id="gsm8k-x",
        target_answer="6",
        rationale="2+3 is incorrectly treated as 6. #### 6",
        mutation_type="arithmetic",
        oracle_status=OracleStatus.PASSED,
        public_summary=public,
        public_summary_tokens=tokens,
        public_summary_hash=hashlib.sha256(public.encode()).hexdigest(),
    )
    protocol = numeric_summary_protocol(count_words)
    envelope = parse_numeric_summary_envelope(protocol.adversarial_output(answer))
    assert envelope.public_summary == public
    assert protocol.parse_answer(envelope.serialize(), finish_reason="stop") == "6"


def test_raw_peer_message_is_rejected() -> None:
    peer = MessageRecord(
        message_id="m",
        run_id="r",
        task_id="gsm8k-x",
        graph_id="g",
        round_index=0,
        sender=0,
        recipients=(1,),
        raw_text="raw",
        answer_state=AnswerState.UNPARSED,
        message_type=MessageType.LEGACY_RAW,
    )
    with pytest.raises(ValueError, match="only summary"):
        build_numeric_summary_messages(
            task(),
            previous_output=NumericSummaryEnvelope(
                "Work", serialize_numeric_public_summary("Short.", None), "stop"
            ).serialize(),
            incoming_messages=(peer,),
        )
