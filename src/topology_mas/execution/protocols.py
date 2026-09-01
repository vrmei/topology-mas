"""Pluggable task semantics for the shared synchronous graph executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from topology_mas.execution.aime import (
    AIME_BOUNDED_PROMPT_VERSION,
    AIME_FULL_RATIONALE_PROMPT_VERSION,
    build_aime_bounded_node_messages,
    build_aime_full_rationale_node_messages,
    parse_aime_answer,
)
from topology_mas.execution.answers import parse_numeric_answer
from topology_mas.execution.prompts import PROMPT_VERSION, build_node_messages
from topology_mas.execution.schemas import ChatMessage
from topology_mas.models import MessageRecord, TaskInstance


class NodeExecutionProtocol(Protocol):
    """Task-specific prompt, parser, and public-message contract."""

    prompt_version: str
    supported_oracle_types: frozenset[str]

    def build_messages(
        self,
        task: TaskInstance,
        *,
        previous_output: str | None,
        incoming_messages: tuple[MessageRecord, ...],
    ) -> tuple[ChatMessage, ...]: ...

    def parse_answer(
        self,
        raw_text: str,
        *,
        finish_reason: str | None,
    ) -> str | None: ...

    def public_message(self, raw_text: str) -> str: ...


@dataclass(frozen=True)
class GSM8KNodeProtocol:
    prompt_version: str = PROMPT_VERSION
    supported_oracle_types: frozenset[str] = frozenset({"numeric"})

    def build_messages(
        self,
        task: TaskInstance,
        *,
        previous_output: str | None,
        incoming_messages: tuple[MessageRecord, ...],
    ) -> tuple[ChatMessage, ...]:
        return build_node_messages(
            task,
            previous_output=previous_output,
            incoming_messages=incoming_messages,
        )

    @staticmethod
    def parse_answer(
        raw_text: str,
        *,
        finish_reason: str | None,
    ) -> str | None:
        del finish_reason
        return parse_numeric_answer(raw_text)

    @staticmethod
    def public_message(raw_text: str) -> str:
        return raw_text


@dataclass(frozen=True)
class AIMEBoundedNodeProtocol:
    """One-pass bounded public-state protocol for AIME.

    The complete completion is public and is capped by the generation request.
    This deliberately does not claim to preserve a separate hidden/private chain
    of thought.
    """

    prompt_version: str = AIME_BOUNDED_PROMPT_VERSION
    supported_oracle_types: frozenset[str] = frozenset({"aime_integer"})

    def build_messages(
        self,
        task: TaskInstance,
        *,
        previous_output: str | None,
        incoming_messages: tuple[MessageRecord, ...],
    ) -> tuple[ChatMessage, ...]:
        return build_aime_bounded_node_messages(
            task,
            previous_output=previous_output,
            incoming_messages=incoming_messages,
        )

    @staticmethod
    def parse_answer(
        raw_text: str,
        *,
        finish_reason: str | None,
    ) -> str | None:
        if finish_reason == "length":
            return None
        return parse_aime_answer(raw_text)

    @staticmethod
    def public_message(raw_text: str) -> str:
        return raw_text


@dataclass(frozen=True)
class AIMEFullRationaleNodeProtocol:
    """One-pass AIME protocol broadcasting each raw completion verbatim."""

    prompt_version: str = AIME_FULL_RATIONALE_PROMPT_VERSION
    supported_oracle_types: frozenset[str] = frozenset({"aime_integer"})

    def build_messages(
        self,
        task: TaskInstance,
        *,
        previous_output: str | None,
        incoming_messages: tuple[MessageRecord, ...],
    ) -> tuple[ChatMessage, ...]:
        return build_aime_full_rationale_node_messages(
            task,
            previous_output=previous_output,
            incoming_messages=incoming_messages,
        )

    @staticmethod
    def parse_answer(
        raw_text: str,
        *,
        finish_reason: str | None,
    ) -> str | None:
        if finish_reason == "length":
            return None
        return parse_aime_answer(raw_text)

    @staticmethod
    def public_message(raw_text: str) -> str:
        return raw_text


GSM8K_PROTOCOL = GSM8KNodeProtocol()
AIME_BOUNDED_PROTOCOL = AIMEBoundedNodeProtocol()
AIME_FULL_RATIONALE_PROTOCOL = AIMEFullRationaleNodeProtocol()
