"""Free-response AIME prompting and explicit integer-answer parsing."""

from __future__ import annotations

import re

from topology_mas.execution.schemas import ChatMessage
from topology_mas.models import TaskInstance

AIME_PROMPT_VERSION = "homogeneous-aime-free-response-v1"

_SYSTEM_PROMPT = """You are one solver in a homogeneous mathematical problem-solving system.
Solve the problem independently and check the derivation.
The answer must be an integer from 000 through 999. End with exactly one final line:
FINAL_ANSWER: \\boxed{ddd}
where ddd is the zero-padded three-digit answer. Do not write anything after that line."""

_EXPLICIT_FINAL = re.compile(
    r"(?im)^\s*(?:\*\*)?FINAL[\s_]+ANSWER\s*:\s*"
    r"(?:\\boxed\{\s*)?([0-9]{1,3})(?:\s*\})?\s*(?:\*\*)?\s*$"
)
_BOXED_INTEGER = re.compile(r"\\boxed\{\s*([0-9]{1,3})\s*\}")


def build_aime_round_zero_messages(task: TaskInstance) -> tuple[ChatMessage, ...]:
    """Build a candidate-free prompt; no evaluator field enters model-visible text."""

    return (
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=task.prompt),
    )


def parse_aime_answer(raw_text: str) -> str | None:
    """Return a normalized AIME integer from an explicit final marker only."""

    matches = list(_EXPLICIT_FINAL.finditer(raw_text))
    if not matches:
        matches = list(_BOXED_INTEGER.finditer(raw_text))
    if not matches:
        return None
    value = int(matches[-1].group(1))
    if not 0 <= value <= 999:
        return None
    return str(value)
