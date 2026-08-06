"""Explicit GSM8K answer extraction and state classification."""

from __future__ import annotations

import re
from fractions import Fraction

from topology_mas.models import AnswerState

_FINAL_ANSWER = re.compile(
    r"(?im)(?:\*\*)?\s*FINAL[\s_]+ANSWER\s*:\s*"
    r"(?:FINAL[\s_]+ANSWER\s*:\s*)?(.+?)\s*(?:\*\*)?\s*$"
)
_GSM8K_MARKER = re.compile(r"####\s*([^\s]+)\s*$")
_BOXED_ANSWER = re.compile(
    r"(?im)^\s*(?:\\\[\s*)?\\boxed\{([^{}]+)\}(?:\s*\\\])?\s*$"
)


def normalize_numeric_answer(value: str) -> str:
    cleaned = value.strip().replace(r"\$", "$")
    cleaned = cleaned.removeprefix("**").removesuffix("**").strip()
    cleaned = cleaned.replace(",", "").removeprefix("$").strip()
    number = Fraction(cleaned)
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def parse_numeric_answer(raw_text: str) -> str | None:
    """Parse an explicit answer marker; never guess from an unmarked trailing number."""

    matches = list(_FINAL_ANSWER.finditer(raw_text))
    if not matches:
        matches = list(_GSM8K_MARKER.finditer(raw_text))
    if not matches:
        matches = list(_BOXED_ANSWER.finditer(raw_text))
    if not matches:
        return None
    try:
        return normalize_numeric_answer(matches[-1].group(1))
    except (ValueError, ZeroDivisionError):
        return None


def classify_numeric_answer(
    parsed_answer: str | None,
    *,
    reference_answer: str,
    target_answer: str | None,
) -> AnswerState:
    if parsed_answer is None:
        return AnswerState.UNPARSED
    normalized_reference = normalize_numeric_answer(reference_answer)
    if parsed_answer == normalized_reference:
        return AnswerState.CORRECT
    if target_answer is not None and parsed_answer == normalize_numeric_answer(target_answer):
        return AnswerState.TARGET_ERROR
    return AnswerState.OTHER_ERROR
