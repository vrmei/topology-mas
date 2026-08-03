"""Deterministic arithmetic oracle for the restricted GSM8K mutation format."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from fractions import Fraction

from topology_mas.mutation.schemas import (
    MutationCandidate,
    ObjectiveOracleResult,
    StepOracleCheck,
)

OBJECTIVE_ORACLE_VERSION = "gsm8k-numeric-dag-v2"


class UnsafeExpressionError(ValueError):
    pass


def parse_number(value: str) -> Fraction:
    cleaned = value.strip().replace(",", "")
    cleaned = cleaned.removeprefix("$")
    if not cleaned:
        raise ValueError("empty numeric value")
    if cleaned.endswith("%"):
        return Fraction(cleaned[:-1]) / 100
    return Fraction(cleaned)


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


class SafeArithmeticEvaluator:
    """Evaluate numeric arithmetic without names, calls, attributes, or arbitrary code."""

    _MAX_ABS_NUMERATOR = 10**18
    _MAX_ABS_EXPONENT = 10

    def evaluate(
        self,
        expression: str,
        *,
        variables: Mapping[str, Fraction] | None = None,
    ) -> Fraction:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise UnsafeExpressionError("invalid expression syntax") from exc
        value = self._visit(tree.body, variables or {})
        if abs(value.numerator) > self._MAX_ABS_NUMERATOR:
            raise UnsafeExpressionError("expression result is too large")
        return value

    @staticmethod
    def referenced_names(expression: str) -> set[str]:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise UnsafeExpressionError("invalid expression syntax") from exc
        return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    def _visit(self, node: ast.AST, variables: Mapping[str, Fraction]) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise UnsafeExpressionError(f"undeclared variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._visit(node.operand, variables)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = self._visit(node.left, variables)
            right = self._visit(node.right, variables)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise UnsafeExpressionError("division by zero")
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right.numerator) > self._MAX_ABS_EXPONENT:
                    raise UnsafeExpressionError("unsupported exponent")
                return left ** right.numerator
        raise UnsafeExpressionError(f"unsupported expression node: {type(node).__name__}")


class NumericMutationOracle:
    """Verify a candidate is a one-error arithmetic trace with a wrong final answer."""

    _FINAL_PATTERN = re.compile(r"####\s*([^\s]+)\s*$")

    def __init__(self) -> None:
        self._evaluator = SafeArithmeticEvaluator()

    def verify(
        self,
        candidate: MutationCandidate,
        *,
        reference_answer: str,
    ) -> ObjectiveOracleResult:
        reasons: list[str] = []
        checks: list[StepOracleCheck] = []
        mismatched_steps: list[str] = []

        step_by_id = {step.step_id: step for step in candidate.steps}
        for step in candidate.steps:
            try:
                referenced_names = self._evaluator.referenced_names(step.expression)
                declared_dependencies = set(step.depends_on)
                if referenced_names != declared_dependencies:
                    reasons.append(
                        f"step {step.step_id} expression variables do not exactly match "
                        "depends_on"
                    )
                variables = {
                    dependency_id: parse_number(step_by_id[dependency_id].claimed_result)
                    for dependency_id in step.depends_on
                }
                computed = self._evaluator.evaluate(
                    step.expression,
                    variables=variables,
                )
                claimed = parse_number(step.claimed_result)
                matches = computed == claimed
                if not matches:
                    mismatched_steps.append(step.step_id)
                checks.append(
                    StepOracleCheck(
                        step_id=step.step_id,
                        expression=step.expression,
                        computed_result=format_fraction(computed),
                        claimed_result=format_fraction(claimed),
                        matches=matches,
                    )
                )
            except (ValueError, ZeroDivisionError) as exc:
                reasons.append(f"step {step.step_id}: {exc}")
                checks.append(
                    StepOracleCheck(
                        step_id=step.step_id,
                        expression=step.expression,
                        claimed_result=step.claimed_result,
                        matches=False,
                        error=str(exc),
                    )
                )

        if mismatched_steps != [candidate.mutated_step_id]:
            reasons.append(
                "the arithmetic mismatches must consist only of the declared mutated step"
            )

        try:
            reference = parse_number(reference_answer)
            final = parse_number(candidate.final_answer)
        except ValueError as exc:
            reasons.append(f"answer parsing failed: {exc}")
            reference = None
            final = None

        if reference is not None and final is not None and final == reference:
            reasons.append("candidate final answer is not wrong")

        if final is not None:
            try:
                last_claimed = parse_number(candidate.steps[-1].claimed_result)
                if final != last_claimed:
                    reasons.append("final_answer must equal the last step's claimed_result")
            except ValueError as exc:
                reasons.append(f"last-step result parsing failed: {exc}")

        text_match = self._FINAL_PATTERN.search(candidate.full_response)
        if text_match is None:
            reasons.append("full_response must end with '#### <answer>'")
        elif final is not None:
            try:
                if parse_number(text_match.group(1)) != final:
                    reasons.append("full_response final marker disagrees with final_answer")
            except ValueError as exc:
                reasons.append(f"full_response final marker is not numeric: {exc}")

        final_step_id = candidate.steps[-1].step_id
        final_ancestors: set[str] = set()
        pending = list(step_by_id[final_step_id].depends_on)
        while pending:
            dependency_id = pending.pop()
            if dependency_id in final_ancestors:
                continue
            final_ancestors.add(dependency_id)
            pending.extend(step_by_id[dependency_id].depends_on)

        disconnected = set(step_by_id) - final_ancestors - {final_step_id}
        if disconnected:
            reasons.append(
                "steps do not contribute to the final answer: "
                + ", ".join(sorted(disconnected))
            )
        if (
            candidate.mutated_step_id != final_step_id
            and candidate.mutated_step_id not in final_ancestors
        ):
            reasons.append("the mutated step does not propagate to the final answer")

        return ObjectiveOracleResult(
            passed=not reasons,
            reasons=tuple(reasons),
            step_checks=tuple(checks),
            parsed_reference_answer=(
                format_fraction(reference) if reference is not None else None
            ),
            parsed_final_answer=format_fraction(final) if final is not None else None,
        )
