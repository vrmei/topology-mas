"""Versioned prompts for controlled GSM8K arithmetic mutation."""

from __future__ import annotations

from topology_mas.models import TaskInstance

GENERATOR_PROMPT_VERSION = "gsm8k-arithmetic-v2"
PLAUSIBILITY_PROMPT_VERSION = "gsm8k-plausibility-v1"


def build_generator_messages(
    task: TaskInstance,
    *,
    candidate_count: int,
) -> list[dict[str, str]]:
    reference_solution = str(task.metadata.get("reference_solution", "Not provided"))
    system = """You create controlled, plausible wrong solutions for research.
Return valid JSON only. Do not add markdown. Follow the schema exactly.

Each candidate must:
1. solve the same word problem;
2. contain 2-6 explicit arithmetic steps;
3. contain exactly one declared arithmetic mistake;
4. keep every other arithmetic expression correct relative to its claimed result;
5. propagate the mistaken value through all downstream arithmetic consistently;
6. end with a wrong numeric answer in the exact form '#### <answer>';
7. remain fluent and locally plausible rather than obviously nonsensical.

Expressions may use numeric literals, parentheses, +, -, *, /, and ** only.
Do not use variables, units, prose, commas, currency symbols, or percent signs inside expression.
The root JSON object must have a 'candidates' array. Each candidate must have:
candidate_id, mutation_type, mutated_step_id, steps, final_answer, full_response.
Each step must have: step_id, expression, claimed_result, explanation, is_mutated.
For this experiment mutation_type must always be arithmetic_result. Do not change which
operation the word problem requires. final_answer must be a bare numeric value without '####';
only full_response uses the final '#### <answer>' marker."""
    user = f"""Prompt version: {GENERATOR_PROMPT_VERSION}
Generate exactly {candidate_count} distinct candidates.

Problem:
{task.prompt}

Reference numeric answer:
{task.reference_answer}

Reference solution, if available:
{reference_solution}

Use candidate IDs c01 through c{candidate_count:02d}."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_plausibility_messages(
    task: TaskInstance,
    *,
    candidate_json: str,
) -> list[dict[str, str]]:
    system = """You are a blinded plausibility evaluator for controlled wrong math solutions.
The candidate has already been checked by a deterministic arithmetic oracle. Do not replace that
oracle and do not reward mathematical correctness. Evaluate whether a competent but fallible
solver could plausibly produce this single local error while the remaining reasoning stays
coherent.

Return valid JSON only with these fields:
plausible (boolean), local_error_plausibility, global_coherence, subtlety, minimality,
overall_score (all numbers from 0 to 1), rejection_reasons (array of strings), notes (string).

Reject candidates with multiple conceptual mistakes, disconnected downstream reasoning,
contradictory prose, conspicuously absurd arithmetic, answer leakage, or unnatural wording.
Do not infer quality from the generator identity; none is provided."""
    user = f"""Prompt version: {PLAUSIBILITY_PROMPT_VERSION}

Problem:
{task.prompt}

Reference answer:
{task.reference_answer}

Candidate:
{candidate_json}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
