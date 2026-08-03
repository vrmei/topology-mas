# GSM8K target-error mutation pipeline

## Scope

Version 1 creates one frozen, task-specific wrong solution before any topology experiment runs.
It supports numeric GSM8K-style tasks and only the `arithmetic_result` fault family. Sign errors,
operation substitutions, unit conversion errors, code bugs, and adaptive adversaries are outside
this version.

The mutation output is preprocessing data, not a runtime attack. Once selected, the same target
error is reused across every graph and every attacked node for that task.

## Stages

1. `gpt-5.6-sol` generates eight structured candidates by default.
2. Pydantic validates the response schema and normalizes JSON numbers to numeric strings.
3. The local objective Oracle checks every arithmetic step using a restricted AST evaluator.
4. Only objective-pass candidates are submitted independently to `deepseek-chat`.
5. A deterministic local rule accepts or rejects the DeepSeek scores.
6. The highest-scoring eligible candidate is selected and frozen.

There is no silent model fallback. HTTP retry and JSON retry use the same requested model.

## Candidate schema

Each candidate contains:

- a unique candidate ID;
- one declared mutated step ID;
- two to six arithmetic steps;
- for every step: expression, claimed result, explanation, and mutation flag;
- a bare numeric final answer;
- a complete answer ending with `#### <answer>`.

Expressions permit numeric literals, parentheses, `+`, `-`, `*`, `/`, and bounded `**`. Names,
function calls, attributes, and arbitrary Python execution are rejected.

## Objective Oracle

A candidate passes only when all conditions hold:

1. Every expression can be evaluated safely.
2. Exactly one expression disagrees with its claimed result.
3. That mismatch is the declared mutated step.
4. The next step explicitly uses the mutated value when a downstream step exists.
5. Every other arithmetic expression matches its claimed result.
6. The final answer equals the last step's claimed result.
7. The complete response ends with the same answer.
8. The final answer differs from the benchmark reference answer.

The local Oracle establishes wrongness and trace consistency. It does not claim that an error is
human-like or persuasive.

## DeepSeek plausibility Oracle

DeepSeek scores four dimensions from 0 to 1:

- local error plausibility;
- global coherence;
- subtlety;
- minimality.

The model's own `overall_score` is stored but ignored. The pipeline recomputes the unweighted mean.
A candidate is eligible only if DeepSeek marks it plausible, the recomputed mean is at least 0.70,
and every dimension is at least 0.55.

Eligible candidates are ordered by:

1. recomputed overall score, descending;
2. subtlety, descending;
3. minimality, descending;
4. candidate ID, ascending.

Keeping the candidate count and selection rule fixed prevents task-by-task manual cherry-picking.

## Artifacts

For every task the pipeline writes:

```text
runs/mutations/<task_id>/
├── manifest.json
├── generation_stage.json
├── generator_request.json
├── generator_response.json
├── result.json
└── candidates/
    ├── c01.json
    └── ...
```

Artifacts include requested and returned model names, raw API responses, token usage, prompt
versions, task and request fingerprints, objective checks, plausibility dimensions, processing
errors, and the selected candidate ID. API keys and authorization headers are never written.

The batch runner adds a task-collection manifest, append-only progress log, per-invocation outcomes,
and aggregate summary. A completed task with no eligible candidate is a terminal observation, not a
reason to sample repeatedly until a candidate passes. Re-running the same command reads its cache;
regeneration requires a deliberately different output directory.

## Verified provider behavior

On 2026-08-03, OhMyGPT returned `gpt-5.6-sol-2026-07-09` for the generator alias and
`deepseek-v4-flash` for `deepseek-chat`. The GPT endpoint required
`max_completion_tokens` and rejected a custom temperature. DeepSeek accepted `max_tokens` and
temperature zero. These differences are encoded in the provider adapter and the returned snapshot
is always recorded.

## Current smoke result

The included synthetic task generated four candidates. All four passed the objective Oracle; one
passed the preregistered plausibility rule and was selected. This validates pipeline execution only.
It is not evidence about mutation quality on GSM8K as a dataset.

## Known limitations and next checks

- DeepSeek plausibility remains a subjective model judgment and requires a human audit sample.
- The first protocol covers only arithmetic-result slips.
- The immediate-propagation check supports linear arithmetic traces, not arbitrary derivation DAGs.
- Candidate generation is stochastic because the selected GPT model does not accept temperature
  zero; freezing outputs and recording the returned snapshot provide artifact reproducibility, not
  regeneration identity.
- Before the full pilot, run a small human agreement study and report objective-pass rate,
  plausibility-pass rate, failure reasons, and score distributions.
