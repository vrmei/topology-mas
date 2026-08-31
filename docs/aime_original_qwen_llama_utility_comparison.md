# Original 2025 AIME utility: Qwen3-4B versus Llama-3.1-8B

## Scope

This is a Round-0, single-agent utility comparison on the 30 unmodified 2025 AIME
I and II problems. There are no problem mutations, target errors, attacks, topologies,
or cross-run output reuse. Each model receives only the free-response problem and the
same answer-format instruction.

Each problem is sampled independently 10 times. The two models use the sampling
parameters in their respective local generation configurations:

- Qwen3-4B-Instruct-2507: temperature `0.7`, top-p `0.8`, top-k `20`;
- Llama-3.1-8B-Instruct: temperature `0.6`, top-p `0.9`.

Both runs use a 32,768-token serving context and a maximum output of 16,384 tokens.
A length-truncated response is invalid even if its unfinished text contains a boxed
integer. Primary utility retains all invalid responses in the denominator.

## Results

| Metric | Qwen3-4B | Llama-3.1-8B |
|---|---:|---:|
| Requests | 300 | 300 |
| Correct answers | 132 | 0 |
| Budgeted Round-0 utility | 44.0% | 0.0% |
| Task-bootstrap 95% CI | 29.0%–59.3% | 0.0%–0.0% |
| Valid parsed answers | 246 (82.0%) | 116 (38.7%) |
| Accuracy among valid answers | 53.7% | 0.0% |
| Length-limit stops | 47 (15.7%) | 168 (56.0%) |
| Median output tokens | 6,652 | 16,384 |
| Mean output tokens | 7,210.9 | 9,656.4 |
| Total output tokens | 2,163,256 | 2,896,934 |
| A800 GPU-hours | 0.507 | 0.987 |

The paired task-level difference, Llama minus Qwen, is `-0.440`, with a task-bootstrap
95% interval of `-0.593` to `-0.297`.

Qwen has 11 tasks in the 20%–80% informative band. Llama has no task outside the
0%–10% floor band.

## Integrity checks

For the Llama run:

- all 300 planned requests completed and no API request failed;
- all 300 request IDs and generation seeds are unique;
- every response reports `meta-llama/Llama-3.1-8B-Instruct`;
- an independent answer-regex recomputation also finds zero correct valid responses;
- even among the 116 normally terminated, parsable responses, zero answers are
  correct;
- no truncated response has a final extractable answer equal to the gold answer.

The zero utility is therefore not an analyzer aggregation bug, and truncation alone
cannot explain it.

## Claim boundary

The supported conclusion is narrow:

> Under this frozen free-response protocol, the original 2025 AIME set is useful for
> Qwen3-4B utility experiments but is a floor task set for Llama-3.1-8B.

This run does **not** establish that Qwen is intrinsically better at mathematical
reasoning. Llama-3.1 predates the 2025 contest, while Qwen3-4B-Instruct-2507 was
released afterward. The exact training exposure of Qwen is not established here, so
benchmark contamination or memorization is a possible confound, not a demonstrated
cause. The models also use their own recommended sampling parameters rather than an
identical decoding distribution.

Consequently, original 2025 AIME should not be used as a shared cross-model topology
benchmark for these two models. A later cross-model study needs a task set with
controlled temporal exposure or independently verified novel instances. That is a
separate design decision and is not part of this utility-only run.

## Runtime implication

Llama's median response reaches the full 16,384-token output budget. Full-rationale
broadcast would therefore exceed the current MAS context budget even more severely
than Qwen. Regardless of task difficulty, this model cannot enter the existing
full-message MAS protocol without first freezing a bounded-message or larger-context
policy.

## Artifacts

- `artifacts/aime_original_llama31_8b_round0_16k_formal_v1/audit_llama.json`
- `artifacts/aime_original_llama31_8b_round0_16k_formal_v1/status.json`
- `artifacts/aime_original_llama31_8b_round0_16k_formal_v1/summary.json`
- `artifacts/aime_original_llama31_8b_round0_16k_formal_v1/summary.md`
- `artifacts/aime_original_llama31_8b_round0_16k_formal_v1/per_task_solve_rates.csv`
- `artifacts/aime_original_llama31_8b_round0_16k_formal_v1/per_problem_solve_rates.png`

Raw responses remain archived on the remote experiment volume.
