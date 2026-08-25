# Original AIME utility calibration: Qwen3-4B-Instruct-2507

## Scope

This run measures only single-agent, Round-0 utility on the 30 original 2025 AIME
problems. It does **not** mutate problems, generate target errors, run attacks, or
execute a multi-agent topology.

The purpose is to check whether the unmodified AIME tasks occupy a useful difficulty
range for the selected local model before any later MAS experiment is designed.

## Frozen protocol

- Dataset: 2025 AIME I and 2025 AIME II, all 30 original problems.
- Response mode: free-response integer answer; no answer choices are shown to the model.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, loaded from the existing system-disk Hugging
  Face cache.
- Sampling: temperature `0.7`, top-p `0.8`, top-k `20`, matching the model-card
  recommendation for non-thinking generation.
- Maximum output: `16,384` tokens.
- Replicates: 10 independent generations per problem, for 300 requests in total.
- Reuse: none. Every request has a distinct request ID and generation seed.
- Scoring: an answer is valid only if the response terminates normally and a final
  integer can be parsed. A length-truncated response is scored as unsuccessful even
  if an intermediate boxed integer appears in its text.
- Primary metric: budgeted Round-0 utility,
  \(U_0=\text{correct requests}/\text{all requests}\). Invalid and truncated outputs
  remain in the denominator.

The source dataset has canonical-LF SHA-256
`2ea54682e3139b7370b6c1dcd575f57de679426c29536a1f91e75b7acde8c388`.

## Calibration decision before the formal run

The first 3,072-token probe was rejected as a utility measurement because 75.3% of
responses stopped at the output limit. Raising the limit to 16,384 tokens reduced
this failure substantially. Adding presence penalty `1.0` did not improve the net
valid-answer rate and slightly reduced observed utility in the one-replicate probe,
so the formal run uses no presence penalty.

## Formal result

| Quantity | Result |
|---|---:|
| Requests | 300 |
| Valid final answers | 246 (82.0%) |
| Correct answers | 132 |
| Primary utility \(U_0\) | 44.0% |
| Task-bootstrap 95% CI for \(U_0\) | 29.0%–59.3% |
| Accuracy conditional on a valid answer | 53.7% |
| Length-limit stops | 47 (15.7%) |
| Other unparsed responses | 7 (2.3%) |
| Mean output tokens | 7,210.9 |
| Median output tokens | 6,652 |
| Total output tokens | 2,163,256 |
| Wall time | 30.42 minutes |
| A800 GPU time | 0.507 GPU-hours |

Contest-level utility is 45.3% on AIME I and 42.7% on AIME II.

Across the 30 tasks:

- 12 tasks are in the floor band (0%–10% success);
- 11 tasks are in the informative middle band (20%–80% success);
- 7 tasks are in the ceiling band (90%–100% success).

## What this result supports

The original AIME set is not an aggregate floor or ceiling for this model. It
contains a meaningful middle band, so it can support a clean-utility pilot without
first creating parameter mutations. The large between-task variation also means
that aggregate utility should always be accompanied by per-task results.

The primary 44.0% utility and the 53.7% conditional accuracy answer different
questions. The former measures performance under the frozen inference budget and is
the correct endpoint metric for this run. The latter is diagnostic: it separates
reasoning mistakes from failures to produce a valid final answer, but must not replace
the primary metric.

## Constraint discovered before an MAS run

The current response format produces long messages. With a median output of 6,652
tokens, a dense five-node readout that receives its own previous response and four
peer responses would receive roughly 33,260 message tokens before the problem and
system instructions are counted. That already exceeds the 32,768-token model length
used by this vLLM service, leaving no room for the readout's next output.

This is an operational extrapolation from the measured message lengths, not yet an
MAS result. It means that a full-topology utility experiment should not be launched
under the present full-rationale broadcast protocol until the context policy is
frozen. The two defensible choices have different scientific meanings:

1. increase the serving context and retain full messages, preserving the current SOP
   at higher memory and latency cost; or
2. define a bounded/structured message format, which makes the experiment tractable
   but changes the agents' local update law and therefore requires a new protocol
   decision.

No mutation or attack experiment is needed to resolve this issue.

## Reproducibility artifacts

- `artifacts/aime_original_qwen3_4b_round0_16k_formal_v1/audit.json`
- `artifacts/aime_original_qwen3_4b_round0_16k_formal_v1/summary.json`
- `artifacts/aime_original_qwen3_4b_round0_16k_formal_v1/summary.md`
- `artifacts/aime_original_qwen3_4b_round0_16k_formal_v1/per_task_solve_rates.csv`
- `artifacts/aime_original_qwen3_4b_round0_16k_formal_v1/per_problem_solve_rates.png`

Raw responses remain archived on the experiment server and are intentionally not
committed because they contain more than 2.16 million generated tokens.
