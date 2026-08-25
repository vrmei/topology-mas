# Original AIME utility calibration protocol

## Scope

This stage measures only single-agent, no-communication utility on the 30 original
problems from 2025 AIME I and 2025 AIME II. It performs no parameter mutation,
target-error generation, attack injection, topology sampling, or MAS rollout.

The purpose is to measure the task-level difficulty distribution before deciding
whether these tasks provide enough non-ceiling/non-floor variation for a later clean
MAS utility experiment. The decision is made after observing the calibration; it is
not encoded as a favorable expected outcome.

## Frozen task and prompt boundary

- Dataset: `data/aime/original_2025.jsonl` (30 tasks).
- Normal-agent visible content: problem statement only.
- Evaluator-only content: task ID and gold integer answer.
- Response mode: free response, not multiple choice.
- Required terminal marker: `FINAL_ANSWER: \boxed{ddd}`.
- No candidate answers, reference solution, target error, or mutation metadata is
  included in the model prompt.

The extraction script and source manifest are committed with the dataset. Problem
identity follows the ordered list rendered by each contest page because its legacy
`aimeProblemNumber` metadata contains duplicate values.

## Qwen phase

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Local system-disk snapshot:
  `/root/hf-system-cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`.
- Replicates: 10 independent generations per task (300 requests).
- Sampling: temperature 0.7, top-p 0.8, top-k 20.
- Maximum output: 16,384 tokens. The initial 3072-token attempt is retained as a
  failed measurement probe rather than treated as utility data.

The Qwen sampling values follow the downloaded model's generation configuration and
model card. This phase estimates Qwen's task distribution; it is not by itself a
controlled claim that Qwen is better or worse than another model with different
recommended decoding.

Two one-replicate probes were frozen before the 10-replicate run. At 16,384 tokens,
adding `presence_penalty=1.0` reduced length stops but did not improve the total valid
parse rate, so the formal run retains the base model-card sampling without a presence
penalty. Length stops remain observable utility failures rather than being silently
repaired with a different decoding condition.

## Execution and storage

Each `(task, replicate)` receives a deterministic generation seed. Replicates never
reuse generated text. Every successful request is stored atomically in its own file;
an interrupted run resumes only missing requests. A failed request is logged without
stopping unrelated requests and remains missing until a later resume succeeds.
Any response with `finish_reason=length` is invalidated even if an intermediate
`\boxed{}` expression happens to be parseable.

## Outputs

Primary metric:

\[
U_0 = \frac{1}{30}\sum_q \hat p_q,
\]

where `p_q` is the 10-replicate solve rate for problem `q`.

The analysis also reports:

- each problem's solve rate;
- AIME I and AIME II utility separately;
- task-bootstrap 95% confidence interval for `U0`;
- parse failure and length-stop rates;
- descriptive task counts at 0–0.1, 0.2–0.8, and 0.9–1.0 solve rate.

No clean MAS topology experiment is launched automatically. Its design is evaluated
only after this calibration is complete.
