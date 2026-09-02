# `homogeneous-mas-scalable-v1`

## Scope

This protocol is a separately versioned execution path. Existing full-rationale and
two-stage bounded-summary artifacts remain valid and are not reinterpreted.

The refactor isolates three engineering changes:

1. a task-indexed Round-0 response pool;
2. one-call local-full/public-summary generation;
3. a global dependency-aware READY scheduler.

No empirical equivalence claim is made before the validation experiments below.

## 1. Round-0 pool

`ScalableRoundZeroPoolGenerator` creates exactly `K` requested slots per task (default
`K=64`). Each slot has a deterministic generation seed and is saved atomically. The
generator retains every returned completion, including wrong, unparsed, and
length-stopped outputs. It never resamples based on answer state.

Each record contains:

- task, pool, response, and request identities;
- generation seed and exact prompt;
- raw response, parsed answer, and C/O/U state;
- input/output tokens, finish reason, latency, and provider metadata;
- SHA-256 of the raw response.

For a fixed `(task, n, Round-0 replicate)`, `build_round_zero_draws` samples `n`
response IDs without replacement. That response set is graph-independent.
`assign_draw_to_graph` then uses the full graph-content fingerprint to derive a
graph-specific permutation over structural node IDs. Both the draw and assignment
are persisted models with their own seeds and IDs.

The draw planner marks a configured 10--20% of replicates as `fresh_audit`. Such a
draw contains no pool IDs and must generate Round 0 inside the scheduler. It cannot
be accidentally materialized as a cache hit.

## 2. Local full solution and public summary

One normal node call must return exactly:

```text
<FULL_SOLUTION>
...
</FULL_SOLUTION>
<PUBLIC_SUMMARY>
...
</PUBLIC_SUMMARY>
```

The next update receives:

- the node's own previous `FULL_SOLUTION`;
- only each predecessor's `PUBLIC_SUMMARY`.

The immutable prompt prefix is ordered as system instruction, problem, and stable
instructions before dynamic local/peer state to support backend prefix caching.

`SinglePassDualChannelGenerator` makes one backend call and rejects the result if:

- either tag is absent, repeated, reordered, or unclosed;
- the provider stops because of the output-length limit;
- the public summary exceeds 512 tokens under the actual model tokenizer;
- the two channels have different parsed answers;
- the full solution is unparsed but the summary invents a parseable answer.

Rejected output raises `DualChannelValidationError`; it is not broadcast. Successful
output records both hashes, token counts, parsed answers, consistency, and
`summary_mode=single_pass`.

Pool construction uses the wrapper's audit mode: malformed dual-channel generations
are still persisted as required by the no-filtering rule, with
`summary_validation_passed=false`. Selecting one later causes an explicit run
failure at the communication boundary; it is never silently repaired or replaced.

The first implementation intentionally does not enable a second-call fallback. Its
necessity will be decided from measured missing-tag, length-stop, and inconsistency
rates on real model output.

## 3. Global READY scheduler

`build_causal_ready_jobs` expands each run into node-round jobs after applying the
existing causal rule:

\[
t+d(v,r)\le H.
\]

At round `t>0`, a job depends on its own previous state and every active predecessor
message from round `t-1`. Round-0 pool results enter as precompleted jobs; fresh audit
Round 0 remains ordinary generation work.

`GlobalReadyScheduler` merges jobs from all tasks, graphs, runs, and rounds into one
READY heap. A job becomes READY only after every dependency succeeds. Failure blocks
only causal descendants; unrelated runs continue.

Scheduling uses:

- direct downstream jobs unlocked (descending);
- distance to readout;
- round;
- estimated prompt-length bucket (`<=8k`, `<=16k`, `<=32k`, `<=64k`, `>64k`).

Each backend entry represents one independent model replica/GPU endpoint. Multiple
workers can consume the same endpoint when vLLM continuous batching supports it.
The scheduler does not claim a server-side batch size unless the backend reports it.

Per-job telemetry includes queue wait, execution time, backend index, actual token
counts, summary tokens, finish reason, and optional provider prefill/decode/batch
metrics. `aggregate_ready_run_costs` produces task-graph-run totals.

## Validation gates before scaling

### Gate A — Round-0 distribution

Run paired pooled and fresh-audit conditions. Compare `U0`, final utility, gain,
difficulty effects, density curves, and graph rankings. Use task + Round-0 draw as a
resampling block for pooled comparisons.

### Gate B — summary fidelity

On a small real-model sample, report tag-completion rate, answer consistency,
unparsed preservation, summary-token distribution, and prompt-token reduction. Do
not substitute an invalid summary.

### Gate C — scheduler equivalence and speed

Use identical deterministic inputs with one-worker sequential scheduling and the
global multi-worker scheduler. Outputs must match exactly. Only after this check
compare wall time, queue wait, GPU utilization, and server-reported batch behavior.

### Acceptance criteria requiring real experiments

The implementation makes these measurable but does not pre-claim them:

- Round-0 physical calls no longer scale with graph count outside fresh audits;
- most `n=20` prompts remain well below the context limit without truncation;
- global scheduling reduces wall-clock time at the same physical-call count.
