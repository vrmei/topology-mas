# Synchronous MAS execution protocol

This module executes one task on one already validated directed graph. It does not sample graphs,
generate target errors, choose attack positions, or aggregate results across runs.

## Homogeneous nodes

All normal nodes use the same model-facing system prompt and update format. Numeric node identifiers
are orchestration metadata and are never shown to the model. Incoming messages use repeated,
unnumbered `<peer_message>` blocks: sender IDs, receiver IDs, graph IDs, and readout identity are not
included. To avoid making arbitrary structural labels determine prompt order, peer texts are ordered
by a stable content hash keyed by `message_order_seed`. This removes label-based ordering, but it does
not hide the number or content of messages received. The readout is not a special aggregator: it
independently answers in round 0 and uses the same update rule as every other normal node.

The prompt is versioned as `homogeneous-gsm8k-v1`. It requires an explicit
`FINAL_ANSWER: <number>` marker. The parser also accepts the cached GSM8K mutation marker `####`,
but never guesses an answer from the last number in free text.

## Strict synchronous rounds

- Round 0 is replayed from a graph-independent cache of anonymous replica slots. Each replica was
  generated independently from only the benchmark problem.
- Round `t > 0`: a normal node receives the problem, its own previous output, and messages sent by
  its in-neighbors in round `t-1`.
- All outputs for a round are completed before any are delivered to the next round. Consequently,
  information crosses at most one directed edge per round.
- A node generates one output and that exact text is copied to all causally active out-neighbors.

## Readout-cone pruning

Let `d(v,r)` be the shortest directed distance from node `v` to readout `r`, and let `T` be the
final round. Node `v` is called in round `t` exactly when:

```text
t + d(v,r) <= T
```

An edge `u -> v` carries a round-`t` output exactly when:

```text
t + 1 + d(v,r) <= T
```

Calls and sends outside this causal cone cannot change the final readout and are omitted. The
schedule is stored in every trace.

## Targeted attack

An attack run requires an offline target error that passed the objective mutation oracle and
belongs to the same task. Whenever the attacked node is active, the engine replays the same cached
wrong rationale and answer without calling the model. The attacker therefore cannot change target,
adapt wording, or have different generation randomness at different graph positions.

Round-zero cache entries are indexed by anonymous replica slot rather than structural node:

```text
seed_round_zero = H("round-zero-replica", experiment_seed, task_id, replica_slot)
```

For each experiment seed, an explicit permutation maps structural nodes to replica slots. The same
assignment is reused for paired graph comparisons. Multiple assignment seeds can later average over
the nuisance effect of where independent initial samples were placed. Structural node IDs remain in
the trace but never enter model-facing text.

Online rounds use deterministic replica-round seeds. The stochastic stream therefore moves with the
assigned initial state when a graph is relabeled:

```text
seed_online = H("online-replica-round", experiment_seed, task_id, replica_slot, round_index)
```

`graph_id` is deliberately excluded from both seed schemes. It remains part of `run_id`, so artifacts
cannot collide.

## Trace contents

`RunTrace` stores the full causal schedule, all node turns, every broadcast, prompt messages,
incoming message IDs, previous node output, parsed answer state, generation seed, token counts,
latency, final readout result, execution settings, the assignment seed and exact replica permutation,
and whether a turn was a model call, cache replay, or attack replay. Runtime token totals exclude the
offline round-zero generation cost; cached records retain their original token and latency metadata
separately.

The current module exposes a provider-neutral `TextGenerator` protocol and an OpenAI-compatible
adapter. Deterministic fake backends are used for protocol invariants; provider smoke tests are used
only for adapter compatibility and model-facing behavior.
