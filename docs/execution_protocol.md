# Synchronous MAS execution protocol

This module executes one task on one already validated directed graph. It does not sample graphs,
generate target errors, choose attack positions, or aggregate results across runs.

## Homogeneous nodes

All normal nodes use the same model-facing system prompt and update format. Numeric node identifiers
are not shown to the model. Incoming messages are ordered by sender ID for reproducibility but are
presented anonymously as `PEER_MESSAGE_1`, `PEER_MESSAGE_2`, and so on. The readout is not a special
aggregator: it independently answers in round 0 and uses the same update rule as every other normal
node.

The prompt is versioned as `homogeneous-gsm8k-v1`. It requires an explicit
`FINAL_ANSWER: <number>` marker. The parser also accepts the cached GSM8K mutation marker `####`,
but never guesses an answer from the last number in free text.

## Strict synchronous rounds

- Round 0: every node independently receives only the benchmark problem.
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

Clean and attack runs use the same per-node generation seeds. Round-zero seeds also match across
different graphs:

```text
seed_node_round = H(experiment_seed, task_id, node_id, round_index)
```

`graph_id` is deliberately excluded so topology comparisons begin from paired independent outputs.
It remains part of `run_id`, so artifacts cannot collide.

## Trace contents

`RunTrace` stores the full causal schedule, all node turns, every broadcast, prompt messages,
incoming message IDs, previous node output, parsed answer state, generation seed, token counts,
latency, final readout result, and whether a turn was a model call or attack replay.

The current module exposes a provider-neutral `TextGenerator` protocol and is tested with a
deterministic fake backend. A separate adapter will connect this interface to an
OpenAI-compatible/vLLM server after the execution semantics are frozen.
