# Execution sanity checks

These checks test whether the implementation matches the controlled experiment. They are not
empirical claims about real LLM behavior.

## 1. Model-facing identity isolation

Structural node IDs, sender IDs, graph IDs, assignment IDs, and readout identity may appear in trace
metadata, but not in model prompts. Peer inputs use repeated anonymous `<peer_message>` blocks.

Expected invariant: changing only structural labels cannot change the visible prompt when the graph,
initial states, and messages are relabeled together.

## 2. Round-zero independence and assignment

Round-zero records are keyed by `(task, replica_slot, experiment_seed)` and generated without a graph.
Every graph run records a permutation from structural nodes to replica slots. Cache replay must not
call the online generator or add offline cache tokens to runtime totals.

Expected invariant: with the same cache and assignment, paired graph runs start from identical
node-level initial texts. Under graph relabeling, the assignment must be relabeled with the graph.

## 3. Isomorphism equivariance

For a deterministic, label-blind update rule, run a graph and an isomorphic relabeling with initial
states moved through the same isomorphism. Compare every relabeled node state at every active round.

Expected invariant: traces are identical after mapping node labels back. Failure indicates label,
ordering, scheduling, assignment, or stochastic-stream leakage in the executor. The comparison also
checks that online generation seeds move with the assigned replica rather than remaining attached to
arbitrary structural labels.

Passing this check does **not** establish that a stochastic provider model is isomorphism invariant.
That requires repeated empirical tests with fixed model snapshots and assignment seeds.

## 4. Anonymous message ordering

Peer messages are sorted by a stable hash of visible text and `message_order_seed`, not sender ID.

Expected invariant: the same multiset of peer texts produces the same sequence after relabeling.
Message count remains observable and is an intended consequence of indegree.
