# Classical dynamics baseline protocol

This protocol fixes the first non-LLM dynamics comparison before its results are inspected.

## Research question

How closely can parameter-free, content-free graph dynamics reproduce the targeted-error outcomes
of the homogeneous LLM multi-agent pilot when graph, Round-zero states, attack position, target
answer, and round horizon are held fixed?

No directional difference between the LLM and either classical baseline is assumed in advance.

## Frozen inputs

For every task--graph cell, the analysis uses the exported `classical_initial_states.jsonl` record.
It contains the exact parsed Round-zero answer assigned to every structural node. An unparsed answer
is represented by a node-specific sentinel so that two parsing failures do not become false
agreement. The attack condition replaces the selected attacker's initial state with the frozen
oracle-verified target answer and clamps that node to the target in every later round.

The graph, directed edges, readout node, and `max_rounds` are copied from `selected_graphs.jsonl`.
All nodes are simulated synchronously. Simulating the full graph is readout-equivalent to the
execution engine's causal pruning because states outside the final readout's causal cone cannot
reach it within the fixed horizon.

## Baselines

### Frozen Round zero

Every non-readout node remains at its frozen initial answer and communication has no effect. Because
the readout is not attacked, clean and attacked outcomes are identical. This is the trivial
no-propagation baseline against which inertial majority must improve.

### Inertial local majority

At each round, a non-attacker considers its previous state and the previous states of all directed
in-neighbors. It changes only when one state has a unique plurality. A tie preserves the node's
previous state. This avoids arbitrary numeric or lexical ordering of answer labels.

### Equal-weight DeGroot on categorical beliefs

Each distinct parsed answer is represented as a one-hot belief dimension. Every node assigns equal
weight to its own previous belief vector and those of all directed in-neighbors. The row-stochastic
matrix is fixed across tasks and rounds. At readout, the state with the largest belief mass is
selected; a tie preserves the previous discrete state. The continuous belief vector, not the
tie-broken discrete label, is propagated in later rounds.

These are finite-state reductions of the existing communication process. They deliberately exclude
rationales, confidence, provenance, task text, message order, and history-dependent weighting.

## Paired outcomes

For every baseline, task, graph, and non-readout attack position, compute:

- clean final correctness;
- attacked final correctness;
- clean-minus-attacked correctness in `{-1, 0, 1}`;
- whether the attack newly induces the fixed target answer at readout.

Node-level aggregate outcomes are compared with the LLM observations using:

- MAE, RMSE, and R-squared;
- pooled and mean within-graph Spearman correlation;
- fractional-credit top-1 vulnerable-node identification.

Task-level exact agreement, balanced accuracy, and Matthews correlation are diagnostic because
rows share tasks and graphs. Primary uncertainty resamples complete held-out graphs and preserves
all node positions and tasks within a graph.

Each dynamic baseline is also compared with frozen Round zero using a graph-paired MAE difference.
The reported quantity is frozen-baseline MAE minus dynamic-baseline MAE, so positive values favor
the dynamic process.

## Integrity requirements

- exactly one initial-state record per task--graph cell;
- exact coverage of every recorded non-readout attack position;
- identical reference and target answers between initial-state and attack artifacts;
- the readout is never treated as an attack position;
- graph-independent Round-zero assignments are audited within each node-count stratum.

## Claim boundary

- Agreement supports compatibility with the specified finite-state update rule; it does not prove
  that the LLM internally implements that rule.
- Disagreement does not establish a semantic mechanism. It may reflect stochastic decoding, answer
  projection, missing confidence, task difficulty, or another omitted state variable.
- The two baselines do not exhaust classical consensus, diffusion, or Byzantine-resilient methods.
- Current intervals cover selected-graph variation only, not task, seed, model, or graph-population
  variation.
