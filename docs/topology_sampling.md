# Constrained directed-topology sampling

## Purpose

The sampler measures a controlled topology space; it does not optimize topology. Utility,
robustness, model outputs, and classical graph templates never influence which random graph is
proposed or accepted.

For one stratum, the following values are fixed:

- node count `n`;
- exact directed-edge count `m`;
- readout node `r`;
- final communication round `T`;
- requested number of distinct labeled graphs;
- pseudorandom seed.

## Proposal space

The readout has no outgoing edges and self-loops are forbidden. The candidate edge pool therefore
contains

```text
(n - 1)^2
```

directed edges. Each proposal samples exactly `m` different edges uniformly with Python's seeded
pseudorandom generator. A SHA-256-derived seed identifies every sample-index and attempt pair, so
accepted artifacts are deterministic for a fixed configuration and implementation version.

The sampler rejects a proposal when:

1. at least one node cannot reach the readout;
2. otherwise, at least one shortest directed distance to readout exceeds `T`;
3. otherwise, the exact labeled edge set was already accepted.

Rejection categories are mutually exclusive. Conditional rejection sampling retains no utility or
robustness preference. It samples labeled graphs and deliberately does not remove graph
isomorphisms; quotienting by isomorphism would define a different distribution.

The requested graph count cannot exceed the proposal-space upper bound
`choose((n - 1)^2, m)`. The number of legal graphs may be smaller and is not generally enumerated;
an explicit maximum-attempt guard fails rather than silently changing the sampling algorithm.

## Legal graph conditions

Every accepted graph satisfies:

```text
outdegree(r) = 0
distance(v, r) <= T for every v != r
```

Multiple indegree-zero source nodes and directed cycles are allowed. Reachability makes the readout
the unique sink automatically.

## Round-aware causal schedule

Round 0 contains every node's independent answer. Under synchronous one-edge-per-round messaging,
a node `v` generates a new output in round `t` only when

```text
t + distance(v, r) <= T
```

An edge `u -> v` carries a round-`t` message only when

```text
t + 1 + distance(v, r) <= T
```

This removes calls and transmissions outside the final readout's causal cone without changing its
round-`T` output, assuming no shared side effects. It also bounds feedback through directed cycles.

Every graph records distances, sources, cycle presence, active node counts, active edge counts, and
total message opportunities. Equal edge count controls the number of static channels, not exact
token or compute cost; actual messages, tokens, and latency must still be measured during MAS runs.

## Artifacts

The CLI writes:

```text
<output-dir>/
|-- graphs.jsonl
`-- manifest.json
```

The manifest includes sampler version, complete configuration, graph IDs, collection fingerprint,
attempt count, mutually exclusive rejection counts, and proposal acceptance rate. Reusing an output
directory is idempotent only when all content is identical; otherwise it fails.

## Structural smoke results

Twenty graphs were sampled per non-degenerate stratum with seed 0 and `T=3`; the complete
`n=5,m=16` endpoint contains only one distinct graph. These values validate implementation and
estimate rejection efficiency only; they are not scientific results.

| n | m | graphs | attempts | acceptance |
|---:|---:|---:|---:|---:|
| 5 | 4 | 20 | 617 | 0.0324 |
| 5 | 8 | 20 | 31 | 0.6452 |
| 5 | 12 | 20 | 20 | 1.0000 |
| 5 | 16 | 1 | 1 | 1.0000 |
| 8 | 7 | 20 | 16,677 | 0.0012 |
| 8 | 14 | 20 | 64 | 0.3125 |
| 8 | 21 | 20 | 28 | 0.7143 |
| 8 | 28 | 20 | 20 | 1.0000 |

Sparse `n=8,m=7` sampling is feasible at pilot scale but inefficient. If later configurations make
rejection sampling impractical, any MCMC or constructive replacement must be introduced as a new
sampler version and validated for sampling bias rather than silently substituted.
