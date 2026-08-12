# 500-task node-round transition protocol

This protocol is frozen before inspecting the 500-task transition results.

## Question

The fixed-`T=3` pilot shows that denser selected graphs expose more internal nodes to the target
error, while the final readout loss is lower at the highest tested density than at intermediate
density. The analysis separates two candidate descriptions:

1. adoption attenuation: target exposure increases, but `P(C -> T | target exposure)` decreases;
2. recovery: adoption does not decrease, but `P(T -> C)` increases.

These are empirical alternatives, not preregistered conclusions. Both may occur, or neither may
explain the aggregate endpoint pattern.

## Unit and pairing

One active benign-node update at round `t >= 1` in a paired clean/attack condition. The attacked
node is excluded. States use the deterministic numeric parser:

- `C`: benchmark-correct answer;
- `T`: the frozen task-specific target error;
- `O`: another parsed answer;
- `U`: unparsed output, reported separately and never silently merged into `O`.

Every attack update is paired with the clean update for the same task, graph, node, round,
experiment seed, assignment seed, and stochastic stream.

## Primary rates

- descriptive adoption: `P(C -> T | at least one incoming target message)`;
- attack-attributed adoption: `P(C -> T and paired clean current != T | at least one incoming
  attack-induced target message)`;
- recovery: `P(T -> C | previous T)`;
- attack-induced recovery: `P(T -> C | previous attack-induced T)`;
- persistence: `P(T -> T | previous T)`;
- collateral transition: `P(C -> O | target exposure)`.

Recovery is additionally stratified by whether the current update continues to receive a target
message. This prevents loss of exposure from being mislabeled as stronger correction.

## Exposure measurements

Report both update-level exposure and the mean number of unique benign receivers exposed per
task--graph--attack condition. The latter corresponds more closely to the statement that an error
"reaches more nodes."

## Regimes

The main estimand uses fixed `T=3`. Experiment C is reconstructed as the exact graph-depth causal
subset satisfying:

```text
round_index + receiver_distance_to_readout <= graph_depth
```

Experiment C is a horizon sensitivity analysis, not a pure density comparison.

## Statistics and claim limits

Rates are ratios of event counts. Confidence intervals use 10,000 task bootstrap replicates,
conditional on the currently selected graphs. Adjacent-density contrasts resample the same task
IDs for both strata. Five selected graphs per non-degenerate stratum do not identify population
topology uncertainty; `n5_m16` contains one unique complete graph.

The transitions are post-treatment intermediate outcomes. They can describe where the endpoint
paradox occurs in the realized propagation chain, but they do not by themselves establish a causal
LLM semantic mechanism.
