# Task-conditioned topology-ranking stability protocol

This protocol is fixed before computing stability results from the completed pilot.

## Research question

Are clean utility, targeted-attack robustness, and vulnerable-node rankings stable properties of a
communication graph, or do their measured rankings depend materially on the sampled tasks?

No claim that rankings are stable or unstable is assumed in advance.

## Frozen inputs and outcomes

Use all completed pilot tasks, selected graphs, clean runs, and valid non-readout attack positions.
Analyze three graph-level outcomes:

- clean readout correctness;
- mean attack correctness across non-readout attack positions;
- worst-position attack correctness, reported with explicit finite-task uncertainty.

Analyze two node-level outcomes:

- induced target-adoption rate;
- paired clean-minus-attacked accuracy drop.

Comparisons of graph quality are restricted within the same `(n,m)` stratum unless a metric is first
residualized for stratum. Complete or otherwise single-graph strata cannot identify a topology rank.

## Split-half stability

Generate 1,000 deterministic task splits from fixed seed 20260807. Each split partitions the 100
tasks into two disjoint halves. For each half, recompute graph and graph--attack-node metrics.

Primary stability summaries:

- mean and 95% interval of within-stratum graph-rank Spearman correlation between halves;
- vulnerable-node rank correlation within each graph;
- fractional agreement on the most vulnerable node;
- overlap of the empirical Pareto set for clean utility and targeted robustness.

## Pairwise reversals

For every pair of distinct graphs in the same stratum, estimate the paired task-level performance
difference. Report:

- full-sample difference and task-bootstrap interval;
- probability that the difference changes sign under task resampling;
- fraction of graph pairs whose ordering is unresolved at the pilot sample size.

These are uncertainty measures, not evidence that a task caused a topology effect.

## Variance decomposition

For clean task--graph outcomes and paired task--graph--attack-node outcomes, compare strictly held-out
predictors containing:

1. task information only through training-fold prevalence;
2. static graph or graph--node features;
3. Round-zero categorical state summaries;
4. their additive combination.

Residual variation is summarized by task and graph but is not labeled an interaction mechanism
without a dedicated intervention.

## Integrity gates

- every task split contains exactly 50 disjoint tasks per half;
- no metric uses a task outside its assigned half;
- graph comparisons never cross unmatched `(n,m)` strata;
- node comparisons retain the structural node identity and exclude readout attacks;
- uncertainty resamples entire tasks and preserves all graph/node observations for a sampled task;
- single-graph strata are reported descriptively and excluded from rank statistics.

## Claim boundary

- High split-half agreement supports measurement stability for this task distribution; it does not
  establish universal topology value.
- Low agreement may reflect finite positive-event counts rather than a semantic graph--task
  interaction.
- Rank reversals under task resampling are uncertainty diagnostics, not proof that graph theory fails.
- Cross-dataset and cross-model transfer still require new GPU experiments later.
