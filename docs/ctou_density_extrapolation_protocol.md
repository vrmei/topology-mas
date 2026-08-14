# CTOU density extrapolation protocol

## Research question

Does the content-free CTOU transition table capture a local update law that transfers to unseen graph densities, or does its recursive success depend on interpolation within the density support observed during fitting?

## Frozen state source

The execution-time numeric Oracle field `answer_state` is the sole primary C/T/O/U label source. Surface parsing is permitted only as a legacy fallback.

## Shared evaluator boundary

All experiments retain the recursive-rollout boundary:

- allowed: graph, readout, attack node, true Round-0 C/T/O/U states, horizon, active-node schedule;
- forbidden: true Round-1+ states, compositions, answers, rationale text, graph/task identity as model features.

The attack node remains clamped to `T`. All later benign states are generated recursively by the fitted transition law.

## Experiment A: leave-(n,m)-out

For every observed `(n,m)` level, remove all transition updates from that level before fitting CTOU and evaluate only endpoints from the removed level.

Two validation scopes are reported:

1. `density_only`: tasks may appear at other density levels. This isolates density shift.
2. `density_task`: additionally exclude the test task fold from fitting. This tests joint unseen-density and unseen-task transfer.

Every test graph is unseen because a graph belongs to exactly one `(n,m)` level.

## Experiment B: range extrapolation

The split boundary is fixed at the median sampled `m` level within each `n`:

- `n=5`: sparse `m<=10`, dense `m>10`;
- `n=8`: sparse `m<=28`, dense `m>28`.

Run both directions:

- `sparse_to_dense`: fit only on sparse updates and evaluate dense endpoints;
- `dense_to_sparse`: fit only on dense updates and evaluate sparse endpoints.

Again report `density_only` and `density_task` scopes.

## Models

- persistence;
- equal-weight DeGroot;
- CTOU transition table.

CTOU table is the primary learned model because it was the strongest recursive model in the in-support experiment and does not impose a parametric density trend. DeGroot and persistence do not require fitting.

The primary rollout uses factorized mean-field. The previous recursive experiment found graph-level particle/mean-field gaps below 0.002 on average for CTOU table, so the deterministic approximation is used for large extrapolation screening.

## Primary metrics

- endpoint four-state, target, and correct Brier/log loss;
- paired loss degradation relative to the original in-support crossed-holdout CTOU prediction;
- fixed-`n` held-out `m`-curve MAE and Spearman correlation;
- graph-level correct/target MAE and Spearman correlation with graph-bootstrap intervals;
- overall target-rate and attack-accuracy calibration by direction.

## Post-hoc support diagnostic

After endpoint prediction is complete, measure what fraction of observed test
updates have a training-supported:

- exact transition cell: previous state, round, and incoming C/T/O/U counts;
- count-only composition cell: incoming C/T/O/U counts.

This diagnostic is not supplied to the rollout. It separates density shift from
lookup-table state-support shift and must be described as post-hoc rather than a
preregistered primary metric.

## Interpretation rule

Strong leave-level interpolation but failed sparse/dense range extrapolation supports only local interpolation within observed density support. Successful range extrapolation supports a stronger claim that the fitted local state dynamics transfer across density regimes.

Neither result creates a topology-only predictor because true Round-0 states remain observed inputs.
