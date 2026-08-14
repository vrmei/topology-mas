# CTOU support-stratified error protocol

## Research question

When CTOU is transferred outside its fitted density range, how much endpoint
prediction error is associated with insufficient support for the local
transition cells visited during recursive rollout?

This is a failure-analysis question. Association between support and error does
not by itself establish that support shortage causes the error.

## Frozen inputs

- Existing Llama-3.1-8B GSM8K dense-50 traces;
- existing sparse-to-dense and dense-to-sparse splits;
- existing density-only and density+task validation scopes;
- the same CTOU table, prior strength, true Round-0 states, graph, attacker,
  horizon, and active-node schedule used in density extrapolation;
- no new LLM inference.

The analysis must reproduce every saved CTOU endpoint probability within
`1e-6` before support results are accepted.

## Two support measurements

### Expected-rollout support

At every recursively predicted normal-node update, enumerate the probability
distribution over:

\[
(S_{t-1},t,C,T,O,U).
\]

Use the fitted split's exact transition count for each cell and accumulate the
probability-weighted number and fraction of visits to:

- unseen cells (`count=0`);
- cells with `count<5`;
- cells with `count<10`;
- cells with `count<20`.

This quantity is available from the surrogate rollout without observing true
Round-1+ states. It is the primary potentially deployable support-risk measure.

### Observed-trace support

Look up the exact training count of every transition cell actually visited by
the saved LLM trace and aggregate the same measures per endpoint.

This is a post-hoc explanatory diagnostic. It is outcome-dependent and cannot
be presented as a prediction-time feature.

## Reporting strata

The continuous fractions are primary. Thresholded strata are descriptive.

Expected-rollout strata use the probability mass assigned to cells with
`count<20`:

- zero mass: `all_high_support`;
- `(0,5%]`: `low_mass_le_5pct`;
- `(5%,20%]`: `low_mass_5_20pct`;
- `>20%`: `low_mass_gt_20pct`.

Observed-trace strata are:

- all cells have at least 20 examples;
- at least one low-support cell but no unseen cell;
- exactly one unseen cell;
- multiple unseen cells.

The threshold 20 is not treated as a discovered boundary. Sensitivity at 5 and
10 and continuous rank associations are retained.

## Outcomes

Endpoint outcomes:

- multiclass Brier;
- target Brier and absolute target-probability error;
- correct Brier and absolute correct-probability error.

Graph outcomes:

- signed target/correct residual;
- absolute target/correct residual;
- Spearman association between graph-average support burden and residuals.

Stratum means receive task-bootstrap 95% intervals. Results are reported
separately by direction, validation scope, and node count.

## Interpretation

- A monotonic association between lower expected-rollout support and larger
  error is evidence that a detectable support shortage accompanies part of the
  extrapolation failure.
- An association found only for observed-trace support is explanatory but not
  available to a topology evaluator at prediction time.
- Weak associations do not prove transition-law shift; they only show that
  exact table support alone is insufficient to explain the residuals.
- No causal support-origin claim is allowed without a controlled support or
  model-form intervention.
