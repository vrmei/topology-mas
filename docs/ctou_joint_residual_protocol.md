# CTOU joint residual co-movement protocol

## Research question

After conditioning each normal receiver on its own CTOU input, do simultaneously
updated receivers still change state together more often than the conditionally
independent rollout assumes, and is any remaining dependence related to shared
upstream graph structure?

This is a CPU-only diagnostic over existing attack traces. It does not fit a joint
transition model and does not call an LLM.

## Hypotheses and falsification criteria

### H1: residual co-movement exists

For outcome `s` in `{correct, target}`, define the out-of-fold residual

\[
r_{i,s}=\mathbb{1}[S_i'=s]-\hat p_i(s).
\]

For two normal receivers updated in the same task, graph, attack run, and round, the
pair statistic is

\[
J_{ij,s}=r_{i,s}r_{j,s}.
\]

Conditional independence implies zero expected pair product when the marginal model
is calibrated. H1 is supported only if the task-cluster bootstrap interval for mean
`J` is above zero. Raw residuals and task-cell-adjusted residuals are reported
separately.

### H2: co-movement is topology-dependent

For each simultaneous receiver pair, measure:

- immediate predecessor overlap;
- immediate predecessor Jaccard similarity;
- overlap of the Round-0 causal cones that can reach each receiver by the current
  round;
- causal-cone Jaccard similarity;
- whether the attacker occurs in both causal cones.

The primary topology test is a within-event fixed-effect slope of adjusted `J` on
causal-cone Jaccard. The event is `(task, graph, attack position, round)`, so the test
compares receiver pairs exposed to the same task, topology, attacker, and round.

H2 is supported only when:

1. the slope is positive;
2. both task-cluster and graph-cluster bootstrap intervals exclude zero;
3. the direction is present for at least one outcome under both CTOU table and CTOU
   logistic predictions;
4. it is not restricted to pairs containing the readout.

Failure of these conditions means simultaneous dependence is not established as the
main explanation of the topology-level CTOU gap.

### H3: provenance absorbs the dependence

Repeat H1 and H2 using provenance-aware out-of-fold probabilities. A material drop in
mean pair product or topology slope would show that source lineage explains part of
the apparent joint dependence. A negligible change means provenance and co-movement
are distinct missing-information candidates.

## Crossed-holdout marginal predictions

Use the same five-by-five crossed graph/task folds as the recursive rollout:

- test graph IDs do not occur in transition-law training;
- test task IDs do not occur in transition-law training;
- CTOU table and multinomial logistic laws are both evaluated;
- provenance table and provenance logistic laws are secondary comparisons.

No graph feature, pair feature, task identity, receiver identity, or other receiver's
outcome is supplied to a marginal transition model.

## Pair construction

Construct all unordered pairs of normal receivers that update within the same:

`stratum + task + graph + run + attack node + round`.

Report three pair scopes:

- all pairs;
- internal--internal pairs;
- readout--internal pairs.

Attacker updates are excluded. If execution pruning makes a node inactive in a round,
it is not included in that round's pair set.

## Controlling hidden task difficulty

Raw residual co-movement can arise because a particular task is jointly easy or hard
for all nodes. Therefore the primary sensitivity residual removes the mean residual
for the same:

`task + round + exact local input cell`

estimated from other graphs only. Rows without support from another graph are omitted
from this sensitivity analysis. This is a post-hoc diagnostic adjustment, not an input
to the prospective topology evaluator.

The exact cell is original CTOU for CTOU models and provenance-aware CTOU for
provenance models.

## Matched-cell sensitivity

Repeat topology tests on receiver pairs having identical previous state and identical
incoming composition. This directly tests whether two nodes with the same marginal
transition probability exhibit different joint behavior depending on shared upstream
structure.

Support counts must be reported. Absence of an effect in a very small matched subset
is not evidence of conditional independence.

## Statistics

- Primary co-movement measure: mean residual product.
- Diagnostic measure: Pearson residual product, clipped away from probabilities 0/1.
- Primary topology measure: within-event fixed-effect slope.
- Uncertainty: 2,000 task-cluster bootstrap replicates and 2,000 graph-cluster
  bootstrap replicates.
- Report estimates even when intervals include zero.
- No multiple subgroup is promoted to a main result after inspection.

## Claim limits

- Positive residual dependence does not identify semantic content as its cause.
- A topology association is observational with respect to sampled graph structure;
  it does not by itself prove that adding a shared predecessor causes co-movement.
- Task-cell adjustment can remove broad task difficulty but not every latent semantic
  variable.
- Results remain conditional on the current Llama-3.1-8B, GSM8K, persistent-target
  attack protocol, and sampled topology family.
