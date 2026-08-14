# CTOU Round-0-free evaluation protocol

## Research question

How much endpoint and graph-ranking performance remains when recursive CTOU rollout no longer observes the held-out task's realized Round-0 node states?

The target progression is:

\[
G + S_0^{\mathrm{real}} \rightarrow \widehat R(G)
\]

to:

\[
G + \mathcal D(S_0 \mid \text{model, dataset}) \rightarrow \widehat R(G).
\]

This experiment does not claim topology-only transfer outside the current Llama/GSM8K regime.

## Preliminary reliability reference

Before evaluating initialization priors, repeatedly split the fixed 50 tasks into two disjoint 25-task halves. For each half, aggregate the true attack endpoints by graph and compute graph-ranking Spearman correlation separately for:

- final target-error rate;
- final correct rate;
- node counts \(n=5\) and \(n=8\).

Report the full split-half distribution and the Spearman–Brown diagnostic. This is an empirical stability reference for the finite task sample, not a mathematical upper bound on predictor performance.

## Frozen transition and attack components

All conditions use the existing:

- four states: correct, target error, other error, and unparsed;
- CTOU multinomial-logistic local transition law;
- crossed five graph-fold by five task-fold validation;
- synchronous active-node schedule and original graph horizon;
- persistent target-error attacker clamped to state `T`;
- joint particle recursive rollout;
- endpoint and robustness definitions.

Only the Round-0 initialization source changes.

## Initialization conditions

### Oracle

Use the observed categorical Round-0 vector from the held-out trace. This is the existing recursive CTOU result and serves as the information-rich reference.

### IID empirical prior

Within each crossed holdout cell, estimate the benign-state marginal from training graphs and training tasks only:

\[
\widehat\pi_{0,n}=P(S_i^0=C,T,O,U\mid n).
\]

Initialize every benign test node independently from this node-count-matched marginal. Clamp the attack node to `T`. The prior sees no held-out task text or realized state.

### Correlated empirical prior

Within each holdout cell and node count, retain each training trace's complete benign Round-0 state vector. For every rollout particle:

1. sample one training vector;
2. randomly permute its benign-node entries;
3. map the entries to the test graph's benign nodes;
4. clamp the attack node to `T`.

This preserves vector-level task/run difficulty correlation and the number of simultaneous errors while discarding training node identity and graph position.

## Leakage controls

For test graph fold \(g\) and task fold \(t\):

- fit the transition law only on rows outside \(g\) and \(t\);
- estimate both initialization priors only from attack traces outside \(g\) and \(t\);
- require zero graph-ID and task-ID overlap;
- use identical held-out endpoints for all initialization conditions.

## Metrics

Primary comparisons are the degradation from oracle initialization in:

- endpoint four-state and binary correct/target Brier score;
- endpoint log loss;
- fixed-\(n\) density-curve MAE and Spearman correlation;
- graph-level correct-rate and target-rate MAE/Spearman;
- graph-level predicted versus observed scatter.

Task bootstrap is used for endpoint loss summaries and graph bootstrap for topology-level metrics.

## Decision rule

- Small degradation under empirical priors supports proceeding toward a distribution-conditioned topology evaluator in the current regime.
- Large degradation means the realized task-specific initialization carries essential information; the next research problem becomes modeling \(P(S_0\mid q,\text{model})\), not further tuning the local CTOU transition law.
- A difference between IID and correlated priors diagnoses whether task-level co-failure structure matters beyond the global marginal.

## Claim limits

- The empirical priors know the current model, dataset, attack protocol, and node count.
- The correlated prior is a nonparametric bootstrap, not a learned task-conditioned initializer.
- Successful prediction here does not establish transfer to another model, dataset, graph scale, horizon, or attack semantics.
- Split-half stability quantifies finite-sample reproducibility; it is not a strict ceiling.
