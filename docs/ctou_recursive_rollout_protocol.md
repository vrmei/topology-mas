# CTOU recursive rollout protocol

## Research question

Can a content-free C/T/O/U transition model predict attack endpoints on an unseen graph and unseen task when it receives only the true Round-0 node states and recursively generates every later state?

## Frozen information boundary

For every held-out attack run, the evaluator may use:

- the directed graph and readout node;
- the attack-node identity;
- the observed Round-0 C/T/O/U state of every node;
- the fixed horizon and synchronous active-node schedule.

Categorical states are read from the execution-time numeric Oracle stored in each trace/message as `answer_state`. Surface-form parsing is used only for legacy traces that lack this field; this keeps equivalent values such as `2.2` and `11/5` in the same state.

It may not use any true state, message composition, parsed answer, or text from Round 1 onward.

## Models

1. Persistence: every benign node retains its previous state.
2. Equal-weight DeGroot: the next categorical distribution is the equal-weight mixture of the receiver's previous state and its incoming neighbors.
3. CTOU table: a smoothed categorical transition table conditioned on previous state, round, and exact incoming C/T/O/U counts.
4. CTOU logistic: multinomial logistic regression using previous-state and round one-hot variables plus incoming counts and fractions.

The learned models exclude message text, task and graph identity, graph features, `n`, `m`, density, and receiver scope.

## Validation split

Use the same crossed 5 graph-fold x 5 task-fold protocol as the one-step benchmark. For a test cell `(g, t)`, train only on rows whose graph fold is not `g` and whose task fold is not `t`. Thus every test endpoint belongs to both an unseen graph and an unseen task.

## Rollout methods

### Primary: joint particle rollout

Initialize every particle with the observed Round-0 state vector. At each synchronous round, compute each active receiver's incoming composition from the particle's sampled parent states, apply the transition model, and sample the next state. The attack node is clamped to `T`. This preserves dependencies induced by shared ancestors within each particle.

Use common random numbers across models for the same graph, task-fold, attack position, and Round-0 state pattern. Cache identical categorical initial-state cases because the evaluator receives no task text.

### Sensitivity: factorized mean-field rollout

Propagate node marginals and construct each incoming-composition distribution under conditional independence. This is deterministic but discards correlations between nodes. Its gap from particle rollout estimates the effect of the factorization approximation.

## Primary metrics

- case-level four-state Brier score and log loss;
- target endpoint Brier score and log loss;
- correct endpoint Brier score and log loss;
- calibration: observed versus predicted endpoint `T` and `C` rates;
- fixed-`n` `m`-curve MAE and Spearman correlation;
- graph-level target-risk and robustness MAE/Spearman across tasks and attack positions.

Task bootstrap is used for endpoint loss contrasts. Graph bootstrap is used for topology-level contrasts. The bootstrap is conditional on the sampled graphs and the current Llama/GSM8K pilot.

## Falsification criterion

The transition model is not yet a topology evaluator if its recursive rollout fails to improve endpoint calibration or topology ranking over DeGroot, even when its one-step prediction is better.

## Claim limits

- Successful rollout shows that the coarse state dynamics are sufficient for prediction under this experiment; it does not prove the LLM internally implements the fitted law.
- Round-0 states are observed inputs, so the evaluator does not yet predict utility or robustness from topology alone.
- Failure can arise from transition misspecification, compounding error, or missing cross-node dependence; it is not automatically evidence of a semantic mechanism.
