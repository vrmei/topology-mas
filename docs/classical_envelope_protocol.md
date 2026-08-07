# Nonlinear classical-envelope protocol

This protocol is fixed before inspecting the new model comparisons.

## Research question

Does a richer content-free model of graph exposure explain held-out LLM target adoption better than
the existing linear calibration of final DeGroot exposure?

The analysis is predictive, not causal. No direction of improvement is assumed.

## Frozen observations

- Completed 100-task homogeneous-agent GSM8K pilot.
- Every selected graph and every valid non-readout attack node.
- Exact graph-independent Round-zero categorical answers.
- Existing binary outcome: the attack newly induces the frozen target error at readout.
- No natural-language message, rationale, token, embedding, confidence, or LLM judge feature.

## Content-free features

For every task--graph--attack-node condition, equal-weight DeGroot dynamics are run from the exact
Round-zero categorical states with the attacker clamped to the target. The following are retained:

- readout target mass at every round, including Round zero;
- mean target mass among non-attacker nodes at every round;
- mean and peak exposure across the finite trajectory;
- the eight previously frozen Round-zero state summaries;
- the previously exported static graph and attacked-node features.

These features can be computed without a language model. They expand the classical baseline beyond
one final exposure scalar while preserving the same information restriction.

## Frozen candidate models

1. `intercept_only`: training-fold prevalence.
2. `final_exposure_linear`: standardized final exposure with logistic regression.
3. `final_exposure_spline`: five-knot cubic spline of final exposure with logistic regression.
4. `classical_trajectory_linear`: all declared classical features with L2 logistic regression.
5. `classical_trajectory_hgb`: fixed histogram gradient boosting with learning rate 0.05,
   200 iterations, 15 leaf nodes, minimum leaf size 100, and L2 regularization 1.0.

Hyperparameters are fixed here rather than selected from the test folds. The flexible models are
included to test whether the previously observed residual is merely functional-form misspecification.

## Validation and metrics

Use the same strict 5-by-5 crossed folds as the previous analysis. A test row is predicted only by a
model trained on different graphs and different tasks. Primary metric is Brier score. Log loss and
average precision are secondary. Graph--attack-node aggregate ranking metrics are diagnostic.

Compare candidates with `final_exposure_linear` using a crossed graph-by-task bootstrap with 2,000
replicates. Positive Brier improvement means the candidate has lower error.

## Integrity gates

- Exact 20,400-condition coverage and no duplicate condition keys.
- No graph or task overlap in any crossed fold.
- Static features contain no outcome columns after merging.
- Final trajectory exposure equals the independently computed existing final exposure.
- All probabilities are finite and strictly inside zero and one after clipping.

## Claim boundary

- Improvement supports a stronger content-free predictive baseline, not DeGroot mechanistic
  equivalence.
- Failure to improve does not identify a semantic mechanism.
- Residual error may still arise from unmodeled non-textual dynamics, finite samples, stochastic
  generation, or projection of free text to categorical answers.
- This analysis cannot replace a matched intervention on message content; it determines how strong
  that intervention's classical control must be.
