# Round-zero-conditioned classical exposure protocol

This protocol is fixed before inspecting model-comparison results.

## Research question

Can target-error adoption in the homogeneous LLM pilot be predicted from categorical Round-zero
states and classical graph exposure, without using natural-language content?

The test separates three sources of predictive information:

1. the base adoption rate;
2. the initial categorical state configuration;
3. the continuous target mass delivered by equal-weight DeGroot dynamics.

## Unit and outcome

The prediction unit is one task–graph–attacked-node condition. The binary outcome is whether the
recorded LLM readout newly adopts the cached target error relative to its clean run.

## Classical exposure

Each distinct parsed Round-zero answer is represented by a one-hot belief. Non-attacking nodes use
the equal-weight self-plus-in-neighbor DeGroot matrix. The malicious node is clamped to the cached
target answer after every synchronous update. After the recorded graph horizon, the continuous
target-answer mass at the readout is retained:

\[
e_{q,G,a}=P^{(T)}_{r,z^{adv}}\in[0,1].
\]

There is no final `argmax` in this exposure variable.

## State-only features

The state-only model receives no adjacency, distance, degree, centrality, graph identifier, task
identifier, or natural-language text. Its fixed features are:

- readout initially correct;
- readout initially equal to the target error;
- attacked node initially correct;
- attacked node initially equal to the target error;
- correct-answer fraction among non-attacking nodes;
- target-error fraction among non-attacking nodes;
- normalized number of distinct parsed answers among non-attacking nodes;
- largest parsed-answer consensus fraction among non-attacking nodes.

## Models

- `intercept_only`: training-fold adoption prevalence.
- `round0_state`: ridge logistic regression on the eight state-only features.
- `classical_exposure`: ridge logistic regression on continuous DeGroot target exposure only.
- `state_plus_exposure`: ridge logistic regression on all nine variables.

Continuous inputs are standardized using training data only. Logistic regularization is fixed at
`C=1`; it is not tuned on the pilot.

## Validation

Graphs are assigned to five deterministic folds, stratified by `(n,m)` stratum. Tasks are assigned
independently to five deterministic folds. For each of the 25 graph-fold × task-fold cells:

- test rows are the intersection of the held-out graph fold and held-out task fold;
- training rows exclude every graph in the held-out graph fold and every task in the held-out task
  fold.

Thus neither a test graph nor a test task may appear in its training set. Every condition receives
one out-of-sample prediction from each model.

## Metrics

Primary:

- task-condition Brier score.

Secondary:

- log loss;
- average precision, reported with the positive prevalence;
- calibration slope and intercept where estimable;
- graph–attack-node aggregate MAE, R-squared, Spearman correlation, and vulnerable-node ranking.

Paired Brier improvements are reported against `intercept_only` and, for the combined model,
against `classical_exposure`. Confidence intervals use crossed graph-by-task bootstrap weights.

## Interpretation boundary

- Improvement by `classical_exposure` shows predictive information compatible with a classical
  diffusion quantity; it does not show that the LLM internally runs DeGroot.
- Improvement by `round0_state` shows that categorical initial conditions matter; it does not
  identify a semantic mechanism.
- Incremental improvement by `state_plus_exposure` shows that exposure is not a sufficient summary
  of the selected categorical state variables.
- Failure of all three models rejects these specified reductions only. It does not by itself prove
  that natural-language semantics causes the residual.
- This pilot does not establish transfer to another model, dataset, seed, or graph population.
