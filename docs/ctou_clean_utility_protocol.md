# CTOU clean utility and unified-law protocol

## Research questions

1. Can the same four-state local transition framework recursively describe benign information aggregation and predict clean utility?
2. Do clean and attack traces require condition-specific local response laws after conditioning on previous state, round, and incoming C/T/O/U composition?
3. Within the current Llama/GSM8K regime, how do clean utility and attack robustness vary with density, and is there evidence of alignment, trade-off, or non-monotonic optima?

## Pre-specified hypotheses and falsification

### H1: CTOU generalizes to clean aggregation

The clean-specific CTOU law should improve clean endpoint calibration and graph/density prediction over persistence and equal-weight DeGroot. Failure to improve means the previous attack result may depend on the regular persistent-target protocol.

### H2: A condition-invariant local law may be sufficient

After controlling the CTOU input cell, a condition-balanced pooled law should approach condition-specific laws on both clean and attack transitions/endpoints. Material degradation on shared-support cells supports condition-dependent local response not represented by CTOU.

This experiment can reject a unified law in the present regime; it cannot prove that clean and attack cognition are intrinsically different.

### H3: Utility and robustness curves need not coincide

Estimate both curves without assuming their direction. Report aligned improvement, trade-off, or non-monotonicity only if supported by observed values and task-bootstrap uncertainty.

## Data and deduplication

Use the completed Llama-3.1-8B, GSM8K 50-task dense pilot.

- Each `(task, graph)` has exactly one clean trace even though it is paired with multiple attack positions.
- Read every clean trace once; never replicate clean updates by attacker count.
- Attack updates and endpoints retain the existing definitions and all non-readout attack positions.
- State labels use execution-time numeric Oracle `answer_state` values.

## Crossed holdout

Retain the strict five graph-fold by five task-fold protocol. For test cell `(g,t)`, every transition law and Round-0 prior uses only rows whose graph fold is not `g` and task fold is not `t`.

Report and enforce zero graph/task overlap.

## Local-law experiment

Construct clean and attack transition rows with the same features:

\[
X=(S_{t-1},t,C,T,O,U).
\]

Fit three multinomial-logistic laws in every training fold:

1. clean-only;
2. attack-only;
3. pooled clean+attack with condition-balanced sample weights.

Evaluate every law on both test conditions. Primary local-law comparison uses exact cells that have nonzero training support in both conditions; all-row results are secondary. This separates response-law mismatch from composition support mismatch.

Metrics are four-state, correct, and target Brier/log loss with paired task bootstrap. Compare domain-specific versus pooled and cross-domain laws.

## Recursive endpoint experiment

### Clean

No node is clamped. Roll out:

- clean-specific law;
- balanced pooled law;
- persistence and DeGroot baselines where useful.

Use two initialization sources:

- oracle clean Round-0 vector;
- training-only correlated empirical clean vector, randomly remapped to test nodes.

### Attack

Retain the persistent `T` attacker and original schedule. Compare the balanced pooled law against the saved attack-specific CTOU results under matched oracle and correlated empirical initialization.

The pre-specified unified surrogate is **balanced pooled law + correlated empirical initialization**. Condition-specific laws are domain comparators, not candidates selected after observing results.

## Utility and robustness outputs

For each graph:

\[
U_0(G)=P(S_r^0=C\mid\text{clean}),
\]

\[
U_T(G)=P(S_r^T=C\mid\text{clean}),
\]

\[
R(G)=P(S_r^T=C\mid\text{attack}),
\]

where `R` retains the current attack-accuracy definition. Also report:

\[
\Delta U(G)=U_T(G)-U_0(G).
\]

Primary analyses:

- fixed-`n` `m -> U` and `m -> R` curves;
- graph-level utility/robustness scatter;
- graph and density-curve MAE/Spearman for observed versus predicted values;
- task-bootstrap intervals for curve points and utility changes;
- observed and predicted Pareto sets, reported descriptively because only five graphs are sampled per density.

## Claim limits

- The pooled law is condition-invariant only with respect to the recorded clean/attack label; unobserved semantic differences may remain.
- Similar endpoint performance does not prove identical transition probabilities.
- Utility/robustness conclusions remain conditional on one model, GSM8K, the sampled graph family, horizon, and attack protocol.
- With 50 tasks and five graphs per density, subtle within-density topology ordering remains noisy.
- The correlated prior is distribution-conditioned and not a task-conditioned predictor.
