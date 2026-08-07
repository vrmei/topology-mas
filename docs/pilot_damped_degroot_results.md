# Pilot result: one global damped-DeGroot susceptibility

## Question

Can the gap between frozen Round-zero states and equal-weight DeGroot dynamics be explained by one
global susceptibility parameter?

The tested update is

\[
P^{(t+1)}=(1-\alpha)P^{(t)}+\alpha W P^{(t)},
\]

where \(W\) averages a node's own state and its incoming neighbors. The malicious node remains
clamped to the cached target error. The grid contains 51 values from 0 to 1. Alpha is selected by
leave-one-entire-graph-out validation on target-error induction; no node from the held-out graph is
used for selection. Accuracy drop and clean utility are evaluated without retuning.

## Data and integrity

- 36 held-out graphs across the completed pilot strata.
- 100 tasks per graph.
- 204 graph–attack-node aggregate rows.
- The alpha-zero predictions exactly match the frozen Round-zero baseline.
- The alpha-one predictions exactly match the previous equal-weight DeGroot baseline.
- The formal run uses 2,000 graph-level bootstrap replicates.

## Results

The selected alpha is stable numerically: 35 folds select 0.38 and one fold selects 0.36. This does
not imply that the model explains the observations. The discrete predictions remain on an almost
frozen plateau at those values.

For the primary target-error-induction outcome:

- observed mean: 0.0518;
- damped-DeGroot predicted mean: 0.000049;
- prediction differs from the frozen baseline for only 1 of 204 attack-node rows;
- held-out MAE: 0.0517, versus 0.0518 for frozen Round zero;
- graph-equal MAE improvement over frozen: 0.00004, 95% graph-bootstrap CI [0.00000, 0.00012];
- the damped model is better on only 1 of 36 graphs;
- \(R^2=-1.046\), so it does not calibrate the node-level outcome.

For the untuned accuracy-drop outcome:

- observed mean: 0.0457;
- damped-DeGroot predicted mean: 0.0059;
- held-out MAE: 0.0477, versus 0.0533 for frozen Round zero;
- graph-equal MAE improvement over frozen: 0.00596, 95% CI [0.0037, 0.0084];
- \(R^2=-0.336\), so the magnitude remains poorly calibrated.

For clean utility, damped DeGroot predicts 0.920 against an observed 0.892. Its graph-equal MAE is
0.0331, worse than the frozen baseline's 0.0222.

Equal-weight DeGroot still contains structural ranking information: its target-induction Spearman
correlation is 0.709 and its within-graph correlation is defined on 35 of 36 graphs. It severely
overpredicts magnitude, however. Damping to the selected global alpha removes nearly all target
propagation and therefore also removes the useful ranking signal; its within-graph target
correlation is defined on only one graph.

## Supported interpretation

One global, content-independent susceptibility is insufficient to reproduce the pilot's targeted
error induction. The result is compatible with the following narrower picture:

1. the directed graph contains useful information about which positions are relatively exposed;
2. equal DeGroot exaggerates how often exposure becomes adoption;
3. one global damping parameter cannot calibrate adoption while preserving DeGroot's ranking signal.

The pilot does **not** identify why susceptibility varies. The remaining gap cannot yet be assigned
to semantic persuasiveness, confidence, task difficulty, node history, or another mechanism without
a matched intervention. It also does not establish generality beyond the current model, dataset,
and pilot graph sample.

## Next discriminating test

The next test should preserve the same graphs and recorded runs, but ask whether susceptibility is
conditional rather than global. Candidate conditioning variables must be defined before fitting and
evaluated by whole-graph holdout. A useful order is:

1. Round-zero state configuration only: number and location of correct, target-wrong, and other-wrong
   states.
2. Structural exposure plus state configuration: distance, path multiplicity, and local agreement.
3. Message-content variables only after the non-semantic conditional baseline is exhausted.

This ordering separates variation already implied by categorical initial states from variation that
requires information in the natural-language messages.
