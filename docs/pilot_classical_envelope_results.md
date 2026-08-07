# Pilot result: nonlinear classical envelope

## Question

Does a richer content-free model of graph exposure explain held-out LLM target adoption better than
the existing linear calibration of final equal-weight DeGroot exposure?

The candidate models were frozen before the new comparisons were inspected.

## Integrity

- 20,400 task--graph--attack-node conditions.
- 36 graphs and 100 tasks.
- 1,056 positive induced-target events.
- 44 content-free state, trajectory, and static graph features.
- Strict 5-by-5 graph--task crossed holdout with zero graph or task overlap.
- 2,000 crossed graph-by-task bootstrap replicates.

## Task-condition prediction

| model | Brier | log loss | average precision |
|---|---:|---:|---:|
| Intercept only | 0.04925 | 0.20540 | 0.0419 |
| Final exposure, linear | 0.04718 | 0.18862 | 0.1221 |
| Final exposure, spline | 0.04712 | 0.18830 | 0.1207 |
| Classical trajectory, linear | 0.04724 | 0.18948 | 0.1323 |
| Classical trajectory, HGB | 0.04722 | 0.18873 | 0.1249 |

Relative to linear final exposure, the spline improvement in Brier score was 0.000055 with a 95%
crossed-bootstrap interval of [-0.000086, 0.000205]. The trajectory-linear difference was -0.000064
[-0.000479, 0.000371], and the trajectory-HGB difference was -0.000049
[-0.000513, 0.000399]. None provides clear held-out Brier improvement.

## Aggregate graph--attack-node prediction

| model | MAE | R2 | global Spearman | within-graph Spearman | top-1 |
|---|---:|---:|---:|---:|---:|
| Final exposure, linear | 0.01538 | 0.827 | 0.835 | 0.737 | 0.653 |
| Final exposure, spline | 0.01432 | 0.852 | 0.831 | 0.737 | 0.653 |
| Classical trajectory, linear | 0.01719 | 0.778 | 0.779 | 0.705 | 0.583 |
| Classical trajectory, HGB | 0.01514 | 0.834 | 0.818 | 0.659 | 0.583 |

The spline improves aggregate calibration and R2 modestly, but it does not improve vulnerable-node
ranking or condition-level Brier score with a decisive interval. Adding all trajectory and graph
features does not improve held-out prediction.

## Supported interpretation

Within the frozen candidate set, the residual left by final DeGroot exposure is not explained by a
more flexible final-exposure mapping or by adding the complete declared content-free trajectory and
static graph feature set. The simple final-exposure model therefore remains the strongest compact
classical baseline for the next analysis.

This is not evidence that the residual is semantic. The next CPU analysis moves to node-round paired
traces and tests whether categorical exposure predicts the adoption transition itself. Only a later
matched message intervention can attribute an effect to rationale content.
