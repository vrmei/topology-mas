# Pilot result: Round-zero-conditioned classical exposure

## Question

Can target-error adoption in the homogeneous LLM pilot be predicted without natural-language
content, using only categorical Round-zero states and continuous equal-weight DeGroot exposure?

The analysis compares an intercept-only predictor, a Round-zero-state model, a classical-exposure
model, and their combination. Every prediction is made under a strict 5×5 crossed split: neither
the test graph nor the test task appears in the corresponding training set.

## Integrity

- 20,400 task–graph–attack-node conditions.
- 36 graphs and 100 tasks.
- 1,056 positive target-adoption events; prevalence 0.0518.
- 81,600 out-of-sample predictions across four models.
- 25 crossed graph-fold × task-fold cells.
- Maximum train/test graph overlap: zero.
- Maximum train/test task overlap: zero.
- Continuous exposure range: [0.03125, 0.875].
- Formal uncertainty uses 2,000 crossed graph-by-task bootstrap replicates.

## Main results

| model | Brier | log loss | average precision |
|---|---:|---:|---:|
| Intercept only | 0.04925 | 0.20540 | 0.0419 |
| Round-zero state | 0.04904 | 0.20292 | 0.0642 |
| Classical exposure | 0.04718 | 0.18862 | 0.1221 |
| State + exposure | 0.04698 | 0.18737 | 0.1347 |

Classical exposure reduces Brier score by 0.002074 relative to the intercept-only predictor, a 4.2%
relative reduction. Its 95% crossed-bootstrap interval is [0.000892, 0.003696]. Average precision
is 0.1221 against a positive prevalence of 0.0518.

Round-zero state alone improves Brier score by 0.000213, with interval
[-0.000096, 0.000505]. The interval includes zero. Adding the eight state variables to exposure
improves Brier score by another 0.000191, with interval [-0.000159, 0.000558]. The current pilot
therefore provides no clear evidence that these selected state summaries add predictive information
beyond continuous exposure.

## Aggregate structural prediction

After averaging over the 100 tasks for each graph–attack-node pair, classical exposure achieves:

- MAE: 0.01538;
- \(R^2=0.827\);
- global Spearman: 0.835;
- mean within-graph Spearman: 0.737;
- top vulnerable-node accuracy: 0.653.

The combined model does not improve aggregate fit: its \(R^2\) is 0.825 and MAE is 0.01537.

The unfitted exposure diagnostic also shows a broad monotonic gradient. Across exposure quantiles,
observed target adoption rises from 1.1% in the lowest group to 17.0% in the highest group. The
standardized exposure coefficient is positive in all 25 crossed folds, ranging from 0.650 to 0.805.

## Interpretation

The previous hard-state DeGroot test and this continuous-exposure test answer different questions.
Hard `argmax` DeGroot required the target error to become the largest categorical belief and therefore
overpredicted at high susceptibility or became almost frozen at lower susceptibility. Continuous
exposure preserves sub-threshold target mass. A simple probabilistic calibration then recovers a
strong graph–node ranking and a modest but reliable task-condition improvement.

The supported claim is:

> Continuous classical exposure contains substantial information about where target errors are more
> likely to be adopted, especially after averaging over tasks.

The result weakens any claim that the current pilot already demonstrates dynamics irreducible to a
classical graph quantity. It does not show that the LLM implements DeGroot: the logistic adoption
mapping is learned from LLM observations, and task-level predictive improvement remains limited.

The residual cannot yet be labeled a semantic effect. Other unmodeled non-textual mechanisms may
include nonlinear exposure response, round-specific susceptibility, correlated node errors, or
state variables not represented by the eight summaries.

## Consequence for the research direction

Static graph features and continuous DeGroot exposure should remain mandatory baselines. A future
LLM-specific claim must demonstrate incremental explanatory or causal value beyond this exposure
baseline. The next discriminating experiment should hold task, graph, attacked position, target
answer, and classical exposure fixed while changing the malicious message's reasoning or
presentation. Without that matched intervention, a residual analysis alone cannot identify
language semantics as the cause.
