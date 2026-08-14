# Provenance-aware recursive CTOU pilot

## Question

Does separating target messages by lineage improve free-running topology prediction?

The extended node state is

\[
\{C,T_n,T_a,O,U\},
\]

where `T_n` is a naturally produced target answer and `T_a` is attack-descended.
Incoming target messages are separated into direct attacker messages, relayed
attack-descended messages, and natural target messages.

The strict test-time information boundary is:

\[
G + S_0 + \text{attacker position}.
\]

All Round-1+ states and provenance are generated recursively inside each particle.
No observed Round-1+ composition, state, provenance, answer, or text is used.

## Integrity audit

- 50 GSM8K tasks;
- 132 graphs;
- 37,050 graph-task-attacker cases;
- 394,800 normal-node updates;
- 5 x 5 crossed graph/task holdout;
- zero graph overlap and zero task overlap in every fold;
- 2,048 particles per rollout;
- zero duplicate update keys.

The provenance support is highly imbalanced. Direct attacker target exposure occurs in
236,950 updates, relayed attack target exposure in 16,396 updates, and natural target
exposure in 1,657 updates. Thus the experiment can estimate direct exposure much more
precisely than the rarer provenance categories.

## One-step prediction

Loss is computed after collapsing `T_n` and `T_a` back to the original target state.
Negative paired differences favor the provenance model.

| Comparison | Metric | CTOU | Provenance | Paired difference (95% CI) |
|---|---:|---:|---:|---:|
| table | multiclass log loss | 0.379818 | 0.378973 | -0.000845 [-0.001289, -0.000372] |
| table | target log loss | 0.118503 | 0.117361 | -0.001142 [-0.001557, -0.000728] |
| logit | multiclass log loss | 0.386860 | 0.385153 | -0.001707 [-0.002149, -0.001297] |
| logit | target log loss | 0.120125 | 0.118704 | -0.001421 [-0.001828, -0.001054] |

Provenance therefore contains reproducible local transition information, but the
absolute gain is small. This is consistent with the earlier matched-cell result:
relayed target messages are disproportionately influential, but they are much rarer
than direct attacker exposure.

## Free-running endpoint prediction

| Comparison | Metric | CTOU | Provenance | Paired difference (95% CI) |
|---|---:|---:|---:|---:|
| table | multiclass log loss | 0.456480 | 0.451635 | -0.004845 [-0.008669, -0.001746] |
| table | target log loss | 0.212035 | 0.206534 | -0.005501 [-0.009276, -0.002205] |
| logit | multiclass log loss | 0.471446 | 0.466984 | -0.004462 [-0.007289, -0.001927] |
| logit | target log loss | 0.220299 | 0.215681 | -0.004618 [-0.007632, -0.002220] |

The recursive provenance model modestly improves endpoint probability calibration.
This establishes that provenance can be generated prospectively rather than supplied
post hoc. It does not establish that provenance is the main missing state variable.

## Topology curves

The stronger requirement is recovery of the observed edge-count response curve and
graph ranking. Results are mixed.

### Target endpoint

| n | Model | Curve MAE | Spearman | Absolute slope error | Slope sign |
|---:|---|---:|---:|---:|---|
| 5 | CTOU table | 0.006667 | 0.096 | 0.000031 | correct |
| 5 | provenance table | 0.009237 | 0.096 | 0.000479 | wrong |
| 5 | CTOU logit | 0.008903 | 0.152 | 0.000932 | correct |
| 5 | provenance logit | 0.008095 | 0.226 | 0.001084 | correct |
| 8 | CTOU table | 0.010126 | 0.344 | 0.000132 | correct |
| 8 | provenance table | 0.009684 | 0.000 | 0.000166 | wrong |
| 8 | CTOU logit | 0.010543 | 0.480 | 0.000467 | correct |
| 8 | provenance logit | 0.011657 | 0.541 | 0.000349 | correct |

For the main n=8 target curve, the logit model improves rank correlation and slope
error but worsens curve MAE. The table model slightly improves MAE but loses the slope
sign and rank correlation. The result is estimator-dependent rather than a robust
topology-level improvement.

### Correct endpoint

The original CTOU table already predicts the n=8 correct curve well: Spearman 0.903
and slope error 0.000095. Provenance table changes these to 0.907 and 0.000123.
Provenance logit improves n=8 rank correlation from 0.455 to 0.695 and slope error
from 0.000817 to 0.000690, while slightly worsening curve MAE from 0.014070 to
0.014569. Again, the evidence is mixed.

At graph level, no uniform improvement appears. For example, n=8 target graph MAE
changes from 0.011695 to 0.011385 for the table model, but from 0.011911 to 0.012981
for the logit model.

## Interpretation

The supported claim is:

> Target lineage is a real, prospectively simulable local predictor. Adding it gives a
> small but consistent improvement in one-step and endpoint probability calibration.

The unsupported claim is:

> Target lineage explains the current CTOU topology-curve error or is sufficient to
> produce a materially better topology evaluator.

The topology-level gains are small and inconsistent across estimators, outcomes, and
node counts. Provenance should therefore remain a diagnostic extension rather than
replace the original CTOU representation at this stage.

## Next implication

Of the two proposed missing-information candidates, source lineage has now been tested
directly. It matters locally but does not resolve the topology curve. The next most
plausible candidate is dependence between simultaneous node transitions: particle
rollout currently samples receivers conditionally independently given the current
state, while real nodes can co-move because they share upstream messages, evidence,
or latent semantic difficulty.

The next analysis should first measure residual co-movement under matched CTOU and
provenance compositions. A joint model should only be implemented if this residual
dependence is both substantial and topology-dependent.
