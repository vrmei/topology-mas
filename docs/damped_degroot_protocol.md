# Damped DeGroot calibration protocol

This protocol is fixed before inspecting the calibrated model's results.

## Research question

Can one graph- and task-independent susceptibility parameter calibrate equal-weight DeGroot to the
targeted-error propagation observed in the homogeneous LLM pilot?

The test does not assume that damping will improve held-out prediction.

## Model

Let `P(t)` be the matrix of categorical answer-belief vectors and `W` the fixed equal-weight
DeGroot matrix over self plus directed in-neighbors. A non-attacker updates as:

```text
P(t + 1) = (1 - alpha) P(t) + alpha W P(t)
```

The attacker's row is replaced by the one-hot target-error vector after every update. `alpha=0`
is frozen Round zero and `alpha=1` is the previously evaluated equal-weight DeGroot baseline.
Final ties preserve the preceding discrete state. All other state projection and graph semantics
follow `classical_dynamics_protocol.md`.

## Parameter grid and selection

- Candidate grid: `alpha in {0.00, 0.02, ..., 1.00}`.
- Outer validation: leave one entire graph out.
- Selection loss: for each candidate, first compute target-induction MAE across attack nodes within
  each training graph, then average those graph-level MAEs equally.
- Tie handling: choose the smallest alpha among equal training losses.
- The selected alpha is applied unchanged to every task and attack position in the held-out graph.

No node from the held-out graph may influence alpha selection. There is no per-stratum, per-task,
per-node, or per-round fitting.

## Outcomes

Primary:

- held-out-graph prediction of node-level induced target-error rate.

Secondary, using the same alpha without retuning:

- paired clean-minus-attacked accuracy drop;
- clean utility;
- vulnerable-node ranking and top-1 identification;
- distribution and stability of selected alpha across outer folds.

## Comparisons and uncertainty

The calibrated model is paired against both endpoints, frozen Round zero and equal-weight DeGroot.
Positive graph-equal MAE improvement means the calibrated model is better. Confidence intervals
resample complete held-out graphs and preserve model pairing.

## Interpretation boundary

- Better calibration would show that global susceptibility explains part of the discrepancy; it
  would not show that the LLM implements DeGroot.
- Failure to calibrate would reject this one-parameter reduction, not classical graph dynamics in
  general.
- Stability across held-out graphs does not establish stability across tasks, seeds, datasets, or
  model families.
- Any claim about natural-language semantics still requires matched message-content interventions.
