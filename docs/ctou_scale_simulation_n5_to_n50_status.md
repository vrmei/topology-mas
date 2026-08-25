# CTOU n=5–50 simulation status

## Claim boundary

This study is a model-based CTOU simulation. Llama-3.1-8B traces exist only at
`n=5,6,7,8,10`; values at `n=11..50` are extrapolations conditional on
GSM8K-50, horizon `H=3`, the current synchronous protocol, and one persistent
non-readout attacker. The output must not be described as measured LLM
performance at `n>10`.

## Frozen validation results

The local response law was selected using `n=6,7,8`, with `n=10` reserved as a
stress check. The selected law uses CTOU proportions plus a bounded evidence
volume `d/(d+2)` and its interactions. Its selection log loss was `0.30566`,
compared with `0.31273` for proportions alone and `0.31976` for raw counts. At
`n=10`, the corresponding log losses were `0.28340`, `0.29036`, and `0.32557`.

The selection is not unique: `d/(d+1)`, `d/(d+4)`, and `log(1+d)` were nearly
indistinguishable at the task level. They therefore define a later model
envelope; the `d/(d+2)` law is only the frozen primary specification.

The hierarchical Round-0 initializer passed its predefined gate on
`n=6,7,8,10`:

- maximum correct-rate bias: `0.00141`;
- mean variance-error ratio versus global IID: `0.00899`;
- mean Wasserstein-distance ratio versus global IID: `0.02332`.

At `n=10`, the observed all-correct rate was `0.4944`; the hierarchical model
predicted `0.4964`, whereas the global IID model predicted `0.1209`. The main
dependence captured here is task-specific difficulty shared by nodes.

Recursive validation on real anchor graphs also passed. Relative to the
proportions law, the selected law's mean Utility/Robustness graph-MAE ratio was
`0.9927`; its minimum graph-ranking Spearman across anchor cells was `0.4514`.
This is a safeguard, not evidence that every individual anchor metric improves.

## Frozen Phase-2 design

- sizes: every integer `n=5..50`;
- density axis: 20 requested normalized excess-density levels, deduplicated
  after integer edge rounding;
- graph samples: 10 per non-complete `(n,m)` cell and one complete graph;
- graph constraints: no self-loop, no readout outgoing edge, exact `m`, and
  every node reaches readout within three rounds;
- attack evaluation: every non-readout attacker position;
- outputs: Utility, Robustness, target risk, attack penalty, Round-0 Utility,
  and communication gain;
- versions: strict `n=5` fitting and calibrated `n=5,6,7,8,10` fitting.

Primary propagation uses expected-composition mean field. A separate 2,048
particle check was completed at `n={5,10,15,20,30,40,50}` and
`delta={0,0.5,1}`.

## Completed outputs

The full mean-field scan completed for every integer `n=5..50`. The final
curve table contains 1,826 `(version,n,m)` rows and passed the coverage,
probability-range, duplicate, and graph-count audit. Requested density levels
were deduplicated only when integer edge rounding produced the same `m`.

The frozen particle grid contains 2,100 task/cell rows across 42
`(version,n,density)` cells. The grid-level gate passed:

- Utility task-MAE: `0.01313`;
- Robustness task-MAE: `0.01456`;
- maximum Utility aggregate cell error: `0.02021`;
- maximum Robustness aggregate cell error: `0.04385`.

The pass is not uniform across sizes. At `n=5`, Robustness task-MAE was
`0.03574`, above the per-size `0.03` diagnostic threshold. In the calibrated
dense cell, mean field predicted Robustness `0.85089` while particles gave
`0.80704`; target risk was `0.04364` versus `0.08174`. The discrepancy shrank
with size: Robustness task-MAE was `0.01828` at `n=10`, `0.01072` at `n=20`,
and `0.00646` at `n=50`. Thus the mean-field gate passes globally, while the
small-graph density bias must remain visible in any report.

The four-law extrapolation envelope is also complete. It includes
proportions, absolute counts, counts plus proportions, and proportions plus
bounded evidence volume. The envelope fails the qualitative-identification
test beyond the real-data boundary. At `n=50` and complete density:

- strict fits predict Robustness from `0.8400` to `0.8868`;
- calibrated fits predict Utility from `0.8600` to `0.8933`;
- some count-based fits predict negative attack penalties, while the frozen
  bounded-volume model predicts a small positive penalty.

From `n=10` to `n=50`, admissible parameterizations predict utility increases,
decreases, or near-constant behavior depending on density. Consequently, the
current traces do not identify a unique `n>10` scaling law. The generated
surface is retained as a conditional model diagnostic, not evidence for
monotonic growth, saturation, or a Utility–Robustness trade-off in real LLM
systems.

Detailed results and the decision boundary are recorded in
`docs/ctou_scale_simulation_n5_to_n50_results.md`.
