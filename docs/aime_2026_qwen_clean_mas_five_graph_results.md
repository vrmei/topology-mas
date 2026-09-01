# 2026 AIME Qwen clean-MAS five-graph analysis

## Scope

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Tasks: all 30 original 2026 AIME I/II problems.
- System: homogeneous `n=5`, readout included, fixed `H=3`.
- Public communication: bounded solution summary; private reasoning is not broadcast.
- Graphs: five rooted-nonisomorphic graphs at each of `m=4,8,12`; the unique
  complete graph at `m=16`.
- Runs: 480 clean task-graph runs. Round 0 is independently regenerated in every
  run and no generation output is reused across graphs or batches.
- Uncertainty: density-level intervals independently resample the task and graph
  axes. The `m=16` point has task-only uncertainty because only one complete graph
  exists.

## Main results

| Edges | Graphs | Round-0 utility | Final utility | Paired gain | Final utility graph SD |
|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 52.0% | 70.0% | +18.0 pp | 3.65 pp |
| 8 | 5 | 56.7% | 70.7% | +14.0 pp | 2.49 pp |
| 12 | 5 | 54.7% | 68.0% | +13.3 pp | 5.42 pp |
| 16 | 1 | 46.7% | 66.7% | +20.0 pp | not identifiable |

Across all 16 graphs, bounded-protocol Round-0 utility is 53.96% and final
utility is 69.38%, for a paired gain of 15.42 percentage points. The task
bootstrap 95% interval is `[9.58, 21.67]` pp. All 16 individual graphs have a
positive observed paired gain; graph-level gains range from +6.67 to +26.67 pp.

The earlier full-rationale single-agent Round-0 accuracy of 51.33% is an external
reference only. It is not used as the paired denominator because the prompt and
output protocol differ.

## What produces the utility gain?

Across all task-graph runs:

- `P(C_final | C_round0) = 95.37%`;
- `P(C_final | O_round0) = 37.57%`;
- `P(C_final | U_round0) = 43.75%`.

Thus the observed net gain combines high preservation of initially correct
readout answers with substantial correction of parsed and unparsed failures. It
is not simply an artifact of discarding initially wrong samples.

For `m=4,8,12`, parsed-error correction decreases descriptively from 43.86% to
36.54% to 30.77%, while correct preservation changes from 94.87% to 95.29% to
96.34%. However, every adjacent hierarchical-bootstrap interval for these
conditional-rate differences contains zero. The mechanism pattern is therefore
suggestive, not established.

## Density and graph-arrangement evidence

For the 15 sampled random graphs at `m=4,8,12`, the bootstrap slope of final
utility is `-0.25 pp` per added edge, with a 95% interval from `-1.42` to `+0.83`
pp per edge. The paired-gain slope is `-0.58 pp` per edge, with a 95% interval
from `-2.25` to `+0.92` pp per edge.

Task-fixed permutation diagnostics give:

| Outcome | Added structure | Partial R² | Permutation p |
|---|---|---:|---:|
| Round-0 correctness | density beyond task | 0.34% | 0.494 |
| Final correctness | density beyond task | 0.24% | 0.638 |
| Paired gain | density beyond task | 0.29% | 0.563 |
| Final correctness | graph arrangement beyond task and density | 3.01% | 0.395 |
| Paired gain | graph arrangement beyond task and density | 1.44% | 0.918 |

These data do not support a monotonic clean-utility benefit from increasing
density, nor a stable graph-arrangement effect after controlling task identity.
They also do not prove that topology has no effect: there are only five graphs
per non-complete density, 30 tasks, and one stochastic realization per
task-graph cell.

## Difficulty stratification

The difficulty bands were frozen using the earlier independent ten-replicate
Round-0 experiment.

| Band | Tasks | Round-0 utility | Final utility | Paired gain |
|---|---:|---:|---:|---:|
| Floor | 9 | 6.25% | 13.89% | +7.64 pp |
| Intermediate | 12 | 63.54% | 88.54% | +25.00 pp |
| Ceiling | 9 | 88.89% | 99.31% | +10.42 pp |

The intermediate-band gain exceeds the floor-band gain by 17.36 pp with a
task-bootstrap interval of `[2.78, 30.56]` pp, and exceeds the ceiling-band gain
by 14.58 pp with interval `[1.91, 26.74]` pp. This is meaningful exploratory
evidence that communication benefit concentrates at intermediate task
difficulty. It is not yet a cross-model or cross-dataset claim.

## Batch-compatibility audit

The two newly sampled graphs per density were run in a later batch. Relative to
the original batch, their final utility differs by 0.0 pp at `m=4`, +4.44 pp at
`m=8`, and +8.89 pp at `m=12`. All hierarchical-bootstrap intervals include
zero. Parsing diagnostics are also similar: summary-answer mismatch remains
below 1.5%, private length termination is about 1.9%--3.2%, and unparsed turns
are about 2.7%--5.0%. No batch incompatibility is detected, although the small
number of graphs makes this audit low-powered.

## Claim calibration

Supported under this exact pilot setting:

1. Three-round bounded-message clean communication improves readout utility on
   the 30 original 2026 AIME tasks for homogeneous Qwen3-4B agents.
2. The gain is primarily compatible with error correction plus high correct
   preservation.
3. The gain is larger in the externally defined intermediate-difficulty band
   than in the floor and ceiling bands.

Not supported by this experiment:

1. Increasing edge density monotonically improves or harms clean utility.
2. A particular random graph arrangement is reliably superior.
3. The complete graph is worse than sparse graphs; it is a single topology with
   wide task uncertainty.
4. The result generalizes to attacks, other models, other node counts, or other
   reasoning datasets.

## Analysis artifacts

All public analysis outputs are under
[`docs/assets/aime_2026_clean_mas_five_graph`](assets/aime_2026_clean_mas_five_graph/):

- `edge_level_metrics.csv`: density-level estimates and crossed bootstrap CIs;
- `graph_metrics.csv`: all 16 graph-level outcomes;
- `density_difficulty_metrics.csv`: density by difficulty-band decomposition;
- `cohort_comparisons.csv`: original-versus-extension audit;
- `conditional_edge_level_differences.csv`: transition-mechanism comparisons;
- `summary.json`: machine-readable results, slope bootstraps, and permutation
  diagnostics;
- `density_utility_transitions.png` and `density_difficulty_gain.png`: figures.
