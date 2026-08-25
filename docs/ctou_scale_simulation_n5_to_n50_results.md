# CTOU n=5–50 model-based simulation: results

## Bottom line

The computational study is complete, but it does **not** support a unique
LLM scaling claim beyond `n=10`.

Three different questions must remain separate:

1. **Can the fitted local law predict held-out real endpoints?** Yes, at the
   available real sizes `n=5,6,7,8,10`, under the frozen anchor gate.
2. **Can expected-composition mean field approximate a discrete rollout of
   that same fitted law?** Yes on the full frozen particle grid, with a
   material small-graph caveat.
3. **Is the fitted law itself identified well enough to extrapolate to
   `n=50`?** No. Plausible CTOU parameterizations produce qualitatively
   different curves beyond the observed scale range.

Passing question 2 does not answer question 3. The particle validation checks
the numerical rollout approximation conditional on a chosen local law; it
cannot validate that law against unobserved LLM behavior.

## What was run

- sizes: every integer `n=5..50`;
- normalized excess-density requests: 20 levels from sparse to complete;
- topology samples: 10 per non-complete cell and one complete graph;
- horizon: three synchronous rounds;
- attack: each non-readout position in the primary mean-field scan;
- task model: correlated hierarchical Round-0 CTOU initialization;
- local laws: strict `n=5` and calibrated `n=5,6,7,8,10` fits;
- primary law: CTOU proportions plus `d/(d+2)` bounded evidence volume;
- validation: 2,048-particle rollouts at seven sizes and three densities;
- uncertainty envelope: proportions, counts, counts plus proportions, and
  bounded volume.

The main table contains 1,826 audited `(version,n,m)` rows. Values above
`n=10` are simulations from an extrapolated transition model, not new LLM
measurements.

## Validation evidence

### Local transition and Round-0 initialization

The frozen bounded-volume law obtained selection log loss `0.30566`, compared
with `0.31273` for proportions and `0.31976` for raw counts. Its `n=10` stress
log loss was `0.28340`. The hierarchical Round-0 model reproduced task-level
cross-node dependence: at `n=10`, the observed all-correct probability was
`0.4944`, the hierarchical prediction was `0.4964`, and global IID predicted
only `0.1209`.

These results support the chosen law within the observed range, but the
near-equivalence of several saturating-volume specifications already warns
that the response outside that range is weakly identified.

### Mean field versus discrete particles

Across all 42 validation cells, the frozen particle gate passed:

| Metric | Frozen result | Threshold |
|---|---:|---:|
| Utility task-MAE | 0.01313 | 0.03 |
| Robustness task-MAE | 0.01456 | 0.03 |
| Max Utility cell bias | 0.02021 | 0.05 |
| Max Robustness cell bias | 0.04385 | 0.05 |

Mean field was systematically optimistic on average: `+0.00649` for Utility
and `+0.00825` for Robustness. The strongest discrepancy occurred at `n=5`.
For the calibrated complete graph, mean-field versus particle estimates were:

| Metric | Mean field | Particle |
|---|---:|---:|
| Utility | 0.88282 | 0.86727 |
| Robustness | 0.85089 | 0.80704 |
| Target risk | 0.04364 | 0.08174 |

The task-MAE decreased with size rather than increasing. This pattern is
consistent with discrete composition fluctuations concentrating as the number
of peers grows, but the present experiment does not isolate that mechanism.
The defensible conclusion is narrower: expected composition is a usable
aggregate approximation for this fitted law, while it can materially
understate dense small-graph target propagation.

## Primary surface: descriptive only

The frozen bounded-volume model produces an apparent rise and plateau in both
Utility and Robustness. For the calibrated fit at normalized excess density
`0.5`, the model-based values are:

| n | Utility | Robustness | Attack penalty |
|---:|---:|---:|---:|
| 5 | 0.87256 | 0.83119 | 0.04137 |
| 10 | 0.88497 | 0.87570 | 0.00927 |
| 20 | 0.89058 | 0.88635 | 0.00424 |
| 30 | 0.89141 | 0.88764 | 0.00377 |
| 40 | 0.89237 | 0.88853 | 0.00384 |
| 50 | 0.89282 | 0.88888 | 0.00394 |

This is a property of the frozen simulator. It is not evidence that real LLM
systems plateau at these values, because the local-law envelope below does not
preserve the curve shape.

Graph-to-graph standard deviations within a fixed `(n,m)` cell were usually
small under this simulator: mean Utility SD was about `0.0017–0.0020` and mean
Robustness SD about `0.0019–0.0021`. Maxima reached roughly `0.015` and `0.025`.
These numbers are conditional on the graph proposal and the extrapolated CTOU
law; they do not establish that topology effects are small in a real LLM run.

## Why the scaling claim is rejected

The local-law envelope is qualitatively incompatible beyond `n=10`.

At `n=50`, complete density:

| Fit family | Quantity | Envelope |
|---|---|---:|
| strict `n=5` | Robustness | 0.8400–0.8868 |
| calibrated real sizes | Utility | 0.8600–0.8933 |
| calibrated real sizes | Robustness | 0.8732–0.8895 |

More importantly, between `n=10` and `n=50`:

- the proportions law makes Utility nearly constant;
- the bounded-volume law makes it increase modestly;
- raw-count laws can make dense Utility decrease;
- counts plus proportions can make dense strict Robustness fall sharply;
- some count-based cells produce negative attack penalties.

These are not small confidence-band differences around one shared trend. They
change the sign or qualitative shape of the extrapolation. The stop condition
therefore applies: no monotonicity, saturation, or Utility–Robustness trade-off
claim is made for `n>10`.

## Research decision

The simulation has done its job by locating the uncertainty. More graph
samples or more mean-field runs will not resolve it, because the dominant
uncertainty is the local response function outside the evidence-volume range
observed in real traces.

The next experiment should identify that response function directly. The
controlled receiver experiment already planned is the appropriate test:

1. hold the C/T/O/U ratio fixed while changing absolute peer-message volume;
2. repeat with the receiver's previous answer present and removed;
3. use diverse real trace messages rather than duplicated text;
4. estimate whether volume has an effect after ratio and self-inertia are
   controlled.

If the response saturates, a bounded-volume law gains causal support. If the
effect disappears after removing the previous answer, the apparent volume
effect is better explained as dilution of self-inertia. If absolute counts
remain important without saturation, substantially larger real-system anchors
are needed before any `n=50` prediction is defensible.

## Reproducible outputs

The main local artifact directory is
`artifacts/ctou-scale-simulation-n5-to-n50-v1/`. It contains:

- `simulated_curves/primary_curves.csv`;
- `model_envelope/model_envelope_curves.csv`;
- `particle_validation/n*/particle_predictions.csv.gz`;
- `analysis/` tables and figures;
- `analysis/particle_validation/particle_overall_gate.json`.

The remote server retains the task-level checkpoints and the same finalized
outputs under the corresponding `runs/` directories.
