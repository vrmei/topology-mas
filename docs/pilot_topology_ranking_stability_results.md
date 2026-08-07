# Pilot topology-ranking stability results

This analysis was frozen in `docs/topology_ranking_stability_protocol.md` before inspecting the
results. It uses the completed 100-task GSM8K pilot, 36 graphs, and 20,400 paired attack conditions.

## Main result

The current 100 tasks do not support a precise total ordering of graphs within a fixed `(n,m)`
stratum. Among 140 pairwise graph comparisons, only 17 had a 95% task-bootstrap interval excluding
zero. The remaining 123 graph orderings were unresolved.

Split-half rank stability depended strongly on the outcome:

- clean utility was unstable in several strata, including negative split-half correlations for
  `n8_m14`, `n8_m21`, and `n8_m28`;
- mean attack accuracy had mostly weak-to-moderate stability;
- worst-node attack accuracy was generally more stable, with mean split-half Spearman values from
  0.424 to 0.845 across the seven rankable strata;
- vulnerable-node induced-target rankings had mean Spearman 0.483 and mean top-node overlap 0.523.

Negative split-half correlations are not interpreted as evidence that topology value reverses by
task. Many graph utilities differ by only a few successes in 50 tasks, so tied and near-tied rates
can create unstable ranks.

## Supported claim

At the present pilot size, graph and vulnerable-node rankings contain substantial task-sampling
uncertainty. Worst-node performance appears more reproducible than clean-utility ranking, but this
is a pilot observation rather than a cross-dataset result.

## Not supported

This analysis does not establish that:

- topology value is inherently task-specific;
- no stable topology ordering exists with more tasks;
- GSM8K ranking stability transfers to MATH, code, another model, or another prompt;
- a graph selected from the current 100-task ranking is genuinely optimal.

The practical implication is to retain graph-level uncertainty and avoid selecting later
experiments solely from point-estimate ranks.
