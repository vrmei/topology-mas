# Classical structural explainability protocol

This protocol fixes the first mechanism-oriented analysis before inspecting its results.

## Research question

How much of node-level targeted-attack vulnerability can be predicted out of graph from static
directed structure alone?

The primary outcome is node-level paired accuracy drop. The secondary outcome is the rate at which
the fixed target error is induced at the readout. No directional result is assumed in advance.

## Unit and leakage boundary

- One row is one non-readout attack position in one selected graph.
- Outcomes aggregate the same 100 paired tasks for that graph and position.
- Node labels are excluded because they are arbitrary structural identifiers.
- Features are computed only from the graph and the fixed communication horizon.
- Outer validation holds out an entire graph. No node from the held-out graph may enter training,
  scaling, feature tuning, or ridge-penalty selection.

The current pilot has 36 selected graphs and 204 node-level rows. Results are conditional on one
model, prompt, assignment seed, experiment seed, and task sample.

## Static features

### Local position

- shortest directed distance to readout;
- direct-to-readout indicator;
- in-degree and out-degree;
- source indicator;
- number of descendants and ancestors.

### Routing and bottlenecks

- number of simple paths and shortest paths to readout;
- number of directed walks reaching readout within the fixed horizon `T`;
- local edge and node connectivity to readout;
- upstream post-dominator count: the number of other nodes that lose all paths to readout when the
  candidate node is removed.

### Cycles, centrality, and graph context

- directed betweenness and outward closeness;
- strongly connected component size and nontrivial-cycle indicator;
- readout in-degree, source count, maximum SCC size, cycle indicator;
- mean and standard deviation of node-to-readout distance;
- active message opportunities recorded by the sampler.

## Predictive baselines

1. `global_mean`: training-set mean.
2. `stratum_mean`: training mean for the same `(n,m)` stratum, with an `n`-level then global
   fallback when a stratum contains one unique graph.
3. `distance_lookup`: training mean for the same stratum and distance, with the same fallbacks.
4. `ridge_distance`: shortest-distance features plus stratum controls.
5. `ridge_local`: local-position features plus stratum controls.
6. `ridge_routing`: routing/bottleneck features plus stratum controls.
7. `ridge_full`: all declared static features plus stratum controls.

Ridge penalties are selected inside each outer training fold using grouped inner cross-validation.

## Evaluation

- leave-one-graph-out MAE and RMSE;
- out-of-graph R-squared relative to the observed target mean;
- pooled Spearman rank correlation;
- mean within-graph Spearman correlation;
- top-1 vulnerable-node identification. If a model ties several nodes, its score is the probability
  that a uniformly selected member of the predicted tie set belongs to the observed worst-node set;
  a constant prediction therefore receives chance-level rather than perfect credit.

Confidence intervals resample held-out graphs. MAE, RMSE, and R-squared use graph-frequency weights;
rank intervals use a graph-cluster bootstrap over fixed rank transforms. They quantify variation
across the selected graph set, not task, seed, model, or graph-population uncertainty.

The primary model comparison is paired by held-out graph. For each graph, node-level absolute
errors are averaged first; the reported improvement is reference MAE minus `ridge_full` MAE.
Positive values therefore favor the full structural model. Its interval resamples whole graphs and
preserves the pairing between models.

## Claim boundary

- Better out-of-graph prediction supports structural explainability in this pilot; it does not prove
  a causal graph law.
- Weak prediction does not prove an LLM-specific semantic mechanism; it only leaves structured
  residual variation for subsequent analysis.
- In-sample correlations are diagnostic and cannot replace grouped out-of-graph evaluation.
- The unique complete `n=5, m=16` graph cannot support within-stratum topology generalization.
