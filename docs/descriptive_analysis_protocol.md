# Descriptive pilot analysis protocol

This document fixes the first post-hoc analysis stage before interpreting topology mechanisms.

## Scope

The input is the completed 100-task, eight-stratum Llama-3.1-8B pilot. This stage answers only:

1. whether the stored paired experiment is internally complete;
2. what utility and attacked performance were observed in each sampled stratum;
3. how much uncertainty is attributable to the sampled tasks and selected graphs.

It does not test DeGroot or another classical dynamic, fit graph-feature explanations, or claim an
LLM-specific mechanism.

## Units

- A clean observation is one task–graph pair under the recorded assignment and experiment seed.
- An attack observation is the paired task–graph–attack-node result using the same clean state.
- Attack nodes are exhaustively enumerated design positions. They are not treated as IID samples.
- Tasks are resampling units.
- Selected graphs are resampling units when a stratum contains multiple graphs.

## Primary descriptive estimands

- `utility`: mean clean readout correctness.
- `r_mean`: mean attacked readout correctness over every non-readout attack position.
- `d_mean`: paired clean-minus-attacked correctness.
- `mean_graph_r_worst`: for each graph, the minimum node-specific attacked accuracy, averaged over
  selected graphs.
- `mean_graph_d_max`: for each graph, the maximum node-specific paired accuracy drop, averaged over
  selected graphs.
- `correct_to_target_flip_rate`: target-error adoption conditional on a correct clean readout.
- `correct_to_non_target_error_rate`: attack-induced error not equal to the fixed target, conditional
  on a correct clean readout.

## Uncertainty

The 95% intervals use a crossed graph-by-task bootstrap. Each replicate samples graphs and tasks
with replacement, then retains every attack position for each sampled graph/task cell. For the
unique complete graph at `n=5, m=16`, only tasks are resampled; topology-distribution uncertainty
is not identifiable.

These intervals do not cover model-family variation, prompt variation, or assignment/experiment
seed variation because the current pilot contains one setting for each.

## Claim boundary

This analysis may describe observed differences. It cannot establish monotonicity, causal graph
mechanisms, failure of classical dynamics, or cross-model generalization.
