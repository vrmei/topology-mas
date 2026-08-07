# Node-round exposure--adoption protocol

This protocol is fixed before extracting node-round outcomes from the completed traces.

## Research question

After a target error reaches a benign node, how much of that node's newly induced target adoption can
be explained without natural-language content?

This separates two events that a final readout metric conflates:

```text
graph and upstream states deliver the target -> receiving LLM adopts the target
```

No semantic residual is assumed in advance.

## Unit of analysis

One active benign node update at round `t >= 1` in a task--graph--attack-node condition. Every attack
turn is paired with the clean turn having the same task, graph, node, round, experiment seed, and
assignment seed. The attacked node is excluded because its output is deterministic replay.

## Paired outcomes

- `induced_target_state`: the attack turn matches the frozen target and the paired clean turn does
  not;
- `new_induced_target_adoption`: `induced_target_state` is true now and the same benign node did not
  match the target in the preceding attack round;
- `induced_target_recovery`: the preceding attack turn was an induced target state and the current
  attack turn no longer matches the target.

The primary outcome is `new_induced_target_adoption`. A coincidental target answer appearing in both
clean and attack traces is not labeled as attack-induced adoption.

## Content-free exposure variables

At the immediately preceding round:

- count and fraction of incoming attack messages matching the target;
- count and fraction of incoming target messages that do not match the target in the paired clean
  trace;
- receiver's previous categorical state;
- counts of correct, target, other, and unparsed incoming states;
- number of distinct incoming categorical answers and unique-plurality indicators;
- equal-weight DeGroot target mass at the receiver and round, computed from frozen Round zero;
- round index and declared static graph/node features.

The observed categorical-message features contain intermediate LLM states but no rationale text.
They estimate a finite-state adoption kernel rather than an ex-ante graph-only predictor.

## Frozen comparisons

1. training-fold prevalence;
2. DeGroot receiver exposure only;
3. categorical incoming target exposure only;
4. receiver state plus categorical neighborhood state;
5. all declared content-free variables with fixed L2 logistic regression;
6. a fixed nonlinear histogram-gradient-boosting model using the same variables.

## Validation and statistics

Use strict 5-by-5 graph--task crossed holdout. Primary metric is Brier score; log loss and average
precision are secondary. Report results on:

- all eligible benign updates;
- updates with at least one target message received;
- updates with at least one attack-induced target message received.

Crossed graph-by-task bootstrap intervals compare each model with DeGroot receiver exposure. Node,
round, and attack-location rows remain clustered inside their sampled graph and task.

## Integrity gates

- exact clean/attack key matching for every analyzed turn;
- identical targets, graph, assignment, prompt version, and Round-zero record references;
- incoming-message senders and rounds agree with the graph and synchronous schedule;
- no attacked-node update appears as an adoption outcome;
- all categorical labels use the local deterministic parser already used by the main analysis;
- no text-derived feature enters a declared content-free model.

## Claim boundary

- Predicting adoption from categorical exposure supports a finite-state reduction, not an internal
  LLM mechanism claim.
- A residual does not establish semantics; omitted confidence, stochasticity, and lossy answer
  projection remain alternatives.
- Observed intermediate states are post-treatment variables. They are suitable for decomposing the
  realized propagation chain, not for estimating the total causal effect of topology.
- Only a later matched message intervention can attribute a difference to message presentation or
  reasoning content.
