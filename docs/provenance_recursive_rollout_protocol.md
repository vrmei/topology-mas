# Provenance-aware recursive rollout protocol

## Research question

Can attack-descended target lineage be generated recursively from

\[
G + S_0 + \text{attacker position}
\]

and improve out-of-graph, out-of-task endpoint prediction relative to the original
CTOU rollout?

The experiment does not use observed Round-1+ node states, compositions, provenance,
answers, or text during rollout.

## State and input representation

Normal-node latent state:

\[
\{C,T_n,T_a,O,U\},
\]

where `T_n` is a natural target state and `T_a` is an attack-descended target state.
The attacker is fixed to `T_a`, but its outgoing target message is marked direct.

For a normal receiver, incoming composition is:

\[
(\#C,\#T_d,\#T_r,\#T_n,\#O,\#U).
\]

- `T_d`: target message sent by the attacker;
- `T_r`: target message sent by a normal node in state `T_a`;
- `T_n`: target message sent by a normal node in state `T_n`.

The fitted local law is

\[
P(S_i^t\mid S_i^{t-1},t,\#C,\#T_d,\#T_r,\#T_n,\#O,\#U).
\]

The original CTOU law remains unchanged and is the primary baseline.

## Training labels

Extended states and incoming provenance counts are reconstructed from the real attack
traces only for fitting the transition law. Target lineage follows the conservative
within-run temporal rule defined in `ctou_provenance_protocol.md`.

Training and evaluation use the same five-by-five crossed graph/task folds as the
existing CTOU recursive rollout. A test graph and test task never occur in the fitted
transition law for that crossed fold.

## Recursive particle rollout

Round 0 is initialized as follows:

- attacker target state -> `T_a`;
- normal target state -> `T_n`;
- all other states retain `C/O/U`.

At each later round and in every particle:

1. derive each receiver's six provenance-aware message counts from the graph, attacker
   identity, and current simulated sender states;
2. query the provenance-aware local transition law;
3. sample the receiver's next extended state;
4. retain the attacker at `T_a`;
5. continue without reading any real Round-1+ trace state.

The final `T_n` and `T_a` masses are collapsed to `T` for comparison with the observed
four-state endpoint.

Particle rollout is primary because it propagates joint state and provenance
uncertainty. Mean-field rollout is not required for this first test.

## Models

1. original `ctou_table` particle rollout;
2. original `ctou_logit` particle rollout;
3. provenance-aware smoothed table particle rollout;
4. provenance-aware multinomial logistic particle rollout.

No topology, task identity, graph identity, node identity, text, or message embedding is
provided to either learned local law.

## Evaluation

### One-step diagnostics

- extended-state multiclass Brier and log loss;
- collapsed four-state Brier and log loss;
- attack-target binary Brier and log loss;
- task-bootstrap paired difference against the corresponding CTOU model.

### Recursive endpoint metrics

- target and correct Brier/log loss;
- graph-level endpoint MAE and Spearman correlation;
- fixed-`n` `m -> target` and `m -> correct` curve MAE;
- observed and predicted per-edge linear slopes, especially the n=8 target/robustness
  slope;
- absolute slope error and sign agreement.

## Interpretation criteria

- If provenance improves one-step losses but not endpoint curve/slope recovery, treat it
  as an explanatory variable, not a required topology-evaluator state.
- If it improves endpoint calibration and materially reduces n=8 slope error under the
  crossed holdout, treat provenance as a useful recursively generable state variable.
- If gains occur only in the table or only in the logistic model, report the estimator
  dependence rather than claiming a general provenance mechanism.
- A successful prediction does not isolate semantic rewriting from trajectory
  selection; provenance remains a low-dimensional proxy for both.
