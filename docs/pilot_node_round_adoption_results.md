# Pilot node-round exposure--adoption results

The protocol in `docs/node_round_adoption_protocol.md` was fixed before extracting node-round
outcomes. The analysis uses the completed 100-task GSM8K pilot and makes no new model calls.

## Integrity and event counts

- paired attack conditions: 20,400;
- eligible active benign-node updates at rounds 1--3: 200,700;
- newly induced target adoptions: 5,036 (2.51% of all eligible updates);
- updates receiving any target message: 85,146;
- updates receiving an attack-induced target message: 84,859;
- new adoptions among the last group: 4,936 (5.82%);
- duplicate update keys or pairing violations: zero.

Almost all newly induced adoptions occurred after a target message was received, but only a small
minority of exposed updates adopted it. This supports separating delivery from adoption in the
measurement framework. It does not identify why an exposed LLM adopts a message.

## Strict crossed graph--task holdout

For updates receiving an attack-induced target, the main results were:

| predictor | Brier | log loss | average precision |
|---|---:|---:|---:|
| intercept only | 0.05497 | 0.22367 | 0.0474 |
| DeGroot receiver exposure | 0.05497 | 0.22379 | 0.0577 |
| incoming target exposure | 0.05278 | 0.20804 | 0.1362 |
| receiver + categorical neighborhood | 0.05177 | 0.20417 | 0.1674 |
| all content-free, linear | 0.05049 | 0.19301 | 0.2062 |
| all content-free, HGB | 0.05031 | 0.19059 | 0.2078 |

Relative to DeGroot receiver exposure, the all-content-free HGB model improved Brier by 0.00481,
with a crossed graph-by-task bootstrap interval of [0.00304, 0.00683]. The all-content-free linear
model improved it by 0.00459 [0.00291, 0.00652]. Incoming categorical target exposure alone also
improved Brier by 0.00214 [0.00126, 0.00320].

The similar performance of the full linear and HGB models does not support a claim that a highly
nonlinear predictor is necessary. It does show that one scalar equal-weight DeGroot mass omits
reproducible information present in the receiver's prior categorical state, the realized incoming
answer configuration, round, and declared structural variables.

## Supported interpretation

The realized LLM propagation chain is not adequately summarized by final equal-weight DeGroot
target mass alone. A richer finite-state, content-free description predicts newly induced adoption
better on unseen graphs and unseen tasks in this pilot.

This is a result about model reduction, not yet about language semantics. The richer predictors use
post-treatment intermediate answer states and may capture state dependence, round dependence,
finite-state thresholds, or graph-conditioned exposure.

## Not supported

The analysis does not establish that:

- rationale wording causes adoption;
- the remaining prediction error is semantic;
- the categorical predictors estimate topology's total causal effect;
- the result transfers beyond this GSM8K, Llama-3.1-8B, and prompt configuration;
- HGB is meaningfully better than the full linear model.

The deferred matched rationale intervention is still needed to isolate message presentation while
holding the target answer, graph, initial states, and stochastic stream fixed.
