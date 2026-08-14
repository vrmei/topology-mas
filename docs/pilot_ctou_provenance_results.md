# CTOU provenance-gap pilot results

## Question

Conditional on exactly the same

\[
(S_{t-1}, t, \#C, \#T, \#O, \#U),
\]

does the origin of incoming messages change the observed next-state probability?

This is a post-hoc analysis of the existing Llama-3.1-8B dense-50 attack traces. It
does not call an LLM and does not refit the CTOU transition model.

## Data and integrity

- 50 GSM8K tasks;
- 132 graphs;
- 37,050 attack conditions;
- 394,800 non-attacker receiver updates;
- zero duplicate update keys;
- all reconstructed messages obey the graph edge and one-hop synchronous schedule.

Target lineage is reconstructed only within the attack trace. A normal-node target
message is called `relayed` only when its target state descends from a direct or prior
relayed target message. Round-0 normal target answers are `natural`. As an integrity
check, Round 1 contains zero relayed-only updates.

## Result 1: direct versus relayed target information

The primary sample restricts the receiver's previous state to `C`, requires at least
one incoming `T`, excludes natural and mixed target provenance, and compares only
exact CTOU cells supported by both direct-only and relayed-only observations.

With at least 30 observations per provenance group in each cell:

| Receiver scope | Matched cells | Matched rows | P(C→T), direct | P(C→T), relayed | Direct − relayed | 95% task/graph bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| All | 18 | 71,479 | 1.52% | 14.30% | −12.78 pp | [−15.28, −10.14] pp |
| Internal | 10 | 35,833 | 1.74% | 14.74% | −13.00 pp | [−16.35, −9.83] pp |
| Readout | 12 | 24,055 | 1.22% | 14.41% | −13.20 pp | [−16.23, −10.08] pp |

The aggregate effect is stable under minimum cell-group support thresholds:

| Minimum rows per group/cell | Matched cells | Direct − relayed |
|---:|---:|---:|
| 10 | 28 | −12.82 pp |
| 30 | 18 | −12.78 pp |
| 50 | 15 | −13.10 pp |

At the threshold-30 setting, 17 of 18 matched cells have lower adoption under direct
than relayed target input. The median cell difference is −7.24 pp; the only positive
cell is approximately +0.08 pp.

Descriptive stricter matching preserves the direction:

- exact CTOU cell + task, minimum 5 per group: −12.02 pp over 195 cells;
- exact CTOU cell + task + `n,m`, minimum 3 per group: −14.60 pp over 304 cells;
- exact CTOU cell + task + graph, minimum 2 per group: −11.03 pp over 330 sparse cells.

These stricter results are sensitivity checks without separate bootstrap intervals.

## Result 2: shared ancestry among correct messages

For previous-`C` receivers with at least one incoming `T` and at least two incoming
`C` messages, correct-message parent overlap is much weaker.

| Comparison | Scope | Outcome | Shared | Independent | Difference | 95% CI |
|---|---|---|---:|---:|---:|---:|
| Immediate parent overlap | All | next `T` | 1.25% | 1.81% | −0.56 pp | [−1.57, +0.21] pp |
| Immediate parent overlap | Internal | next `T` | 1.62% | 1.69% | −0.07 pp | [−1.36, +0.85] pp |
| Immediate parent overlap | Readout | next `T` | 0.90% | 1.41% | −0.51 pp | [−1.66, +0.44] pp |
| Recursive lineage overlap | All | next `T` | 1.39% | 1.82% | −0.42 pp | [−1.46, +0.41] pp |

All target-adoption intervals include zero, and the point estimates are below the
pre-specified two-percentage-point practical threshold. A small readout difference in
remaining correct under immediate overlap (+1.50 pp) is not reproduced cleanly by the
recursive definition and should not be treated as a mechanism result.

Within direct-only target exposure, the shared-versus-independent target-adoption
difference is approximately −0.13 pp under a minimum-five support sensitivity check.
The relayed-only subgroup is too small for a stable fine-grained overlap analysis.

## Interpretation

The current evidence supports a narrow claim:

> Target-message lineage contains substantial next-state information that is absent
> from CTOU counts, while the tested common-parent overlap summaries do not show a
> comparably stable effect.

This does **not** establish that relaying causally increases persuasiveness. Peer
messages are anonymous in the execution prompt, so receivers cannot explicitly trust
one node identity more than another. Direct and relayed messages can still differ in:

1. semantic wording after an LLM adopts and rewrites the target rationale;
2. the sender's unobserved interaction history;
3. selection: a relayed message exists only on trajectories where an upstream normal
   node already found the target sufficiently persuasive.

Therefore the result identifies a missing predictive variable—attack-descended
message lineage/path—but does not yet isolate whether the operative mechanism is
semantic transformation, trajectory selection, or both.

The common-parent test is also only a proxy for joint dependence. A null result here
does not rule out correlated node transitions more generally.

## Most direct next test

Before adding a large model extension, add one provenance variable to the existing
transition evaluator and measure out-of-task/out-of-graph log-loss and recursive
endpoint improvement. If provenance improves only one-step fit but not recursive
rollout, it is explanatory but not sufficient for topology evaluation. If it improves
both, the smallest useful extension is a provenance-aware CTOU model.

To isolate mechanism after that, use a paired message-replay experiment that holds the
receiver state and CTOU composition fixed while swapping the raw direct rationale with
the raw relayed rewrite. That intervention requires model calls; the current analysis
does not.
