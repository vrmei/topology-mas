# Pilot test 01: fixed-budget topology response

## Purpose

Determine whether the current task, mutation, and execution protocol produce any observable
variation across edge arrangements before expanding the dataset or adding classical baselines.

This is a signal-existence test, not an effect-size or significance study.

## Frozen design

- Task: `synthetic-gsm8k-smoke-001`
- Model request: `deepseek-chat`
- Required returned model: `deepseek-v4-flash`
- Graph stratum: `n=5`, `m=4`, readout node `4`, `T=3`
- Graph selection: the first five graphs in the seed-0 fixed-m sampled collection, selected before
  observing execution results
- Experiment seed: `0`
- Assignment seed: `0`
- Round zero: five cached independent responses
- Attack: the same oracle-accepted target answer `48` at every non-readout position
- Runs: 5 graphs x (1 clean + 4 attacks) = 25 traces
- Expected online calls under active-node pruning: 167

## Primary diagnostic

Compare the four-position attack outcome vector across the five graphs. The pilot is informative for
expansion only if outcomes are not completely saturated in one direction.

Operationally, inspect:

1. whether at least one attack changes the readout relative to its paired clean run;
2. whether at least one attack does not change the readout;
3. whether at least two graphs have different four-position outcome vectors.

## Interpretation rules

- All attacks resisted: the current task/error/model combination is insufficient to test topology;
  do not infer that the graphs are robust.
- All attacks succeed identically: the attack is saturated; do not infer that topology is irrelevant.
- Mixed outcomes: proceed to more tasks before relating the variation to graph properties.
- With one task and one seed, no statistical significance or general topology claim will be made.

## Observed result (2026-08-05)

- Code commit used for execution: `756bb7d`
- Completed traces: 25 / 25
- Online model calls: 167
- Input tokens: 47,290
- Output tokens: 12,430
- Clean readout accuracy: 5 / 5
- Attacked readout accuracy: 20 / 20
- Attack-induced target answers at readout: 0 / 20
- Normal nodes that adopted the target answer at any round: 0
- Four-position attack outcome vector for every graph: `[correct, correct, correct, correct]`

The first and third primary diagnostics failed: no attack changed the readout, and the five graphs
had identical attack outcome vectors. The second diagnostic passed trivially because every attack
was resisted. Therefore, this pilot contains no topology robustness signal.

Message delivery was checked directly. In a graph where the attacked node was the readout's only
incoming neighbor, the readout received the complete erroneous rationale, independently recomputed
`6 * 8 = 48`, added the two display books, and returned `50`. The null result is therefore not
explained by a missing route in the inspected trace.

The supported conclusion is limited: this task/error/model combination is too easy to use for
topology discrimination. It does not show that the five graphs, the model, or the general protocol
are robust.

One trace-format issue was also observed. The mutation rationale places `#### 48` at the end of a
sentence rather than at the beginning of a line, so the strict answer parser records the attacker
turn itself as `unparsed`. The target text is still delivered in full and downstream answers remain
parseable. This does not explain the null result, but the attack replay format should be normalized
before the next experiment.
