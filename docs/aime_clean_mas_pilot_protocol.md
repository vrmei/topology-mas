# First 2026 AIME clean-MAS pilot protocol

## Research question

Under a homogeneous, bounded-message Qwen protocol, does three-round
communication change clean free-response AIME utility relative to the same
readout agent's independently generated Round-0 state?

This is a clean-only pilot. It does not contain mutation, a target error, an
attacker, or a robustness claim.

## Frozen task set

- All 30 tasks in `data/aime/original_2026.jsonl`.
- No post-hoc selection by the earlier task success rate.
- The earlier 10-replicate single-agent task bands are used only for diagnostic
  stratification: floor 0%--10%, informative 20%--80%, ceiling 90%--100%.

## Frozen graph design

- Homogeneous nodes: `n=5`, including readout node 4.
- Synchronous local broadcast, one directed edge per round.
- Fixed horizon: `H=3` for every graph.
- Readout has no outgoing edges; every other node can reach it within three hops.
- Active-node pruning removes only node-round calls outside the final readout's
  causal cone.
- Edge strata: `m in {4, 8, 12, 16}`.
- Graphs: three independently sampled labeled graphs for each of `m=4,8,12`,
  and the single possible complete graph for `m=16`; ten graphs total.
- Round 0 is regenerated inside every task-graph run. No output is reused across
  graphs, conditions, or runs. A node generates once per round and copies that
  same message to all out-neighbors.

## Bounded-message AIME protocol

Protocol version: `homogeneous-aime-private-solve-bounded-message-v2`.

Each logical node update has two local model calls. The first produces a private
solution draft with a 16,384-token ceiling. The second receives that draft and
compresses it into an auditable `SOLUTION_SUMMARY` followed by
`FINAL_ANSWER: \\boxed{ddd}`. The public summary has a hard 1,024-token output cap
and a 512--768-token target. Only the summary is broadcast; private drafts remain
in the audit trace, and node identifiers and sender labels are omitted.

The summarizer is deterministic (`temperature=0`) and is instructed not to re-solve
or change the answer extracted from the private draft. The private solver uses the
official Qwen sampling settings. A length-truncated private solution is marked
unparsed even if incidental digits occur in the partial text.

## Model

- `Qwen/Qwen3-4B-Instruct-2507`, local vLLM.
- Temperature 0.7, top-p 0.8, top-k 20.
- One experiment seed and one assignment seed; 300 task-graph clean runs.
- Independent-per-run Round 0 and no state-replay cache.
- 4,200 logical node updates and 8,400 physical backend calls.

## Primary estimands

For each graph `G`:

- `U0_bounded(G)`: Round-0 readout accuracy under the new bounded protocol.
- `UH(G)`: final Round-3 readout accuracy.
- `delta_U(G) = UH(G) - U0_bounded(G)`.

The paired primary comparison is `UH` versus `U0_bounded` within the same
task-graph run. The earlier 51.3% full-rationale Round-0 result is an external
reference, not the paired denominator, because the prompt and output budget differ.

## Transition decomposition

Readout transitions are reported with explicit denominators:

- correct preservation: `P(C_H | C_0)`;
- correct corruption: `P(not C_H | C_0)`;
- parsed-other-error correction: `P(C_H | O_0)`;
- unparsed-state correction: `P(C_H | U_0)`;
- complete `C/O/U -> C/O/U` transition counts.

The clean experiment has no task-specific target state `T`.

## Diagnostic analyses

- Per-graph and per-edge-level utility, paired delta, input/output tokens, calls,
  and latency when available.
- Per-task utility and paired delta over all ten graphs.
- Descriptive results by the frozen external floor/middle/ceiling bands.
- Paired uncertainty is computed by resampling tasks; graph variation is reported
  directly. With only one `m=16` graph, no claim about within-stratum graph
  variance is made there.

The possible claim that communication benefit peaks at intermediate task
difficulty is not preregistered as true. It is supported only if the observed
middle-band paired gain exceeds the other bands with uncertainty that excludes a
trivial difference; otherwise it remains unsupported.

## Stop condition

This pilot determines whether an attack run is informative. No attack or mutation
experiment is launched automatically. First inspect parsing/length failures,
bounded Round-0 utility, final utility, and the correction-versus-corruption
decomposition.
