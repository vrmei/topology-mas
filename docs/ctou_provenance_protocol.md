# CTOU provenance-gap analysis protocol

## Research question

Conditional on the exact CTOU state cell

\[
(S_{t-1}, t, \#C, \#T, \#O, \#U),
\]

does structural message provenance change the real next-state distribution?

This analysis tests whether CTOU loses predictive information by counting states without recording where those states came from. It does not modify or refit the recursive CTOU model.

## Data

- Existing Llama-3.1-8B GSM8K 50-task attack traces.
- All non-attacker receiver updates already used in the CTOU analysis.
- No new LLM calls.
- Execution-time `answer_state` is the state oracle.

## Structural provenance definitions

All definitions are computed from the actual synchronous messages listed in each trace.

### Target provenance

For every incoming `T` message:

- `direct`: sender is the attacker;
- `relayed`: sender is a normal node whose target state descends temporally from a
  direct or already-relayed target message in the same attack trace;
- `natural`: sender emitted the target at Round 0, or first entered the target state
  without receiving an attacker-descended target message.

The target-state lineage is computed recursively. A normal node that remains in `T`
inherits its previous origin; a normal node that newly changes into `T` becomes
`relayed` only if it received a direct or relayed `T` in that update. This deliberately
uses no paired-clean output, because Round 0 was independently regenerated across
conditions in this pilot.

At the receiver-update level:

- `direct_only`: all incoming `T` messages are direct;
- `relayed_only`: all incoming `T` messages are induced relays;
- `natural_only`: all incoming `T` messages have natural target lineage;
- `mixed`: more than one provenance type is present.

The primary contrast is pure `direct_only` versus pure `relayed_only`, conditional on
the exact CTOU cell. Natural and mixed target inputs are excluded. Because the attacker
can send at most one direct message, common support automatically compares one direct
target message with one induced relayed target message.

### Correct-source overlap

For each incoming `C` sender at round `t-1`, define its immediate structural parent set as:

- the sender itself, representing persistence of its previous state; plus
- the actual senders of messages it received while producing its round-`t-1` output.

For receiver updates with at least two incoming `C` messages:

- `independent_immediate_C`: no pair of C senders has overlapping parent sets;
- `shared_immediate_C`: at least one pair overlaps.

A recursive potential-lineage set is also computed by unioning structural ancestors through earlier turns. This is secondary because it may become nearly saturated in dense cyclic graphs.

These variables describe potential structural ancestry, not proven causal influence.

## Primary analysis

Restrict to:

- previous receiver state `C`;
- at least one incoming `T`;
- non-attacker receivers.

Outcome:

\[
Y=\mathbb{1}[S_t=T],
\]

that is, `C -> T` adoption.

Compare direct versus relayed target provenance only inside exact CTOU cells represented in both groups. Report:

1. raw group counts and transition probabilities per cell;
2. a common-support standardized risk difference, weighting each cell by the smaller of its two group counts;
3. two-way task/graph bootstrap intervals;
4. readout and internal-node results separately when both groups have adequate support.

The pre-specified practical-effect threshold is an absolute `C -> T` probability difference of 2 percentage points. This threshold is descriptive, not an equivalence margin established by domain consensus.

## Secondary analysis

Among exact cells with at least two incoming `C` messages, compare:

- shared versus independent immediate C ancestry;
- shared versus independent recursive ancestry.

Report both `P(next=T)` and `P(next=C)` for previous-`C` receivers. Require at least 30 rows per provenance group in the aggregate matched support; show sensitivity thresholds of 10 and 50 per exact cell.

## Integrity and support checks

- Incoming message sender and round must match the graph and synchronous schedule.
- Every update key `(task, graph, attacker, receiver, round)` must be unique.
- The reconstructed CTOU counts must equal the existing update extractor counts.
- Direct target senders must equal the attack node.
- No comparison is reported without common exact-cell support.
- Counts, number of tasks, graphs, runs, exact cells, and readout/internal coverage are always reported.

## Claim limits

- A direct-versus-relayed difference combines sender identity, path length, and possible semantic rewriting; this analysis cannot isolate those mechanisms from one another.
- Structural parent overlap is potential dependence, not proof that the common source affected both nodes.
- Small observed differences do not prove provenance irrelevant outside the current model, task, horizon, and attack protocol.
- Bootstrap intervals condition on the existing finite graph/task collection and do not provide a theorem about all graphs.
