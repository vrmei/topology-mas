# Round-phase mechanism analysis and replay plan

## Research question

Why does attacked readout accuracy usually decrease after the first update,
recover after the second update, and change little or inconsistently after the
third update in the completed Llama/GSM8K experiments?

## Competing explanations

- **H0 — composition sufficiency:** after conditioning on the receiver's
  previous CTOU state and exact incoming CTOU counts, update round has no
  practically important residual association with the next state. Topology
  changes the composition and timing of evidence, not the local response law.
- **H1 — provenance/history residual:** exact CTOU counts are insufficient.
  Direct versus relayed target evidence, shared ancestry, or accumulated local
  history changes the transition probability.
- **H2 — textual-semantic residual:** count and provenance controls remain
  insufficient because the concrete rationale text changes adoption or
  correction.

These are analysis hypotheses, not assumed conclusions.

## Phase 1 — trace-only analysis

### Data

- Llama-3.1-8B-Instruct + 50 fixed GSM8K tasks.
- 259 graphs: n=5 (61), n=6 (51), n=7 (76), n=8 (71).
- T=1 and T=2 are prefix endpoints from the same T=3 traces.
- Readout and internal receivers are reported separately.

### Primary estimands

For each receiver scope and system size, compare matched transition risks
between rounds while holding fixed:

1. previous state;
2. exact incoming counts (C,T,O,U);
3. system size n.

Primary transitions are C->T, C->O, O->C, T->C, and T->T.  The primary
contrast is Round 2 minus Round 1. Round 3 minus Round 2 is secondary.

Two matching levels are reported:

- CTOU-cell matching;
- task + CTOU-cell matching, to control task-specific latent difficulty more
  tightly.

Uncertainty is estimated by task-cluster and graph-cluster bootstrap. Multiple
transition comparisons use Benjamini-Hochberg correction.

### Provenance extension

On traces for which exact message ancestry can be reconstructed, add:

- direct, relayed, natural, or mixed target origin;
- immediate and recursive overlap among correct-message lineages;
- previous and next target provenance state.

The provenance analysis asks whether these variables reduce the matched round
contrast and improve crossed task/graph holdout prediction.

### Practical gate for Phase 2

A replay factor is promoted to the GPU experiment only if it satisfies both:

1. an absolute matched risk difference of at least 2 percentage points for a
   primary readout transition; and
2. a 95% task- or graph-cluster interval excluding zero, with directionally
   consistent evidence in more than one system size or matching level.

This gate prevents large samples from turning negligible effects into an
expensive follow-up.

## Phase 2 — paired single-receiver replay

The replay does not run a complete MAS. It reuses real task-matched receiver
history and peer messages from recorded traces and changes one factor at a
time. The exact factor is selected by Phase 1.

Candidate paired interventions:

1. same previous state and CTOU composition, direct versus relayed target;
2. independent versus shared correct-message ancestry;
3. actual accumulated previous response versus a state-matched independent
   previous response;
4. same CTOU and provenance, alternative real rationale texts.

All paired conditions use the same task, target answer, message count, message
representation, generation parameters, and deterministic message ordering.
Primary outcomes are C->T and O/T->C. Each cell uses multiple real message-set
replicates. Technical failures remain missing and are never mapped to O.

## Claim boundary and stop rule

- If matched round effects are practically negligible, do not run a generic
  round-label replay; treat the observed phase pattern as composition/timing
  until contradicted.
- If provenance removes the residual, do not claim a text-semantic mechanism.
- If classical/CTOU models recover endpoint accuracy and graph ranking without
  meaningful residuals, the project should emphasize low-cost topology
  evaluation rather than a new LLM-specific graph law.
