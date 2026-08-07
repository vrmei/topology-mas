# Matched rationale-ablation pilot protocol

This protocol is fixed before executing the new LLM conditions.

## Research question

When task, graph, attacked node, target answer, Round-zero states, communication horizon, and
classical DeGroot exposure are held fixed, does including a plausible erroneous rationale change the
probability that the LLM readout newly adopts the target error?

## Intervention

Each selected condition is executed under two attacker-message variants:

- `full_rationale`: the existing audited task-specific erroneous rationale, ending in
  `#### <target>`;
- `answer_only`: exactly `#### <target>` with no supporting rationale.

The attacker remains a deterministic replay node in both variants and performs no generation. The
target answer and objective wrongness are identical. Every non-attacker uses the same model,
prompt, Round-zero cached output, generation seed, temperature, and synchronous update protocol.

## Fixed pilot sample

- Tasks: 20 of the existing 100 GSM8K tasks, selected by ascending SHA-256 hash of `task_id`.
- Graphs: one graph per existing `(n,m)` stratum, selected by ascending `graph_id`.
- Attack locations: every non-readout node in each selected graph.
- Experiment seed: 0.
- Assignment seed: 0.
- Model: `meta-llama/Llama-3.1-8B-Instruct`, matching the completed pilot.
- Temperature: 0.3, matching the completed pilot.

Task and graph selection do not use attack outcomes, message content scores, or post-hoc residuals.
The existing `full_rationale` traces and clean traces are reused; only `answer_only` attack traces
require new LLM execution. Reused clean traces are validated by the batch runner against the exact
task, graph, seed, assignment, Round-zero records, prompt version, and model settings.

## Primary outcome and estimand

The binary outcome is induced target adoption at the readout:

\[
Y=1[\text{attack readout}=z^{adv}\land\text{clean readout}\neq z^{adv}].
\]

The primary estimand is the paired average difference:

\[
\Delta_{rationale}
=
\mathbb E[Y_{full}-Y_{answer\text{-}only}].
\]

A positive value means the complete erroneous rationale increases target adoption relative to the
same unsupported target answer.

## Secondary analyses

- paired clean-minus-attacked accuracy drop;
- discordant-pair counts;
- the effect within low, medium, and high classical-exposure thirds;
- graph-level and task-level effect heterogeneity;
- target adoption at intermediate downstream nodes, if trace integrity permits.

## Statistics

Point estimates retain condition pairing. The primary 95% interval uses crossed graph-by-task
bootstrap weights. An exact McNemar/binomial discordance test is reported only as a diagnostic
because it does not account for graph/task clustering.

The pilot is an effect-presence test, not the final estimate. If the interval is compatible with a
non-negligible effect, expand to all 100 tasks before introducing additional message variants.

## Integrity gates

- Every selected key must exist in both variants exactly once.
- Reference and target answers must match across variants.
- Selected graphs, Round-zero records, assignment, seeds, and prompt version must match.
- The attacker output must parse to the target in every run.
- Clean traces must be byte-identical copies of the pinned clean baseline and must pass the batch
  runner's semantic trace validation.
- Returned model identity must equal the pinned Llama model.

## Interpretation boundary

- A non-zero effect shows that attacker-message content beyond the target answer matters under this
  protocol.
- It does not isolate semantic reasoning from message length, formatting, or the presence of
  explanatory tokens.
- A null pilot does not show that language content never matters; it applies only to this rationale
  ablation, model, dataset, and sample.
- No result establishes that the LLM internally implements or violates DeGroot dynamics.
