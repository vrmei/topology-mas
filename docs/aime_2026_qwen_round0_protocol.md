# 2026 AIME Qwen Round-0 Utility Protocol

## Purpose

Estimate the single-agent Round-0 utility of Qwen3-4B-Instruct-2507 on the
30 original problems in the LIVE-hosted 2026 AIME I and II form before any
mutation, topology, communication, or attack experiment is started.

The decision question is whether this task/model pair avoids a severe ceiling
or floor effect. The preregistered useful range for unconditional Round-0
accuracy is 20%--70%. This range is a pilot feasibility criterion, not a claim
about model quality.

## Frozen data

- Dataset: `data/aime/original_2026.jsonl`
- Manifest: `data/aime/original_2026.manifest.json`
- Tasks: all 30 problems, 15 from each contest
- Replicates: 10 independent generations per problem (300 requests total)
- Normal-agent-visible field: `problem` only
- Evaluator-only field: `gold_answer`
- Mutated or candidate answers: none
- Target error: none

The exact contest form is identified by the source URLs and source hashes in
the manifest. This matters because similarly named regional forms may differ.
Two source questions include illustrative images. The frozen text-only prompts
retain their complete mathematical definitions and replace those image tags
with the explicit marker `[Illustrative diagram omitted.]`; no solution
information is added.

## Frozen model and decoding

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Backend: local vLLM OpenAI-compatible server
- Temperature: 0.7
- Top-p: 0.8
- Top-k: 20
- Maximum output tokens: 16,384
- Service maximum context: 32,768 tokens
- Cross-task/topology/condition output reuse: disabled

These decoding settings follow the model-family settings already calibrated in
the preceding 2025 AIME Round-0 experiment. Each request is generated anew.

## Outcomes

Primary:

- `U0_all`: exact-match accuracy over all 300 scheduled generations. Parse,
  request, and length failures count as incorrect.

Secondary:

- parsed-answer rate;
- accuracy conditional on a parsed answer;
- request-error and length-limit rates;
- per-problem accuracy and AIME I/AIME II accuracy;
- bootstrap 95% confidence interval over problems, preserving the 10 replicate
  results within each problem.

The 2025 AIME result may be shown as a descriptive reference only. It is not a
paired task comparison because the problem sets differ.

## Decision rule and claim boundary

- If `U0_all` is within 20%--70%, the task/model pair is suitable for a larger
  utility/topology pilot without an obvious global floor or ceiling effect.
- If it falls outside that range, inspect parsing and length failures before
  attributing the result to mathematical ability.
- Do not launch a MAS experiment automatically from this run.

Because the contest postdates the fixed Qwen3-4B-Instruct-2507 snapshot, it is
a cleaner temporal holdout than the 2025 set. Date ordering alone does not
prove the absence of benchmark contamination or related-problem exposure.
