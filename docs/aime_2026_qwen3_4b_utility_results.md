# 2026 AIME Round-0 utility: Qwen3-4B-Instruct-2507

## Scope

This run evaluates only single-agent Round-0 utility on all 30 original problems
in the LIVE-hosted 2026 AIME I and II form. It contains no problem mutation,
target error, attack, communication, or topology execution.

The exact source form is frozen by the URLs and hashes in
`data/aime/original_2026.manifest.json`. Two auxiliary source illustrations were
replaced by an explicit omission marker after verifying that the problem text
contains the complete mathematical definition.

## Frozen protocol

- Model: `Qwen/Qwen3-4B-Instruct-2507`, served locally with vLLM on one A800 80GB.
- Tasks: 30; independent replicates per task: 10; total requests: 300.
- Sampling: temperature `0.7`, top-p `0.8`, top-k `20`.
- Maximum output: 16,384 tokens; service context: 32,768 tokens.
- Reuse: none. Each request has its own request ID and generation seed.
- Primary metric: correct requests divided by all 300 scheduled requests.
  Truncated, unparsed, and failed requests remain in the denominator.
- Uncertainty: 10,000-replicate bootstrap over tasks, preserving each task's ten
  generations.

The preregistered feasibility interval was 20%--70% aggregate utility. It is a
pilot decision range, not a model-quality benchmark.

## Result

| Quantity | Result |
|---|---:|
| Requests | 300 |
| Request failures | 0 |
| Valid parsed answers | 256 (85.3%) |
| Correct answers | 154 |
| Primary utility, `U0_all` | 51.3% |
| Task-bootstrap 95% CI | 37.3%--65.0% |
| Accuracy conditional on a parsed answer | 60.2% |
| Length-limit stops | 40 (13.3%) |
| Other unparsed responses | 4 (1.3%) |
| AIME I utility | 54.0% |
| AIME II utility | 48.7% |
| Mean output tokens | 6,779.2 |
| Median output tokens | 5,708.5 |
| Total output tokens | 2,033,752 |
| Wall time | 28.18 minutes |
| A800 wall-clock GPU time | 0.470 GPU-hours |

Across tasks, 9 are in the 0%--10% floor band, 12 in the 20%--80%
informative band, and 9 in the 90%--100% ceiling band. The aggregate is therefore
not at a global floor or ceiling, but the task-level distribution is strongly
heterogeneous.

## Interpretation

The frozen decision criterion is met: 51.3% lies inside the preregistered
20%--70% range, and 12 of 30 tasks have non-extreme empirical success rates.
Therefore, the 2026 set is suitable for a later utility/topology pilot without
first generating parameter mutations solely to adjust aggregate difficulty.

This does not establish that every problem is informative. Nine tasks are at the
empirical floor and nine at the ceiling under ten samples. Any later topology
analysis must retain task-level outcomes or use a preregistered task-selection rule;
it must not infer homogeneity from the 51.3% aggregate.

The 2025 reference run obtained 44.0% utility, compared with 51.3% here. This
7.3-point descriptive difference is not a paired model comparison: the problem
sets differ and the task-bootstrap confidence intervals overlap. The defensible
conclusion is only that both sets avoid an aggregate floor or ceiling for Qwen.

The 2026 contest postdates the fixed Qwen3-4B-Instruct-2507 snapshot, making this a
cleaner temporal holdout than 2025. Date ordering alone cannot prove zero benchmark
contamination or absence of related-problem exposure.

## Operational constraint retained

The median response is still 5,708.5 tokens. Full-rationale multi-agent broadcast
would exceed a 32,768-token context quickly at moderate in-degree. This utility
result does not resolve that systems constraint; a later MAS protocol still needs
a frozen bounded-message policy or a larger context budget before it is launched.

## Reproducibility

Committed summaries:

- `artifacts/aime_original_2026_qwen3_4b_round0_16k_formal_v1/audit.json`
- `artifacts/aime_original_2026_qwen3_4b_round0_16k_formal_v1/summary.json`
- `artifacts/aime_original_2026_qwen3_4b_round0_16k_formal_v1/summary.md`
- `artifacts/aime_original_2026_qwen3_4b_round0_16k_formal_v1/per_task_solve_rates.csv`
- `artifacts/aime_original_2026_qwen3_4b_round0_16k_formal_v1/per_problem_solve_rates.png`

Raw responses and the request plan are retained locally and on the experiment
server. They are not committed because the raw responses contain more than two
million generated tokens. Their SHA-256 hashes are recorded in `audit.json`.
