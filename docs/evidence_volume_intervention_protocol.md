# Controlled evidence-volume intervention

Status: frozen before any intervention outcome is generated.

## Research question

After fixing the relative C/error composition seen by one homogeneous Llama receiver,
does multiplying the absolute number of distinct peer rationales change its one-step
state transition?

This experiment is an intervention on a receiver prompt, not a complete MAS run. It is
designed to distinguish a ratio-response mechanism from an absolute-evidence-volume
mechanism suggested by the frozen n=5 to n=10 CTOU extrapolation result.

## Model-facing boundary

- model: `meta-llama/Llama-3.1-8B-Instruct`;
- prompt: the existing homogeneous GSM8K node-update prompt;
- sampling: temperature 0.6, top-p 0.9, maximum output 768;
- server context: 8192 tokens, with every rendered request audited before generation;
- tasks: the same fixed GSM8K-50 set used in the previous Llama pilots;
- no topology, sender identity, receiver identity, graph identity, role label, or attack
  metadata is shown to the model.

## Natural-language stimulus pool

Messages are drawn from completed Llama n=5/6/7/8/10 traces for the same task. Attacker
replays are excluded. Text is deduplicated after whitespace normalization, delimiter-like
text is rejected, and every retained item preserves its parsed answer and source
provenance.

- `C`: a normal-node output objectively matching the GSM8K reference answer;
- `T`: a non-attacker normal-node output matching the frozen task-specific target error;
- `O`: a normal-node output matching neither the reference nor the target error.

The pre-generation pool audit requires, for every task, at least 12 distinct `C`, five
distinct `T`, and five distinct `O` rationales. No peer message is duplicated inside one
prompt. Previous output is also distinct from every peer message in that prompt.

## Factorial intervention

The five base C/error count vectors are multiplied by 1, 2, and 3:

| Correct share | Error share | 1x | 2x | 3x |
| ---: | ---: | --- | --- | --- |
| 100% | 0% | 2C+0E | 4C+0E | 6C+0E |
| 80% | 20% | 4C+1E | 8C+2E | 12C+3E |
| 75% | 25% | 3C+1E | 6C+2E | 9C+3E |
| 66.7% | 33.3% | 2C+1E | 4C+2E | 6C+3E |
| 50% | 50% | 1C+1E | 2C+2E | 3C+3E |

Two scenarios reuse this design:

1. **attack adoption**: previous state `C`, error state `T`; primary outcome is
   `I(next=T)` and secondary outcome is `I(next=C)`;
2. **benign correction**: previous state `O`, error state `O`; primary outcome is
   `I(next=C)` and secondary outcome is `I(next=O)`.

For each task, scenario, ratio, and replicate, the 3x condition is sampled first. The 1x
and 2x peer sets are nested prefixes of the 3x set within each state. All three scales use
the same previous solution and generation seed. Peer display order uses the existing
anonymous content-hash ordering. Thus the paired scale contrast adds distinct evidence;
it does not replace the original evidence or repeat the same text.

## Sample size

- 50 tasks;
- 2 scenarios;
- 5 ratios;
- 3 volume multipliers;
- 5 independently sampled message-set replicates.

Total: 7,500 one-step Llama calls. The replicate varies both the natural message set and
the stochastic generation seed; this pilot does not separately identify these two
variance components.

## Estimands and decision rules

The primary estimand is the task-paired change in primary-outcome probability at fixed
ratio:

`ATE_ratio(3x-1x) = E[Y(3x)-Y(1x)]`.

Also report `2x-1x` and `3x-2x` to detect saturation. Uncertainty uses task-cluster
bootstrap, resampling tasks and retaining all paired message-set replicates and volumes.

The smallest effect size of interest is frozen at five percentage points:

- a 95% interval wholly inside `[-0.05, 0.05]` is evidence of practical equivalence for
  that contrast in this pilot;
- an interval excluding zero is evidence of a directional absolute-volume effect;
- an interval overlapping both zero and an equivalence boundary is inconclusive.

The pooled ratio-adjusted 3x-1x contrast is primary. Ratio-specific contrasts and
state-distribution changes are secondary. Family-wise p-values, if reported, use Holm
correction within each scenario; conclusions do not rely on an uncorrected p-value.

## Interpretation cases

- **Ratio-dominant**: pooled and ratio-specific volume contrasts are practically
  equivalent while outcome probability changes substantially across ratios.
- **Saturation**: 2x-1x is material but 3x-2x is practically equivalent or much smaller.
- **Absolute-volume dependence**: a directional paired volume effect replicates across
  tasks and is not confined to one ratio or a small set of message pools.
- **Inconclusive**: intervals are too wide or heterogeneous to distinguish the above.

These outcomes concern the number of distinct natural-language peer rationales under the
frozen prompt. They do not isolate message count from total input-token volume, prove
arbitrary-scale invariance, or establish the mechanism for another model or dataset.
