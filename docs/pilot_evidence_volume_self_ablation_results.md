# Previous-solution ablation results

## Outcome

The no-self follow-up completed 7,500/7,500 Llama calls with zero failures. It reused
the completed with-self experiment's exact task, peer-message sets, ratios, volume
levels, replicates, and generation seeds. The prompt audit found zero previous-section
leaks, zero previous-stimulus IDs among peers, and zero incidental previous-text matches.

Removing the explicit previous solution produced an asymmetric result:

| Primary outcome | With-self 3x-1x | No-self 3x-1x | With-minus-no interaction |
| --- | ---: | ---: | ---: |
| Target selection | +13.36 pp | +14.00 pp | -0.64 pp |
| 95% task-cluster CI | [10.80, 16.00] | [10.72, 17.12] | [-3.04, 1.84] |
| Correct selection | +17.52 pp | +7.84 pp | +9.68 pp |
| 95% task-cluster CI | [15.12, 19.92] | [4.72, 10.88] | [6.16, 13.36] |

The target-selection volume slope survived intact after removing the previous correct
solution. The correct-selection slope became substantially smaller after removing the
previous erroneous solution, but did not disappear under the pre-specified unconditional
estimand.

## Interpretation by mechanism

### Target-error condition

The frozen rule classifies this as **strong peer-volume persistence**. The no-self
3x-1x effect was +14.00 pp and its lower confidence bound exceeded the +5 pp smallest
effect of interest. The interaction was practically equivalent to zero.

Showing a previous correct solution shifted the target-selection level downward, but it
did not measurably change the volume slope. For example, at a 50/50 peer ratio, target
selection rose from 32.0% to 59.2% without self and from 18.8% to 47.2% with self. The
previous correct solution therefore supplied resistance at each volume, while the
additional distinct peer rationales retained a similar marginal effect.

This rejects the explanation that the attack-side volume effect was mainly caused by
diluting one explicit previous answer.

### Benign C/O condition

The unconditional no-self correct-selection effect remained positive at +7.84 pp, but
the with-minus-no interaction was +9.68 pp. Thus the explicit previous erroneous
solution materially amplified the original correction slope: more peers increasingly
overcame that fixed erroneous prior.

The state-mass decomposition from 1x to 3x is informative:

| Condition | Correct | Other error | Unparsed |
| --- | ---: | ---: | ---: |
| With previous O | +17.52 pp | -14.40 pp | -3.28 pp |
| Without previous | +7.84 pp | -2.24 pp | -5.68 pp |

Without the previous O solution, most of the unconditional gain came from fewer
unparsed outputs rather than conversion of parsed other errors to correct answers.

## Parsed-only diagnostic

Conditioning on parsed outputs is post-treatment and is therefore not the primary causal
estimand. It is useful for diagnosing whether the unconditional effect is a formatting
artifact:

| Primary outcome | With-self 3x-1x | No-self 3x-1x | Interaction |
| --- | ---: | ---: | ---: |
| Target given parsed | +14.15 pp | +12.47 pp | +1.69 pp |
| 95% CI | [11.22, 17.27] | [9.01, 15.79] | [-1.18, 4.64] |
| Correct given parsed | +16.95 pp | +3.29 pp | +13.67 pp |
| 95% CI | [14.17, 19.75] | [0.63, 5.78] | [10.22, 17.25] |

The attack-side result remains strong after this diagnostic. The no-self correction
effect becomes small and close to the practical-equivalence boundary.

## What this establishes

The experiment supports three calibrated claims:

1. An explicit previous correct answer is not necessary for the attack-side absolute
   volume effect. At fixed peer ratio, more distinct task-matched peer rationales caused
   more target selection even after the previous solution was removed.
2. The original benign-correction slope was partly a classical self-inertia phenomenon:
   increasing peer volume helped overcome one fixed erroneous previous solution.
3. Absolute volume is not a single universal mechanism. Its role differs between target
   adoption, correction of an explicit prior, and output parseability.

This is stronger than the first pilot's ratio-only falsification, but it still does not
prove a uniquely LLM-specific semantic mechanism. The no-self receiver sees the task and
can reconstruct an implicit model belief. Message count and total token volume also
change together. Classical Bayesian evidence accumulation can likewise depend on sample
count even when normalized DeGroot peer averaging does not.

## Consequences for CTOU

A scale-transfer model should not use a universal raw degree coefficient or discard
absolute counts. The present data motivate:

- a bounded/nonlinear peer-volume term;
- interactions with previous state and explicit-previous availability;
- a separate model or state for unparsed outputs;
- validation of attack and benign transitions separately rather than assuming one
  shared volume law.

The next step should be CPU-only: fit these alternatives on existing transition traces
and compare frozen recursive rollout to n=10. A further GPU mechanism experiment is only
needed to separate message count, total token volume, and semantic diversity, for example
by comparing distinct rationales with duplicated or length-matched evidence.

## Execution details

- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Sampling: temperature 0.6, top-p 0.9, maximum output 768.
- Requests: 7,500 completed, zero failed.
- Generation runtime: approximately 27 minutes 33 seconds.
- Prompt input tokens: median 1,471; P95 3,431; maximum 5,114.
- Pairing audit: zero frozen-field mismatches.
- GPU service was stopped after the CPU analysis completed.

Local artifacts are stored under
`artifacts/evidence-volume-self-ablation-v1/analysis/`.
