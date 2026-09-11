# Priority 1–2 report: round-phase mechanism analysis

## Scope

- Model/task: Llama-3.1-8B-Instruct + GSM8K.
- Observational trace analysis: 749,950 normal-node updates from 259 graphs, with `n in {5,6,7,8}`.
- `T=1` and `T=2` are prefix states from the recorded `T=3` trajectories, not independently rerun systems.
- Receiver replay: 20 support-eligible tasks, 3 replicates, 1 or 2 correct background peers, 480 requests / 240 pairs.

## Priority 1: what changes between Round 1 and Round 2?

For the readout, comparisons were matched on previous state and exact incoming `(C,T,O,U)` counts. Thus the following differences cannot be explained only by a different number of correct, target, other-error, or unparsed incoming messages.

| Transition | n=5 | n=6 | n=7 | n=8 |
|:---|---:|---:|---:|---:|
| `C -> T`, R2 minus R1 | -3.22 pp | -2.79 pp | -2.52 pp | -2.31 pp |
| `C -> not-C`, R2 minus R1 | -6.26 pp | -5.73 pp | -5.30 pp | -5.17 pp |
| `O -> C`, R2 minus R1 | -3.32 pp | -2.56 pp | -3.26 pp | -5.28 pp |

The `C -> T` and `C -> not-C` confidence intervals exclude zero for every `n`; task- and graph-cluster bootstrap BH-adjusted q-values are approximately 0.001 in the primary contrasts. `O -> C` does not increase in Round 2. Therefore the observed later-round improvement should not be summarized as a universal increase in correction. It is more consistent with stronger preservation of already-correct readout states plus a favorable change in the distribution of states/compositions. Because matching remains observational, hidden rationale/history and survivor selection are still possible explanations.

### Target provenance diagnostic

Within the same previous state, round, exact CTOU counts, and receiver scope, target messages propagated by normal nodes were associated with much higher subsequent target adoption than target messages sent directly by the attacker.

| n | Readout direct `P(next=T)` | Readout relayed `P(next=T)` | Direct minus relayed |
|---:|---:|---:|---:|
| 5 | 7.80% | 25.64% | -17.84 pp |
| 6 | 1.70% | 12.83% | -11.14 pp |
| 7 | 0.99% | 13.51% | -12.52 pp |
| 8 | 0.50% | 10.38% | -9.88 pp |

All four task-cluster 95% confidence intervals exclude zero. This is an observational result. Since receivers never see an attacker/provenance label, provenance indexes the generation history and resulting text, not a trusted source identity presented to the receiver.

## Priority 2: paired single-receiver replay

The replay used the historical Llama sampling configuration (`temperature=0.6`, `top_p=0.9`, `max_output_tokens=768`). Within each pair, task, peer order, correct background messages, and generation seed were fixed.

### A. Target-message intervention

Only one peer message changed: a direct attacker target rationale was replaced by a target rationale produced by a normal relay node.

| Correct background peers | Direct target adoption | Relayed target adoption | Paired effect | Task-bootstrap 95% CI | BH q |
|---:|---:|---:|---:|:---:|---:|
| 1 | 8.33% | 21.67% | +13.33 pp | [+6.67, +20.00] | 0.0018 |
| 2 | 0.00% | 18.33% | +18.33 pp | [+8.33, +30.00] | 0.0018 |
| pooled | 4.17% | 20.00% | +15.83 pp | [+9.17, +23.33] | 0.0018 |

The pooled correct-answer rate fell from 89.17% to 70.83% (effect -18.33 pp, 95% CI [-28.33, -8.33], BH q=0.0018). Restricting analysis to pairs where both generations ended with `finish_reason=stop` gave a similar target-adoption effect of +16.38 pp.

### B. Previous-history intervention

Only the receiver's previous correct response changed: a correct Round-0 response was replaced by a correct response generated after deliberation. Peer messages were identical.

Pooled target adoption changed from 17.50% to 14.17% (effect -3.33 pp, 95% CI [-9.17, +2.50], BH q=0.5630). Pooled correctness changed by +3.33 pp (95% CI [-5.83, +12.50], BH q=0.6405). This experiment does not provide evidence that a deliberated correct history alone explains the Round-2 preservation effect.

## Engineering audit

- Completed: 480/480; technical failures: 0.
- Pair-invariant violations: 0.
- Input tokens: median 908, maximum 1,573.
- Output tokens: median 244, maximum 768.
- Finish reasons: 469 `stop`, 11 `length`.
- Runtime: about 77 seconds on one A800 with a 64-worker request pool.

## Claim boundary and next discriminator

The replay supports a causal difference between the *actual message texts* used in the direct and relayed conditions. It does not yet isolate the cause of that text-level difference. The direct target messages had a median length of 44 tokens, whereas relayed target messages had a median length of 224.5 tokens. Consequently, semantic transformation, argument completeness, and evidence volume remain confounded.

The defensible result is:

> On support-eligible GSM8K tasks, target rationales produced after propagation through a normal Llama node caused substantially more downstream target adoption than the original direct attacker rationales under matched local state/count conditions.

The current data do **not** justify the stronger claim that source identity itself matters, or that semantic laundering rather than message length is the unique mechanism. A source/style-by-length intervention is required for that separation.
