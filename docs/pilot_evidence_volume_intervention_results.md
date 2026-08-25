# Evidence-volume intervention pilot results

## Outcome

The frozen one-step intervention completed all 7,500 requests with zero failures. At a
fixed incoming C/error ratio, increasing the number of distinct, task-matched peer
rationales from 1x to 3x materially changed the Llama-3.1-8B receiver's transition:

| Scenario | Primary transition | 3x - 1x | Task-cluster bootstrap 95% CI |
| --- | --- | ---: | ---: |
| Attack adoption | previous C -> next T | +13.36 pp | [+10.72, +16.00] pp |
| Benign correction | previous O -> next C | +17.52 pp | [+15.12, +19.84] pp |

Both intervals exclude zero and the frozen +/-5 pp practical-equivalence region. The
ratio-only local law is therefore not adequate for this model, dataset, and prompt.

This result must **not** yet be described as a uniquely LLM-specific violation of
classical graph dynamics. A DeGroot-style reference that treats the receiver's previous
state as one equal-weight self vote also predicts a same-direction volume effect because
more peer messages dilute self inertia. The experiment rejects a peer-ratio-only model;
it does not by itself isolate semantic aggregation from classical self-weight dilution.

## Frozen design and execution audit

- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Tasks: the fixed GSM8K-50 subset used by the preceding Llama pilots.
- Sampling: temperature 0.6, top-p 0.9, maximum output 768 tokens.
- Design: 2 scenarios x 5 ratios x 3 volume multipliers x 5 message-set replicates x
  50 tasks = 7,500 one-step receiver updates.
- Pairing: 1x, 2x, and 3x used the same previous output and generation seed. Their peer
  sets were nested and contained no repeated raw message.
- Stimuli: real Llama traces from n=5/6/7/8/10; attacker replays were excluded. `T`
  messages came only from normal nodes that had adopted the target error.
- Information boundary: the receiver saw no graph, source node, receiver node, role, or
  attack identity.
- Prompt-token audit: median 1,702, P95 3,673, maximum 5,413; all below the 7,424-token
  input allowance.
- Runtime: approximately 30 minutes 50 seconds at concurrency 96; 7,500 completed and
  zero failed.
- Pairing audit: zero incomplete pairs and zero nesting failures.

## Response shape

### Attack adoption

The pooled target-adoption increase was +8.24 pp from 1x to 2x and another +5.12 pp
from 2x to 3x. The later increase remained directional overall, but the response was
ratio dependent:

| Incoming C share | 3x - 1x target adoption | 3x - 2x | Interpretation |
| ---: | ---: | ---: | --- |
| 100% | 0.0 pp | 0.0 pp | Negative-control sanity check |
| 80% | +8.4 pp | +2.0 pp | Late contrast inconclusive |
| 75% | +9.6 pp | +2.4 pp | Late contrast inconclusive |
| 66.7% | +20.4 pp | +6.4 pp | Continued increase |
| 50% | +28.4 pp | +14.8 pp | Strong continued increase |

Absolute volume therefore amplified target adoption most strongly when the target error
already occupied a substantial share of incoming evidence. At low target shares, the
response showed signs of saturation.

### Benign correction

The pooled correction increase was +11.28 pp from 1x to 2x and +6.24 pp from 2x to 3x.
Every ratio had a positive 3x - 1x contrast, ranging from +13.6 to +22.4 pp. Some
ratio-specific 3x - 2x contrasts were inconclusive, again suggesting a nonlinear or
saturating response rather than a raw linear degree law.

### Parsing sensitivity

Unparsed output decreased by 3.36 pp in attack adoption and 3.28 pp in benign
correction from 1x to 3x. Treating unparsed outputs as non-primary outcomes is the
pre-specified intention-to-treat analysis. Among parsed outputs only, the corresponding
diagnostic effects remained positive: +11.65 pp and +15.62 pp. The main findings are
therefore not caused by a growth in parser failures. Parsed-only conditioning is
post-treatment and is not used as the causal estimand.

## Classical baseline audit

Two simple exposure references clarify what this pilot does and does not establish.

1. **Peer-only equal-weight ratio.** At fixed peer composition, scaling all peer counts
   leaves the exposure ratio unchanged and predicts a zero 3x - 1x contrast.
2. **One self vote plus equal-weight peers.** The receiver's previous state contributes
   one additional vote. Increasing peer count dilutes that self vote, so the primary
   exposure score changes even at a fixed peer ratio.

For the pooled 3x - 1x contrast, the second reference changes by +0.039 in attack
adoption and +0.112 in benign correction. These are continuous exposure-score changes,
whereas the observed +0.134 and +0.175 values are categorical transition-probability
changes. They are shown side by side but cannot be subtracted as if they used the same
response scale.

The defensible conclusion is:

> Incoming peer proportions alone are insufficient. Absolute peer volume changes the
> receiver response, but part of the direction is already expected when peer volume
> changes the balance between self inertia and social evidence.

The stronger claim that natural-language evidence volume creates an effect beyond any
classical inertia model requires either a calibrated classical response link or a new
intervention that holds effective self weight constant.

## Consequences for P0 and CTOU

The n=10 P0 result should now be interpreted narrowly:

- proportions extrapolate more stably than raw linear counts;
- this does **not** mean the true local law is ratio-only;
- raw degree likely requires a bounded/nonlinear representation and interaction with
  ratio and previous state;
- plausible candidates for the CPU model are `log(1 + degree)`, saturating splines, and
  `previous_state x ratio x volume` interactions.

Before another GPU run, these alternatives should be fitted on the existing traces and
evaluated by frozen cross-size rollout, especially n=10. A later controlled follow-up
can remove the remaining classical confound by holding receiver self weight constant
while peer evidence volume changes.

The completed explicit previous-solution ablation and its revised interpretation are
reported in `docs/pilot_evidence_volume_self_ablation_results.md`.

## Claim limits

- Message count and total input-token volume changed together.
- Distinct texts are not guaranteed to be semantically independent evidence.
- Five replicates jointly vary message set and stochastic generation; those variance
  sources are not separately identified.
- Results currently cover one model, one dataset, and one node-update prompt.
- Ratio-specific comparisons are secondary; the paired within-ratio volume contrast is
  the main causal comparison.
- The experiment measures one-step local response, not a full topology-level endpoint.

## Artifacts

The local analysis bundle is under
`artifacts/evidence-volume-intervention-v1/analysis/`. It contains the response figure,
cell summary, task-cluster contrasts, sensitivity analyses, classical exposure
comparison, compact outcomes, and an immutable analysis manifest.
