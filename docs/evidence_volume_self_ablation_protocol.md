# Previous-solution ablation for the evidence-volume intervention

Status: frozen before any no-self outcome is generated.

## Research question

Does the fixed-ratio evidence-volume effect persist when the receiver's own previous
solution is removed from its prompt?

The completed with-self experiment changed both peer volume and the relative weight of
one fixed previous solution. This paired ablation separates a peer-only volume response
from the classical explanation that additional peers dilute self inertia.

## Paired intervention

The no-self condition reuses the completed experiment's exact:

- 50 GSM8K task IDs;
- five C/error ratios and 1x/2x/3x count multipliers;
- five message-set replicates;
- nested peer stimulus IDs and anonymous display order;
- generation seeds;
- Llama-3.1-8B model, temperature 0.6, top-p 0.9, and 768-token output limit.

The only intended model-visible change is:

- remove the `YOUR_PREVIOUS_SOLUTION` section and its text;
- change `Reconsider ... using your own work and the candidate peer reasoning` to
  `Reconsider ... using the candidate peer reasoning`.

The problem, peer messages, system prompt, output contract, and sampling stay fixed.

## Outcomes and terminology

The original scenario labels are retained as paired data keys, but their no-self
interpretation changes:

- `attack_adoption`: primary outcome is peer-only target selection, `I(next=T)`;
- `benign_correction`: primary outcome is peer-only correct selection, `I(next=C)`.

Without a previous state these are not literally C-to-T adoption or O-to-C recovery.

## Estimands

For each scenario, the no-self primary contrast is:

`Delta_no-self = E[Y_no-self(3x) - Y_no-self(1x)]`.

The paired difference-in-differences is:

`Interaction = Delta_with-self - Delta_no-self`.

A positive interaction means that showing the previous solution amplifies the measured
volume response. Both estimands use task-cluster bootstrap with 10,000 resamples. The
1x/2x and 2x/3x contrasts remain secondary saturation diagnostics.

## Frozen decision rules

The smallest effect size of interest remains +/-5 percentage points.

- **Self-dilution dominated:** the no-self 3x-1x 95% interval lies wholly inside
  `[-0.05, 0.05]`, and the interaction interval is above zero.
- **Peer-volume persists:** the no-self 3x-1x interval excludes zero and is not wholly
  inside the equivalence interval. Persistence is called strong only if its lower bound
  exceeds +5 percentage points.
- **Mixed/inconclusive:** neither condition is met, or effects differ materially by
  ratio/scenario.

These rules are applied separately to target selection and correct selection. The
experiment does not assume they must share a mechanism.

## Required audits and limits

- Verify exact equality of task, ratio, multiplier, replicate, peer IDs, display order,
  and seed between with-self and no-self requests.
- Verify that no no-self rendered prompt contains the previous-solution heading or uses
  the previous stimulus ID as a peer message. Record incidental cases where the previous
  raw text is a substring of a different frozen peer message, and repeat the main
  analysis after excluding those request pairs.
- Re-run the prompt-token audit before generation.
- Report unparsed output as an unconditional secondary outcome.
- Removing self evidence changes the decision context, so the interaction identifies
  the effect of this prompt component; it does not by itself reveal attention weights.
- Message count and token volume remain coupled.
