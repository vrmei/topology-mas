# Evidence-volume response curve: pilot results

Experiment: `evidence-volume-response-curve-v1`  
Model: `meta-llama/Llama-3.1-8B-Instruct`  
Dataset boundary: 40 preselected GSM8K tasks with sufficient natural C/T/O
stimulus support  
Requests: 13,840 completed, 0 failed  
Inference: temperature 0.6, top-p 0.9, maximum output 768 tokens

## 1. What this experiment identifies

This is a one-step receiver intervention. It identifies how the probability of
the receiver's next answer changes when the number of distinct peer rationales
is increased while their C:T or C:O composition ratio is fixed. It does not
directly estimate a complete MAS endpoint or a topology-level robustness score.

## 2. Attack adoption increases with absolute evidence volume

The primary attack outcome is selection of the fixed target error T. The
receiver either retains a previous correct solution (`include`) or receives
only peer messages (`omit`).

| Peer composition | Previous C | Lowest degree | Target rate | Degree 30 | Maximum degree | Target rate |
|---|---:|---:|---:|---:|---:|---:|
| 50% C / 50% T | included | 2 | 14.5% | 78.0% | 50 | 80.5% |
| 50% C / 50% T | omitted | 2 | 32.5% | 83.0% | 50 | 86.5% |
| 67% C / 33% T | included | 3 | 11.5% | 46.5% | 48 | 49.5% |
| 67% C / 33% T | omitted | 3 | 26.5% | 58.0% | 48 | 62.5% |
| 80% C / 20% T | included | 5 | 6.0% | 17.5% | 50 | 19.5% |
| 80% C / 20% T | omitted | 5 | 16.0% | 21.0% | 50 | 27.0% |

Holding the peer-state ratio fixed does not hold the Llama transition
probability fixed. The largest changes occur at low and medium degrees. The
effect is strongest when T constitutes 50% of peer messages and much weaker at
20% T.

This rejects a ratio-only local transition law for this model and prompt over
the tested range. It does not establish that every additional peer is an
independent evidence source.

## 3. The volume effect is not only dilution of the receiver's old answer

Removing the receiver's previous correct solution increases target adoption at
almost every matched degree. At low and medium degrees the increase is often
10--20 percentage points. This confirms a substantial inertia or self-anchor
effect.

However, target adoption still rises with peer count after the previous answer
is removed. For example, in the 50/50 condition it rises from 32.5% at degree 2
to 86.5% at degree 50. Therefore, increasing peer volume does more than merely
dilute one explicit self message.

The with-self/no-self gap generally narrows at high volume, but does not vanish
uniformly. The previous solution should consequently be modeled as a privileged
state input rather than as one ordinary C message.

## 4. Attack-side high-volume growth diminishes but is not proven to saturate

The preregistered pooled contrast from degree 30 to the ratio-specific maximum
is:

| Previous solution | High-tail change | 95% task-cluster bootstrap CI |
|---|---:|---:|
| Included | +2.5 pp | [-1.3, +6.5] pp |
| Omitted | +4.7 pp | [+1.2, +7.8] pp |

Every ratio-specific high-tail interval includes zero. The pooled no-self
contrast is positive, but small relative to the earlier low-degree changes.
Under the frozen five-percentage-point smallest effect of interest, both curves
are classified as `diminishing_but_unresolved`, not as fast saturation.

The data support a flattening attack response. They do not support either a
strict plateau or a linear continuation to arbitrarily high degree.

## 5. Message count has an effect beyond total token volume

The token-matched control compares 4 long rationales (2C+2T) with 8 short
rationales (4C+4T), with no previous answer. The mean peer-token totals are
1,469.3 and 1,478.9 respectively.

Eight short messages increase target selection by 12.2 percentage points over
four long messages (95% task-cluster bootstrap CI: [+2.8, +21.1] pp; 36 tasks).
The task-level contrast is positive for 21 tasks, zero for 9, and negative for
6.

This rules out total text volume as a sufficient explanation in this matched
condition. The estimand still combines message multiplicity, natural rationale
diversity, and per-message length. It does not identify independent human-like
sources because sender identity is hidden and semantic argument quality is not
matched.

## 6. Benign correction follows a different high-volume law

When the receiver starts from O, the outcome is correction to C.

| Peer composition | Lowest-degree correction | Degree 30 | Maximum-degree correction |
|---|---:|---:|---:|
| 50% C / 50% O | 40.8% at d=2 | 62.5% | 74.2% at d=50 |
| 67% C / 33% O | 49.2% at d=3 | 84.2% | 81.7% at d=48 |

For 50/50 C/O, degree 30 to 50 adds 11.7 points, with a 95% interval of
[+0.8, +23.3] points. Benign correction therefore has not saturated on this
curve. For 67/33 C/O, correction reaches 90.0% at degree 39 and then falls to
81.7% at degree 48; the paired 39-to-48 contrast is -8.3 points with a 95%
interval of [-15.0, -2.5] points.

This is evidence against a single monotone degree effect shared by all CTOU
transitions. The high-degree decrease is a pilot result, not yet a general
claim about context crowding or overcommunication.

## 7. Output-format failures do not explain the main curves

Unparsed output is more common at low degree and falls as peer count increases.
For example, the no-self 50/50 attack condition falls from 24.0% unparsed at
degree 2 to 1.0% at degree 50. This means raw state probabilities partly include
a format-compliance change.

The primary analysis keeps unparsed responses in the denominator, as frozen.
A parsed-only sensitivity analysis preserves the substantive response:

- with previous C, 50/50 target selection changes from 17.7% to 83.5%;
- without the previous answer, it changes from 40.9% to 87.4%;
- with previous C, 80/20 changes from 6.7% to 19.5%.

The magnitude at the low endpoint changes, but the absolute-volume response and
high-end flattening remain.

## 8. Consequences for extrapolation

Models fitted only on the earlier low-degree support were evaluated at held-out
higher degrees. Averaged equally over the six attack ratio/previous-answer
cells:

| Link family | Mean held-out log loss | Mean Brier | Mean absolute calibration error |
|---|---:|---:|---:|
| bounded `d/(d+4)` | 0.58 | 0.20 | 0.05 |
| bounded `d/(d+2)` | 0.59 | 0.20 | 0.05 |
| log(1+d) | 0.60 | 0.20 | 0.06 |
| ratio only | 0.71 | 0.25 | 0.21 |
| raw degree | 0.99 | 0.24 | 0.17 |

The bounded family has the best aggregate extrapolation, while the raw linear
degree link strongly overpredicts several high-volume conditions. No fixed k is
best in every ratio. These comparisons use the same tasks at new degrees, so
they test degree extrapolation rather than transfer to unseen tasks or models.

For CTOU simulation beyond n=10, the current evidence supports using a bounded
or log-volume link and propagating functional-form uncertainty. It does not yet
justify treating the degree-50 response as the asymptotic law for complete
n=50 MAS rollouts.

## 9. Claims supported now

Supported within this intervention boundary:

1. At fixed state proportions, absolute distinct-message count changes the
   Llama receiver's local transition probability.
2. This effect persists without an explicit previous answer and is therefore
   not reducible to self-answer dilution.
3. Matched total peer-token volume does not remove the message-count effect in
   the tested 50/50 attack condition.
4. Attack adoption flattens at high volume, but complete saturation is not
   established.
5. Attack adoption and benign correction do not share one simple monotone
   volume law.

Not supported by this experiment alone:

- a topology-level utility or robustness conclusion;
- an n=50 complete-MAS endpoint claim;
- a universal response law across tasks, prompts, or models;
- the claim that each message acts as statistically independent evidence;
- a semantic mechanism explaining why additional messages change the response.
