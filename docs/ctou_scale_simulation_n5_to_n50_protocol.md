# CTOU n=5–50 model-based scale simulation protocol

## Scope and claim boundary

The output is a **simulation / model-based extrapolation**, not a measurement
of true LLM performance at `n>10`. Real LLM traces exist at `n=5,6,7,8,10`;
only `n=11..50` are extrapolated. The protocol remains conditional on
Llama-3.1-8B, GSM8K-50, horizon `H=3`, the existing synchronous schedule, and
one persistent non-readout attacker.

The experiment has a gate. No `n>10` surface is interpreted until both the local
transition law and the correlated Round-0 initializer are evaluated against
the real `n=6,7,8,10` anchors.

## Phase 1A: local-law selection

All candidate laws use previous CTOU state and round. They differ only in how
incoming evidence is represented:

1. `proportions`: C/T/O/U proportions;
2. `absolute_counts`: raw C/T/O/U counts;
3. `counts_plus_proportions`: both representations;
4. `proportions_log1p_volume`: proportions, `log(1+d)`, and interactions;
5. `proportions_saturating_volume_k{1,2,4}`: proportions,
   `d/(d+k)`, and interactions.

Volume interactions include `proportion × volume` and
`previous-state × volume`. Models are multinomial logistic regressions with
the same regularization and fold partition.

For task fold `f`, training contains only `n=5` rows whose task is not in
`f`. Evaluation contains the same held-out tasks at a larger size. Conditions
are fitted separately so the comparison concerns feature extrapolation rather
than a clean/attack domain-mixture choice.

Model selection uses `n=6,7,8` only. The primary score is multiclass log loss,
averaged equally over task, system size, and condition. Brier score,
classification error, calibration, and the worst size-condition cell are
secondary diagnostics. `n=10` is a stress check and does not change the
predefined selection score. A candidate is marked unstable if its `n=10` log
loss is non-finite or exceeds the proportions baseline by more than 25%.

This phase selects a local response model, not a complete topology surrogate.
One-step evaluation uses realized composition and is post-treatment. Recursive
endpoint validation with observed Round 0 is reported separately before full
simulation.

Before Phase 2, the selected law is rolled out recursively on the real
`n=6,7,8,10` graphs with observed categorical Round 0. This is a safeguard,
not a second model-selection step. It must not increase the mean graph-level
Utility/Robustness MAE by more than 10% relative to the proportions law, and
its Utility/Robustness graph Spearman must not be negative at any anchor size.

## Phase 1B: correlated Round-0 initializer

Round-0 groups are extracted from clean cases. One group is one
`task × independent run` vector, not one expanded graph/attacker row. Clean
normal-node states are modeled as C/O/U; T is introduced only by the persistent
attacker during attack simulation.

For task `q` and run `g`:

```text
theta_qg ~ Dirichlet(kappa * mu_q)
K_qg | theta_qg ~ Multinomial(n, theta_qg)
```

`mu_q` is estimated from `n=5` with shrinkage toward the global state
distribution. The shared concentration `kappa` is estimated from `n=5`
group-level compositions by maximum likelihood. This preserves both task
difficulty and within-run node correlation.

Validation on `n=6,7,8,10` reports:

- state-proportion error;
- mean and variance of the correct fraction;
- all-correct and majority-correct rates;
- task-level correct-fraction MAE and Spearman;
- Wasserstein distance between observed and posterior-predictive correct
  fractions.

The initializer passes the descriptive gate only if, averaged over
`n=6,7,8,10`, it reduces both correct-fraction Wasserstein distance and
absolute correct-fraction variance error by at least 10% relative to the IID
global-state baseline, while the maximum absolute mean-correct-fraction bias
is at most 3 percentage points. These thresholds are frozen before running the
validation and are not changed retroactively.

## Phase 2: frozen simulation

Two versions are produced:

- **strict**: local law and Round-0 initializer use `n=5` only;
- **calibrated**: parameters use real `n=5,6,7,8,10`, then extrapolate only
  `n=11..50`.

For each `n`, edge levels are derived from normalized excess density

```text
delta = (m - (n-1)) / ((n-1)^2 - (n-1))
```

and deduplicated after integer rounding. Average degree `m/(n-1)` is retained
as a second scale variable. Graphs have no self-loops, no readout outgoing
edges, exactly `m` edges, and every node reaches readout within `H=3`.

Backbone-plus-extra-edge construction is only an initializer: by itself it is
not uniform over legal graphs. The primary simulator therefore records the
proposal distribution explicitly and uses symmetric legal edge swaps for
mixing before retaining a graph. Multiple starts and graph-feature diagnostics
are required; claims are conditional on the sampled graph distribution.

Primary scanning uses mean-field probability propagation. Particle validation
uses 2,048 particles at selected sizes and sparse/medium/dense edge levels.
The outputs are Utility `U`, Robustness `R`, attack penalty `U-R`, communication
gain `U-U0`, target risk, and uncertainty components from graph sampling,
tasks, Round-0 parameters, and the local-law model envelope.

The frozen particle grid is `n={5,10,15,20,30,40,50}` and
`delta={0.0,0.5,1.0}`, using graph index zero at each cell. All attacker
positions are checked at `n<=10`; at larger sizes, five deterministic attacker
positions are checked to keep the validation diagnostic bounded. Particle and
mean-field estimates use the identical graph, task, attacker subset, local law,
and Round-0 model. The approximation gate requires mean absolute endpoint
error at most 0.03 for both Utility and Robustness and maximum aggregate cell
error at most 0.05. This gate diagnoses the plug-in approximation; it does not
validate the extrapolated local law against a real LLM at `n>10`.

CPU runs freeze `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`. This is an execution control,
not a model change: uncapped BLAS pools caused severe process-level
oversubscription under the server's 14-core cgroup quota.

## Stop conditions

Do not interpret `n>10` curves if:

- all candidate local laws fail badly or diverge at the real `n=10` anchor;
- the Round-0 model collapses to an IID-like distribution that misses observed
  cross-agent overdispersion;
- mean-field and particle rollouts materially disagree at the validation grid;
- admissible local laws give qualitatively incompatible trends immediately
  beyond `n=10`.
