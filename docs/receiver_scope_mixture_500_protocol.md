# 500-task receiver-scope and incoming-mixture protocol

This protocol is fixed before inspecting receiver-stratified or mixture-stratified results. It
reuses the completed 500-task traces and makes no new LLM calls.

## Questions

1. Do the exposure, adoption, recovery, and persistence patterns observed after pooling all benign
   receivers also hold at the readout, whose last state is the system output?
2. Is lower adoption at high density descriptively explained by a lower target share among the
   parsed incoming answers, or does a density association remain after conditioning on the realized
   incoming answer mixture?

These are empirical alternatives. The analysis will not presume that the pooled mechanism holds at
the readout or that any residual association is semantic or LLM-specific.

## Units, states, and pairing

The unit is one active benign-node update at round `t >= 1` in a paired clean/attack condition. The
attacker is excluded. Every update remains paired by task, graph, node, round, experiment seed,
assignment seed, and stochastic stream.

States are:

- `C`: benchmark-correct answer;
- `T`: the frozen task-specific target error;
- `O`: another parsed answer;
- `U`: unparsed output, always separate from `O`.

Receivers are divided without overlap into:

- `internal`: benign non-readout receivers;
- `readout`: the unique readout receiver.

## Priority 1: receiver-stratified transitions

For each receiver class and `(n,m)` stratum, report:

- attack-induced exposure per eligible update;
- attack-attributed adoption:
  `P(C -> T and paired clean current != T | attack-induced target exposed)`;
- attack-induced recovery: `P(T -> C | previous attack-induced T)`;
- attack-induced persistence:
  `P(current attack-induced T | previous attack-induced T)`.

Raw descriptive C/T/O/U transitions are retained as diagnostics. Results are also stratified by
round because readout and internal nodes can have different active-round compositions.

## Priority 2: incoming answer mixture

For updates whose previous attack state is C and that receive an attack-induced target, define:

```text
target_share = #T / (#C + #T + #O)
```

`U` messages are excluded from this denominator but their count is retained. Report:

- mean `#C`, `#T`, `#O`, `#U`, parsed-message total, and target share by receiver class and density;
- adoption by exact `(T,C,O,U)` composition when adequately observed;
- adoption by predeclared target-share bins:
  `(0,.25]`, `(.25,.5)`, `.5`, `(.5,.75)`, `[.75,1)`, and `1`;
- a descriptive composition decomposition using pooled exact `(T,C,O,U)`-composition-and-round
  transition rates. Target-share bins are visualization diagnostics, not the primary decomposition.

The decomposition asks how much of the observed density difference is reproduced by changing only
the empirical mixture of incoming states under a pooled transition law. It is not a causal
mediation analysis: incoming composition is post-treatment, density changes support, and the LLM
transition law may itself differ by density.

## Regimes and statistics

Fixed `T=3` is primary. The graph-depth reconstruction is a horizon sensitivity analysis only.
Confidence intervals use 10,000 task bootstrap replicates and remain conditional on the currently
selected graphs. Adjacent-density contrasts pair task IDs. With only five selected graphs per
non-degenerate stratum, and one graph for `n5_m16`, graph-population claims are not supported.

## Decision logic

- If readout adoption decreases and/or readout recovery increases at the highest density, the
  transition pattern is directly aligned with the endpoint pattern, but is not automatically its
  complete cause.
- If only internal nodes change, the pooled transition mechanism cannot by itself explain why the
  final readout is protected.
- If target share falls and a mixture-only decomposition reproduces most of the adoption change,
  classical information mixing remains a sufficient descriptive candidate.
- If a density association remains after mixture and round stratification, it identifies an
  unexplained residual, not proof of a semantic mechanism.
