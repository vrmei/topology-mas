# Research direction decision record

- Decision date: 2026-08-07
- Status: active direction; revise only when new evidence or stronger prior work requires it
- Scope: homogeneous, role-free LLM multi-agent systems on directed communication graphs

## Central decision

The paper will not treat the utility--robustness curve as its primary novelty. That curve remains a
necessary descriptive measurement and a way to select informative graph pairs. The primary research
direction is instead:

> Determine how much LLM-MAS utility and robustness can be explained by static graph structure and
> classical update dynamics, then identify and causally test any remaining task-conditioned semantic
> influence mechanisms.

This direction preserves the existing execution experiment. It changes how the resulting traces are
analyzed and which small follow-up interventions are prioritized.

## Research questions

1. How much of clean utility, average robustness, worst-node robustness, and target-error propagation
   can be predicted from static graph structure and classical graph dynamics?
2. Does the static communication graph differ systematically from the task- and round-conditioned
   effective influence graph?
3. Can any residual difference be attributed to semantic adoption or message transformation rather
   than sampling noise, task difficulty, initial-state placement, or graph reachability?
4. Under which restrictions does an LLM network become adequately described by an ordinary graph
   dynamical model?

These are questions, not established claims.

## Conceptual separation

The static communication graph records which messages may be delivered. It does not by itself say
whether a receiver adopts, rejects, corrects, or rewrites a message.

Classical fixed-update dynamics can be represented schematically as:

```text
x(t + 1) = F_fixed(G, x(t))
```

The LLM execution process permits a task-, content-, and history-conditioned update:

```text
h_i(t + 1) = F_model(q, h_i(t), incoming_messages_i(t))
```

The candidate LLM-specific object is therefore an effective influence graph whose edge effects may
depend on the task, round, current belief, message content, and message ancestry. Its existence and
importance must be measured rather than assumed.

## Staged route

### Current execution order

As of 2026-08-07, all analyses that require only completed traces are executed before any new GPU
condition. The order is:

1. nonlinear and trajectory-level classical baselines;
2. node-round exposure--adoption analysis;
3. task-conditioned topology-ranking stability;
4. matched rationale ablation on the pinned local model;
5. broader model and dataset generalization.

The rationale-ablation protocol and implementation are retained, but its new inference is deferred.
This is a scheduling decision, not a change to its frozen sample or estimand.

### Stage 0: utility--robustness map

Use the existing homogeneous-agent experiment to report:

- clean utility;
- mean and worst-node robustness;
- complete node-attack vectors;
- target-error arrival, adoption, correction, and propagation;
- cost and active-round measurements.

This stage answers what happened. It is a foundation, not the principal differentiator.

### Stage 1: classical explainability

Run CPU-side baselines on the same graphs and frozen Round-zero states:

- DeGroot-style fixed-weight updating;
- local majority dynamics;
- voter or fixed-probability adoption models;
- predictors using only distance, degree, readout indegree, path redundancy, dominators, cuts,
  cycles, and strongly connected components.

Evaluate held-out prediction of node-round transitions, final correctness, target adoption, graph
ranking, and vulnerable-node ranking. Do not compare only in-sample fit.

### Stage 2: exposure--adoption decomposition

Decompose a successful attack into:

```text
target error reaches a node -> node adopts it -> transformed/adopted error reaches readout
```

Static graph features are expected to be relevant to exposure. Whether they also explain adoption
is an empirical question. Paired clean/attack traces must prevent coincidental independent errors
from being labeled as infection.

### Stage 3: minimal semantic interventions

Only after the available CPU analyses in Stages 0--2 are complete, run matched follow-up conditions
on a selected set of graph pairs:

1. answer-only messages;
2. full-rationale messages;
3. the same objectively wrong target answer with controlled differences in rationale plausibility.

Hold graph, initial states, target answer, generation stream, and decoding settings fixed wherever
the intervention permits. These experiments test whether natural-language content changes adoption
beyond the discrete answer state.

### Stage 4: boundary and generalization tests

Test surviving effects across assignment seeds, task families, and model families. Full dataset
coverage is required. The analysis must retain task-level heterogeneity rather than only aggregate
all tasks into one curve.

### Stage 5: topology design, only if justified

Topology optimization or a defense method is deferred. It becomes worthwhile only if the mechanism
analysis reveals a reproducible structural-semantic failure mode that an intervention can target.

## Degeneration toward an ordinary graph process

The LLM system should become increasingly compatible with a classical graph dynamical model when:

- messages are projected to a fixed finite state such as `correct`, `target`, or `other`;
- nodes update only from counts or fixed weights over neighboring states;
- the update kernel is time-homogeneous and task-independent;
- nodes cannot add evidence, reinterpret messages, or rewrite their provenance;
- stochastic streams and initial states are controlled.

The project should measure the boundary of this reduction. It must not assume in advance that the
full-rationale condition differs from the reduced condition.

## Evidence and claim discipline

The following statements require experimental support and must not be written as conclusions in
advance:

- LLM semantic dynamics invalidate a classical graph prediction;
- structural redundancy amplifies one error into false independent evidence;
- a distant attacker is more influential after intermediate rewriting;
- effective influence graphs vary materially across tasks;
- utility or robustness graph rankings transfer across models or datasets.

Currently defensible statements are limited to protocol properties, measured descriptive results,
and research questions supported by cited prior work.

## Kill criteria

The mechanism-first direction should be reduced or abandoned if any of the following holds after a
proper held-out analysis:

1. Fixed-update classical baselines predict node transitions, graph rankings, and vulnerable-node
   rankings within the uncertainty of the LLM measurements.
2. Answer-only and full-rationale conditions have no stable paired difference in adoption behavior.
3. Apparent residuals disappear after controlling assignment seed, stochastic replay, task
   difficulty, and target-error plausibility.
4. A residual appears for only one model or task family and has no reproducible boundary condition.

If these criteria are met, the work should be presented as a benchmark, a negative result, or a
classical topology study rather than claiming an LLM-specific mechanism.

## Decisions on candidate paths

| Path | Decision | Role in the project |
| --- | --- | --- |
| Utility--robustness frontier alone | Fold | Descriptive map and graph selection tool |
| Generic claim that LLMs differ from DeGroot | Drop | Already too broad and partially occupied |
| Static graph versus effective influence graph | Do | Primary analytical direction |
| Semantic content as a cause of target adoption | Do | Primary intervention direction |
| New topology generator or defense | Park | Reconsider only after mechanism evidence |

## Experimental invariants retained

- homogeneous model, prompt, and update protocol across nodes;
- no exogenous Planner, Auditor, or other role labels in the main experiment;
- fixed `n`, `m`, readout, and round budget within a graph stratum;
- all valid non-readout attack positions evaluated;
- objective task oracle and one frozen target error per task and condition;
- graph-independent Round-zero states;
- exact state-consistent replay for new paired experiments;
- logical model transitions reported separately from physical backend calls.

Full task coverage and 20 sampled graphs per non-degenerate architecture remain the intended final
scale. A complete graph stratum with only one possible adjacency cannot supply 20 distinct graphs;
additional repetitions there are assignment or stochastic replications, not topology samples.

## Closest prior-work boundary

- [Understanding the Information Propagation Effects of Communication Topologies in LLM-based
  Multi-Agent Systems](https://aclanthology.org/2025.emnlp-main.623/) studies counterfactual
  error/insight propagation and sparsity.
- [NetSafe](https://arxiv.org/abs/2410.15686) studies topology-oriented safety metrics and attack
  propagation.
- [ResMAS](https://arxiv.org/abs/2601.04694) optimizes topology and prompts for resilience under
  perturbations.
- [Opinion Consensus Formation Among Networked Large Language
  Models](https://arxiv.org/abs/2601.21540) compares networked LLM opinion consensus with DeGroot.
- [Characterizing Opinion Evolution of Networked LLMs](https://arxiv.org/abs/2606.18276) studies
  modifications needed for classical opinion models to fit LLM networks.

The intended differentiator is not another generic topology curve or generic DeGroot comparison. It
is objective task solving under targeted, oracle-verified error propagation, with explicit separation
of structural exposure from semantic adoption and matched interventions on the semantic channel.
