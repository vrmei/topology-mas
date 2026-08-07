# topology-mas

Controlled experiments for separating graph-structural effects from LLM update dynamics in
multi-agent systems.

The repository is being built incrementally. It currently contains validated experiment records,
an auditable offline GSM8K target-error mutation pipeline, constrained graph sampling, a
provider-neutral synchronous MAS execution kernel, an OpenAI-compatible adapter, and a resumable
paired batch runner. Classical baselines and analysis will be added as separate modules.

The active research framing and its evidence gates are recorded in
[the research direction decision](docs/research_direction.md). Utility--robustness mapping remains a
foundation; the primary direction is to separate static graph effects from task-conditioned semantic
influence without assuming an LLM-specific mechanism in advance.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

No API credentials are stored in configuration files. Later provider adapters will read secret
values only from environment variables named by `model.api_key_env`.

## Prepare the pinned GSM8K subsets

The loader downloads the official files at a fixed Git commit, verifies SHA-256 and line counts,
then makes deterministic calibration and main-study subsets. Raw and prepared data remain local.

```powershell
topology-mas-prepare-gsm8k `
  --calibration-count 20 `
  --main-count 50 `
  --seed 0
```

Calibration examples come from the official training split; the topology pilot comes from the
official test split. See [the GSM8K data card](docs/data_card_gsm8k.md).

## Offline target-error mutation

The first mutation protocol intentionally supports only one fault family: a single wrong
arithmetic result whose value is propagated consistently through the rest of a numeric solution.

- `gpt-5.6-sol` generates a fixed number of structured candidates.
- A local restricted-AST oracle verifies all arithmetic and accepts exactly one declared mismatch.
- `deepseek-chat` evaluates plausibility only after objective verification.
- Candidate selection uses a fixed local rule over four DeepSeek dimensions.
- All requests, raw responses, candidates, rejection reasons, model snapshots, and the selected
  target error are persisted under `runs/`, which is excluded from Git.

Set `OHMYGPT_API_KEY` in the environment and run the included synthetic smoke task:

```powershell
python -m topology_mas.mutation.cli `
  --task examples/gsm8k_smoke_task.json `
  --output-dir runs/mutation-smoke `
  --candidate-count 4
```

See [the mutation pipeline documentation](docs/mutation_pipeline.md) for the schema, Oracle rules,
selection policy, and known limitations.

Run mutation over a prepared task manifest:

```powershell
topology-mas-mutate-batch `
  --tasks data/prepared/gsm8k/calibration.jsonl `
  --output-dir runs/gsm8k-calibration `
  --candidate-count 8
```

The output directory is resume-safe. A terminal result—including “no eligible candidate”—is
cached and never silently regenerated. Changing the tasks or mutation configuration requires a new
output directory.

## Sample controlled directed topologies

Sample 20 distinct labeled graphs with five nodes, eight directed edges, readout node 4, and at
most three communication hops to readout:

```powershell
topology-mas-sample-graphs `
  --node-count 5 `
  --edge-count 8 `
  --readout-node 4 `
  --max-rounds 3 `
  --graph-count 20 `
  --seed 0 `
  --output-dir data/prepared/graphs/n5_m8_t3_seed0
```

The sampler has no performance reward. It proposes fixed-edge graphs and conditions only on
readout reachability and round depth. See [the topology sampling protocol](docs/topology_sampling.md).

## Execute a graph synchronously

The execution kernel enforces round snapshots, one-edge-per-round delivery, a homogeneous update
prompt, deterministic per-node seeds, target-error replay, and readout-cone pruning. It currently
depends on a narrow `TextGenerator` protocol. An OpenAI-compatible adapter now connects that
protocol to vLLM or a gateway without changing execution semantics. See
[the execution protocol](docs/execution_protocol.md) and
[the model-adapter protocol](docs/model_adapter.md).

Generate graph-independent round-zero answers before the topology runs:

```powershell
topology-mas-generate-round-zero `
  --tasks data/prepared/gsm8k/main.jsonl `
  --output-dir runs/round-zero/pilot-v1 `
  --replica-count 5 `
  --seeds 0,1,2 `
  --model Qwen/Qwen3-8B `
  --expected-returned-model Qwen/Qwen3-8B `
  --base-url http://127.0.0.1:8000/v1 `
  --api-key-env VLLM_API_KEY
```

Each task-replica-seed answer is atomically cached. A recorded assignment permutation later maps
anonymous replicas to structural nodes; neither identity is exposed to the model. See [the
round-zero cache protocol](docs/round_zero_cache.md).

## Run the paired graph experiment

The batch runner materializes the full clean and per-node attack plan before execution, validates all
Round-zero and mutation inputs, writes each trace atomically, and resumes only missing cells:

```powershell
topology-mas-run-batch `
  --tasks data/prepared/gsm8k/main.jsonl `
  --graphs data/prepared/graphs-v2/n5_m8_t3_seed0/graphs.jsonl `
  --round-zero-dir runs/round-zero/pilot-v1 `
  --mutations-dir runs/mutations/pilot-v1 `
  --output-dir runs/execution/n5_m8_t3 `
  --experiment-seeds 0,1,2 `
  --assignment-seeds 0,1,2 `
  --model deepseek-chat `
  --expected-returned-model deepseek-v4-flash
```

For paired counterfactual runs, exact post-Round-zero states can optionally share one realized
stochastic transition. This is an experimental common-random-numbers policy, not approximate prompt
caching: the complete messages, generation seed, decoding settings, prompt version, namespace, and
pinned model fingerprint must all match.

```powershell
--state-replay-cache-dir runs/state-replay/pilot-v1 `
--state-replay-model-fingerprint <64-character-sha256> `
--state-replay-namespace llama31-8b-pilot-v1
```

See [the paired batch protocol](docs/batch_execution.md) for matrix size, preflight checks, cache
identity, and cost semantics.

## Audit a completed scale pilot

The first post-hoc stage validates every task–graph and task–graph–attack-node cell, then reports
descriptive estimates with a crossed graph-by-task bootstrap. It does not fit a graph mechanism or
test a classical dynamics baseline.

```powershell
python scripts/analyze_scale_pilot_descriptive.py `
  --run-root runs/scale-pilot100-g5-r8-temp03-v1 `
  --output-dir runs/scale-pilot100-g5-r8-temp03-v1/posthoc-descriptive-v1 `
  --bootstrap-replicates 2000 `
  --seed 20260807
```

See [the descriptive analysis protocol](docs/descriptive_analysis_protocol.md) for estimands,
resampling units, and claim boundaries.

## Test classical structural explainability

The first mechanism-oriented analysis asks how well static directed-graph features predict the
effect of targeting a non-readout node. It uses nested leave-one-graph-out validation so that no
node from a held-out graph is used for feature scaling or ridge-penalty selection.

```powershell
python scripts/analyze_classical_structure.py `
  --run-root runs/scale-pilot100-g5-r8-temp03-v1 `
  --output-dir runs/scale-pilot100-g5-r8-temp03-v1/posthoc-classical-structure-v1 `
  --bootstrap-replicates 2000 `
  --seed 20260807
```

See [the classical structural explainability protocol](docs/classical_structure_analysis_protocol.md)
for features, baselines, validation, and claim boundaries.

## Compare with classical graph dynamics

Run parameter-free finite-state dynamics from the exact frozen Round-zero node answers. The
analysis pairs frozen-state, inertial-majority, and equal-weight DeGroot outcomes with the recorded
LLM clean and targeted-attack outcomes.

```powershell
python scripts/analyze_classical_dynamics.py `
  --run-root runs/scale-pilot100-g5-r8-temp03-v1 `
  --output-dir runs/scale-pilot100-g5-r8-temp03-v1/posthoc-classical-dynamics-v1 `
  --bootstrap-replicates 2000 `
  --seed 20260807
```

See [the classical dynamics protocol](docs/classical_dynamics_protocol.md) for state projection,
update rules, tie handling, metrics, and claim boundaries.

## Test one global DeGroot susceptibility

Sweep a fixed damping grid and select one susceptibility parameter using leave-one-entire-graph-out
validation. Selection uses target-error induction only; accuracy drop and clean utility remain
untuned secondary tests.

```powershell
python scripts/analyze_damped_degroot.py `
  --run-root runs/scale-pilot100-g5-r8-temp03-v1 `
  --output-dir runs/scale-pilot100-g5-r8-temp03-v1/posthoc-damped-degroot-v1 `
  --bootstrap-replicates 2000 `
  --seed 20260807
```

See [the damped DeGroot protocol](docs/damped_degroot_protocol.md) for the update equation,
held-out-graph selection rule, endpoint checks, and claim boundaries. The first completed pilot is
summarized in [the damped DeGroot pilot result](docs/pilot_damped_degroot_results.md).

## Calibrate continuous classical exposure

Predict task-level LLM target-error adoption from Round-zero categorical states and/or continuous
equal-weight DeGroot target mass. The analysis uses strict crossed graph-and-task holdout.

```powershell
python scripts/analyze_conditional_classical_exposure.py `
  --run-root runs/scale-pilot100-g5-r8-temp03-v1 `
  --output-dir runs/scale-pilot100-g5-r8-temp03-v1/posthoc-conditional-exposure-v1 `
  --bootstrap-replicates 2000 `
  --seed 20260807
```

See [the conditional exposure protocol](docs/conditional_classical_exposure_protocol.md) and
[the first pilot result](docs/pilot_conditional_exposure_results.md).

## Run the matched rationale-ablation pilot

Freeze a result-independent 20-task sample and one graph from each existing `(n,m)` stratum:

```powershell
python scripts/prepare_rationale_ablation_pilot.py `
  --source-run-root runs/scale-pilot100-g5-r8-temp03-v1 `
  --output-dir runs/rationale-ablation-pilot20-v1/prepared `
  --task-count 20
```

With the pinned local Llama vLLM endpoint running, execute only the new answer-only attacks. Clean
traces are copied from and validated against the completed pilot.

```powershell
python scripts/run_rationale_ablation_pilot.py `
  --project-root . `
  --prepared-dir runs/rationale-ablation-pilot20-v1/prepared `
  --output-dir runs/rationale-ablation-pilot20-v1/answer-only `
  --base-url http://127.0.0.1:8000/v1 `
  --max-workers 16
```

After all eight strata complete, compare the new runs with the pinned full-rationale traces:

```powershell
python scripts/analyze_rationale_ablation_pilot.py `
  --prepared-dir runs/rationale-ablation-pilot20-v1/prepared `
  --answer-only-run-root runs/rationale-ablation-pilot20-v1/answer-only `
  --output-dir runs/rationale-ablation-pilot20-v1/paired-analysis-v1 `
  --bootstrap-replicates 2000 `
  --seed 20260807
```

See [the frozen rationale-ablation protocol](docs/rationale_ablation_pilot_protocol.md) for the
estimand, sampling rule, integrity gates, and claim boundary.
