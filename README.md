# topology-mas

Controlled experiments for separating graph-structural effects from LLM update dynamics in
multi-agent systems.

The repository is being built incrementally. It currently contains validated experiment records,
an auditable offline GSM8K target-error mutation pipeline, constrained graph sampling, and a
provider-neutral synchronous MAS execution kernel. Provider adapters, classical baselines, and
analysis will be added as separate modules.

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
