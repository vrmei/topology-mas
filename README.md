# topology-mas

Controlled experiments for separating graph-structural effects from LLM update dynamics in
multi-agent systems.

The repository is being built incrementally. It currently contains validated experiment records
and an auditable, offline GSM8K target-error mutation pipeline. Graph sampling, MAS execution,
classical baselines, and analysis will be added as separate modules.

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
