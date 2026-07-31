# topology-mas

Controlled experiments for separating graph-structural effects from LLM update dynamics in
multi-agent systems.

The repository is being built incrementally. The current module defines validated experiment
configuration and serializable domain records; graph sampling, execution, mutation, classical
baselines, and analysis will be added as separate modules.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

No API credentials are stored in configuration files. Later provider adapters will read secret
values only from environment variables named by `model.api_key_env`.

