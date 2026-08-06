# OpenAI-compatible text adapter

The execution engine depends on `TextGenerator`, not on vLLM, an SDK, or an API gateway. The
`OpenAICompatibleTextGenerator` adapter sends non-streaming `POST /chat/completions` requests and
normalizes the result into `TextGenerationResult`.

## Request contract

Every node request sends:

- the configured model identifier;
- system/user messages from the versioned execution prompt;
- `max_tokens`;
- `temperature`;
- the deterministic node-round `seed`;
- `stream: false`.

The adapter records requested and returned model names, request ID, finish reason, prompt and
completion token counts, end-to-end latency, HTTP attempt count, and the raw provider response. API
keys are read from an environment variable and never added to a trace. For a private vLLM
endpoint with no authentication, omit the API-key environment setting; the adapter then sends no
`Authorization` header. Do not expose an unauthenticated endpoint publicly.

The optional `expected_returned_model` setting makes a model mismatch a terminal error. Formal
experiments should enable this check when the server returns a stable snapshot identifier. Model
aliases without a pinned returned model are suitable for protocol calibration, not for a
reproducible main experiment.

## Failure behavior

The adapter retries only connection failures, timeouts, and HTTP 408/409/429/5xx statuses, using
bounded exponential backoff. It does not switch models. Non-retryable client errors and successful
responses without usable text fail immediately. A later batch runner will persist these failures.

## Gateway calibration on 2026-08-04

Five short calls were sent through OhMyGPT solely to calibrate the adapter. The requested
`deepseek-chat` alias returned `deepseek-v4-flash`; usage, finish reason, and model fields were
present, and no HTTP retry was required.

The gateway accepted `seed`, but repeated requests with the same prompt, seed, and temperature were
not deterministic. In the discriminating nonce check, the visible outputs were `qzxvj` and `qxrzj`;
completion token counts also differed. This is consistent with the gateway documentation warning
that not all models guarantee determinism.

Consequences:

- the gateway remains acceptable for offline target-error preprocessing;
- it must not be used to claim paired common-random-number control in the topology experiment;
- the rented vLLM server must undergo the same repeated-request calibration before the pilot;
- round-zero outputs should be explicitly cached and reused across graph conditions if strict
  pairing cannot be established at the server level.

## Self-hosted server preflight

Before generating Round 0, run the versioned server probe with an identical request repeated at
least three times:

```powershell
topology-mas-probe-server `
  --base-url http://SERVER:8000/v1 `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --expected-returned-model meta-llama/Llama-3.1-8B-Instruct `
  --output runs/server-probes/llama-3.1-8b.json
```

The report records the returned model, exact output hashes, answer parsing, token-usage fields,
latency, and request IDs. `exact_repeat_observed` means only that this fixed probe repeated exactly;
it is not a general proof that all prompts are deterministic. If the vLLM server is configured with
an API key, add `--api-key-env <ENVIRONMENT_VARIABLE>`.

For the later Round 0 and batch commands, add `--no-auth` when connecting to the same private
unauthenticated endpoint. The OhMyGPT-compatible default remains authenticated.
