# Graph-independent round-zero cache

Round zero is generated before graph execution. For each task `q`, node `i`, and experiment seed
`s`, the cache stores one independent model response:

```text
I[q,i,s] = M(q; node_round_seed(q,i,s))
```

The same collection of node states is later reused across every graph and attack position. This
preserves node-level sampling diversity while preventing graph comparisons from starting with
different random answers.

## Cache identity

Every record fingerprint includes the complete task, node ID, experiment and generation seeds,
prompt text and version, requested and expected returned model names, temperature, token limit, and
cache protocol version. `graph_id` is intentionally absent. The collection manifest additionally
pins the ordered task IDs and task-collection fingerprint.

An existing manifest or record is reused only when its identity matches exactly. Otherwise the
program fails and requires a new output directory; it never overwrites or mixes experimental
conditions.

## Atomic storage

JSON is first written to a temporary file in the destination directory. The writer flushes Python's
buffer, calls `fsync`, and only then replaces the final path with `os.replace`. A crash during JSON
serialization therefore leaves the previous final file unchanged. The incomplete temporary file is
removed on handled write failures.

Atomic replacement protects individual files from partial writes. It does not make the entire
collection transactional. The manifest declares the intended number of records, and a resumed run
generates only missing records until that count is reached.

## Current scope

The cache currently supports numeric tasks and the `homogeneous-gsm8k-v1` round-zero prompt. It
stores explicit-answer parsing, correctness, token use, latency, returned model, and the provider
metadata needed for audit.

Temperature is intentionally configurable. The infrastructure does not assume that temperature
zero or positive-temperature sampling is scientifically preferable; that choice must be fixed by
the pilot protocol after measuring initial-answer diversity.

## Command

```powershell
topology-mas-generate-round-zero `
  --tasks data/prepared/gsm8k/main.jsonl `
  --output-dir runs/round-zero/<condition-id> `
  --node-count 5 `
  --seeds 0,1,2 `
  --model <served-model> `
  --expected-returned-model <pinned-returned-model> `
  --base-url http://127.0.0.1:8000/v1 `
  --api-key-env VLLM_API_KEY
```
