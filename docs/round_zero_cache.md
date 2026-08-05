# Graph-independent round-zero cache

Round zero is generated before graph execution. For each task `q`, anonymous replica slot `k`, and
experiment seed `s`, the cache stores one independent model response:

```text
I[q,k,s] = M(q; round_zero_replica_seed(q,k,s))
```

Here `k` is an independent replica slot, not a structural graph-node identifier. A separate,
recorded permutation maps replica slots onto graph nodes. The model never sees either identifier.
The same collection and assignment are later reused across every paired graph and attack position.
This preserves sampling diversity while preventing graph comparisons from starting with different
random answers. Multiple assignment seeds can be used to average over initial-state placement.

## Cache identity

Every record fingerprint includes the complete task, replica slot, experiment and generation seeds,
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

The cache currently supports numeric tasks and the `homogeneous-gsm8k-v2` round-zero prompt. It
stores explicit-answer parsing, correctness, token use, latency, returned model, and the provider
metadata needed for audit.

The v2 prompt requires a plain-text `FINAL_ANSWER: <number>` final line. The parser accepts only
explicit answer markers, including conservative provider formatting variants such as
`Final Answer:`, Markdown-wrapped markers, and a standalone LaTeX `\\boxed{}` answer. It never
guesses from an unmarked trailing number. Parser changes that affect stored classifications require
a cache-protocol version bump; the current version is `round-zero-cache-v3`.

Temperature is intentionally configurable. The infrastructure does not assume that temperature
zero or positive-temperature sampling is scientifically preferable; that choice must be fixed by
the pilot protocol after measuring initial-answer diversity.

## Command

```powershell
topology-mas-generate-round-zero `
  --tasks data/prepared/gsm8k/main.jsonl `
  --output-dir runs/round-zero/<condition-id> `
  --replica-count 5 `
  --seeds 0,1,2 `
  --model <served-model> `
  --expected-returned-model <pinned-returned-model> `
  --base-url http://127.0.0.1:8000/v1 `
  --api-key-env VLLM_API_KEY
```
