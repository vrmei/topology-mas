# 2026 AIME Qwen full-rationale clean-MAS status

## Frozen primary protocol

- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Sampling: `temperature=0.7`, `top_p=0.8`, `top_k=20`.
- Tasks: all 30 original 2026 AIME I/II problems.
- Graphs: homogeneous `n=5`, readout node 4, fixed `H=3`; five
  rooted-nonisomorphic graphs at each of `m=4,8,12`, plus the unique complete
  graph at `m=16`.
- Matrix: 16 graphs x 30 tasks = 480 independent clean runs.
- Each task-graph run regenerates every Round-0 node. No generated output is
  reused across graphs, tasks, batches, or conditions.
- Every normal node update makes exactly one physical model call with a maximum
  output budget of 16,384 tokens.
- A node's full raw completion is its public message. Its next update receives
  its own full previous completion and every in-neighbor's full previous
  completion. There is no summarizer, second-stage call, message compression,
  post-generation crop, or answer-only communication.

The immutable prompt version is `homogeneous-aime-full-rationale-v1`. The older
480-run bounded-message collection is retained separately as the
`AIME bounded-message ablation`; it is not the primary AIME result.

## Context and failure handling

The model configuration declares a 262,144-token native context. The vLLM
service exposes 131,072 tokens for this experiment, replacing the earlier
32,768-token serving limit. Requests use strict context mode: a context overflow
is persisted under the batch `failures/` directory and is never handled by
reducing `max_tokens`, summarizing, or cropping a peer response.

Every successful turn stores the receiver and round, token count and SHA-256 of
the receiver's previous response, sender ID/token count/SHA-256 for every
incoming response, provider prompt-token count, generated-token count, stop
reason, and explicit overflow/compression/summarization flags. The trace also
stores the complete prompt and all raw responses.

## Complete-graph stress smoke

The smoke uses 2026 AIME I Problem 15 and the `m=16` complete graph. It passed
the protocol audit:

- 1 task-graph run, 16 node turns;
- 16 logical model calls and 16 physical backend calls;
- 12 broadcast records, all byte-for-byte equal to their source raw outputs;
- 0 summarization, 0 message compression, 0 context overflow, and 0 context
  truncation;
- 327,398 input tokens and 79,810 output tokens.

Two of the 16 node generations ended with `finish_reason=length` at the frozen
16,384-token generation budget. Those responses were not post-processed: the
exact partial raw completions were stored and broadcast, and the answer parser
marked them unparsed. This is distinct from context truncation, which did not
occur.

### Actual Round-1 readout construction

- Receiver: node 4; incoming senders: nodes 0--3.
- Own previous raw response: 8,114 tokens.
- Incoming raw responses: 8,304, 16,384, 7,455, and 7,997 tokens.
- Provider-counted total prompt: 48,635 tokens.
- Generated response: 1,158 tokens; stop reason `stop`.

### Actual Round-2 readout construction

- Receiver: node 4; incoming senders: nodes 0--3.
- Own previous raw response: 1,158 tokens.
- Incoming raw responses: 1,305, 16,384, 1,522, and 1,510 tokens.
- Provider-counted total prompt: 22,259 tokens.
- Generated response: 1,584 tokens; stop reason `stop`.

The audit artifact contains the complete raw responses and actual prompt
messages for both examples, plus content hashes that prove each embedded peer
response equals the corresponding source output.

## Execution and post-processing

The 480-run main batch and its protocol audit are complete. All 480 runs
succeeded. The audit covered 6,810 turns and 4,890 broadcast messages and found
zero summarization, compression, context-overflow, context-truncation, or raw
broadcast mismatch events. Primary results are reported in
[`aime_2026_qwen_full_rationale_clean_mas_results.md`](aime_2026_qwen_full_rationale_clean_mas_results.md).
