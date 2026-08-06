# Qwen2.5-7B Round-zero calibration

## Purpose

This is a model gate before any topology run. It measures output parseability, independent-replica
accuracy, and answer diversity on the exact 64 GSM8K tasks with frozen adversarial answers. It does
not estimate topology utility or robustness.

## Frozen configuration

- Requested model: `TA/Qwen/Qwen2.5-7B-Instruct-Turbo`
- Returned model: `Qwen/Qwen2.5-7B-Instruct-Turbo`
- Provider: OhMyGPT routing to TogetherAI
- Task count: 64
- Anonymous replicas per task: 5
- Experiment seeds: `0`
- Temperature: `0.6`
- Maximum output tokens: 768
- Prompt: `homogeneous-gsm8k-v2`
- Cache protocol: `round-zero-cache-v3`
- Task JSONL SHA-256: `967ef84c8efb2d2cf53067b635e370c4d257c8e386d5b6beec7775650db45b49`

Local cache:

```text
runs/round-zero/qwen2p5-7b-pilot64-r5-s0-temp0p6-promptv2-cachev3/
```

## Results

| Measure | Result |
|---|---:|
| Records | 320 |
| Parsed records | 320 (100%) |
| Correct records | 304 (95.0%) |
| Tasks with 5/5 correct replicas | 55 |
| Tasks with mixed correct/incorrect replicas | 8 |
| Tasks with 0/5 correct replicas | 1 |
| Tasks with more than one parsed answer | 9 |
| Natural matches to the frozen target error | 0 records |
| Input tokens | 54,020 |
| Output tokens | 65,906 |
| Median request latency | 2,156 ms |

The correct-replica-count distribution over tasks was:

| Correct replicas | Task count |
|---:|---:|
| 0/5 | 1 |
| 3/5 | 3 |
| 4/5 | 5 |
| 5/5 | 55 |

## Interpretation boundary

The provider adapter and explicit-answer contract passed after calibration. The model is suitable
for an attack-propagation sanity run because clean accuracy is high and the frozen target errors did
not occur naturally. It is not yet justified as the sole model for estimating a clean
utility--robustness frontier: only 9 of 64 tasks exhibited any independent-answer disagreement, so
the clean-utility axis may have limited resolution. A topology experiment is required before making
any claim about whether this ceiling materially suppresses graph-level differences.

At positive temperature, three repeated requests with the same seed produced different surface
forms through the gateway. Round-zero caching therefore remains mandatory. Post-Round-zero API
inference should initially use temperature zero; this is a control choice, not evidence that the
gateway is deterministic for every prompt.
