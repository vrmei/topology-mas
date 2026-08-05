# GSM8K 100-task mutation batch record

## Input

- Dataset: official GSM8K test split
- Upstream revision: `3101c7d5072418e28b9008a6636bde82a006892c`
- Sampling: deterministic 100-task sample, seed 0
- Input fingerprint: `f1e7e3ccd845590d7858de84692eb87f54b9a661246a605cf338012d0503005c`

## Frozen protocol

- Generator: requested `gpt-5.6-sol`
- Plausibility Oracle: requested `deepseek-chat`; provider returned `deepseek-v4-flash`
- Candidates per task: 8
- Wrongness: deterministic arithmetic and final-answer Oracle
- Plausibility eligibility: provider verdict, mean score at least 0.70, and every dimension at
  least 0.55
- Selection: highest eligible score with deterministic tie-breaking

## Canonical audited result

| Measure | Count |
|---|---:|
| Tasks attempted | 100 |
| Task results present | 100 |
| Task-level errors | 0 |
| Candidates generated | 800 |
| Objective-Oracle passes | 800 |
| Plausibility-eligible candidates | 131 |
| Tasks with one frozen error | 64 |
| Tasks with no eligible candidate | 36 |
| Candidate processing errors | 14 |

Selected-answer index fingerprint:
`6cd56124ffb51eacb64792fb5a9401979353360e4b4ab50e7e38300079b08872`.

The trusted counts above come from a clean, single-process cache reload followed by an independent
artifact audit. Intermediate live summaries produced while an unintended residual batch process
was still writing to the same directory are invalid and must not be cited.

## Local artifacts

The full raw cache is intentionally ignored by Git because it contains large provider responses.
In the experiment workspace it is stored under:

```text
data/prepared/gsm8k-pilot100-seed0/main.jsonl
runs/mutations-gsm8k-pilot100-seed0-v1/
|-- batch_manifest.json
|-- batch_summary.json
|-- tasks/
`-- selection-index/
    |-- audit.json
    `-- selected_adversarial_answers.jsonl
```

Later experiments should consume only `selected_adversarial_answers.jsonl`, while retaining the
full task directories for traceability.

## Interpretation

This batch establishes a cache of 64 strictly accepted target errors from 100 attempted tasks. It
does not yet establish that every selected mutation is persuasive to humans. A blinded human audit
sample remains necessary before treating the plausibility score as validated measurement.
