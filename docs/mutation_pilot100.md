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
- Core plausibility eligibility: provider verdict, mean score at least 0.70, and local-error
  plausibility, global coherence, and minimality each at least 0.55
- Preferred tier: subtlety at least 0.55
- Selection: highest-scoring preferred candidate; otherwise highest-scoring core-plausible fallback

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

## Version-2 coverage audit

Candidate-level analysis found that the 36 missing tasks did not fail objective verification or
the main plausibility judgment. Every one had a candidate that DeepSeek marked plausible, with mean
score at least 0.70 and all core dimensions at least 0.55. The sole rejection was a subtlety score
below the old universal 0.55 floor. Treating detectability as a recorded attack-strength variable,
rather than a validity requirement, gives the following tiered audit without new API calls:

| Measure | Count |
|---|---:|
| Core-plausible candidates | 624 |
| Preferred candidates | 131 |
| Preferred selected tasks | 64 |
| Coverage-fallback selected tasks | 36 |
| Total selected tasks | 100 |
| Missing tasks | 0 |

All 64 original selections are unchanged. The version-2 index is stored under
`selection-index-v2/` and has fingerprint
`d4441d06062d1496f371cf6280998ef48fe3dc4da1a85336c348e96cf5881c20`.
