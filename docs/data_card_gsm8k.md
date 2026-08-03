# GSM8K data card for the topology pilot

## Source and version

- Dataset: GSM8K, from the official OpenAI `grade-school-math` repository.
- Pinned Git commit: `3101c7d5072418e28b9008a6636bde82a006892c`.
- License: MIT in the source repository.
- Official sizes: 7,473 training examples and 1,319 test examples.
- Repository status observed on 2026-08-03: archived, so the commit pin remains important even
  though upstream is read-only.

The repository does not redistribute GSM8K. The preparation command downloads the two official
JSONL files directly and accepts them only after both SHA-256 and line-count checks pass.

| Split | SHA-256 |
|---|---|
| train | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| test | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

## Record conversion

Each official record contains `question` and `answer`. The loader:

1. preserves the complete raw answer in metadata;
2. extracts the numeric value following the final `####` marker;
3. removes GSM8K `<<expression=result>>` calculator annotations from the stored reference rationale;
4. assigns a stable ID from the source split and zero-based source line;
5. rejects malformed, empty, or duplicate-question records.

The reference rationale and final answer are available only to offline preprocessing and objective
evaluation. They must not be inserted into the prompts of normal MAS nodes during clean or attack
runs.

## Pilot selection

- Mutation calibration: deterministic sample from `train`, namespace `mutation-calibration`.
- Main topology pilot: deterministic sample from `test`, namespace `topology-main-study`.
- Default seed: 0.

Selection ranks every task by SHA-256 of `(seed, namespace, task_id)`. This avoids dependence on
Python RNG implementation details and makes selection invariant to input ordering. The generated
manifest records task IDs and a fingerprint over the complete selected records.

## Intended use

GSM8K is initially used to validate numeric answer parsing, target-error mutation, synchronous MAS
execution, and graph-versus-semantic analysis. It is not sufficient evidence for generalization to
other mathematical reasoning, code, or open-ended tasks.

## Known risks and reporting requirements

- Modern LLMs may have encountered GSM8K during pretraining. Results measure system dynamics on a
  reproducible benchmark, not unseen-problem generalization.
- The official test split is reserved for the main pilot; mutation thresholds and implementation
  debugging use training examples.
- A model-generated plausibility judgment is not ground truth. Human audit size, agreement, and
  disagreement handling must be reported before the full study.
- Report exact source commit, file hashes, selection seed, namespaces, task IDs or selection
  fingerprints, excluded records, and all mutation failure counts.
