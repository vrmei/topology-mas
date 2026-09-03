# AIME summary protocol v1 audit

Source: the complete 1,920-response Qwen Round-0 pool generated for the 2026 AIME
summary-clean pilot.

## Main rates

| Metric | Count | Rate |
|---|---:|---:|
| Structurally valid dual-channel output | 1,614 / 1,920 | 84.06% |
| Exact parsed-answer preservation, conditional on structural validity | 1,534 / 1,614 | 95.04% |
| CTOU-state preservation, conditional on structural validity | 1,544 / 1,614 | 95.66% |
| Exact parsed-answer preservation, conditional on a parseable full answer | 1,519 / 1,543 | 98.44% |
| End-to-end answer-preserving interface | 1,534 / 1,920 | 79.90% |
| End-to-end state-preserving interface | 1,544 / 1,920 | 80.42% |

The conditional rates answer whether a structurally valid summary preserves its source.
The end-to-end rates additionally count malformed or missing dual-channel outputs as
interface failures. Both must be reported; the conditional rate alone overstates the
reliability of the current one-call protocol.

## Files

- `protocol_manifest.json`: machine-readable prompts, prompt hashes, model, sampling,
  parser, validation rules, and cross-node invariants.
- `fidelity_audit.json`: counts, rates, CTOU transition table, and validation failures.
- `examples_full.md`: ten human-readable exact full-response to summary examples
  (4 C, 4 O, 2 U).
- `examples_full.jsonl`: the same exact examples for programmatic inspection.

## Interpretation boundary

This audit freezes and measures the protocol; it does not establish semantic-equivalence
between the complete reasoning and its summary. The parser validates answer/state
fidelity and the communication boundary, not whether every decisive proof step was
preserved. The observed 79.90% end-to-end validation rate is below a reasonable formal
experiment gate, so the failed 480-run attempt must not be analyzed as a completed clean
baseline.
