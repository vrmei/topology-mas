#!/usr/bin/env python3
"""Audit paired replay invariants, engineering outcomes, and stop-only sensitivity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


CONTRASTS = {
    "target_laundering": ("direct", "relayed"),
    "history_maturity": ("initial", "deliberated"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer")
    args = parser.parse_args()

    requests = pd.DataFrame(read_jsonl(args.prepared_dir / "requests.jsonl"))
    stimuli = {row["stimulus_id"]: row for row in read_jsonl(args.prepared_dir / "stimuli.jsonl")}
    results = pd.DataFrame(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.run_dir / "results").glob("*.json"))
    )
    failures = list((args.run_dir / "failures").glob("*.json"))
    violations: list[str] = []
    pair_rows: list[dict] = []
    for (mechanism, pair_id), group in requests.groupby(["mechanism", "pair_id"]):
        reference, candidate = CONTRASTS[mechanism]
        if len(group) != 2 or set(group.condition) != {reference, candidate}:
            violations.append(f"{pair_id}: incomplete conditions")
            continue
        left = group[group.condition.eq(reference)].iloc[0]
        right = group[group.condition.eq(candidate)].iloc[0]
        invariant_fields = ("task_id", "replicate", "correct_peer_count", "target_slot", "generation_seed")
        if any(left[field] != right[field] for field in invariant_fields):
            violations.append(f"{pair_id}: invariant field differs")
        if mechanism == "target_laundering":
            if left.previous_stimulus_id != right.previous_stimulus_id:
                violations.append(f"{pair_id}: previous response differs")
            changed = [
                index for index, (a, b) in enumerate(zip(left.peer_stimulus_ids, right.peer_stimulus_ids))
                if a != b
            ]
            if changed != [int(left.target_slot)]:
                violations.append(f"{pair_id}: peer difference is not exactly target slot")
            treatment_ids = (left.peer_stimulus_ids[int(left.target_slot)], right.peer_stimulus_ids[int(right.target_slot)])
        else:
            if left.peer_stimulus_ids != right.peer_stimulus_ids:
                violations.append(f"{pair_id}: peer messages differ")
            if left.previous_stimulus_id == right.previous_stimulus_id:
                violations.append(f"{pair_id}: previous response did not change")
            treatment_ids = (left.previous_stimulus_id, right.previous_stimulus_id)
        pair_rows.append({
            "mechanism": mechanism,
            "pair_id": pair_id,
            "correct_peer_count": int(left.correct_peer_count),
            "reference_stimulus_id": treatment_ids[0],
            "candidate_stimulus_id": treatment_ids[1],
            "reference_chars": len(stimuli[treatment_ids[0]]["raw_text"]),
            "candidate_chars": len(stimuli[treatment_ids[1]]["raw_text"]),
        })

    pairs = pd.DataFrame(pair_rows)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        token_cache = {
            stimulus_id: len(tokenizer.encode(row["raw_text"], add_special_tokens=False))
            for stimulus_id, row in stimuli.items()
        }
        pairs["reference_tokens"] = pairs.reference_stimulus_id.map(token_cache)
        pairs["candidate_tokens"] = pairs.candidate_stimulus_id.map(token_cache)

    merged = requests.merge(
        results[["request_id", "finish_reason", "input_tokens", "output_tokens", "latency_ms",
                 "is_target", "is_correct", "is_unparsed"]],
        on="request_id", how="left", validate="one_to_one",
    )
    sensitivity_rows = []
    for mechanism, (reference, candidate) in CONTRASTS.items():
        selected = merged[merged.mechanism.eq(mechanism)]
        for peers in [1, 2, "all"]:
            group = selected if peers == "all" else selected[selected.correct_peer_count.eq(peers)]
            pivot = group.pivot(index="pair_id", columns="condition",
                                values=["finish_reason", "is_target", "is_correct"])
            pivot = pivot[(pivot[("finish_reason", reference)] == "stop")
                          & (pivot[("finish_reason", candidate)] == "stop")]
            sensitivity_rows.append({
                "mechanism": mechanism,
                "correct_peer_count": peers,
                "stop_only_pairs": len(pivot),
                "target_difference": (
                    pivot[("is_target", candidate)].astype(float)
                    - pivot[("is_target", reference)].astype(float)
                ).mean(),
                "correct_difference": (
                    pivot[("is_correct", candidate)].astype(float)
                    - pivot[("is_correct", reference)].astype(float)
                ).mean(),
            })
    sensitivity = pd.DataFrame(sensitivity_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out / "pair_invariants_and_treatment_lengths.csv", index=False)
    sensitivity.to_csv(args.out / "stop_only_sensitivity.csv", index=False)
    audit = {
        "requests": len(requests),
        "results": len(results),
        "failure_artifacts": len(failures),
        "pairs": int(requests.pair_id.nunique()),
        "pair_invariant_violations": violations,
        "finish_reasons": dict(Counter(results.finish_reason)),
        "input_tokens": results.input_tokens.describe(percentiles=[0.5, 0.95]).to_dict(),
        "output_tokens": results.output_tokens.describe(percentiles=[0.5, 0.95]).to_dict(),
        "latency_ms": results.latency_ms.describe(percentiles=[0.5, 0.95]).to_dict(),
    }
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Replay audit", "",
        f"- Requests/results: {len(requests)}/{len(results)}",
        f"- Technical failures: {len(failures)}",
        f"- Pair invariant violations: {len(violations)}",
        f"- Finish reasons: {dict(Counter(results.finish_reason))}", "",
        "## Treatment length audit", "",
    ]
    length_columns = [column for column in ("reference_tokens", "candidate_tokens", "reference_chars", "candidate_chars") if column in pairs]
    lines.append("```text")
    lines.append(
        pairs.groupby("mechanism")[length_columns].median().round(1)
        .rename_axis(None).to_string()
    )
    lines.append("```")
    lines += ["", "## Stop-only sensitivity", "", "```text", sensitivity.to_string(index=False), "```", ""]
    (args.out / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"violations": len(violations), "results": len(results), "failures": len(failures)}))


if __name__ == "__main__":
    main()
